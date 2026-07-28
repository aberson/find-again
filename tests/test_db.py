"""Tests for the SQLite/FTS5 storage layer (plan.md Step 2, 5 db.py, 9 smoke).

Covers the done-when conditions:

* one real document writes and reads back through FTS (round-trip),
* a failed write rolls back and preserves the prior index (metadata <-> FTS
  never diverge),
* schema upgrades are deterministic + idempotent, and a newer DB is refused,
* an FTS5-less build fails with a clean error.
"""

from __future__ import annotations

import contextlib
import sqlite3
from collections.abc import Iterator
from pathlib import Path

import pytest

from find_again import db as dbmod
from find_again.db import (
    DB_DIRNAME,
    DB_FILENAME,
    Database,
    FTS5UnavailableError,
    MigrationError,
    NewerDatabaseError,
    apply_migrations,
    current_schema_version,
    default_db_path,
    latest_migration_version,
)
from find_again.models import ArtifactType, IndexedDocument, content_hash


def _doc(
    *,
    doc_id: str = "docs/lessons-learned.md",
    source_path: str = "docs/lessons-learned.md",
    body: str = "never dump secret file contents from a bash pipeline",
    artifact_type: ArtifactType = ArtifactType.MARKDOWN,
    project: str = "find-again",
    timestamp: str = "2026-07-27T00:00:00Z",
    record_locator: str | None = None,
) -> IndexedDocument:
    return IndexedDocument(
        doc_id=doc_id,
        source_path=source_path,
        artifact_type=artifact_type,
        project=project,
        timestamp=timestamp,
        content_hash=content_hash(body.encode("utf-8")),
        body=body,
        record_locator=record_locator,
    )


@pytest.fixture
def store() -> Iterator[Database]:
    database = Database.open_memory()
    try:
        yield database
    finally:
        database.close()


# --------------------------------------------------------------------------- #
# Open / migrate
# --------------------------------------------------------------------------- #
def test_open_memory_migrates_to_latest(store: Database) -> None:
    assert store.schema_version == latest_migration_version()
    assert latest_migration_version() >= 1
    names = {
        row["name"]
        for row in store.connection.execute(
            "SELECT name FROM sqlite_master WHERE type IN ('table','view')"
        )
    }
    assert "documents" in names
    assert "documents_fts" in names


def test_default_db_path() -> None:
    root = Path("/some/root")
    assert default_db_path(root) == root / DB_DIRNAME / DB_FILENAME


def test_open_creates_find_again_dir_and_persists(tmp_path: Path) -> None:
    db_path = default_db_path(tmp_path)
    assert not db_path.parent.exists()
    with Database.open(db_path) as database:
        database.upsert_document(_doc())
    assert db_path.parent.is_dir()
    assert db_path.name == DB_FILENAME
    assert db_path.is_file()

    # Reopen: the write persisted and the reopen is an idempotent no-op migrate.
    with Database.open(db_path) as reopened:
        assert reopened.schema_version == latest_migration_version()
        got = reopened.get_document("docs/lessons-learned.md")
        assert got is not None
        assert got.body == "never dump secret file contents from a bash pipeline"


# --------------------------------------------------------------------------- #
# Write / read round-trip through FTS
# --------------------------------------------------------------------------- #
def test_write_read_roundtrip_through_fts(store: Database) -> None:
    doc = _doc()
    store.upsert_document(doc)

    # Read back by doc_id (metadata + body reconstructed).
    got = store.get_document(doc.doc_id)
    assert got == doc

    # Query the FTS body and get the document back.
    hits = store.match_documents("secret")
    assert [h.doc_id for h in hits] == [doc.doc_id]
    assert hits[0].body == doc.body
    assert hits[0].artifact_type is ArtifactType.MARKDOWN

    # A term not in the body matches nothing.
    assert store.match_documents("nonexistentterm") == []
    assert store.count_documents() == 1
    assert store.check_consistency()


def test_record_locator_and_enum_roundtrip(store: Database) -> None:
    doc = _doc(
        doc_id="runs/telemetry.jsonl::L42",
        source_path="runs/telemetry.jsonl",
        record_locator="L42",
        artifact_type=ArtifactType.JSONL,
        body="token usage levers investigation entry",
    )
    store.upsert_document(doc)
    got = store.get_document("runs/telemetry.jsonl::L42")
    assert got is not None
    assert got.record_locator == "L42"
    assert got.artifact_type is ArtifactType.JSONL
    assert got == doc


def test_get_missing_returns_none(store: Database) -> None:
    assert store.get_document("nope.md") is None


def test_upsert_replaces_body_in_metadata_and_fts(store: Database) -> None:
    store.upsert_document(_doc(body="first body alpha"))
    store.upsert_document(_doc(body="second body beta"))

    assert store.count_documents() == 1
    got = store.get_document("docs/lessons-learned.md")
    assert got is not None
    assert got.body == "second body beta"

    # The FTS index reflects the new body, not the stale one.
    assert store.match_documents("beta")
    assert store.match_documents("alpha") == []
    assert store.check_consistency()


def test_multiple_documents_match_shared_and_unique_terms(store: Database) -> None:
    store.upsert_document(_doc(doc_id="a.md", source_path="a.md", body="shared token alpha"))
    store.upsert_document(_doc(doc_id="b.md", source_path="b.md", body="shared token beta"))

    assert {h.doc_id for h in store.match_documents("shared")} == {"a.md", "b.md"}
    assert [h.doc_id for h in store.match_documents("alpha")] == ["a.md"]
    assert store.count_documents() == 2
    assert store.check_consistency()


# --------------------------------------------------------------------------- #
# Delete
# --------------------------------------------------------------------------- #
def test_delete_removes_metadata_and_fts(store: Database) -> None:
    doc = _doc()
    store.upsert_document(doc)
    assert store.delete_document(doc.doc_id) is True

    assert store.get_document(doc.doc_id) is None
    assert store.match_documents("secret") == []
    assert store.count_documents() == 0
    assert store.check_consistency()

    # Deleting again is a clean no-op.
    assert store.delete_document(doc.doc_id) is False


# --------------------------------------------------------------------------- #
# Rollback preserves the prior index (metadata <-> FTS consistency)
# --------------------------------------------------------------------------- #
def test_rollback_on_fresh_insert_preserves_prior_index(
    store: Database, monkeypatch: pytest.MonkeyPatch
) -> None:
    prior = _doc(doc_id="a.md", source_path="a.md", body="prior committed body")
    store.upsert_document(prior)

    # Force the FTS half of the write to fail *after* the metadata insert.
    def boom(_self: Database, _rowid: int, _body: str) -> None:
        raise RuntimeError("injected mid-transaction failure")

    monkeypatch.setattr(Database, "_insert_fts", boom)

    with pytest.raises(RuntimeError, match="injected mid-transaction failure"):
        store.upsert_document(_doc(doc_id="b.md", source_path="b.md", body="doomed body"))

    # The failed write left NO partial row, and the prior document is intact.
    monkeypatch.undo()
    assert store.count_documents() == 1
    assert store.get_document("b.md") is None
    survivor = store.get_document("a.md")
    assert survivor is not None
    assert survivor == prior
    assert store.match_documents("prior") == [prior]
    assert store.match_documents("doomed") == []
    assert store.check_consistency()


def test_rollback_on_update_preserves_old_version(
    store: Database, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = _doc(body="original body keeps standing")
    store.upsert_document(original)

    def boom(_self: Database, _rowid: int, _body: str) -> None:
        raise RuntimeError("injected failure during re-index")

    monkeypatch.setattr(Database, "_insert_fts", boom)

    with pytest.raises(RuntimeError):
        store.upsert_document(_doc(body="replacement body that never lands"))

    monkeypatch.undo()
    # The prior version survives whole; the delete-then-insert rolled back.
    got = store.get_document("docs/lessons-learned.md")
    assert got is not None
    assert got == original
    assert store.match_documents("original") == [original]
    assert store.match_documents("replacement") == []
    assert store.count_documents() == 1
    assert store.check_consistency()


# --------------------------------------------------------------------------- #
# Deterministic + idempotent migrations; newer DB refused
# --------------------------------------------------------------------------- #
def _schema_snapshot(conn: sqlite3.Connection) -> list[tuple[str, str, str]]:
    return sorted(
        (row[0], row[1], row[2] or "")
        for row in conn.execute(
            "SELECT type, name, sql FROM sqlite_master WHERE name NOT LIKE 'sqlite_%'"
        )
    )


def test_migration_fresh_and_older_reach_identical_schema() -> None:
    latest = latest_migration_version()
    assert latest >= 2, "determinism test needs the two-step (metadata, fts) upgrade"

    # Fresh DB: migrate straight to latest.
    fresh = sqlite3.connect(":memory:", isolation_level=None)
    fresh.row_factory = sqlite3.Row
    assert apply_migrations(fresh) == latest
    fresh_schema = _schema_snapshot(fresh)

    # Older DB: stop at v1 (metadata only, no FTS), then upgrade the rest of the way.
    older = sqlite3.connect(":memory:", isolation_level=None)
    older.row_factory = sqlite3.Row
    assert apply_migrations(older, target_version=1) == 1
    older_names = {row["name"] for row in older.execute("SELECT name FROM sqlite_master")}
    assert "documents" in older_names
    assert "documents_fts" not in older_names  # FTS arrives only in v2

    assert apply_migrations(older) == latest
    assert _schema_snapshot(older) == fresh_schema

    fresh.close()
    older.close()


def test_migration_reapply_is_a_noop() -> None:
    conn = sqlite3.connect(":memory:", isolation_level=None)
    conn.row_factory = sqlite3.Row
    latest = apply_migrations(conn)
    before = _schema_snapshot(conn)

    # Re-running at the latest version changes nothing and does not error.
    assert apply_migrations(conn) == latest
    assert current_schema_version(conn) == latest
    assert _schema_snapshot(conn) == before
    conn.close()


def test_newer_database_is_refused() -> None:
    conn = sqlite3.connect(":memory:", isolation_level=None)
    conn.row_factory = sqlite3.Row
    apply_migrations(conn)
    conn.execute(f"PRAGMA user_version = {latest_migration_version() + 1}")

    with pytest.raises(NewerDatabaseError):
        apply_migrations(conn)
    conn.close()


def test_open_refuses_newer_database(tmp_path: Path) -> None:
    db_path = default_db_path(tmp_path)
    Database.open(db_path).close()
    # Bump the on-disk version past what this build supports.
    raw = sqlite3.connect(str(db_path), isolation_level=None)
    raw.execute(f"PRAGMA user_version = {latest_migration_version() + 5}")
    raw.close()

    with pytest.raises(NewerDatabaseError):
        Database.open(db_path)


def test_migration_target_beyond_latest_is_error() -> None:
    conn = sqlite3.connect(":memory:", isolation_level=None)
    conn.row_factory = sqlite3.Row
    with pytest.raises(MigrationError):
        apply_migrations(conn, target_version=latest_migration_version() + 1)
    conn.close()


# --------------------------------------------------------------------------- #
# FTS5 missing -> clean error
# --------------------------------------------------------------------------- #
def test_fts5_missing_raises_clean_error(monkeypatch: pytest.MonkeyPatch) -> None:
    # Simulate a SQLite build without FTS5.
    monkeypatch.setattr(dbmod, "_probe_fts5", lambda _conn: False)
    with pytest.raises(FTS5UnavailableError, match="FTS5"):
        Database.open_memory()


# --------------------------------------------------------------------------- #
# Bootstrap failure releases the connection / file lock (no leak on Windows)
# --------------------------------------------------------------------------- #
def test_open_fts5_missing_does_not_lock_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db_path = default_db_path(tmp_path)

    with monkeypatch.context() as m:
        m.setattr(dbmod, "_probe_fts5", lambda _conn: False)
        with pytest.raises(FTS5UnavailableError):
            Database.open(db_path)

    # The failed open closed its connection: on Windows a leaked handle would make
    # unlink() raise PermissionError. Any file created by the aborted open deletes.
    if db_path.exists():
        db_path.unlink()

    # And the same path opens cleanly now that FTS5 is "available" again.
    with Database.open(db_path) as database:
        assert database.schema_version == latest_migration_version()


def test_open_newer_database_does_not_lock_file(tmp_path: Path) -> None:
    db_path = default_db_path(tmp_path)
    Database.open(db_path).close()
    # Bump the on-disk version past what this build supports.
    raw = sqlite3.connect(str(db_path), isolation_level=None)
    raw.execute(f"PRAGMA user_version = {latest_migration_version() + 5}")
    raw.close()

    with pytest.raises(NewerDatabaseError):
        Database.open(db_path)

    # The refusal closed its connection: the DB file is not locked, so it deletes
    # (a leaked handle would raise PermissionError on Windows).
    db_path.unlink()
    assert not db_path.exists()


# --------------------------------------------------------------------------- #
# Migration atomicity: a mid-apply failure rolls the whole migration back
# --------------------------------------------------------------------------- #
def test_migration_failure_mid_apply_rolls_back_to_prior_version(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A poisoned v2 whose SECOND statement fails *after* the first schema change.
    poisoned = (
        (1, "0001_a.sql", "CREATE TABLE m1 (id INTEGER);"),
        (2, "0002_b.sql", "CREATE TABLE m2 (id INTEGER);\nINSERT INTO nonexistent VALUES (1);"),
    )
    monkeypatch.setattr(dbmod, "_load_migrations", lambda: poisoned)

    conn = sqlite3.connect(":memory:", isolation_level=None)
    conn.row_factory = sqlite3.Row
    assert apply_migrations(conn, target_version=1) == 1  # v1 lands cleanly

    with pytest.raises(sqlite3.OperationalError):
        apply_migrations(conn)  # v2's second statement fails mid-transaction

    # The v2 transaction rolled back wholesale: prior user_version + prior schema.
    assert current_schema_version(conn) == 1
    names = {row["name"] for row in conn.execute("SELECT name FROM sqlite_master")}
    assert "m1" in names
    assert "m2" not in names  # v2's partial change (the CREATE TABLE m2) is gone

    # Re-running with a fixed v2 applies cleanly the rest of the way.
    fixed = (
        (1, "0001_a.sql", "CREATE TABLE m1 (id INTEGER);"),
        (2, "0002_b.sql", "CREATE TABLE m2 (id INTEGER);"),
    )
    monkeypatch.setattr(dbmod, "_load_migrations", lambda: fixed)
    assert apply_migrations(conn) == 2
    names = {row["name"] for row in conn.execute("SELECT name FROM sqlite_master")}
    assert "m2" in names
    conn.close()


def test_migration_with_trigger_body_and_string_semicolons_applies() -> None:
    # A multi-statement migration whose trigger body and string literals contain
    # semicolons -- the exact shape a naive ;-splitter would mangle. executescript
    # hands it to SQLite's parser, so it applies as authored.
    migration_sql = (
        "CREATE TABLE t (id INTEGER PRIMARY KEY, note TEXT);\n"
        "CREATE TRIGGER trg AFTER INSERT ON t BEGIN\n"
        "  UPDATE t SET note = 'semi;colon;value' WHERE id = NEW.id;\n"
        "  UPDATE t SET note = note || ';more' WHERE id = NEW.id;\n"
        "END;\n"
        "INSERT INTO t (id) VALUES (1);"
    )
    conn = sqlite3.connect(":memory:", isolation_level=None)
    conn.row_factory = sqlite3.Row
    dbmod._apply_migration(conn, 1, migration_sql)

    assert current_schema_version(conn) == 1
    names = {row["name"] for row in conn.execute("SELECT name FROM sqlite_master")}
    assert {"t", "trg"} <= names
    # The trigger (with its ;-laden body) fired and both UPDATEs ran.
    assert conn.execute("SELECT note FROM t WHERE id = 1").fetchone()[0] == "semi;colon;value;more"
    conn.close()


# --------------------------------------------------------------------------- #
# FTS5 special-character robustness (metacharacter body + parameterized MATCH)
# --------------------------------------------------------------------------- #
_FTS_METACHARS_BODY = (
    'body with "double quotes" and AND OR NEAR NOT operators, a star* and caret^, '
    'a column:filter colon, a NEAR/2 form, and a bare " quote'
)


def test_fts_metacharacter_body_stored_and_read_back_intact(store: Database) -> None:
    doc = _doc(doc_id="meta.md", source_path="meta.md", body=_FTS_METACHARS_BODY)
    store.upsert_document(doc)  # must not raise despite FTS5 metacharacters in body

    got = store.get_document("meta.md")
    assert got is not None
    assert got.body == _FTS_METACHARS_BODY  # stored verbatim, no mangling
    assert got == doc
    assert store.check_consistency()

    # A later plain-term MATCH still works and finds the metachar doc.
    assert [h.doc_id for h in store.match_documents("operators")] == ["meta.md"]


def test_match_query_is_parameterized_no_injection(store: Database) -> None:
    store.upsert_document(
        _doc(doc_id="safe.md", source_path="safe.md", body="ordinary content here")
    )

    # A malicious match string is BOUND as a parameter, never executed as SQL.
    # FTS5 may reject it as a query-syntax error, but it must not run injected SQL.
    malicious = 'content"; DROP TABLE documents; --'
    with contextlib.suppress(sqlite3.OperationalError):
        store.match_documents(malicious)

    # The table still exists and the row is intact -> no injection occurred.
    assert store.count_documents() == 1
    assert store.get_document("safe.md") is not None
    assert store.check_consistency()


def test_upsert_replace_with_metachar_body_drops_old_from_fts(store: Database) -> None:
    store.upsert_document(_doc(doc_id="m.md", source_path="m.md", body="alpha uniqueoldterm"))
    store.upsert_document(_doc(doc_id="m.md", source_path="m.md", body=_FTS_METACHARS_BODY))

    assert store.count_documents() == 1
    assert store.match_documents("uniqueoldterm") == []  # old body no longer searchable
    assert [h.doc_id for h in store.match_documents("operators")] == ["m.md"]
    assert store.check_consistency()


def test_delete_metachar_body_removes_metadata_and_fts(store: Database) -> None:
    store.upsert_document(_doc(doc_id="m.md", source_path="m.md", body=_FTS_METACHARS_BODY))
    assert store.delete_document("m.md") is True

    assert store.get_document("m.md") is None
    assert store.match_documents("operators") == []  # FTS row removed too
    assert store.count_documents() == 0
    assert store.check_consistency()
