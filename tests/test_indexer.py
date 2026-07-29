"""Tests for incremental indexing + reconciliation (plan.md Step 4, §6).

Covers the done-when conditions:

* unchanged files are SKIPPED (no re-parse, no re-write),
* a CHANGED file re-indexes (new content searchable, old gone),
* a DELETED file's documents are removed on refresh,
* a JSONL that shrank reconciles its stale records away,
* an INTERRUPTED refresh leaves the index consistent (prior or cleanly-advanced,
  never torn),
* excluded / secret files never enter the index, and
* diagnostics are preserved and content-free.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

import pytest

import find_again.indexer as indexer_mod
from find_again.adapters import select_adapter_for as real_select_adapter_for
from find_again.config import Config
from find_again.db import Database
from find_again.indexer import refresh_index

# A live AWS-shaped access-key id (fixture only) the content scanner must catch. It
# is the sentinel we assert never reaches the index or a diagnostic message.
SECRET_TOKEN = "AKIAIOSFODNN7EXAMPLE"  # noqa: S105 -- test fixture, not a real credential


def _config(root: Path, roots: tuple[str, ...] = ("docs",), **over: object) -> Config:
    return Config(
        root=root,
        roots=roots,
        exclude=tuple(over.get("exclude", ())),  # type: ignore[arg-type]
        max_file_kb=int(over.get("max_file_kb", 512)),  # type: ignore[arg-type]
        schema_version=1,
    )


def _write(root: Path, rel: str, text: str) -> None:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


@pytest.fixture
def store() -> Iterator[Database]:
    database = Database.open_memory()
    try:
        yield database
    finally:
        database.close()


def _refresh(config: Config, store: Database) -> indexer_mod.RefreshResult:
    return refresh_index(config, store, use_git_ignore=False)


# --------------------------------------------------------------------------- #
# Unchanged files are skipped (no re-parse, no re-write)
# --------------------------------------------------------------------------- #
def test_unchanged_files_are_skipped_without_reparsing(
    tmp_path: Path, store: Database, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write(tmp_path, "docs/a.md", "# Alpha\n\nunique_alpha content\n")
    _write(tmp_path, "docs/b.jsonl", '{"note": "unique_beta"}\n')
    config = _config(tmp_path)

    calls = {"n": 0}

    def spy(source: object) -> object:
        calls["n"] += 1
        return real_select_adapter_for(source)  # type: ignore[arg-type]

    monkeypatch.setattr(indexer_mod, "select_adapter_for", spy)

    first = _refresh(config, store)
    assert first.indexed == 2
    assert first.skipped == 0
    parsed_first = calls["n"]
    assert parsed_first == 2  # both files routed to an adapter

    # Snapshot the stored hashes so we can prove they are untouched by the no-op run.
    hashes_before = {doc_id: h for doc_id, _sp, h in store.iter_document_index()}

    calls["n"] = 0
    second = _refresh(config, store)

    # Nothing changed on disk: every file is skipped, and NOTHING is re-parsed.
    assert second.skipped == 2
    assert second.indexed == 0
    assert second.updated == 0
    assert second.deleted == 0
    assert calls["n"] == 0  # select_adapter_for never reached -> no re-parse
    assert {doc_id: h for doc_id, _sp, h in store.iter_document_index()} == hashes_before


# --------------------------------------------------------------------------- #
# A changed file re-indexes: new content searchable, old content gone
# --------------------------------------------------------------------------- #
def test_changed_file_reindexes_new_content_and_drops_old(tmp_path: Path, store: Database) -> None:
    _write(tmp_path, "docs/note.md", "old_marker_alpha lives here\n")
    config = _config(tmp_path)
    _refresh(config, store)
    assert store.match_documents("old_marker_alpha")

    _write(tmp_path, "docs/note.md", "new_marker_beta replaced it\n")
    result = _refresh(config, store)

    assert result.updated == 1
    assert result.indexed == 0
    assert store.match_documents("new_marker_beta")  # new content searchable
    assert store.match_documents("old_marker_alpha") == []  # old content gone
    assert store.count_documents() == 1
    assert store.check_consistency()


# --------------------------------------------------------------------------- #
# A deleted file's documents are removed on refresh
# --------------------------------------------------------------------------- #
def test_deleted_file_documents_removed_on_refresh(tmp_path: Path, store: Database) -> None:
    _write(tmp_path, "docs/keep.md", "keep_me_marker\n")
    _write(tmp_path, "docs/gone.md", "delete_me_marker\n")
    config = _config(tmp_path)
    _refresh(config, store)
    assert store.count_documents() == 2

    (tmp_path / "docs" / "gone.md").unlink()
    result = _refresh(config, store)

    assert result.deleted == 1
    assert store.get_document("docs/gone.md") is None
    assert store.get_document("docs/keep.md") is not None
    assert store.match_documents("delete_me_marker") == []
    assert store.count_documents() == 1
    assert store.check_consistency()


# --------------------------------------------------------------------------- #
# A JSONL that shrank reconciles its stale records away
# --------------------------------------------------------------------------- #
def test_shrunk_jsonl_reconciles_stale_records(tmp_path: Path, store: Database) -> None:
    _write(
        tmp_path,
        "docs/data.jsonl",
        '{"v": "rec_one"}\n{"v": "rec_two"}\n{"v": "rec_three"}\n',
    )
    config = _config(tmp_path)
    _refresh(config, store)
    assert store.count_documents() == 3
    assert store.get_document("docs/data.jsonl::L3") is not None

    # Shrink to a single record.
    _write(tmp_path, "docs/data.jsonl", '{"v": "rec_one"}\n')
    result = _refresh(config, store)

    assert result.deleted == 2  # L2 + L3 reconciled away
    assert result.updated == 1
    assert store.get_document("docs/data.jsonl::L1") is not None
    assert store.get_document("docs/data.jsonl::L2") is None
    assert store.get_document("docs/data.jsonl::L3") is None
    assert store.match_documents("rec_three") == []
    assert store.count_documents() == 1
    assert store.check_consistency()


# --------------------------------------------------------------------------- #
# An interrupted refresh leaves a consistent index (prior or cleanly-advanced)
# --------------------------------------------------------------------------- #
def test_interrupted_refresh_leaves_consistent_state(
    tmp_path: Path, store: Database, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write(tmp_path, "docs/a.md", "a_before marker\n")
    _write(tmp_path, "docs/b.md", "b_before marker\n")
    _write(tmp_path, "docs/c.md", "c_before marker\n")
    config = _config(tmp_path)
    _refresh(config, store)
    assert store.count_documents() == 3

    # Change all three; sorted order is a.md, b.md, c.md.
    _write(tmp_path, "docs/a.md", "a_after unique_a_advanced\n")
    _write(tmp_path, "docs/b.md", "b_after unique_b_doomed\n")
    _write(tmp_path, "docs/c.md", "c_after unique_c_untouched\n")

    original_insert_fts = Database._insert_fts

    def boom(self: Database, rowid: int, body: str) -> None:
        # Fail mid-transaction while b.md is being re-indexed (after a.md committed,
        # before c.md is reached).
        if "unique_b_doomed" in body:
            raise RuntimeError("injected mid-file failure")
        original_insert_fts(self, rowid, body)

    monkeypatch.setattr(Database, "_insert_fts", boom)

    with pytest.raises(RuntimeError, match="injected mid-file failure"):
        _refresh(config, store)

    monkeypatch.undo()

    # The index is never torn: metadata and FTS stay one-to-one, count unchanged.
    assert store.check_consistency()
    assert store.count_documents() == 3

    # a.md cleanly advanced; b.md rolled back to prior; c.md never touched (prior).
    assert store.match_documents("unique_a_advanced")
    assert store.match_documents("a_before") == []
    assert store.match_documents("unique_b_doomed") == []
    assert store.match_documents("b_before")  # prior content survives intact
    assert store.match_documents("unique_c_untouched") == []
    assert store.match_documents("c_before")

    # A clean re-refresh completes the interrupted advance.
    result = _refresh(config, store)
    assert store.match_documents("unique_b_doomed")
    assert store.match_documents("unique_c_untouched")
    assert result.deleted == 0
    assert store.check_consistency()


# --------------------------------------------------------------------------- #
# Excluded / secret files never enter the index
# --------------------------------------------------------------------------- #
def test_secret_and_excluded_files_never_enter_index(tmp_path: Path, store: Database) -> None:
    _write(tmp_path, "docs/clean.md", "ordinary indexable content\n")
    # Path-glob deny (never even read): a .env file.
    _write(tmp_path, "docs/app.env", f"API_TOKEN={SECRET_TOKEN}\n")
    # Content-scan deny: a Markdown file carrying an AWS-shaped key.
    _write(tmp_path, "docs/creds.md", f"here is a key: {SECRET_TOKEN}\n")
    config = _config(tmp_path)

    result = _refresh(config, store)

    # Only the clean file is indexed.
    assert store.count_documents() == 1
    assert store.get_document("docs/clean.md") is not None
    assert store.get_document("docs/app.env") is None
    assert store.get_document("docs/creds.md") is None

    # The secret text reached NEITHER a stored body NOR any diagnostic message.
    for doc_id, _sp, _h in store.iter_document_index():
        body = store.get_document(doc_id)
        assert body is not None
        assert SECRET_TOKEN not in body.body
    assert store.match_documents("AKIAIOSFODNN7EXAMPLE") == []

    codes = {d.code for d in result.diagnostics}
    assert "secret-path-glob" in codes  # docs/app.env
    assert "secret-content" in codes  # docs/creds.md
    for diag in result.diagnostics:
        assert SECRET_TOKEN not in diag.message  # content-free diagnostics


def test_path_glob_excluded_secret_is_never_read(
    tmp_path: Path, store: Database, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Finding 2: the path-glob deny layer runs BEFORE any read, so a path-denied
    # secret file's bytes never load into memory. Spy on every read and assert the
    # .env file is never among them, while the clean file IS read.
    _write(tmp_path, "docs/clean.md", "ordinary indexable content\n")
    _write(tmp_path, "docs/app.env", f"API_TOKEN={SECRET_TOKEN}\n")
    config = _config(tmp_path)

    real_read_bytes = Path.read_bytes
    read_paths: list[str] = []

    def spy_read(self: Path) -> bytes:
        read_paths.append(self.name)
        return real_read_bytes(self)

    monkeypatch.setattr(Path, "read_bytes", spy_read)
    _refresh(config, store)
    monkeypatch.undo()

    assert "app.env" not in read_paths  # path-denied bytes never read
    assert "clean.md" in read_paths  # a clean file is still read + indexed
    assert store.get_document("docs/app.env") is None


# --------------------------------------------------------------------------- #
# Diagnostics are preserved (persisted) and replaced each refresh
# --------------------------------------------------------------------------- #
def test_diagnostics_persist_and_are_replaced(tmp_path: Path, store: Database) -> None:
    _write(tmp_path, "docs/clean.md", "clean content\n")
    _write(tmp_path, "docs/creds.md", f"secret {SECRET_TOKEN}\n")
    config = _config(tmp_path)
    _refresh(config, store)

    # Diagnostics survive the refresh so `status` can read them back.
    persisted = store.get_diagnostics()
    assert any(d.code == "secret-content" for d in persisted)
    assert all(SECRET_TOKEN not in d.message for d in persisted)

    # Remove the offending file; the next refresh REPLACES the diagnostic set.
    (tmp_path / "docs" / "creds.md").unlink()
    _refresh(config, store)
    assert store.get_diagnostics() == []


# --------------------------------------------------------------------------- #
# Index age: a successful refresh stamps last_refreshed
# --------------------------------------------------------------------------- #
def test_refresh_stamps_last_refreshed(tmp_path: Path, store: Database) -> None:
    _write(tmp_path, "docs/a.md", "content\n")
    config = _config(tmp_path)
    fixed = datetime(2026, 7, 28, 12, 0, 0, tzinfo=UTC)
    result = refresh_index(config, store, now=fixed, use_git_ignore=False)

    assert result.refreshed_at == "2026-07-28T12:00:00Z"
    assert store.get_meta("last_refreshed") == "2026-07-28T12:00:00Z"


# --------------------------------------------------------------------------- #
# A configured root that does not exist surfaces a diagnostic (not silent)
# --------------------------------------------------------------------------- #
def test_missing_root_emits_diagnostic(tmp_path: Path, store: Database) -> None:
    config = _config(tmp_path, roots=("docs", "nonexistent"))
    _write(tmp_path, "docs/a.md", "content\n")
    result = _refresh(config, store)

    missing = [d for d in result.diagnostics if d.code == "missing-root"]
    assert len(missing) == 1
    assert missing[0].source_path == "nonexistent"
    assert store.count_documents() == 1


# --------------------------------------------------------------------------- #
# Finding 1: a PRESENT-but-unreadable file keeps its prior docs (not purged) and
# diagnoses; a genuinely-deleted file IS purged. A transient Windows file lock or
# permission error must never drop a file's search coverage.
# --------------------------------------------------------------------------- #
def test_present_but_unreadable_file_keeps_docs_and_diagnoses(
    tmp_path: Path, store: Database, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write(tmp_path, "docs/locked.md", "locked_marker stays indexed\n")
    _write(tmp_path, "docs/gone.md", "gone_marker will vanish\n")
    config = _config(tmp_path)
    _refresh(config, store)
    assert store.count_documents() == 2
    assert store.match_documents("locked_marker")

    # locked.md becomes transiently unreadable (a Windows lock / permission blip);
    # gone.md is genuinely deleted.
    (tmp_path / "docs" / "gone.md").unlink()
    real_read_bytes = Path.read_bytes

    def flaky_read(self: Path) -> bytes:
        if self.name == "locked.md":
            raise OSError("file is locked by another process")
        return real_read_bytes(self)

    monkeypatch.setattr(Path, "read_bytes", flaky_read)
    result = _refresh(config, store)
    monkeypatch.undo()

    # The present-but-unreadable file KEEPS its prior documents (never purged) and
    # emits a content-free unreadable-input diagnostic naming only its path.
    assert store.match_documents("locked_marker")  # coverage preserved
    assert store.get_document("docs/locked.md") is not None
    unreadable = [d for d in result.diagnostics if d.code == "unreadable-input"]
    assert len(unreadable) == 1
    assert unreadable[0].source_path == "docs/locked.md"
    assert "locked_marker" not in unreadable[0].message  # no content leak

    # The genuinely-deleted file IS purged (only its one document removed).
    assert result.deleted == 1
    assert store.get_document("docs/gone.md") is None
    assert store.match_documents("gone_marker") == []
    assert store.count_documents() == 1
    assert store.check_consistency()


# --------------------------------------------------------------------------- #
# Finding 5: a file that WAS indexed clean but now becomes EXCLUDED has its prior
# documents removed on refresh (and the secret/denied bytes never enter).
# --------------------------------------------------------------------------- #
def test_file_that_gains_a_secret_has_prior_docs_removed(tmp_path: Path, store: Database) -> None:
    _write(tmp_path, "docs/notes.md", "clean_marker ordinary content\n")
    config = _config(tmp_path)
    _refresh(config, store)
    assert store.match_documents("clean_marker")
    assert store.count_documents() == 1

    # The file gains a secret -> the content-scan layer now excludes it, so its
    # prior documents reconcile away and the secret never enters the index.
    _write(tmp_path, "docs/notes.md", f"clean_marker plus a key {SECRET_TOKEN}\n")
    result = _refresh(config, store)

    assert result.deleted == 1
    assert store.get_document("docs/notes.md") is None  # prior docs removed
    assert store.match_documents("clean_marker") == []
    assert store.match_documents("AKIAIOSFODNN7EXAMPLE") == []  # secret never indexed
    assert store.count_documents() == 0
    codes = {d.code for d in result.diagnostics}
    assert "secret-content" in codes
    for diag in result.diagnostics:
        assert SECRET_TOKEN not in diag.message  # content-free diagnostics
    assert store.check_consistency()


def test_file_that_becomes_deny_globbed_has_prior_docs_removed(
    tmp_path: Path, store: Database
) -> None:
    _write(tmp_path, "docs/wip.md", "draft_marker content\n")
    _refresh(_config(tmp_path), store)
    assert store.count_documents() == 1

    # The operator adds a deny glob that now matches the file -> its prior documents
    # are removed on refresh, and the path layer excludes it before any read.
    result = _refresh(_config(tmp_path, exclude=("docs/wip.md",)), store)

    assert result.deleted == 1
    assert store.get_document("docs/wip.md") is None
    assert store.match_documents("draft_marker") == []
    assert store.count_documents() == 0
    assert any(d.code == "excluded-glob" for d in result.diagnostics)
    assert store.check_consistency()
