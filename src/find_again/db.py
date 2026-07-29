"""SQLite + FTS5 storage layer (plan.md 5 db.py role, 6 Derived local index, 10).

This module owns the derived, rebuildable index. Source artifacts remain
authoritative (plan.md 6); the database at ``<root>/.find-again/index.db`` is
gitignored and may be deleted and rebuilt at any time.

Schema (two migrations under :mod:`find_again.migrations`):

* ``documents`` -- metadata table, one row per :class:`~find_again.models.
  IndexedDocument` (``doc_id`` unique, plus ``source_path``, ``record_locator``,
  ``artifact_type``, ``project``, ``timestamp``, ``content_hash``). ``id`` is the
  integer rowid.
* ``documents_fts`` -- a standalone (content-owning) FTS5 virtual table holding
  the document ``body``. Each FTS row's rowid equals the owning ``documents.id``.

FTS5 design: a *content-owning* (not contentless, not external-content) FTS5
table. The body is stored only here -- the metadata table has no body column --
so there is no duplication, deletes and updates are ordinary SQL, and the search
layer can produce ``snippet()`` excerpts (a contentless table cannot).

Consistency: every write (upsert, delete) touches both the metadata row and its
FTS row inside a single transaction, so the two never diverge. A mid-transaction
failure rolls the whole write back, preserving the prior index state.

Scope: this is the storage layer only. Discovery/hashing/reconciliation
(:mod:`find_again.indexer`, Step 4) and ranked/filtered querying
(:mod:`find_again.search`, Step 5) live elsewhere. :meth:`Database.match_documents`
here is the minimal FTS read used by the Step 2 write/read round-trip, not the
Step 5 search API.
"""

from __future__ import annotations

import contextlib
import functools
import importlib.resources
import sqlite3
from collections.abc import Iterable, Iterator, Sequence
from pathlib import Path
from types import TracebackType

from .models import ArtifactType, Diagnostic, IndexedDocument, Severity

__all__ = [
    "DB_DIRNAME",
    "DB_FILENAME",
    "Database",
    "DatabaseError",
    "FTS5UnavailableError",
    "MigrationError",
    "NewerDatabaseError",
    "apply_migrations",
    "current_schema_version",
    "default_db_path",
    "latest_migration_version",
]

DB_DIRNAME = ".find-again"
DB_FILENAME = "index.db"

_MIGRATIONS_PACKAGE = "find_again.migrations"


# --------------------------------------------------------------------------- #
# Errors
# --------------------------------------------------------------------------- #
class DatabaseError(Exception):
    """Base class for storage-layer failures."""


class FTS5UnavailableError(DatabaseError):
    """The bundled SQLite build was compiled without the FTS5 extension."""


class NewerDatabaseError(DatabaseError):
    """The DB ``user_version`` is newer than this build's latest migration.

    Consistent with the config ``schema_version`` policy (plan.md 6): older
    databases migrate up, a newer one is refused with an explicit error rather
    than silently downgraded.
    """


class MigrationError(DatabaseError):
    """A migration file is malformed (bad version, empty, or duplicate)."""


# --------------------------------------------------------------------------- #
# FTS5 availability
# --------------------------------------------------------------------------- #
def _probe_fts5(conn: sqlite3.Connection) -> bool:
    """Return ``True`` if this SQLite build has FTS5 (create+drop a temp table)."""
    try:
        conn.execute("CREATE VIRTUAL TABLE temp.__find_again_fts5_probe USING fts5(x)")
    except sqlite3.OperationalError:
        return False
    conn.execute("DROP TABLE temp.__find_again_fts5_probe")
    return True


def _assert_fts5_available(conn: sqlite3.Connection) -> None:
    """Raise :class:`FTS5UnavailableError` unless FTS5 is compiled in."""
    if not _probe_fts5(conn):
        raise FTS5UnavailableError(
            "SQLite FTS5 extension is not available in this Python build; "
            "find-again requires sqlite3 compiled with FTS5 (rebuild Python "
            "against a SQLite with -DSQLITE_ENABLE_FTS5, or use a build that has it)"
        )


# --------------------------------------------------------------------------- #
# Transactions (explicit, in autocommit mode)
# --------------------------------------------------------------------------- #
@contextlib.contextmanager
def _transaction(conn: sqlite3.Connection) -> Iterator[None]:
    """Run a block inside one BEGIN/COMMIT, rolling back on any exception.

    The connection is opened with ``isolation_level=None`` (autocommit), so this
    is the sole transaction boundary: on success the block commits, and on any
    exception -- including a failure between the metadata write and the FTS write
    -- everything rolls back, leaving the prior index state untouched.
    """
    conn.execute("BEGIN")
    try:
        yield
    except BaseException:
        conn.execute("ROLLBACK")
        raise
    else:
        conn.execute("COMMIT")


# --------------------------------------------------------------------------- #
# Migrations
# --------------------------------------------------------------------------- #
def _has_executable_sql(sql: str) -> bool:
    """Return ``True`` if ``sql`` has any statement content.

    Blank lines and full-line ``--`` comments do not count. This is only used to
    reject an empty migration file; the SQL itself is never split by hand -- it is
    executed verbatim via :meth:`sqlite3.Connection.executescript`, so SQLite's own
    parser handles multi-statement DDL, ``;`` inside string literals, and trigger
    bodies (no fragile hand-rolled ``;`` splitter to trip over).
    """
    for line in sql.splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("--"):
            return True
    return False


@functools.lru_cache(maxsize=1)
def _load_migrations() -> tuple[tuple[int, str, str], ...]:
    """Load and validate the packaged migrations, sorted by ascending version.

    Returns a tuple of ``(version, filename, sql)`` where ``sql`` is the raw file
    text (run verbatim by :func:`apply_migrations` via ``executescript``). Raises
    :class:`MigrationError` for a file without a leading integer version, a
    duplicate version, or a file with no executable SQL.
    """
    resources = importlib.resources.files(_MIGRATIONS_PACKAGE)
    loaded: list[tuple[int, str, str]] = []
    seen: set[int] = set()
    for entry in resources.iterdir():
        name = entry.name
        if not name.endswith(".sql"):
            continue
        try:
            version = int(name.split("_", 1)[0])
        except ValueError as exc:
            raise MigrationError(
                f"migration file {name!r} lacks a leading integer version"
            ) from exc
        if version in seen:
            raise MigrationError(f"duplicate migration version {version} ({name!r})")
        seen.add(version)
        sql = entry.read_text(encoding="utf-8")
        if not _has_executable_sql(sql):
            raise MigrationError(f"migration file {name!r} contains no statements")
        loaded.append((version, name, sql))
    loaded.sort(key=lambda item: item[0])
    return tuple(loaded)


def _apply_migration(conn: sqlite3.Connection, version: int, sql: str) -> None:
    """Apply one migration's SQL atomically and stamp ``PRAGMA user_version``.

    ``BEGIN``/``COMMIT`` are embedded in the script so the whole migration -- DDL
    plus the version stamp -- is one transaction handed to SQLite's parser via
    ``executescript``. A mid-script failure leaves an open transaction, which we
    roll back, so the DB is left at the *prior* ``user_version`` with the prior
    schema (SQLite DDL is transactional). ``version`` is a validated int parsed
    from the migration filename, so the interpolation is safe.
    """
    script = f"BEGIN;\n{sql}\nPRAGMA user_version = {int(version)};\nCOMMIT;"  # noqa: S608
    try:
        conn.executescript(script)
    except BaseException:
        # Roll back the partially-applied migration. suppress() guards the edge
        # where no transaction is open (e.g. BEGIN itself never took).
        with contextlib.suppress(sqlite3.OperationalError):
            conn.execute("ROLLBACK")
        raise


def latest_migration_version() -> int:
    """Return the highest packaged migration version (0 if there are none)."""
    migrations = _load_migrations()
    return migrations[-1][0] if migrations else 0


def current_schema_version(conn: sqlite3.Connection) -> int:
    """Return the DB's applied schema version (``PRAGMA user_version``)."""
    return int(conn.execute("PRAGMA user_version").fetchone()[0])


def apply_migrations(conn: sqlite3.Connection, target_version: int | None = None) -> int:
    """Migrate ``conn`` up to ``target_version`` (default: the latest packaged).

    Deterministic and idempotent: pending migrations (those with version in the
    half-open range ``(current, target]``) are applied in ascending order, each
    in its own transaction, and ``PRAGMA user_version`` records the applied
    version. Re-running once at the latest version is a no-op. A DB whose
    ``user_version`` is *newer* than the latest packaged migration is refused
    with :class:`NewerDatabaseError`.

    Returns the resulting schema version.
    """
    migrations = _load_migrations()
    max_version = migrations[-1][0] if migrations else 0
    current = current_schema_version(conn)
    if current > max_version:
        raise NewerDatabaseError(
            f"database schema version {current} is newer than the latest supported "
            f"version {max_version}; upgrade find-again (refusing to open)"
        )
    target = max_version if target_version is None else target_version
    if target > max_version:
        raise MigrationError(
            f"requested migration target {target} exceeds the latest available "
            f"version {max_version}"
        )
    for version, _name, sql in migrations:
        if not (current < version <= target):
            continue
        _apply_migration(conn, version, sql)
    return current_schema_version(conn)


# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
def default_db_path(root: Path) -> Path:
    """Return ``<root>/.find-again/index.db`` (plan.md 10)."""
    return Path(root) / DB_DIRNAME / DB_FILENAME


# --------------------------------------------------------------------------- #
# Row reconstruction
# --------------------------------------------------------------------------- #
def _row_to_document(row: sqlite3.Row) -> IndexedDocument:
    """Rebuild an :class:`IndexedDocument` from a joined documents+FTS row."""
    return IndexedDocument(
        doc_id=row["doc_id"],
        source_path=row["source_path"],
        artifact_type=ArtifactType(row["artifact_type"]),
        project=row["project"],
        timestamp=row["timestamp"],
        content_hash=row["content_hash"],
        body=row["body"],
        record_locator=row["record_locator"],
    )


# --------------------------------------------------------------------------- #
# The storage layer
# --------------------------------------------------------------------------- #
class Database:
    """Transactional SQLite/FTS5 store for indexed documents.

    Construct via :meth:`open` (file-backed), :meth:`open_root` (default path
    under a root), or :meth:`open_memory` (in-memory, for tests). Each opener
    verifies FTS5 availability and migrates the schema up to the latest version.
    """

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    # -- construction ------------------------------------------------------- #
    @classmethod
    def _bootstrap(cls, conn: sqlite3.Connection) -> Database:
        try:
            conn.row_factory = sqlite3.Row
            _assert_fts5_available(conn)
            apply_migrations(conn)
        except BaseException:
            # Close on ANY bootstrap failure (FTS5-missing, newer-DB refusal, a
            # bad migration). Otherwise a file-backed open would leak the
            # connection -- and, on Windows, the OS file lock on index.db -- which
            # can block a subsequent open or delete of the same database.
            conn.close()
            raise
        return cls(conn)

    @classmethod
    def open(cls, path: Path | str) -> Database:
        """Open (creating if needed) a file-backed DB at ``path``.

        The parent directory (typically ``.find-again``) is created if missing.
        """
        db_path = Path(path)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(db_path), isolation_level=None)
        return cls._bootstrap(conn)

    @classmethod
    def open_root(cls, root: Path | str) -> Database:
        """Open the DB at ``<root>/.find-again/index.db``."""
        return cls.open(default_db_path(Path(root)))

    @classmethod
    def open_memory(cls) -> Database:
        """Open an in-memory DB (tests / ephemeral use)."""
        conn = sqlite3.connect(":memory:", isolation_level=None)
        return cls._bootstrap(conn)

    # -- lifecycle ---------------------------------------------------------- #
    @property
    def connection(self) -> sqlite3.Connection:
        """The underlying connection (for later steps that read directly)."""
        return self._conn

    @property
    def schema_version(self) -> int:
        """The applied schema version (``PRAGMA user_version``)."""
        return current_schema_version(self._conn)

    def close(self) -> None:
        """Close the underlying connection."""
        self._conn.close()

    def __enter__(self) -> Database:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    # -- writes ------------------------------------------------------------- #
    def upsert_document(self, doc: IndexedDocument) -> None:
        """Insert or replace ``doc`` by ``doc_id``, metadata + FTS in one txn.

        If a row with the same ``doc_id`` exists it is fully removed (metadata +
        FTS) and re-inserted, so the FTS body always matches the current
        metadata. The whole operation is one transaction: a failure at any point
        leaves the prior state intact.
        """
        with _transaction(self._conn):
            existing = self._conn.execute(
                "SELECT id FROM documents WHERE doc_id = ?", (doc.doc_id,)
            ).fetchone()
            if existing is not None:
                self._delete_rowid(int(existing["id"]))
            self._insert_document(doc)

    def delete_document(self, doc_id: str) -> bool:
        """Delete a document (metadata + FTS) by ``doc_id`` in one transaction.

        Returns ``True`` if a row was removed, ``False`` if none matched.
        """
        with _transaction(self._conn):
            existing = self._conn.execute(
                "SELECT id FROM documents WHERE doc_id = ?", (doc_id,)
            ).fetchone()
            if existing is None:
                return False
            self._delete_rowid(int(existing["id"]))
            return True

    def _insert_document(self, doc: IndexedDocument) -> None:
        cursor = self._conn.execute(
            "INSERT INTO documents "
            "(doc_id, source_path, record_locator, artifact_type, project, "
            "timestamp, content_hash) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                doc.doc_id,
                doc.source_path,
                doc.record_locator,
                doc.artifact_type.value,
                doc.project,
                doc.timestamp,
                doc.content_hash,
            ),
        )
        rowid = cursor.lastrowid
        if rowid is None:  # pragma: no cover -- INSERT always yields a rowid
            raise DatabaseError("metadata insert did not return a rowid")
        self._insert_fts(rowid, doc.body)

    def _insert_fts(self, rowid: int, body: str) -> None:
        # FTS row's rowid is pinned to the owning documents.id so the two tables
        # stay joinable and consistent.
        self._conn.execute("INSERT INTO documents_fts (rowid, body) VALUES (?, ?)", (rowid, body))

    def _delete_rowid(self, rowid: int) -> None:
        self._conn.execute("DELETE FROM documents_fts WHERE rowid = ?", (rowid,))
        self._conn.execute("DELETE FROM documents WHERE id = ?", (rowid,))

    # -- reads -------------------------------------------------------------- #
    def get_document(self, doc_id: str) -> IndexedDocument | None:
        """Read a document (metadata + FTS body) by ``doc_id``, or ``None``."""
        row = self._conn.execute(
            "SELECT d.doc_id, d.source_path, d.record_locator, d.artifact_type, "
            "d.project, d.timestamp, d.content_hash, f.body "
            "FROM documents d JOIN documents_fts f ON f.rowid = d.id "
            "WHERE d.doc_id = ?",
            (doc_id,),
        ).fetchone()
        return None if row is None else _row_to_document(row)

    def match_documents(self, match_query: str) -> list[IndexedDocument]:
        """Return documents whose body matches an FTS5 ``MATCH`` query.

        Minimal storage-level full-text read used by the Step 2 round-trip test;
        results are ordered by rowid, not ranked. Ranked/filtered querying with
        excerpts is Step 5 (:mod:`find_again.search`).
        """
        rows = self._conn.execute(
            "SELECT d.doc_id, d.source_path, d.record_locator, d.artifact_type, "
            "d.project, d.timestamp, d.content_hash, documents_fts.body "
            "FROM documents_fts JOIN documents d ON d.id = documents_fts.rowid "
            "WHERE documents_fts MATCH ? ORDER BY d.id",
            (match_query,),
        ).fetchall()
        return [_row_to_document(row) for row in rows]

    def count_documents(self) -> int:
        """Return the number of indexed documents."""
        return int(self._conn.execute("SELECT count(*) FROM documents").fetchone()[0])

    def check_consistency(self) -> bool:
        """Return ``True`` iff metadata and FTS rows are one-to-one by rowid.

        Verifies (a) equal row counts and (b) no orphan on either side. A healthy
        store always satisfies this; a broken transaction boundary would not.
        """
        meta_count = int(self._conn.execute("SELECT count(*) FROM documents").fetchone()[0])
        fts_count = int(self._conn.execute("SELECT count(*) FROM documents_fts").fetchone()[0])
        if meta_count != fts_count:
            return False
        orphan_meta = int(
            self._conn.execute(
                "SELECT count(*) FROM documents d "
                "WHERE NOT EXISTS (SELECT 1 FROM documents_fts f WHERE f.rowid = d.id)"
            ).fetchone()[0]
        )
        orphan_fts = int(
            self._conn.execute(
                "SELECT count(*) FROM documents_fts f "
                "WHERE NOT EXISTS (SELECT 1 FROM documents d WHERE d.id = f.rowid)"
            ).fetchone()[0]
        )
        return orphan_meta == 0 and orphan_fts == 0

    # -- incremental reconciliation (Step 4 indexer support) ---------------- #
    def iter_document_index(self) -> Iterator[tuple[str, str, str]]:
        """Yield ``(doc_id, source_path, content_hash)`` for every stored document.

        The refresh reconciler reads this once at the start of a run to build its
        prior-state snapshot (which files changed, which vanished). Bodies are not
        read -- only the metadata the incremental decision needs.
        """
        for row in self._conn.execute("SELECT doc_id, source_path, content_hash FROM documents"):
            yield (row["doc_id"], row["source_path"], row["content_hash"])

    def reconcile_source(
        self,
        documents: Sequence[IndexedDocument],
        stale_doc_ids: Iterable[str],
    ) -> int:
        """Atomically advance ONE source file: drop ``stale_doc_ids``, upsert ``documents``.

        This is the per-file atomicity unit of a refresh (plan.md Step 4 "interrupted
        refresh does not leave partial state"). Every delete + insert for the file
        runs inside ONE transaction, so a crash mid-file leaves the file at its PRIOR
        state (rolled back), never a torn mix of old and new records. ``stale_doc_ids``
        are the file's previously-indexed doc_ids that its new parse no longer
        produces (e.g. a JSONL that shrank); passing them here removes them in the
        same transaction that lands the new records, so the file transitions cleanly.

        Returns the number of stale documents actually removed.
        """
        removed = 0
        with _transaction(self._conn):
            for doc_id in stale_doc_ids:
                existing = self._conn.execute(
                    "SELECT id FROM documents WHERE doc_id = ?", (doc_id,)
                ).fetchone()
                if existing is not None:
                    self._delete_rowid(int(existing["id"]))
                    removed += 1
            for doc in documents:
                existing = self._conn.execute(
                    "SELECT id FROM documents WHERE doc_id = ?", (doc.doc_id,)
                ).fetchone()
                if existing is not None:
                    self._delete_rowid(int(existing["id"]))
                self._insert_document(doc)
        return removed

    def remove_documents(self, doc_ids: Iterable[str]) -> int:
        """Delete a batch of documents by ``doc_id`` in ONE transaction.

        Used for whole-file delete reconciliation (a source file that vanished or is
        now excluded loses ALL its documents). Missing ids are skipped. Returns the
        number of rows actually removed.
        """
        removed = 0
        with _transaction(self._conn):
            for doc_id in doc_ids:
                existing = self._conn.execute(
                    "SELECT id FROM documents WHERE doc_id = ?", (doc_id,)
                ).fetchone()
                if existing is not None:
                    self._delete_rowid(int(existing["id"]))
                    removed += 1
        return removed

    # -- refresh metadata + persisted diagnostics --------------------------- #
    def set_meta(self, key: str, value: str) -> None:
        """Upsert a single index-meta ``key`` -> ``value`` (e.g. ``last_refreshed``)."""
        with _transaction(self._conn):
            self._conn.execute(
                "INSERT INTO index_meta (key, value) VALUES (?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (key, value),
            )

    def get_meta(self, key: str) -> str | None:
        """Return the value stored for ``key`` in index-meta, or ``None`` if unset."""
        row = self._conn.execute("SELECT value FROM index_meta WHERE key = ?", (key,)).fetchone()
        return None if row is None else str(row["value"])

    def replace_diagnostics(self, diagnostics: Sequence[Diagnostic]) -> None:
        """Replace the persisted diagnostic set with ``diagnostics`` in ONE transaction.

        The indexer calls this once per refresh so ``status`` reports the LAST
        refresh's diagnostics without re-indexing. Each row is path + stable code +
        message only -- the message never carries file content (plan.md §5 Diagnostic
        safety contract; enforced by the producing exclusion/adapter code).
        """
        with _transaction(self._conn):
            self._conn.execute("DELETE FROM diagnostics")
            self._conn.executemany(
                "INSERT INTO diagnostics (source_path, adapter, severity, code, message) "
                "VALUES (?, ?, ?, ?, ?)",
                [
                    (d.source_path, d.adapter, d.severity.value, d.code, d.message)
                    for d in diagnostics
                ],
            )

    def get_diagnostics(self) -> list[Diagnostic]:
        """Return the persisted diagnostics from the last refresh, in insertion order."""
        rows = self._conn.execute(
            "SELECT source_path, adapter, severity, code, message FROM diagnostics ORDER BY id"
        ).fetchall()
        return [
            Diagnostic(
                source_path=row["source_path"],
                adapter=row["adapter"],
                severity=Severity(row["severity"]),
                code=row["code"],
                message=row["message"],
            )
            for row in rows
        ]
