"""Incremental indexing + reconciliation (plan.md Step 4, §5 indexer.py, §6).

This module is the wiring layer: it walks the configured roots
(:mod:`find_again.config`), applies the PATH-based exclusion layers BEFORE reading
a file (:func:`find_again.config.evaluate_path`) so a path-denied secret's bytes
never load, reads the bytes, runs the CONTENT secret scan on those exact bytes
(:func:`find_again.config.evaluate_content`), hashes content, routes only NEW or
CHANGED files through the adapters (:func:`find_again.adapters.select_adapter_for`),
and reconciles the store (:class:`find_again.db.Database`) so it exactly mirrors
the current filesystem. It owns no storage, no parsing, and no exclusion logic of
its own -- every one of those is reused from Steps 1-3b.

The refresh is driven by the file-level ``content_hash`` (SHA-256 over raw bytes,
plan.md §6 Identifiers):

* **Unchanged** -- every stored document for a discovered file already carries the
  file's current hash. The file is SKIPPED entirely: no re-parse, no re-write.
* **New / changed** -- the file is parsed and its documents upserted. Any of the
  file's previously-indexed doc_ids the new parse no longer produces (a JSONL that
  shrank, a plan whose sections moved) are dropped in the SAME transaction.
* **Deleted / now-excluded** -- a source file that is gone, or newly excluded (a
  new deny glob, a newly-present secret), or newly unhandled loses ALL its
  documents in the delete-reconciliation pass.
* **Present but unreadable** -- a file that still exists but cannot be read this
  refresh (a transient Windows file lock, a momentary permission error) KEEPS its
  prior documents and emits an ``unreadable-input`` diagnostic. Only genuinely
  absent/excluded sources are pruned, so a momentary lock never drops a file's
  search coverage (plan.md §6 "preserves diagnostics for unreadable inputs").

Atomicity granularity (plan.md Step 4 "interrupted refresh does not leave partial
state"): **per source file**. Each changed file's upserts + stale-record deletes
run inside one :meth:`find_again.db.Database.reconcile_source` transaction, and the
whole-file delete pass is one :meth:`~find_again.db.Database.remove_documents`
transaction. A crash mid-refresh therefore leaves every file either at its prior
state or cleanly advanced -- never a torn mix of old and new records, and metadata
never diverges from the FTS index. A whole-refresh single transaction was rejected:
it would hold one write lock across an arbitrarily long walk and force the storage
layer to expose its transaction boundary to the indexer; per-file transactions give
the same "no partial/corrupt state" guarantee (each file all-or-nothing) while
reusing db.py's existing transaction ownership unchanged.

Diagnostics (exclusion + adapter) are collected across the whole refresh and
persisted (replacing the prior set) so ``find-again status`` can report why inputs
were skipped without re-indexing. Every diagnostic is path + stable code only --
never file content (the Step-1/3 safety invariant).
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from .adapters import SourceFile, select_adapter_for
from .config import Config, evaluate_content, evaluate_path, git_ignored
from .db import Database
from .models import Diagnostic, IndexedDocument, Severity, content_hash, normalize_source_path

__all__ = ["RefreshResult", "refresh_index"]

# Directories never walked: git's own store and find-again's own derived index
# (the DB lives at <root>/.find-again/index.db and must never index itself).
_PRUNED_DIRS = frozenset({".git", ".find-again"})


# --------------------------------------------------------------------------- #
# Result
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class RefreshResult:
    """Outcome of one :func:`refresh_index` run.

    ``indexed``/``updated``/``skipped`` count FILES (new, changed, unchanged);
    ``deleted`` counts DOCUMENTS removed (whole vanished/excluded files plus stale
    records of files that shrank). ``documents`` is the total document count after
    the refresh. ``refreshed_at`` is the ISO 8601 UTC completion time stamped into
    the index. ``diagnostics`` are every exclusion/adapter diagnostic from this run.
    """

    indexed: int
    updated: int
    skipped: int
    deleted: int
    documents: int
    refreshed_at: str
    diagnostics: tuple[Diagnostic, ...] = ()


# --------------------------------------------------------------------------- #
# Discovery
# --------------------------------------------------------------------------- #
def _normalized_roots(config: Config) -> list[str]:
    """The configured include roots, normalized to POSIX-relative form."""
    return [raw.replace("\\", "/").strip("/") for raw in config.roots]


def _discover(config: Config) -> tuple[dict[str, Path], list[Diagnostic]]:
    """Walk the configured roots, returning ``{rel_posix: abs_path}`` + discovery diagnostics.

    Overlapping roots are de-duplicated by root-relative path. ``.git`` and
    ``.find-again`` subtrees are pruned. A configured root that does not exist emits
    a WARN diagnostic (surfacing misconfiguration) rather than silently indexing
    nothing. No file is read here -- discovery only enumerates candidate paths.
    """
    candidates: dict[str, Path] = {}
    diagnostics: list[Diagnostic] = []
    root = config.root
    for rel_root in _normalized_roots(config):
        base = root if rel_root in ("", ".") else root / rel_root
        if base.is_file():
            candidates[normalize_source_path(base.relative_to(root))] = base
            continue
        if not base.is_dir():
            diagnostics.append(
                Diagnostic(
                    source_path=rel_root or ".",
                    adapter="discovery",
                    severity=Severity.WARN,
                    code="missing-root",
                    message="configured root does not exist or is not a directory",
                )
            )
            continue
        for dirpath, dirnames, filenames in os.walk(base):
            dirnames[:] = sorted(d for d in dirnames if d not in _PRUNED_DIRS)
            for filename in sorted(filenames):
                abs_path = Path(dirpath) / filename
                rel = normalize_source_path(abs_path.relative_to(root))
                candidates.setdefault(rel, abs_path)
    return candidates, diagnostics


def _iso_utc(epoch_seconds: float) -> str:
    """Render an epoch time as ``YYYY-MM-DDTHH:MM:SSZ`` (ISO 8601 UTC, second precision)."""
    stamped = datetime.fromtimestamp(epoch_seconds, tz=UTC).replace(microsecond=0)
    return stamped.isoformat().replace("+00:00", "Z")


# --------------------------------------------------------------------------- #
# Per-file evaluation state
# --------------------------------------------------------------------------- #
@dataclass(slots=True)
class _Tally:
    """Mutable counters accumulated across a refresh."""

    indexed: int = 0
    updated: int = 0
    skipped: int = 0
    deleted: int = 0
    diagnostics: list[Diagnostic] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class _ReadOutcome:
    """Classified result of reading a candidate file's bytes for indexing.

    * ``content`` not None -> the bytes were read (size within the limit); index them.
    * ``oversized``        -> stat size exceeds the limit; exclude (its documents purge).
    * ``unreadable``       -> present but could not be stat'd/read this refresh; KEEP
                              its prior documents and diagnose (a transient lock must
                              never drop coverage), never purge.
    * all-default (``content`` None, both flags ``False``) -> the file vanished
      before it could be read; let delete-reconciliation purge it.
    """

    content: bytes | None = None
    oversized: bool = False
    unreadable: bool = False
    size: int = 0


def _read_source(config: Config, abs_path: Path) -> _ReadOutcome:
    """Stat-gate on size, then read ``abs_path``'s bytes, classifying any failure.

    The size ceiling is checked from the stat size BEFORE the bytes are read, so an
    oversized file is never slurped into memory (Layer 3 applied pre-read). A
    stat/read failure is classified by its cause: ``FileNotFoundError`` means the
    file vanished after discovery (genuinely gone -> purge); any other ``OSError``
    means the entry is present but momentarily unreadable (a Windows lock, a
    transient permission error) and its prior documents must be PRESERVED, never
    purged. The classification is deliberately conservative -- only a definitively
    absent file is treated as deleted.
    """
    limit_bytes = config.max_file_kb * 1024
    try:
        size = abs_path.stat().st_size
    except FileNotFoundError:
        return _ReadOutcome()  # vanished after discovery -> purge
    except OSError:
        return _ReadOutcome(unreadable=True)  # present but unreadable -> keep + diagnose
    if size > limit_bytes:
        return _ReadOutcome(oversized=True, size=size)
    try:
        content = abs_path.read_bytes()
    except FileNotFoundError:
        return _ReadOutcome()  # vanished between stat and read -> purge
    except OSError:
        return _ReadOutcome(unreadable=True)  # present but unreadable -> keep + diagnose
    return _ReadOutcome(content=content)


def _timestamp_for(abs_path: Path) -> str:
    """ISO 8601 UTC modification time of ``abs_path`` (falls back to now on stat error)."""
    try:
        mtime = abs_path.stat().st_mtime
    except OSError:
        return _iso_utc(datetime.now(tz=UTC).timestamp())
    return _iso_utc(mtime)


# --------------------------------------------------------------------------- #
# Refresh
# --------------------------------------------------------------------------- #
def refresh_index(
    config: Config,
    db: Database,
    *,
    now: datetime | None = None,
    use_git_ignore: bool = True,
) -> RefreshResult:
    """Reconcile ``db`` to match the files currently under ``config``'s roots.

    Discovers candidate files, applies exclusion, skips unchanged files by content
    hash, (re)indexes new/changed files, removes documents for vanished/excluded
    files, persists diagnostics + the refresh timestamp, and returns a
    :class:`RefreshResult`. ``now`` overrides the completion timestamp (for tests);
    ``use_git_ignore`` may be disabled when a git probe is undesirable.
    """
    project = config.root.name or "root"
    tally = _Tally()

    # Prior-state snapshot: which doc_ids exist, grouped by source file + its hash.
    existing_by_source: dict[str, dict[str, str]] = {}
    for doc_id, source_path, stored_hash in db.iter_document_index():
        existing_by_source.setdefault(source_path, {})[doc_id] = stored_hash

    candidates, discovery_diagnostics = _discover(config)
    tally.diagnostics.extend(discovery_diagnostics)

    ignored: set[str] = git_ignored(config.root, list(candidates)) if use_git_ignore else set()

    # Source files that are present + included this refresh (unchanged or reindexed).
    # Their documents survive delete-reconciliation; everything else is pruned.
    live_sources: set[str] = set()

    for rel, abs_path in sorted(candidates.items()):
        # Phase A -- path-only layers (outside-roots, path-glob deny, git-ignored),
        # applied BEFORE any read so a path-denied secret file's bytes never load.
        path_decision = evaluate_path(config, abs_path, is_git_ignored=rel in ignored)
        if not path_decision.include:
            if path_decision.diagnostic is not None:  # exclusion always attaches one
                tally.diagnostics.append(path_decision.diagnostic)
            continue  # excluded -> not live -> its documents reconcile away

        # Phase B -- size-gate (stat) then read, classifying any failure. A present-
        # but-unreadable file KEEPS its prior documents (never purge on a transient
        # lock); an oversized file is excluded; a vanished file is purged.
        outcome = _read_source(config, abs_path)
        if outcome.oversized:
            tally.diagnostics.append(
                Diagnostic(
                    source_path=rel,
                    adapter="exclusion",
                    severity=Severity.WARN,
                    code="oversized",
                    message=(
                        f"file size {outcome.size} bytes exceeds max_file_kb={config.max_file_kb}"
                    ),
                )
            )
            continue  # oversized -> excluded -> purge
        if outcome.unreadable:
            # Present but unreadable this refresh: preserve prior coverage (do NOT
            # purge). Diagnostic is path + code only -- never file content.
            tally.diagnostics.append(
                Diagnostic(
                    source_path=rel,
                    adapter="indexer",
                    severity=Severity.WARN,
                    code="unreadable-input",
                    message=(
                        "file present but could not be read this refresh; prior documents preserved"
                    ),
                )
            )
            live_sources.add(rel)  # KEEP its documents -> excluded from the purge pass
            continue
        if outcome.content is None:
            continue  # vanished before read -> genuinely gone -> reconciliation purges
        raw = outcome.content

        # Phase C -- content layers (size-by-length + secret scan) on the EXACT bytes
        # that will be indexed. A file that now carries a secret (or is undecodable)
        # is excluded -> its prior documents reconcile away. Fail-closed: no bytes
        # ever reach indexing without passing this scan.
        content_decision = evaluate_content(config, abs_path, raw)
        if not content_decision.include:
            if content_decision.diagnostic is not None:  # exclusion always attaches one
                tally.diagnostics.append(content_decision.diagnostic)
            continue

        file_hash = content_hash(raw)
        existing_for_source = existing_by_source.get(rel)

        # Unchanged: every stored document for this file already carries its current
        # hash. Skip entirely -- no re-parse, no re-write.
        if existing_for_source and set(existing_for_source.values()) == {file_hash}:
            live_sources.add(rel)
            tally.skipped += 1
            continue

        source = SourceFile(
            source_path=rel,
            raw=raw,
            project=project,
            timestamp=_timestamp_for(abs_path),
        )
        adapter = select_adapter_for(source)
        if adapter is None:
            # Includable but no adapter owns this extension (e.g. a stray .csv). Not
            # live -> any documents it used to have reconcile away.
            tally.diagnostics.append(
                Diagnostic(
                    source_path=rel,
                    adapter="indexer",
                    severity=Severity.WARN,
                    code="no-adapter",
                    message="no adapter handles this file type; skipped",
                )
            )
            continue

        result = adapter.parse(source)
        tally.diagnostics.extend(result.diagnostics)
        new_documents: Sequence[IndexedDocument] = [pd.document for pd in result.documents]
        new_doc_ids = {doc.doc_id for doc in new_documents}
        stale_doc_ids = set(existing_for_source or {}) - new_doc_ids

        # Per-file atomic advance: drop stale records + upsert new ones in one txn.
        tally.deleted += db.reconcile_source(new_documents, stale_doc_ids)
        live_sources.add(rel)
        if existing_for_source is None:
            tally.indexed += 1
        else:
            tally.updated += 1

    # Delete reconciliation: every document whose source file is gone / excluded /
    # unhandled this refresh (its source_path is not live) is removed in one txn.
    to_remove = [
        doc_id
        for source_path, docs in existing_by_source.items()
        if source_path not in live_sources
        for doc_id in docs
    ]
    tally.deleted += db.remove_documents(to_remove)

    refreshed_at = _iso_utc((now or datetime.now(tz=UTC)).timestamp())
    db.replace_diagnostics(tally.diagnostics)
    db.set_meta("last_refreshed", refreshed_at)

    return RefreshResult(
        indexed=tally.indexed,
        updated=tally.updated,
        skipped=tally.skipped,
        deleted=tally.deleted,
        documents=db.count_documents(),
        refreshed_at=refreshed_at,
        diagnostics=tuple(tally.diagnostics),
    )
