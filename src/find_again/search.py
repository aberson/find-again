"""Deterministic FTS retrieval + filtering (plan.md Step 5, §5 search.py, §6 Retrieval only).

This is the read side of the index. It turns an operator query plus optional
artifact/project/date filters into a deterministically ordered list of
:class:`~find_again.models.SearchResult` -- each carrying the document's
``type``/``timestamp``, a ranked ``snippet()`` **excerpt** around the match, an
openable :class:`~find_again.models.Locator`, and its deterministic FTS ``rank``.

Retrieval only (plan.md §6): nothing here summarizes, answers, or calls a model.
Every excerpt is source text produced by SQLite's ``snippet()`` (empty markers, so
it is a contiguous substring of the stored body bracketed by an ellipsis where
truncated), never synthesized prose.

Query safety (plan.md Step 5 "passed to MATCH SAFELY"):

* The query never reaches SQL as text -- it is always a BOUND parameter, so SQL
  injection is impossible (the ``documents`` table cannot be dropped by a crafted
  query; see the storage-layer round-trip test).
* Exposed FTS syntax is deliberately minimal and SAFE BY CONSTRUCTION: the query is
  split on whitespace and each token is emitted as a **quoted FTS5 phrase** (inner
  ``"`` doubled per FTS5 escaping). A quoted phrase is a literal in FTS5, so every
  operator character (``"`` ``*`` ``:`` ``^`` ``(`` ``AND`` ``OR`` ``NEAR`` ...) is
  treated as ordinary text rather than query syntax -- a hostile or malformed query
  therefore cannot raise an FTS syntax error or inject anything; at worst it matches
  nothing (a clean empty result). Tokens with no alphanumeric character (pure
  punctuation) are dropped; a query with no searchable token yields an empty result.
* Belt-and-suspenders: an FTS5 syntax-class ``OperationalError`` (unreachable given
  the sanitizer, but defended anyway) is turned into a clean empty result; any other
  ``OperationalError`` (a genuine DB failure) propagates for the CLI to report.

Ranking is deterministic (plan.md §5 "rank deterministic FTS rank"): rows are
ordered by ``bm25()`` ascending (most relevant first) with a stable ``doc_id``
tiebreak. ``doc_id`` is UNIQUE, so the order is a total order -- the SAME query over
the SAME index always returns the SAME ordering, run to run.
"""

from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime

from .db import Database
from .models import ArtifactType, Locator, SearchResult

__all__ = [
    "DEFAULT_LIMIT",
    "EXCERPT_ELLIPSIS",
    "SEARCH_SCHEMA_VERSION",
    "SearchQuery",
    "build_match_query",
    "normalize_date_bound",
    "result_to_dict",
    "search",
]

# JSON output contract version (plan.md Step 5 "JSON schema_version").
SEARCH_SCHEMA_VERSION = 1

# Default top-N when the caller passes no explicit limit (plan.md "top 5+ / a --limit").
DEFAULT_LIMIT = 10

# A record locator of the shape ``L<n>`` (a JSONL/record line) renders back to the
# openable ``path:<n>`` form the adapters produced (json_lines.py). Any other
# record locator is rendered as ``path::<key>``.
_LINE_RECORD_RE = re.compile(r"L(\d+)")

# A bare calendar date (no time component) supplied to a date filter.
_DATE_ONLY_RE = re.compile(r"\d{4}-\d{2}-\d{2}")

# Excerpt window is a fixed SQL literal (``snippet(documents_fts, 0, '', '', '...',
# 20)`` below): the empty highlight markers ('', '') keep the excerpt a faithful
# substring of the source body (retrieval-only, no synthesized text); the '...'
# ellipsis marks a truncated end; the 20-token cap is within FTS5's required 1..64
# range and keeps the excerpt to roughly one line.
EXCERPT_ELLIPSIS = "..."


@dataclass(frozen=True, slots=True)
class SearchQuery:
    """A resolved search request (plan.md Step 5 filters).

    ``query`` is the operator's raw text (sanitized into a safe FTS phrase query by
    :func:`build_match_query`). ``artifact_types`` (empty = any), ``project``
    (``None`` = any), and the ``since``/``until`` timestamp bounds (inclusive,
    already normalized ISO 8601 strings) are the composable filters. ``limit`` caps
    the returned rows (top-N).
    """

    query: str
    artifact_types: tuple[ArtifactType, ...] = field(default_factory=tuple)
    project: str | None = None
    since: str | None = None
    until: str | None = None
    limit: int = DEFAULT_LIMIT


def build_match_query(query: str) -> str | None:
    """Sanitize an operator query into a safe FTS5 MATCH expression, or ``None``.

    Each whitespace-delimited token that contains at least one alphanumeric
    character becomes a quoted FTS5 phrase (inner ``"`` doubled). Quoting makes every
    FTS operator character literal, so the result can never be a malformed/hostile
    FTS query. Tokens are joined by spaces (implicit AND -- every token must match).
    Returns ``None`` when no searchable token remains (an all-punctuation or empty
    query), which the caller treats as a clean no-results search.
    """
    phrases: list[str] = []
    for token in query.split():
        if not any(char.isalnum() for char in token):
            continue  # pure punctuation -> not a searchable term; drop it
        escaped = token.replace('"', '""')
        phrases.append(f'"{escaped}"')
    if not phrases:
        return None
    return " ".join(phrases)


def normalize_date_bound(value: str, *, end: bool) -> str:
    """Validate + normalize a date/timestamp filter bound to an ISO 8601 UTC string.

    Accepts either a bare calendar date (``YYYY-MM-DD``) or a full ISO 8601
    timestamp. A bare date is expanded to the start of that day for a lower bound and
    to the last second of that day for an upper bound (``end=True``), so an inclusive
    ``--until 2026-07-27`` includes everything stamped on that day. Comparison against
    the stored ``YYYY-MM-DDTHH:MM:SSZ`` timestamps is lexicographic (both are
    fixed-width UTC), so string bounds order correctly. Raises :class:`ValueError` on
    an unparseable value (the CLI maps this to a usage error).
    """
    text = value.strip()
    try:
        datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"invalid date/timestamp filter: {value!r}") from exc
    if _DATE_ONLY_RE.fullmatch(text):
        return f"{text}T23:59:59Z" if end else f"{text}T00:00:00Z"
    return text


def _locator_from_row(source_path: str, record_locator: str | None) -> Locator:
    """Rebuild an openable :class:`Locator` from the persisted path + record locator.

    A ``L<n>`` record locator (a JSONL record) is rendered back as the openable
    ``path:<n>`` line form the JSON/JSONL adapter produced; any other record locator
    keeps its ``path::<key>`` record form; a whole-file document (no record locator)
    renders as the plain, still-openable ``path``.
    """
    if record_locator:
        match = _LINE_RECORD_RE.fullmatch(record_locator)
        if match is not None:
            line = int(match.group(1))
            return Locator(
                source_path=source_path,
                line_start=line,
                line_end=line,
                record_key=record_locator,
            )
        return Locator(source_path=source_path, record_key=record_locator)
    return Locator(source_path=source_path)


def _is_fts_query_error(exc: sqlite3.OperationalError) -> bool:
    """True for an FTS5 *query-syntax* error (as opposed to a real DB failure)."""
    message = str(exc).lower()
    return "fts5" in message or "syntax error" in message


def search(db: Database, query: SearchQuery) -> list[SearchResult]:
    """Run ``query`` against ``db``'s index, returning ranked, filtered results.

    Deterministic: ordered by ``bm25()`` ascending (most relevant first) with a
    stable ``doc_id`` tiebreak, so the same query over the same index is byte-stable
    across runs. Retrieval only -- each result's ``excerpt`` is a ``snippet()`` of the
    stored body (no synthesized text). A query with no searchable term, or one that
    matches nothing, returns an empty list (a clean no-results search, never an
    error).
    """
    if query.limit <= 0:
        raise ValueError(f"limit must be a positive integer, got {query.limit}")

    match_expr = build_match_query(query.query)
    if match_expr is None:
        return []  # no searchable term -> clean empty result

    conditions: list[str] = ["documents_fts MATCH ?"]
    params: list[object] = [match_expr]

    if query.artifact_types:
        placeholders = ", ".join("?" for _ in query.artifact_types)
        conditions.append(f"d.artifact_type IN ({placeholders})")
        params.extend(artifact_type.value for artifact_type in query.artifact_types)
    if query.project is not None:
        conditions.append("d.project = ?")
        params.append(query.project)
    if query.since is not None:
        conditions.append("d.timestamp >= ?")
        params.append(query.since)
    if query.until is not None:
        conditions.append("d.timestamp <= ?")
        params.append(query.until)

    # ``where_clause`` is assembled only from the fixed condition fragments above --
    # every one is a literal SQL string whose variable part is a bound ``?`` parameter
    # (the operator query, the filter values). Nothing operator-supplied is formatted
    # into the SQL text, so this interpolation is injection-safe; S608 is a heuristic
    # that flags any f-string SQL, hence the scoped suppression.
    where_clause = " AND ".join(conditions)
    sql = (
        "SELECT d.doc_id AS doc_id, d.source_path AS source_path, "  # noqa: S608
        "d.record_locator AS record_locator, d.artifact_type AS artifact_type, "
        "d.timestamp AS timestamp, "
        "bm25(documents_fts) AS bm25_rank, "
        "snippet(documents_fts, 0, '', '', '...', 20) AS excerpt "
        "FROM documents_fts JOIN documents d ON d.id = documents_fts.rowid "
        f"WHERE {where_clause} "
        "ORDER BY bm25_rank ASC, d.doc_id ASC "
        "LIMIT ?"
    )
    params.append(query.limit)

    try:
        rows = db.connection.execute(sql, params).fetchall()
    except sqlite3.OperationalError as exc:
        if _is_fts_query_error(exc):
            # Defensive: the phrase sanitizer makes this unreachable for operator
            # input, but a syntax-class error is still a clean no-results, not a crash.
            return []
        raise

    return [
        SearchResult(
            doc_id=row["doc_id"],
            type=ArtifactType(row["artifact_type"]),
            timestamp=row["timestamp"],
            excerpt=row["excerpt"],
            locator=_locator_from_row(row["source_path"], row["record_locator"]),
            rank=float(row["bm25_rank"]),
        )
        for row in rows
    ]


def result_to_dict(result: SearchResult) -> dict[str, object]:
    """Render one :class:`SearchResult` as a deterministic JSON-serializable dict."""
    return {
        "doc_id": result.doc_id,
        "type": result.type.value,
        "timestamp": result.timestamp,
        "rank": result.rank,
        "excerpt": result.excerpt,
        "locator": result.locator.rendered(),
    }
