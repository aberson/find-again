"""Tests for deterministic FTS search + filtering (plan.md Step 5, §5 search.py, §9).

Covers the done-when + scope conditions:

* SEEDED retrieval: index a small fixture corpus, run queries, and assert the
  expected source is in the TOP FIVE and every result carries type / timestamp /
  excerpt / locator;
* DETERMINISTIC ordering: the same query returns an identical result order across
  runs (bm25 rank + stable doc_id tiebreak);
* FILTERS: artifact_type, project, and date-range narrow correctly and compose;
* FTS query SAFETY: a malformed / hostile query is handled cleanly (no crash, no SQL
  injection -- the documents table survives);
* NO-RESULTS: a query that matches nothing is a clean empty result / exit 0;
* EXCERPT fidelity: the excerpt surrounds the matched term and is genuine source
  text (retrieval-only -- no synthesized text).

Retrieval corpora are built through the real indexer into temp index DBs; the
metadata-only filter cases (project, date) upsert documents directly so those axes
can be varied independently of a single root/mtime.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import pytest

from find_again.cli import main
from find_again.config import Config
from find_again.db import Database
from find_again.indexer import refresh_index
from find_again.models import ArtifactType, IndexedDocument, content_hash
from find_again.search import (
    DEFAULT_LIMIT,
    EXCERPT_ELLIPSIS,
    SEARCH_SCHEMA_VERSION,
    SearchQuery,
    build_match_query,
    normalize_date_bound,
    result_to_dict,
    search,
)

# A live AWS-shaped access-key id (fixture only) the content scanner must catch; the
# CLI injection test also proves a crafted query never drops a table.
SECRET_TOKEN = "AKIAIOSFODNN7EXAMPLE"  # noqa: S105 -- test fixture, not a real credential


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _write(root: Path, rel: str, text: str) -> None:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _config(root: Path, roots: tuple[str, ...] = ("docs",)) -> Config:
    return Config(root=root, roots=roots, exclude=(), max_file_kb=512, schema_version=1)


def _seed_corpus(tmp_path: Path, files: dict[str, str]) -> Database:
    """Write ``files`` under ``tmp_path`` and index them into a fresh in-memory store."""
    for rel, text in files.items():
        _write(tmp_path, rel, text)
    database = Database.open_memory()
    refresh_index(_config(tmp_path), database, use_git_ignore=False)
    return database


def _upsert(
    database: Database,
    *,
    doc_id: str,
    body: str,
    artifact_type: ArtifactType = ArtifactType.MARKDOWN,
    project: str = "find-again",
    timestamp: str = "2026-07-27T00:00:00Z",
    source_path: str | None = None,
    record_locator: str | None = None,
) -> None:
    """Insert one document with fully-controlled metadata (for filter-axis tests)."""
    database.upsert_document(
        IndexedDocument(
            doc_id=doc_id,
            source_path=source_path or doc_id,
            artifact_type=artifact_type,
            project=project,
            timestamp=timestamp,
            content_hash=content_hash(body.encode("utf-8")),
            body=body,
            record_locator=record_locator,
        )
    )


@pytest.fixture
def store() -> Iterator[Database]:
    database = Database.open_memory()
    try:
        yield database
    finally:
        database.close()


# --------------------------------------------------------------------------- #
# build_match_query: safe phrase sanitization
# --------------------------------------------------------------------------- #
def test_build_match_query_quotes_each_token() -> None:
    assert build_match_query("token usage levers") == '"token" "usage" "levers"'


def test_build_match_query_escapes_inner_quotes() -> None:
    # An inner double-quote is doubled (FTS5 phrase escaping), never left to open a
    # new phrase or a syntax error.
    assert build_match_query('a"b') == '"a""b"'


def test_build_match_query_drops_pure_punctuation() -> None:
    assert build_match_query("   ") is None
    assert build_match_query("!@#  ---") is None
    # A mix keeps only the alphanumeric-bearing tokens.
    assert build_match_query("-- foo ;;") == '"foo"'


# --------------------------------------------------------------------------- #
# normalize_date_bound
# --------------------------------------------------------------------------- #
def test_normalize_date_bound_expands_bare_dates() -> None:
    assert normalize_date_bound("2026-07-27", end=False) == "2026-07-27T00:00:00Z"
    assert normalize_date_bound("2026-07-27", end=True) == "2026-07-27T23:59:59Z"


def test_normalize_date_bound_passes_through_full_timestamps() -> None:
    assert normalize_date_bound("2026-07-27T12:30:00Z", end=True) == "2026-07-27T12:30:00Z"


def test_normalize_date_bound_rejects_garbage() -> None:
    with pytest.raises(ValueError, match="invalid date"):
        normalize_date_bound("not-a-date", end=False)


# --------------------------------------------------------------------------- #
# Seeded retrieval: expected source in the top five + result completeness
# --------------------------------------------------------------------------- #
def test_seeded_retrieval_expected_source_in_top_five_and_complete(tmp_path: Path) -> None:
    corpus = {
        "docs/lessons-learned.md": (
            "Never dump secret file contents from a bash pipeline; the filter defenses "
            "fail silently across shell boundaries. Metadata-only checks are safe."
        ),
        "docs/decoy-a.md": "a document about scheduling and calendars\n",
        "docs/decoy-b.md": "notes about vite dev ports and windows\n",
        "docs/decoy-c.md": "a memory about worktree hygiene\n",
        "docs/decoy-d.md": "an incident about a database lock\n",
        "docs/decoy-e.md": "unrelated filler content about colors\n",
    }
    database = _seed_corpus(tmp_path, corpus)

    results = search(database, SearchQuery(query="never dump secret file contents"))

    assert results, "expected at least one hit"
    top_five_ids = [r.doc_id for r in results[:5]]
    assert "docs/lessons-learned.md" in top_five_ids

    # Every result is complete: type (enum), timestamp (ISO Z), a non-empty excerpt,
    # and an openable locator.
    for result in results:
        assert isinstance(result.type, ArtifactType)
        assert result.timestamp.endswith("Z")
        assert isinstance(result.excerpt, str) and result.excerpt.strip()
        assert result.locator.rendered()  # openable, non-empty
        assert isinstance(result.rank, float)


def test_search_respects_limit(tmp_path: Path) -> None:
    corpus = {f"docs/n{i}.md": f"shared_marker doc number {i}\n" for i in range(8)}
    database = _seed_corpus(tmp_path, corpus)

    assert len(search(database, SearchQuery(query="shared_marker", limit=3))) == 3
    assert len(search(database, SearchQuery(query="shared_marker"))) == min(8, DEFAULT_LIMIT)


# --------------------------------------------------------------------------- #
# Deterministic ordering: same query -> identical order across runs
# --------------------------------------------------------------------------- #
def test_ordering_is_identical_across_runs(tmp_path: Path) -> None:
    corpus = {
        "docs/rich.md": "token token token token token usage here\n",
        "docs/mid.md": "token token appears twice with usage\n",
        "docs/poor.md": "token appears once amid much other filler text and words\n",
    }
    database = _seed_corpus(tmp_path, corpus)

    first = search(database, SearchQuery(query="token"))
    second = search(database, SearchQuery(query="token"))
    # Frozen dataclasses compare by value: identical rank, excerpt, locator, order.
    assert first == second
    assert [r.doc_id for r in first] == [r.doc_id for r in second]


def test_ranking_prefers_more_relevant_document(tmp_path: Path) -> None:
    corpus = {
        "docs/rich.md": "token token token token token usage here\n",
        "docs/poor.md": "token appears once amid much other filler text and words here\n",
    }
    database = _seed_corpus(tmp_path, corpus)

    results = search(database, SearchQuery(query="token"))
    assert [r.doc_id for r in results][0] == "docs/rich.md"
    # bm25: a better match sorts first (ascending), so rank is non-decreasing.
    assert results[0].rank <= results[-1].rank


def test_tie_breaks_deterministically_by_doc_id(store: Database) -> None:
    # Byte-identical bodies -> identical bm25 -> the doc_id tiebreak decides, stably.
    _upsert(store, doc_id="docs/zzz.md", source_path="docs/zzz.md", body="tiebreak marker term")
    _upsert(store, doc_id="docs/aaa.md", source_path="docs/aaa.md", body="tiebreak marker term")

    results = search(store, SearchQuery(query="tiebreak"))
    assert [r.doc_id for r in results] == ["docs/aaa.md", "docs/zzz.md"]


# --------------------------------------------------------------------------- #
# Filters: artifact_type / project / date, composable
# --------------------------------------------------------------------------- #
def test_artifact_type_filter_narrows(tmp_path: Path) -> None:
    corpus = {
        "docs/note.md": "widget alpha content\n",
        "docs/data.jsonl": '{"k": "widget beta content"}\n',
    }
    database = _seed_corpus(tmp_path, corpus)

    md_only = search(database, SearchQuery(query="widget", artifact_types=(ArtifactType.MARKDOWN,)))
    assert [r.doc_id for r in md_only] == ["docs/note.md"]
    assert md_only[0].type is ArtifactType.MARKDOWN

    jsonl_only = search(database, SearchQuery(query="widget", artifact_types=(ArtifactType.JSONL,)))
    assert [r.doc_id for r in jsonl_only] == ["docs/data.jsonl::L1"]
    assert jsonl_only[0].type is ArtifactType.JSONL
    # The JSONL record locator renders as the openable path:line form.
    assert jsonl_only[0].locator.rendered() == "docs/data.jsonl:1"

    both = search(
        database,
        SearchQuery(query="widget", artifact_types=(ArtifactType.MARKDOWN, ArtifactType.JSONL)),
    )
    assert {r.doc_id for r in both} == {"docs/note.md", "docs/data.jsonl::L1"}


def test_project_filter_narrows(store: Database) -> None:
    _upsert(store, doc_id="a.md", body="projterm one", project="alpha")
    _upsert(store, doc_id="b.md", body="projterm two", project="beta")
    _upsert(store, doc_id="c.md", body="projterm three", project="beta")

    beta = search(store, SearchQuery(query="projterm", project="beta"))
    assert {r.doc_id for r in beta} == {"b.md", "c.md"}

    alpha = search(store, SearchQuery(query="projterm", project="alpha"))
    assert {r.doc_id for r in alpha} == {"a.md"}


def test_date_range_filter_narrows_and_composes(store: Database) -> None:
    _upsert(store, doc_id="jan.md", body="dateterm one", timestamp="2026-01-15T00:00:00Z")
    _upsert(store, doc_id="jun.md", body="dateterm two", timestamp="2026-06-15T00:00:00Z")
    _upsert(store, doc_id="dec.md", body="dateterm three", timestamp="2026-12-15T00:00:00Z")

    since = search(
        store,
        SearchQuery(query="dateterm", since=normalize_date_bound("2026-06-01", end=False)),
    )
    assert {r.doc_id for r in since} == {"jun.md", "dec.md"}

    until = search(
        store,
        SearchQuery(query="dateterm", until=normalize_date_bound("2026-06-15", end=True)),
    )
    assert {r.doc_id for r in until} == {"jan.md", "jun.md"}

    # since + until compose to a window that isolates a single month.
    window = search(
        store,
        SearchQuery(
            query="dateterm",
            since=normalize_date_bound("2026-06-01", end=False),
            until=normalize_date_bound("2026-06-30", end=True),
        ),
    )
    assert {r.doc_id for r in window} == {"jun.md"}


def test_filters_compose_with_query_and_type(store: Database) -> None:
    _upsert(
        store,
        doc_id="a.jsonl::L1",
        source_path="a.jsonl",
        record_locator="L1",
        body="composeterm hit",
        artifact_type=ArtifactType.JSONL,
        project="beta",
        timestamp="2026-06-15T00:00:00Z",
    )
    _upsert(
        store,
        doc_id="b.md",
        body="composeterm miss-type",
        artifact_type=ArtifactType.MARKDOWN,
        project="beta",
        timestamp="2026-06-15T00:00:00Z",
    )

    results = search(
        store,
        SearchQuery(
            query="composeterm",
            artifact_types=(ArtifactType.JSONL,),
            project="beta",
            since=normalize_date_bound("2026-06-01", end=False),
            until=normalize_date_bound("2026-06-30", end=True),
        ),
    )
    assert [r.doc_id for r in results] == ["a.jsonl::L1"]


# --------------------------------------------------------------------------- #
# Excerpt fidelity: surrounds the match; retrieval-only (genuine source text)
# --------------------------------------------------------------------------- #
def test_excerpt_contains_matched_term_and_is_source_text(tmp_path: Path) -> None:
    body = (
        "lots of leading context words here before the distinctive_needle term "
        "and then a good deal of trailing context words after it to force a window"
    )
    database = _seed_corpus(tmp_path, {"docs/hay.md": body + "\n"})

    results = search(database, SearchQuery(query="distinctive_needle"))
    assert results
    excerpt = results[0].excerpt
    # The excerpt surrounds / contains the matched term.
    assert "distinctive_needle" in excerpt

    # Retrieval-only: every excerpt segment (between ellipses) is genuine, contiguous
    # source text -- nothing is synthesized. Fetch the stored body to compare.
    stored = database.get_document("docs/hay.md")
    assert stored is not None
    for segment in excerpt.split(EXCERPT_ELLIPSIS):
        segment = segment.strip()
        if segment:
            assert segment in stored.body


# --------------------------------------------------------------------------- #
# No-results is a clean empty search (not an error)
# --------------------------------------------------------------------------- #
def test_no_match_returns_empty(tmp_path: Path) -> None:
    database = _seed_corpus(tmp_path, {"docs/a.md": "ordinary content here\n"})
    assert search(database, SearchQuery(query="zzz_absent_term_zzz")) == []


def test_empty_query_returns_empty(tmp_path: Path) -> None:
    database = _seed_corpus(tmp_path, {"docs/a.md": "ordinary content here\n"})
    assert search(database, SearchQuery(query="   ")) == []
    assert search(database, SearchQuery(query="!@#---")) == []


# --------------------------------------------------------------------------- #
# FTS query safety: malformed / hostile queries are handled cleanly, no injection
# --------------------------------------------------------------------------- #
def test_malformed_and_hostile_queries_do_not_crash_or_inject(tmp_path: Path) -> None:
    database = _seed_corpus(tmp_path, {"docs/a.md": "ordinary content here\n"})
    before = database.count_documents()

    hostile = [
        'content"; DROP TABLE documents; --',
        "AND OR NOT NEAR",
        '"',
        "foo* (bar",
        "col:val ^ NEAR/2",
        "*",
        ")))(((",
    ]
    for query in hostile:
        # Must not raise, must return a list -- a clean result or empty.
        assert isinstance(search(database, SearchQuery(query=query)), list)

    # No injection occurred: the table and its row are intact.
    assert database.count_documents() == before
    assert database.get_document("docs/a.md") is not None
    assert database.check_consistency()


def test_search_rejects_non_positive_limit(store: Database) -> None:
    with pytest.raises(ValueError, match="positive"):
        search(store, SearchQuery(query="anything", limit=0))


# --------------------------------------------------------------------------- #
# result_to_dict: deterministic JSON-serializable shape
# --------------------------------------------------------------------------- #
def test_result_to_dict_shape(tmp_path: Path) -> None:
    database = _seed_corpus(tmp_path, {"docs/a.md": "serialize_me content here\n"})
    result = search(database, SearchQuery(query="serialize_me"))[0]
    payload = result_to_dict(result)
    assert set(payload) == {"doc_id", "type", "timestamp", "rank", "excerpt", "locator"}
    assert payload["type"] == "markdown"
    assert payload["locator"] == "docs/a.md"
    # Round-trips through json without error (deterministic, serializable).
    assert json.loads(json.dumps(payload))["doc_id"] == "docs/a.md"


# --------------------------------------------------------------------------- #
# CLI: text + JSON output, exit codes, filters, injection safety
# --------------------------------------------------------------------------- #
def _project_with_index(tmp_path: Path, files: dict[str, str]) -> None:
    _write(tmp_path, "find-again.toml", 'schema_version = 1\nroots = ["docs"]\n')
    for rel, text in files.items():
        _write(tmp_path, rel, text)
    assert main(["--root", str(tmp_path), "index", "--no-git-ignore"]) == 0


def test_cli_search_text_output(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _project_with_index(tmp_path, {"docs/note.md": "the special_marker appears here\n"})
    capsys.readouterr()  # drain the index output

    assert main(["--root", str(tmp_path), "search", "special_marker"]) == 0
    out = capsys.readouterr().out
    assert "special_marker" in out
    assert "docs/note.md" in out
    assert "1 result for" in out


def test_cli_search_json_output(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _project_with_index(tmp_path, {"docs/note.md": "the special_marker appears here\n"})
    capsys.readouterr()

    assert main(["--root", str(tmp_path), "--json", "search", "special_marker"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["schema_version"] == SEARCH_SCHEMA_VERSION
    assert payload["query"] == "special_marker"
    assert payload["count"] == 1
    result = payload["results"][0]
    assert set(result) == {"doc_id", "type", "timestamp", "rank", "excerpt", "locator"}
    assert result["type"] == "markdown"
    assert result["locator"].startswith("docs/note.md")
    assert "special_marker" in result["excerpt"]


def test_cli_search_no_match_is_exit_0(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _project_with_index(tmp_path, {"docs/note.md": "content here\n"})
    capsys.readouterr()

    assert main(["--root", str(tmp_path), "search", "zzz_no_such_term"]) == 0
    assert "No matches" in capsys.readouterr().out


def test_cli_search_empty_index_hint(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    # A configured root that was never indexed: clean exit 0 with a hint to index.
    _write(tmp_path, "find-again.toml", 'roots = ["docs"]\n')
    assert main(["--root", str(tmp_path), "search", "anything"]) == 0
    out = capsys.readouterr().out
    assert "No matches" in out
    assert "index is empty" in out


def test_cli_search_type_filter(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _project_with_index(
        tmp_path,
        {"docs/note.md": "widget alpha\n", "docs/data.jsonl": '{"k": "widget beta"}\n'},
    )
    capsys.readouterr()

    assert main(["--root", str(tmp_path), "--json", "search", "widget", "--type", "markdown"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["count"] == 1
    assert payload["results"][0]["type"] == "markdown"

    assert main(["--root", str(tmp_path), "--json", "search", "widget", "--type", "jsonl"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["count"] == 1
    assert payload["results"][0]["type"] == "jsonl"
    assert payload["results"][0]["locator"] == "docs/data.jsonl:1"


def test_cli_search_bad_type_is_usage_error(tmp_path: Path) -> None:
    _write(tmp_path, "find-again.toml", 'roots = ["docs"]\n')
    with pytest.raises(SystemExit) as excinfo:
        main(["--root", str(tmp_path), "search", "x", "--type", "bogus"])
    assert excinfo.value.code == 2


def test_cli_search_bad_limit_is_usage_error(tmp_path: Path) -> None:
    _write(tmp_path, "find-again.toml", 'roots = ["docs"]\n')
    with pytest.raises(SystemExit) as excinfo:
        main(["--root", str(tmp_path), "search", "x", "--limit", "0"])
    assert excinfo.value.code == 2


def test_cli_search_bad_date_is_exit_2(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _project_with_index(tmp_path, {"docs/note.md": "content here\n"})
    capsys.readouterr()

    code = main(["--root", str(tmp_path), "search", "content", "--since", "not-a-date"])
    assert code == 2
    assert "find-again:" in capsys.readouterr().err


def test_cli_search_malformed_query_is_clean(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _project_with_index(tmp_path, {"docs/note.md": "ordinary content here\n"})
    capsys.readouterr()

    # A crafted injection-shaped query exits cleanly (0), never a crash or a dropped table.
    code = main(["--root", str(tmp_path), "search", 'x"; DROP TABLE documents; --'])
    assert code == 0
    capsys.readouterr()

    # The index survived: status still reports the one document.
    assert main(["--root", str(tmp_path), "--json", "status"]) == 0
    assert json.loads(capsys.readouterr().out)["documents"] == 1
