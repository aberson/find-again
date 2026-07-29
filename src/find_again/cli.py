"""Command-line entry point (plan.md §5 cli.py, §10 Build and Run Contract).

The ``index``, ``search``, and ``status`` verbs land here. ``index`` resolves the
target root, loads config, and reconciles the index
(:func:`find_again.indexer.refresh_index`), reporting new/updated/unchanged/removed
counts plus diagnostics. ``search`` runs a deterministic FTS query
(:func:`find_again.search.search`) with artifact/project/date filters over the
resolved-root index and renders ranked excerpts + openable locators. ``status``
reports index age, document count, a per-diagnostic summary, and the configured
roots. All support ``--json`` (machine output) and a human text form, and use sane
exit codes:

* ``0`` -- success (including a clean no-results search).
* ``2`` -- configuration / root-resolution / usage failure (bad or missing config,
  no repo and no ``--root``, a schema newer than this build, a bad ``--since`` /
  ``--until`` / ``--limit`` value).
* ``1`` -- an unexpected storage/database failure.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from datetime import UTC, datetime
from pathlib import Path

from . import __version__
from .config import Config, ConfigError, load_config, resolve_root
from .db import Database, DatabaseError, NewerDatabaseError
from .indexer import RefreshResult, refresh_index
from .models import ArtifactType, Diagnostic, SearchResult
from .search import (
    DEFAULT_LIMIT,
    SEARCH_SCHEMA_VERSION,
    SearchQuery,
    normalize_date_bound,
    result_to_dict,
    search,
)

__all__ = ["build_parser", "main"]


def build_parser() -> argparse.ArgumentParser:
    """Construct the argument parser for the ``find-again`` CLI."""
    parser = argparse.ArgumentParser(
        prog="find-again",
        description="Local-first full-text retrieval over development-memory artifacts.",
    )
    parser.add_argument("--version", action="version", version=f"find-again {__version__}")
    parser.add_argument(
        "--root",
        default=None,
        help="Override the target root (default: the enclosing git root of cwd).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON instead of text.",
    )

    subparsers = parser.add_subparsers(dest="command", metavar="{index,search,status}")
    index = subparsers.add_parser("index", help="Build or refresh the index from configured roots.")
    index.add_argument(
        "--no-git-ignore",
        action="store_true",
        help="Skip the git-ignore check (path-glob + content secret layers still apply).",
    )
    search_parser = subparsers.add_parser("search", help="Search the index for a query.")
    search_parser.add_argument("query", help="Full-text query string (plain terms; AND-ed).")
    search_parser.add_argument(
        "--type",
        action="append",
        dest="types",
        default=None,
        metavar="TYPE",
        choices=[artifact_type.value for artifact_type in ArtifactType],
        help="Filter by artifact type (repeatable; OR-ed). One of: "
        + ", ".join(artifact_type.value for artifact_type in ArtifactType),
    )
    search_parser.add_argument(
        "--project",
        default=None,
        help="Filter to a single owning project.",
    )
    search_parser.add_argument(
        "--since",
        default=None,
        metavar="DATE",
        help="Only results at/after this date or ISO timestamp (inclusive).",
    )
    search_parser.add_argument(
        "--until",
        default=None,
        metavar="DATE",
        help="Only results at/before this date or ISO timestamp (inclusive).",
    )
    search_parser.add_argument(
        "--limit",
        type=_positive_int,
        default=DEFAULT_LIMIT,
        metavar="N",
        help=f"Maximum number of results (default: {DEFAULT_LIMIT}).",
    )
    subparsers.add_parser("status", help="Show index age, document count, and diagnostics.")
    return parser


def _positive_int(value: str) -> int:
    """argparse type for ``--limit``: a strictly positive integer (else a usage error)."""
    try:
        parsed = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"invalid int value: {value!r}") from None
    if parsed <= 0:
        raise argparse.ArgumentTypeError(f"must be a positive integer, got {parsed}")
    return parsed


# --------------------------------------------------------------------------- #
# Shared helpers
# --------------------------------------------------------------------------- #
def _resolve(args: argparse.Namespace) -> tuple[Path, Config]:
    """Resolve the target root and load its config (raises :class:`ConfigError`)."""
    root = resolve_root(Path.cwd(), args.root)
    return root, load_config(root)


def _diagnostic_dict(diag: Diagnostic) -> dict[str, str]:
    return {
        "source_path": diag.source_path,
        "adapter": diag.adapter,
        "severity": diag.severity.value,
        "code": diag.code,
        "message": diag.message,
    }


def _summarize_diagnostics(diagnostics: list[Diagnostic]) -> list[str]:
    """One line per (severity, adapter, code) group with a count, sorted for stability."""
    counts: dict[tuple[str, str, str], int] = {}
    for diag in diagnostics:
        key = (diag.severity.value, diag.adapter, diag.code)
        counts[key] = counts.get(key, 0) + 1
    return [
        f"  {severity:<5} {adapter}/{code}: {count}"
        for (severity, adapter, code), count in sorted(counts.items())
    ]


def _humanize_age(seconds: float) -> str:
    """Render an age in seconds as a compact ``Nd Nh Nm Ns`` string (largest 2 units)."""
    total = int(max(0, seconds))
    days, rem = divmod(total, 86_400)
    hours, rem = divmod(rem, 3_600)
    minutes, secs = divmod(rem, 60)
    parts = [(days, "d"), (hours, "h"), (minutes, "m"), (secs, "s")]
    nonzero = [f"{value}{unit}" for value, unit in parts if value]
    return " ".join(nonzero[:2]) if nonzero else "0s"


# --------------------------------------------------------------------------- #
# index
# --------------------------------------------------------------------------- #
def _print_index_text(root: Path, result: RefreshResult) -> None:
    print(f"Indexed {root} ({root.name or 'root'})")
    print(f"  new:       {result.indexed}")
    print(f"  updated:   {result.updated}")
    print(f"  unchanged: {result.skipped}")
    print(f"  removed:   {result.deleted}")
    print(f"  documents: {result.documents} total")
    diagnostics = list(result.diagnostics)
    warns = sum(1 for d in diagnostics if d.severity.value == "warn")
    errors = sum(1 for d in diagnostics if d.severity.value == "error")
    print(f"  diagnostics: {len(diagnostics)} (warn: {warns}, error: {errors})")
    for line in _summarize_diagnostics(diagnostics):
        print(line)


def _run_index(args: argparse.Namespace, root: Path, config: Config) -> int:
    with Database.open_root(root) as db:
        result = refresh_index(config, db, use_git_ignore=not args.no_git_ignore)
    if args.json:
        print(
            json.dumps(
                {
                    "root": str(root),
                    "project": root.name or "root",
                    "indexed": result.indexed,
                    "updated": result.updated,
                    "skipped": result.skipped,
                    "deleted": result.deleted,
                    "documents": result.documents,
                    "refreshed_at": result.refreshed_at,
                    "diagnostics": [_diagnostic_dict(d) for d in result.diagnostics],
                },
                indent=2,
            )
        )
    else:
        _print_index_text(root, result)
    return 0


# --------------------------------------------------------------------------- #
# status
# --------------------------------------------------------------------------- #
def _parse_iso(value: str) -> datetime | None:
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _age_seconds(last_refreshed: str | None) -> float | None:
    if last_refreshed is None:
        return None
    stamped = _parse_iso(last_refreshed)
    if stamped is None:
        return None
    return (datetime.now(tz=UTC) - stamped).total_seconds()


def _run_status(args: argparse.Namespace, root: Path, config: Config) -> int:
    with Database.open_root(root) as db:
        documents = db.count_documents()
        last_refreshed = db.get_meta("last_refreshed")
        diagnostics = db.get_diagnostics()
    age = _age_seconds(last_refreshed)

    if args.json:
        print(
            json.dumps(
                {
                    "root": str(root),
                    "project": root.name or "root",
                    "roots": list(config.roots),
                    "documents": documents,
                    "last_refreshed": last_refreshed,
                    "age_seconds": None if age is None else int(age),
                    "diagnostics": [_diagnostic_dict(d) for d in diagnostics],
                },
                indent=2,
            )
        )
        return 0

    print(f"Index status for {root} ({root.name or 'root'})")
    if last_refreshed is None:
        print("  last refreshed: never (run `find-again index`)")
    else:
        age_text = "unknown" if age is None else f"{_humanize_age(age)} ago"
        print(f"  last refreshed: {last_refreshed} ({age_text})")
    print(f"  documents: {documents}")
    print(f"  roots: {', '.join(config.roots) if config.roots else '(none configured)'}")
    warns = sum(1 for d in diagnostics if d.severity.value == "warn")
    errors = sum(1 for d in diagnostics if d.severity.value == "error")
    print(f"  diagnostics: {len(diagnostics)} (warn: {warns}, error: {errors})")
    for line in _summarize_diagnostics(diagnostics):
        print(line)
    return 0


# --------------------------------------------------------------------------- #
# search
# --------------------------------------------------------------------------- #
def _build_search_query(args: argparse.Namespace) -> SearchQuery:
    """Assemble a :class:`SearchQuery` from parsed args (raises ``ValueError`` on bad dates).

    ``--type`` values are validated by argparse ``choices`` (safe to map straight to
    the enum); ``--since`` / ``--until`` are normalized here so an unparseable date
    surfaces as a usage error.
    """
    types = tuple(ArtifactType(value) for value in (args.types or ()))
    since = None if args.since is None else normalize_date_bound(args.since, end=False)
    until = None if args.until is None else normalize_date_bound(args.until, end=True)
    return SearchQuery(
        query=args.query,
        artifact_types=types,
        project=args.project,
        since=since,
        until=until,
        limit=args.limit,
    )


def _print_search_text(query: str, results: list[SearchResult], *, empty_index: bool) -> None:
    if not results:
        print(f'No matches for "{query}".')
        if empty_index:
            print("  (the index is empty -- run `find-again index` first)")
        return
    noun = "result" if len(results) == 1 else "results"
    print(f'{len(results)} {noun} for "{query}":')
    for position, result in enumerate(results, start=1):
        print(f"{position}. {result.locator.rendered()}  ({result.type.value}, {result.timestamp})")
        excerpt = " ".join(result.excerpt.split())  # collapse newlines for a one-line excerpt
        if excerpt:
            print(f"   {excerpt}")


def _run_search(args: argparse.Namespace, root: Path, config: Config) -> int:
    query = _build_search_query(args)
    with Database.open_root(root) as db:
        results = search(db, query)
        empty_index = db.count_documents() == 0
    if args.json:
        print(
            json.dumps(
                {
                    "schema_version": SEARCH_SCHEMA_VERSION,
                    "query": query.query,
                    "count": len(results),
                    "results": [result_to_dict(result) for result in results],
                },
                indent=2,
            )
        )
    else:
        _print_search_text(query.query, results, empty_index=empty_index)
    return 0


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns a process exit code."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return 2

    try:
        root, config = _resolve(args)
    except ConfigError as exc:
        print(f"find-again: {exc}", file=sys.stderr)
        return 2

    try:
        if args.command == "index":
            return _run_index(args, root, config)
        if args.command == "search":
            return _run_search(args, root, config)
        return _run_status(args, root, config)
    except ValueError as exc:
        # A bad --since / --until date is a usage-class failure (exit 2), consistent
        # with the argparse-validated --type / --limit checks.
        print(f"find-again: {exc}", file=sys.stderr)
        return 2
    except NewerDatabaseError as exc:
        # A schema newer than this build is a usage/config-class failure (exit 2),
        # consistent with the config schema_version policy and the docstring above.
        print(f"find-again: {exc}", file=sys.stderr)
        return 2
    except DatabaseError as exc:
        print(f"find-again: {exc}", file=sys.stderr)
        return 1
    except sqlite3.Error as exc:
        # A corrupt / locked / otherwise-unopenable index DB surfaces here as a raw
        # sqlite3 error; report it cleanly (exit 1) instead of leaking a traceback.
        print(f"find-again: could not open or read the index database: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
