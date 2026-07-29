"""Command-line entry point (plan.md §5 cli.py, §10 Build and Run Contract).

The ``index`` and ``status`` verbs land here (Step 4); ``search`` remains a stub
until Step 5. ``index`` resolves the target root, loads config, and reconciles the
index (:func:`find_again.indexer.refresh_index`), reporting new/updated/unchanged/
removed counts plus diagnostics. ``status`` reports index age, document count, a
per-diagnostic summary, and the configured roots. Both support ``--json`` (machine
output) and a human text form, and use sane exit codes:

* ``0`` -- success.
* ``2`` -- configuration / root-resolution failure (bad or missing config, no repo
  and no ``--root``, a schema newer than this build).
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
from .models import Diagnostic

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
    search = subparsers.add_parser("search", help="Search the index for a query.")
    search.add_argument("query", help="Full-text query string.")
    subparsers.add_parser("status", help="Show index age, document count, and diagnostics.")
    return parser


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
# Entry point
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns a process exit code."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return 2

    if args.command == "search":
        print("find-again search: not implemented yet (Step 5).", file=sys.stderr)
        return 2

    try:
        root, config = _resolve(args)
    except ConfigError as exc:
        print(f"find-again: {exc}", file=sys.stderr)
        return 2

    try:
        if args.command == "index":
            return _run_index(args, root, config)
        return _run_status(args, root, config)
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
