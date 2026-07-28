"""Command-line entry point (Step 1 stub).

The ``index``, ``search``, and ``status`` verbs are declared here so the console
entry point and ``--help`` are real, but their behavior lands in later build
steps (indexer -> Step 4, search -> Step 5). This stub performs no indexing and
touches no database.
"""

from __future__ import annotations

import argparse

from . import __version__

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

    subparsers = parser.add_subparsers(dest="command", metavar="{index,search,status}")
    subparsers.add_parser("index", help="Build or refresh the index from configured roots.")
    search = subparsers.add_parser("search", help="Search the index for a query.")
    search.add_argument("query", help="Full-text query string.")
    subparsers.add_parser("status", help="Show index age and adapter diagnostics.")
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns a process exit code."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return 2

    print(f"find-again {args.command}: not implemented yet (Step 1 scaffold).")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
