# find-again

A local-first utility for **unified full-text retrieval** across explicitly configured
development-memory artifacts. It incrementally indexes plans, handoffs, session state, decision
records, incident notes, and memory files, then returns ranked matches with openable source
locators — not generated answers.

## Stack

| Tool | Why |
|---|---|
| Python 3.12+ | Implementation language |
| uv | Environment + dependency management |
| SQLite + FTS5 (stdlib) | Local full-text index (no external service) |
| argparse | CLI |
| pytest | Adapter, migration, exclusion, and ranking tests |
| Ruff | Lint + format |
| mypy (strict) | Static typing |

## Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/)

## Setup

```bash
uv sync --extra dev
uv run pytest -q
uv run ruff check .
uv run mypy --strict src
```

## Usage

```bash
find-again index    # build/refresh the index from configured roots
find-again search "<query>"   # ranked results with openable path:line locators
find-again status   # index age + adapter diagnostics
```

Configuration is discovered as `find-again.toml` at the enclosing git root of cwd (`--root <path>`
override; no repo and no `--root` is an explicit error — never an implicit crawl). The index DB
lives at `<root>/.find-again/index.db` (gitignored).

```toml
schema_version = 1
roots = ["docs", ".claude/task-state"]
exclude = ["**/*.env", "**/secrets*"]
max_file_kb = 512
```

## Design decisions

- **Retrieval only.** Search returns excerpts and exact locators; it never summarizes, answers, or
  calls an LLM.
- **Explicit roots.** Configuration names every searchable root and adapter family. Git ignores,
  deny patterns, and size limits apply before content enters the database.
- **Two-layer secret exclusion.** Path-glob deny patterns (e.g. `**/*.env`, `**/secrets*`) plus a
  content-based secret-pattern scan before any DB insert. A content hit **skips the entire file**
  (never redact-and-index) and emits a diagnostic naming path + pattern id only — never the matched
  text.
- **Derived local index.** SQLite is gitignored and rebuildable; source artifacts remain
  authoritative.
- **Incremental correctness.** Content hashes (SHA-256 of file bytes) identify changes; successful
  refresh removes deleted documents and preserves diagnostics for unreadable inputs.

## Project structure

```
src/find_again/
  models.py    indexed-document / locator / diagnostic / result shapes
  config.py    explicit roots, adapter selection, exclusion policy
  db.py + migrations/   SQLite/FTS schema + migrations
  adapters/    markdown/text, json/jsonl, plan, handoff/session, decision, incident, memory
  indexer.py   discovery, hashing, update/delete reconciliation
  search.py    deterministic FTS querying + filters
  cli.py       index / search / status
```

The decision-record adapter reads YAML-frontmatter Markdown as a **format contract** (indexed keys
pinned in paper-trail's plan) — no build dependency on Paper Trail shipping. See
[plans/plan.md](plans/plan.md) for the full component + field pins.
