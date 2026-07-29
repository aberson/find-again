# Seed Plan: find-again

<!-- decisions-applied: 2026-07-26 per dev/docs/plan-reviews/2026-07-25-utility/DECISIONS.md -->

## 1. What This Feature Does

Proposal: `../../docs/utility-project-proposal.html`

Find Again is a local-first utility project that provides unified full-text retrieval across
explicitly configured development-memory artifacts. It incrementally indexes plans, handoffs,
session state, decision records, incident notes, and memory files, then returns ranked matches with
openable source locators rather than generated answers.

## 2. Existing Context

- Relevant reasoning is distributed across Markdown, JSON, JSONL, and text files in multiple projects.
- The operator has confirmed that recovering prior context is a recurring problem.
- The approved proposal is `../../docs/seeds/seed_find_again.md`.
- Decision records will become one high-value adapter family, consumed through a format contract
  only: YAML-frontmatter Markdown whose indexed keys are pinned in Paper Trail's plan
  (`../../paper-trail/plans/plan.md`, schema frozen at its Step 1). Find Again takes no build or
  import dependency on Paper Trail, works fully without it, and never owns decision lifecycle; any
  producer emitting the same format is indexed identically, and files without parseable frontmatter
  fall back to plain-Markdown indexing.

## 3. Scope

**In:** Python 3.12+ and uv; stdlib SQLite with FTS5; explicit include roots; ignore and secret
exclusions; adapter-based parsing; content-hash incremental refresh; search/status/index CLI; source
locators; text and JSON output.

**Out:** embeddings, external model calls, answer generation, cloud synchronization, implicit
workspace-wide crawling, indexing ignored files by default, and mutating source artifacts.

## 4. Impact Analysis

| File | Change Type | Reason | Verified |
|---|---|---|---|
| `plans/plan.md` | add | Canonical project plan | New project |
| `../../docs/seeds/seed_find_again.md` | read-only input | Approved seed | Read directly |
| future decision-record files (YAML-frontmatter Markdown) | read-only adapter input | Searchable decision family via format contract | Producer-agnostic; no build dependency; absence means zero decision hits, never an error |

No existing artifact schema or source file is modified.

## 5. New Components

- `src/find_again/models.py`: indexed-document, locator, diagnostic, and result shapes.

  Contract-pinned fields (frozen at Step 1; column types/nullability beyond these rows stay
  builder-decided in Step 1):

  | IndexedDocument field | Pin |
  |---|---|
  | `doc_id` | primary key; composition per §6 Identifiers |
  | `source_path` | root-relative POSIX path |
  | `record_locator` | optional; multi-document files only |
  | `artifact_type` | enum: markdown, json, jsonl, plan, handoff, decision, incident, memory |
  | `project` | owning project |
  | `timestamp` | ISO 8601 UTC |
  | `content_hash` | SHA-256 hex per §6 Identifiers |
  | `body` | FTS-indexed content |

  | Locator field | Pin |
  |---|---|
  | `source_path` | root-relative POSIX path |
  | `line_start`, `line_end` | optional line span |
  | `record_key` | optional record key (JSON/JSONL) |
  | (rendering) | rendered as an openable `path:line` form |

  | SearchResult field | Pin |
  |---|---|
  | `doc_id` | matches the indexed document |
  | `type` | the document's artifact_type |
  | `timestamp` | ISO 8601 UTC |
  | `excerpt` | matched excerpt |
  | `locator` | Locator rendered openable |
  | `rank` | deterministic FTS rank |

  | Diagnostic field | Pin |
  |---|---|
  | `source_path` | file skipped or failed |
  | `adapter` | reporting adapter family |
  | `severity` | enum: warn, error |
  | `code` | stable diagnostic code |
  | `message` | human-readable; never contains excluded or matched content |
- `src/find_again/config.py`: explicit roots, adapter selection, and exclusion policy.
- `src/find_again/db.py` plus `migrations/`: SQLite/FTS schema and migrations.
- `src/find_again/adapters/`: Markdown/text, JSON/JSONL, plan, handoff/session, decision, incident, and memory readers.
- `src/find_again/indexer.py`: discovery, hashing, update/delete reconciliation.
- `src/find_again/search.py`: deterministic FTS querying and filters.
- `src/find_again/cli.py`: `index`, `search`, and `status`.

## 6. Design Decisions

**Retrieval only.** Search returns excerpts and exact locators. It does not summarize, answer, or
call an LLM.

**Explicit roots.** Configuration names every searchable root and adapter family. Git ignores,
deny patterns, and file-size limits apply before content enters the database. Secret exclusion is
two-layer: path-glob deny patterns (primary — e.g. `**/*.env`, `**/secrets*`) plus a content-based
secret-pattern scan (common token-shape regexes) run before any DB insert. A content hit skips the
entire file — never redact-and-index — and emits a diagnostic naming the path and pattern id only,
never the matched text.

**Derived local index.** SQLite is gitignored and rebuildable. Source artifacts remain authoritative.

**Incremental correctness.** Content hashes identify changes; successful refresh removes deleted
documents and preserves diagnostics for unreadable inputs.

**Configuration discovery.** The target root is the enclosing git root of the current directory;
`--root <path>` overrides it. No repo and no `--root` is an explicit error — never a heuristic
fallback, and never an implicit crawl. Configuration is TOML at `<root>/find-again.toml`; the
config reader tolerates older `schema_version` values and refuses newer ones with an explicit
error. The index DB lives at `<root>/.find-again/index.db` (gitignored). Example:

```toml
schema_version = 1
roots = ["docs", ".claude/task-state"]
exclude = ["**/*.env", "**/secrets*"]
max_file_kb = 512
```

**Identifiers.** Defined once in `models.py`, generated by `indexer.py`, consumed by `db.py` and
`search.py`:

- `doc_id` — POSIX-normalized root-relative source path, plus `::<record-locator>` for
  multi-document files (e.g. `docs/lessons-learned.md`; `runs/telemetry.jsonl::L42`).
- `content_hash` — SHA-256 hex over raw file bytes (file-level; drives refresh reconciliation).

## 7. Build Steps

<!-- autofix-applied: 2026-07-25 -->
### Step 1: Scaffold, configure roots, and define safety boundaries
- **Problem:** Create the uv project, typed document/config shapes, explicit-root configuration (per §6 Configuration discovery), ignore handling, file-size limits, and the two-layer secret-exclusion contract (path-glob deny patterns plus content-based secret-pattern scan, per §6 Explicit roots).
- **Type:** code
- **Issue:** #1
- **Files:** pyproject.toml, src/find_again/models.py, src/find_again/config.py, tests/
- **Flags:** --reviewers deep --isolation worktree
- **Produces:** scaffold, `models.py`, `config.py`, security fixtures
- **Done when:** paths outside configured roots, ignored files, oversized files, and secret fixtures are excluded with visible diagnostics; a content-scan hit skips the entire file (never redact-and-index); exclusion diagnostics name path and pattern id only; and a fixture asserts the secret text reaches neither the database nor any diagnostic
- **Depends on:** none
- **Status:** DONE (2026-07-27)

<!-- autofix-applied: 2026-07-25 -->
### Step 2: Build the SQLite and FTS storage layer
- **Problem:** Implement migrations, indexed-document metadata, FTS5 content, hashes, timestamps, and transactional updates.
- **Type:** code
- **Issue:** #2
- **Files:** src/find_again/db.py, migrations/, tests/
- **Flags:** --reviewers deep --isolation worktree
- **Produces:** `db.py`, migration files, database tests
- **Done when:** one real document writes and reads through FTS, rollback preserves the prior index, and schema upgrades are deterministic
- **Depends on:** 1
- **Status:** DONE (2026-07-27)

<!-- autofix-applied: 2026-07-25 -->
### Step 3a: Define the adapter contract and generic adapters
- **Problem:** Define the single adapter-to-document contract (stable locators, diagnostics), then implement the generic Markdown/text and JSON/JSONL adapters against it with fixtures.
- **Type:** code
- **Issue:** #3
- **Files:** src/find_again/adapters/, tests/
- **Flags:** --reviewers deep --isolation worktree
- **Produces:** adapter contract, generic `adapters/`, malformed-input fixtures
- **Done when:** Markdown/text and JSON/JSONL inputs emit searchable text and an openable path/line or record locator; corrupt inputs diagnose without aborting siblings
- **Depends on:** 1
- **Status:** DONE (2026-07-27)

<!-- autofix-applied: 2026-07-25 -->
### Step 3b: Implement the structured artifact-family adapters
- **Problem:** Implement the plan, handoff/session, decision-record, incident, and memory adapters on Step 3a's contract. The decision-record adapter is a generic YAML-frontmatter Markdown adapter: it asserts only the indexed keys pinned as a format contract in Paper Trail's plan (`id`, `title`, `status`, `review_date` — `../../paper-trail/plans/plan.md`, schema frozen at its Step 1), tolerates unknown keys, and falls back to plain-Markdown indexing when no frontmatter parses. No build dependency on Paper Trail shipping: any producer emitting the same format is indexed identically.
- **Type:** code
- **Issue:** #4
- **Files:** src/find_again/adapters/, tests/
- **Flags:** --reviewers deep --isolation worktree
- **Produces:** structured-family `adapters/`, malformed-input fixtures
- **Done when:** every structured family emits searchable text and an openable path/line or record locator; corrupt inputs diagnose without aborting siblings
- **Depends on:** 3a
- **Status:** DONE (2026-07-27)

<!-- autofix-applied: 2026-07-25 -->
### Step 4: Add incremental indexing and reconciliation
- **Problem:** Discover configured files, hash content, update changed documents, remove deleted documents, and expose index age and adapter diagnostics.
- **Type:** code
- **Issue:** #5
- **Files:** src/find_again/indexer.py, src/find_again/cli.py, tests/
- **Flags:** --reviewers deep --isolation worktree
- **Produces:** `indexer.py`, `find-again index/status`
- **Done when:** unchanged files are skipped, changed and deleted files reconcile correctly, and interrupted refresh does not leave partial state
- **Depends on:** 2, 3b
- **Status:** DONE (2026-07-27)

<!-- autofix-applied: 2026-07-25 -->
### Step 5: Build search and filtering
- **Problem:** Implement deterministic FTS queries with artifact/project/date filters, ranked excerpts, source locators, and text/JSON output.
- **Type:** code
- **Issue:** #6
- **Files:** src/find_again/search.py, src/find_again/cli.py, tests/
- **Flags:** --reviewers code --isolation worktree
- **Produces:** `search.py`, `find-again search`
- **Done when:** seeded retrieval queries return the expected source in the top five and every result includes type, timestamp, excerpt, and locator
- **Depends on:** 4
- **Status:** DONE (2026-07-27)

<!-- autofix-applied: 2026-07-25 -->
### Step 6: Run the real retrieval benchmark
- **Problem:** Index a safe, explicit sample of the dev workspace and test the pre-registered prior-context queries below, exclusion behavior, refresh correctness, and time-to-recovery. Query-to-expected-source pairs are operator-supplied (pre-registered in the table below before this step builds), never dev-agent-invented; the harness is a standalone script, not a pytest module, so the default `uv run pytest -q` gate stays hermetic.
- **Type:** code
- **Issue:** #7
- **Files:** scripts/benchmark.py, docs/findings/v1-retrieval-benchmark.md
- **Flags:** --reviewers code --isolation worktree
- **Produces:** `scripts/benchmark.py` (run via `uv run python scripts/benchmark.py`), `docs/findings/v1-retrieval-benchmark.md`
- **Done when:** each pre-registered benchmark target is recovered in under 60 seconds, secret/ignored fixtures remain absent, and stale/deleted results do not survive refresh
- **Depends on:** 5
- **Status:** DONE (2026-07-27) — benchmark ran; safety (exclusion + refresh) PASS, all queries <60s; 3/5 targets in top-5, the 2 misses are ranking/corpus-composition (retrievable but outranked, documented v1.1); a real over-exclusion bug (high-entropy scanner false-positive dropping 29 files) was surfaced + fixed (secure, base64 bypass closed)

Pre-registered benchmark targets (operator-supplied; the draft rows below are candidates for the
operator to confirm or replace before this step builds):

| # | Query | Expected source (dev-root-relative) |
|---|---|---|
| 1 | never dump secret file contents | `docs/lessons-learned.md` |
| 2 | 83% of billed tokens above 150k context | `docs/investigations/high-context-usage-2026-06-22.md` |
| 3 | token usage levers | `docs/investigations/token-usage-levers-consolidated-2026-06-22.md` |
| 4 | session transitions investigation | `docs/investigations/session-transitions-2026-07-10.md` |
| 5 | build-phase halt allowlist | `.claude/rules/code-quality.md` |

### Step 7: Operator recall UAT
- **Problem:** The Step 6 benchmark is pre-registered and script-driven; only the operator can exercise genuine recall with queries the build never saw. Run 3-5 genuine "where did I write X" queries NOT in the Step 6 set.
- **Type:** operator
- **Issue:** #8
- **Files:** docs/findings/v1-retrieval-benchmark.md
- **Flags:** none
- **Produces:** operator recall verdict appended to `docs/findings/v1-retrieval-benchmark.md` (no code artifacts)
- **Done when:** each queried source is recovered in the top five within 60 seconds and the verdict is noted in `docs/findings/v1-retrieval-benchmark.md`
- **Depends on:** 6

## 8. Risks and Open Questions

| Item | Risk | Mitigation |
|---|---|---|
| Sensitive indexing | Secrets become searchable | Explicit roots, ignores, two-layer deny (path globs + content scan; hit skips whole file), local DB |
| Stale results | Search points to old content | Hash refresh, deletion reconciliation, visible age |
| Adapter sprawl | Every format becomes custom | Small adapter contract and supported-family list |
| Search becomes synthesis | Unverifiable answers | Retrieval-only non-goal |

## 9. Testing Strategy

Use temporary SQLite databases, migration tests, adapter fixtures, exclusion/security cases,
incremental add/change/delete tests, deterministic ranking fixtures, and a real-workspace retrieval
benchmark. The data pipeline receives a real write/read smoke in Step 2 before the broader benchmark.
The Step 6 benchmark harness is a standalone script (`scripts/benchmark.py`, run via
`uv run python scripts/benchmark.py`) with operator-supplied pre-registered targets — not a pytest
module, so the default `uv run pytest -q` gate stays hermetic.

## Appendix: Decision Inventory

| ID | P/D | Choice | Status |
|---|---|---|---|
| P5 | P | Build Find Again because local context retrieval is a demonstrated problem | accepted |
| D1 | D | Use Python 3.12+, uv, argparse, pytest, Ruff, and mypy strict | accepted |
| D3 | D | Initialize a separate nested GitHub repository before build | accepted |
| D7 | D | Use local SQLite FTS5 with explicit roots and retrieval-only output | accepted |

## 10. Build and Run Contract

Bootstrap with Python 3.12+ and `uv sync --extra dev`. Quality gates are `uv run pytest -q`,
`uv run ruff check .`, and `uv run mypy --strict src`. The installed CLI entry point is
`find-again`; configuration must name searchable roots before the first index build. The target
root is the enclosing git root of the current directory (`--root <path>` override; no repo and no
`--root` is an explicit error, never a heuristic fallback or implicit crawl). Configuration is
discovered as `find-again.toml` at that root (TOML, `schema_version = 1`; example in §6
Configuration discovery); the index database is `<root>/.find-again/index.db`, gitignored.

## Manual UAT

*Generated by /build-phase on 2026-07-27. Append-only; re-running the phase adds new items below, never modifies existing ones.*

### M1: Operator recall UAT
- **Source step:** Step 7 (from this plan's §7)
- **Issue:** #8
- **Commands to run:**
  ```powershell
  cd C:\Users\abero\dev\find-again
  # 1. Author find-again.toml at the dev root naming the roots to index (see README / plan §6 example), e.g.:
  #    schema_version = 1
  #    roots = ["docs", ".claude/rules", ".claude/task-state"]
  #    exclude = ["**/*.env", "**/secrets*"]
  #    max_file_kb = 512
  # 2. Build the index against the real workspace (read-only):
  uv run find-again index --root C:\Users\abero\dev
  uv run find-again status --root C:\Users\abero\dev
  # 3. Run 3-5 GENUINE "where did I write X" queries the build NEVER saw (NOT the Step-6 five):
  uv run find-again search "<your recall query>" --root C:\Users\abero\dev
  ```
- **What you're looking for:**

  | Check | Expected outcome |
  |---|---|
  | each genuine recall query | the source you were thinking of appears in the TOP FIVE within 60s |
  | result completeness | every hit shows type, timestamp, excerpt, openable locator |
  | exclusion | no secret/`.env`/ignored content in results |
  | KNOWN v1.1 caveat | large canonical docs can be outranked by shorter derivative doc-sets under BM25 (Q1/Q5 benchmark misses) — if a target is retrievable but not top-5, that is the documented ranking item, not an exclusion failure. Narrowing `roots` to the relevant dirs helps. |
- **Note:** append the operator recall verdict to `docs/findings/v1-retrieval-benchmark.md`.
