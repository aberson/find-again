-- Migration 0003: index-refresh metadata + persisted diagnostics (plan.md Step 4).
-- The incremental indexer (find_again.indexer) needs two things that outlive a
-- single refresh so `find-again status` can report them without re-indexing:
--   * index_meta -- a tiny key/value table; the indexer stamps `last_refreshed`
--     (ISO 8601 UTC) here after a successful refresh, so status can show index age.
--   * diagnostics -- the exclusion + adapter diagnostics from the LAST refresh
--     (path + stable code only, never file content -- the Step-1/3 invariant), so
--     status can surface a per-diagnostic summary for skipped/unreadable inputs.
-- Both are rewritten wholesale each refresh (the indexer replaces the diagnostics
-- set and upserts the meta key), so no history accumulates.
CREATE TABLE index_meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE diagnostics (
    id          INTEGER PRIMARY KEY,
    source_path TEXT NOT NULL,
    adapter     TEXT NOT NULL,
    severity    TEXT NOT NULL,
    code        TEXT NOT NULL,
    message     TEXT NOT NULL
);
