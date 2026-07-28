-- Migration 0001: document metadata table.
-- One row per indexed document. The full-text body lives in the FTS5 table
-- created by migration 0002; the two are linked by rowid (documents.id ==
-- documents_fts.rowid) and kept consistent transactionally by the storage layer.
-- Columns mirror the contract-pinned IndexedDocument fields (plan.md 5); the
-- body is intentionally absent here (it belongs to the FTS index).
CREATE TABLE documents (
    id             INTEGER PRIMARY KEY,
    doc_id         TEXT NOT NULL UNIQUE,
    source_path    TEXT NOT NULL,
    record_locator TEXT,
    artifact_type  TEXT NOT NULL,
    project        TEXT NOT NULL,
    timestamp      TEXT NOT NULL,
    content_hash   TEXT NOT NULL
);

-- Secondary lookup paths used by later steps (reconciliation by path in Step 4;
-- artifact/project filters in Step 5). doc_id is already indexed by UNIQUE.
CREATE INDEX idx_documents_source_path ON documents (source_path);
CREATE INDEX idx_documents_artifact_type ON documents (artifact_type);
CREATE INDEX idx_documents_project ON documents (project);
