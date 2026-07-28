"""Tests for the contract-pinned shapes and identifier helpers (plan.md §5, §6)."""

from __future__ import annotations

import hashlib

from find_again.models import (
    ArtifactType,
    Diagnostic,
    IndexedDocument,
    Locator,
    SearchResult,
    Severity,
    content_hash,
    make_doc_id,
    normalize_source_path,
)


def test_artifact_type_enum_covers_pinned_families() -> None:
    values = {member.value for member in ArtifactType}
    assert values == {
        "markdown",
        "json",
        "jsonl",
        "plan",
        "handoff",
        "decision",
        "incident",
        "memory",
    }


def test_severity_enum() -> None:
    assert {member.value for member in Severity} == {"warn", "error"}


def test_normalize_source_path_is_posix() -> None:
    assert normalize_source_path("docs\\sub\\file.md") == "docs/sub/file.md"
    assert normalize_source_path("docs/already/posix.md") == "docs/already/posix.md"


def test_make_doc_id_plain_and_multi_document() -> None:
    assert make_doc_id("docs/lessons-learned.md") == "docs/lessons-learned.md"
    assert make_doc_id("runs\\telemetry.jsonl", "L42") == "runs/telemetry.jsonl::L42"
    # An empty/None record locator does not add the separator.
    assert make_doc_id("docs/x.md", None) == "docs/x.md"
    assert make_doc_id("docs/x.md", "") == "docs/x.md"


def test_content_hash_is_sha256_hex_over_bytes() -> None:
    data = b"raw file bytes\n"
    assert content_hash(data) == hashlib.sha256(data).hexdigest()
    assert len(content_hash(data)) == 64


def test_locator_rendered_forms() -> None:
    assert Locator("docs/x.md").rendered() == "docs/x.md"
    assert Locator("docs/x.md", line_start=12).rendered() == "docs/x.md:12"
    assert Locator("docs/x.md", line_start=12, line_end=12).rendered() == "docs/x.md:12"
    assert Locator("docs/x.md", line_start=12, line_end=20).rendered() == "docs/x.md:12-20"
    assert Locator("runs/t.jsonl", record_key="L42").rendered() == "runs/t.jsonl::L42"


def test_dataclasses_are_frozen() -> None:
    doc = IndexedDocument(
        doc_id="docs/x.md",
        source_path="docs/x.md",
        artifact_type=ArtifactType.MARKDOWN,
        project="find-again",
        timestamp="2026-07-27T00:00:00Z",
        content_hash=content_hash(b"x"),
        body="hello",
    )
    assert doc.record_locator is None
    try:
        doc.body = "mutated"  # type: ignore[misc]
    except AttributeError:
        pass
    else:  # pragma: no cover
        raise AssertionError("IndexedDocument should be frozen")


def test_search_result_and_diagnostic_shapes() -> None:
    result = SearchResult(
        doc_id="docs/x.md",
        type=ArtifactType.MARKDOWN,
        timestamp="2026-07-27T00:00:00Z",
        excerpt="a matched excerpt",
        locator=Locator("docs/x.md", line_start=3),
        rank=-1.5,
    )
    assert result.locator.rendered() == "docs/x.md:3"

    diag = Diagnostic(
        source_path="docs/x.md",
        adapter="exclusion",
        severity=Severity.WARN,
        code="outside-roots",
        message="path is outside the configured roots",
    )
    assert diag.severity is Severity.WARN
