"""Structured artifact-family Markdown adapters (plan.md Step 3b, §5 adapters/, §8).

Four families -- plan, handoff/session, incident, memory -- share ONE section-aware
Markdown reader (:class:`_StructuredMarkdownAdapter`); each concrete adapter only
declares its stable ``name``, its :class:`~find_again.models.ArtifactType`, and the
path/name predicate that decides which files it :meth:`~.base.Adapter.handles`.
The decision-record family is a separate frontmatter-aware adapter
(:mod:`find_again.adapters.decision`).

Section-aware indexing (plan.md Step 3b "section-aware locators where feasible"):
a file is split at level-2/level-3 ATX headings (``## Phase`` / ``### Step N`` /
``## <section>``) that are NOT inside a fenced code block, and EACH section becomes
one document whose locator points at the section's heading line -- so a match lands
on the right section, not merely the file. The preamble before the first heading
(the ``# Title`` + intro) is its own section. A file with no such headings falls
back to a single whole-file document (the same shape the generic Markdown adapter
emits). Per-section ``doc_id`` is ``path::L<start-line>`` (the multi-document
convention already used by the JSONL adapter), and every section shares the
file-level ``content_hash`` (plan.md §6).

Corruption: a genuinely binary file (NUL bytes) mislabeled with a text extension is
diagnosed (path + stable ``code`` only, never content) and SKIPPED -- ``parse``
returns the diagnostic instead of raising, so one bad file never aborts its
siblings. Output is deterministic: sections are emitted in file order.
"""

from __future__ import annotations

import re
from pathlib import PurePosixPath
from typing import ClassVar

from ..models import ArtifactType, Diagnostic, IndexedDocument, Locator, Severity, make_doc_id
from .base import (
    TEXT_EXTENSIONS,
    Adapter,
    AdapterResult,
    ParsedDocument,
    SourceFile,
    _looks_binary,
    _split_physical_lines,
    _suffix,
    decode_text,
)

# A section boundary is a level-2 or level-3 ATX heading with a title (`## Foo` /
# `### Step 1`). Level 1 (`# Title`) stays in the preamble; level 4+ stays inside
# its parent section. The trailing `\S` requires an actual title, so a bare `## `
# is not a boundary.
_SECTION_HEADING = re.compile(r"^#{2,3}[ \t]+\S")


def _section_start_lines(lines: list[str]) -> list[int]:
    """1-based line numbers of section headings, ignoring lines inside code fences.

    A ``#`` inside a ```` ``` ```` / ``~~~`` fenced code block (common in plans that
    quote shell snippets) is NOT a heading; fence state is tracked so such lines are
    never mistaken for section boundaries.
    """
    starts: list[int] = []
    in_fence = False
    fence_marker = ""
    for line_number, line in enumerate(lines, start=1):
        stripped = line.lstrip()
        if stripped.startswith("```") or stripped.startswith("~~~"):
            marker = stripped[:3]
            if not in_fence:
                in_fence, fence_marker = True, marker
            elif stripped.startswith(fence_marker):
                in_fence, fence_marker = False, ""
            continue
        if in_fence:
            continue
        if _SECTION_HEADING.match(line):
            starts.append(line_number)
    return starts


def _section_bounds(starts: list[int], total_lines: int) -> list[tuple[int, int]]:
    """Inclusive 1-based ``(start, end)`` spans: an optional preamble, then one per heading."""
    bounds: list[tuple[int, int]] = []
    if starts[0] > 1:
        bounds.append((1, starts[0] - 1))  # preamble: title + intro before first heading
    for index, start in enumerate(starts):
        end = starts[index + 1] - 1 if index + 1 < len(starts) else total_lines
        bounds.append((start, end))
    return bounds


class _StructuredMarkdownAdapter(Adapter):
    """Shared section-aware Markdown reader; subclasses set name/type/``handles``."""

    artifact_type: ClassVar[ArtifactType]

    def _binary_skip(self, source: SourceFile) -> AdapterResult:
        return AdapterResult(
            diagnostics=(
                Diagnostic(
                    source_path=source.source_path,
                    adapter=self.name,
                    severity=Severity.WARN,
                    code="binary-content",
                    message="file appears to be binary (NUL bytes); skipped",
                ),
            )
        )

    def _whole_file_document(
        self, source: SourceFile, body: str, line_count: int
    ) -> ParsedDocument:
        document = IndexedDocument(
            doc_id=make_doc_id(source.source_path),
            source_path=source.source_path,
            artifact_type=self.artifact_type,
            project=source.project,
            timestamp=source.timestamp,
            content_hash=source.content_hash,
            body=body,
            record_locator=None,
        )
        locator = Locator(source_path=source.source_path, line_start=1, line_end=line_count)
        return ParsedDocument(document, locator)

    def _section_document(
        self, source: SourceFile, file_hash: str, start: int, end: int, body: str
    ) -> ParsedDocument:
        record_key = f"L{start}"
        document = IndexedDocument(
            doc_id=make_doc_id(source.source_path, record_key),
            source_path=source.source_path,
            artifact_type=self.artifact_type,
            project=source.project,
            timestamp=source.timestamp,
            content_hash=file_hash,  # file-level: every section shares the file hash
            body=body,
            record_locator=record_key,
        )
        locator = Locator(
            source_path=source.source_path,
            line_start=start,
            line_end=end,
            record_key=record_key,
        )
        return ParsedDocument(document, locator)

    def parse(self, source: SourceFile) -> AdapterResult:
        if _looks_binary(source.raw):
            return self._binary_skip(source)
        # Physical-newline split (NOT str.splitlines()): a U+2028/U+2029/U+0085/FF in
        # a section body must not split a line and shift every later section locator.
        lines = _split_physical_lines(decode_text(source.raw)) or [""]
        total_lines = len(lines)
        starts = _section_start_lines(lines)
        if not starts:
            # No section headings -> single whole-file document (generic shape).
            body = "\n".join(lines)
            return AdapterResult(documents=(self._whole_file_document(source, body, total_lines),))
        file_hash = source.content_hash  # hashed once, shared by every section
        documents: list[ParsedDocument] = []
        for start, end in _section_bounds(starts, total_lines):
            body = "\n".join(lines[start - 1 : end])
            if not body.strip():
                continue  # a blank preamble carries nothing searchable -- skip it
            documents.append(self._section_document(source, file_hash, start, end, body))
        if not documents:
            body = "\n".join(lines)
            return AdapterResult(documents=(self._whole_file_document(source, body, total_lines),))
        return AdapterResult(documents=tuple(documents))


class PlanAdapter(_StructuredMarkdownAdapter):
    """Plan documents (``plan.md`` / ``master_plan.md``): section-aware per Step/Phase."""

    name: ClassVar[str] = "plan"
    artifact_type: ClassVar[ArtifactType] = ArtifactType.PLAN

    def handles(self, source: SourceFile) -> bool:
        if _suffix(source.source_path) not in TEXT_EXTENSIONS:
            return False
        return PurePosixPath(source.source_path).name.lower() in {"plan.md", "master_plan.md"}


class HandoffAdapter(_StructuredMarkdownAdapter):
    """Handoff / session state: ``handoff-prompt.md``, ``current.md``, ``sessions/*.md``."""

    name: ClassVar[str] = "handoff"
    artifact_type: ClassVar[ArtifactType] = ArtifactType.HANDOFF

    def handles(self, source: SourceFile) -> bool:
        path = PurePosixPath(source.source_path)
        if path.suffix.lower() not in TEXT_EXTENSIONS:
            return False
        if path.name.lower() in {"handoff-prompt.md", "current.md"}:
            return True
        return "sessions" in {part.lower() for part in path.parts[:-1]}


class IncidentAdapter(_StructuredMarkdownAdapter):
    """Incident notes: any ``*.md`` whose name says ``incident`` or lives under ``incidents/``."""

    name: ClassVar[str] = "incident"
    artifact_type: ClassVar[ArtifactType] = ArtifactType.INCIDENT

    def handles(self, source: SourceFile) -> bool:
        path = PurePosixPath(source.source_path)
        if path.suffix.lower() not in TEXT_EXTENSIONS:
            return False
        if "incident" in path.name.lower():
            return True
        return "incidents" in {part.lower() for part in path.parts[:-1]}


class MemoryAdapter(_StructuredMarkdownAdapter):
    """Memory files: ``MEMORY.md``, ``feedback_*.md``, or anything under a ``memory/`` dir."""

    name: ClassVar[str] = "memory"
    artifact_type: ClassVar[ArtifactType] = ArtifactType.MEMORY

    def handles(self, source: SourceFile) -> bool:
        path = PurePosixPath(source.source_path)
        if path.suffix.lower() not in TEXT_EXTENSIONS:
            return False
        name = path.name.lower()
        if name == "memory.md" or name.startswith("feedback_"):
            return True
        return "memory" in {part.lower() for part in path.parts[:-1]}
