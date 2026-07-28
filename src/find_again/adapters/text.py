"""Generic Markdown / plain-text adapter (plan.md §5 adapters/, Step 3a).

The whole file body becomes searchable text; the document's ``doc_id`` is the
root-relative POSIX path (no record locator) and its ``content_hash`` is taken
over the raw bytes. The locator is the whole-file line span, rendered as an
openable ``path:line`` (``path:1`` for a single-line file, ``path:1-N``
otherwise). A file that is actually binary (contains NUL bytes) is diagnosed and
skipped rather than indexed as mojibake.
"""

from __future__ import annotations

from typing import ClassVar

from ..models import ArtifactType, Diagnostic, IndexedDocument, Locator, Severity, make_doc_id
from .base import (
    Adapter,
    AdapterResult,
    ParsedDocument,
    SourceFile,
    _looks_binary,
    _suffix,
    decode_text,
)

_TEXT_EXTENSIONS = frozenset({".md", ".markdown", ".mdown", ".mkd", ".txt", ".text"})


class MarkdownAdapter(Adapter):
    """Generic Markdown/text adapter: whole-file body -> one searchable document."""

    name: ClassVar[str] = "markdown"

    def handles(self, source: SourceFile) -> bool:
        return _suffix(source.source_path) in _TEXT_EXTENSIONS

    def parse(self, source: SourceFile) -> AdapterResult:
        if _looks_binary(source.raw):
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
        text = decode_text(source.raw)
        line_count = max(1, len(text.splitlines()))
        document = IndexedDocument(
            doc_id=make_doc_id(source.source_path),
            source_path=source.source_path,
            artifact_type=ArtifactType.MARKDOWN,
            project=source.project,
            timestamp=source.timestamp,
            content_hash=source.content_hash,
            body=text,
            record_locator=None,
        )
        locator = Locator(source_path=source.source_path, line_start=1, line_end=line_count)
        return AdapterResult(documents=(ParsedDocument(document, locator),))
