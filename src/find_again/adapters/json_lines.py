"""Generic JSON / JSONL adapter (plan.md §5 adapters/, Step 3a).

* ``.json`` -> ONE document for the whole file. The object's keys and scalar
  values are flattened into a deterministic, key-sorted searchable body; the
  locator is the whole-file line span (openable ``path:line``).
* ``.jsonl`` / ``.ndjson`` -> ONE document PER record (line). Each document's
  ``doc_id`` is ``path::L<n>`` (1-based line number), its ``record_locator`` is
  ``L<n>``, and its locator renders as the openable ``path:<n>``. Every record
  shares the file-level ``content_hash`` (plan.md §6).

A malformed JSON document (a whole ``.json`` file) or a malformed JSONL line
yields a :class:`~find_again.models.Diagnostic` (naming the path + line only,
never file content) and is SKIPPED; siblings -- other files, and the other
records of the same JSONL file -- still index. Blank JSONL lines are skipped
silently (they are not an error).

Determinism: object keys are sorted before flattening, so the same input always
produces byte-identical searchable text.
"""

from __future__ import annotations

import json
from typing import Any, ClassVar

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

_JSON_EXTENSIONS = frozenset({".json"})
_JSONL_EXTENSIONS = frozenset({".jsonl", ".ndjson"})


def _scalar_text(value: Any) -> str:
    """Render a JSON scalar as stable searchable text."""
    if value is None:
        return "null"
    if value is True:
        return "true"
    if value is False:
        return "false"
    return str(value)


def _flatten_body(value: Any) -> str:
    """Deterministic searchable text for a parsed JSON value.

    Iterative (explicit stack) rather than recursive so a pathologically deep
    document cannot blow the Python stack -- ``parse`` must never raise. Object
    keys are sorted and children are pushed in reverse so the emitted order is the
    same key-sorted pre-order DFS the recursive form produced (determinism).
    """
    out: list[str] = []
    stack: list[tuple[Any, str]] = [(value, "")]
    while stack:
        current, prefix = stack.pop()
        if isinstance(current, dict):
            children = [
                (current[key], f"{prefix}.{key}" if prefix else str(key))
                for key in sorted(current.keys())
            ]
            stack.extend(reversed(children))
        elif isinstance(current, list):
            children = [(item, f"{prefix}[{index}]") for index, item in enumerate(current)]
            stack.extend(reversed(children))
        else:
            text = _scalar_text(current)
            out.append(f"{prefix}: {text}" if prefix else text)
    return "\n".join(out)


class JsonAdapter(Adapter):
    """Generic JSON/JSONL adapter (one whole-file document, or one per record)."""

    name: ClassVar[str] = "json"

    def handles(self, source: SourceFile) -> bool:
        return _suffix(source.source_path) in (_JSON_EXTENSIONS | _JSONL_EXTENSIONS)

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
        if _suffix(source.source_path) in _JSONL_EXTENSIONS:
            return self._parse_jsonl(source)
        return self._parse_json(source)

    def _malformed_json(self, source: SourceFile, message: str) -> AdapterResult:
        """A one-diagnostic result for an unparseable whole-file JSON document."""
        return AdapterResult(
            diagnostics=(
                Diagnostic(
                    source_path=source.source_path,
                    adapter=self.name,
                    severity=Severity.WARN,
                    code="malformed-json",
                    message=message,
                ),
            )
        )

    def _parse_json(self, source: SourceFile) -> AdapterResult:
        text = decode_text(source.raw)
        try:
            parsed: Any = json.loads(text)
        except json.JSONDecodeError as exc:
            return self._malformed_json(
                source, f"could not parse JSON document (near line {exc.lineno}); skipped"
            )
        except Exception:
            # RecursionError (pathologically deep nesting) and any other unexpected
            # failure -- parse() must never raise or it aborts every sibling file.
            return self._malformed_json(source, "could not parse JSON document; skipped")
        line_count = max(1, len(text.splitlines()))
        document = IndexedDocument(
            doc_id=make_doc_id(source.source_path),
            source_path=source.source_path,
            artifact_type=ArtifactType.JSON,
            project=source.project,
            timestamp=source.timestamp,
            content_hash=source.content_hash,
            body=_flatten_body(parsed),
            record_locator=None,
        )
        locator = Locator(source_path=source.source_path, line_start=1, line_end=line_count)
        return AdapterResult(documents=(ParsedDocument(document, locator),))

    def _parse_jsonl(self, source: SourceFile) -> AdapterResult:
        text = decode_text(source.raw)
        # Split on PHYSICAL newlines only. text.splitlines() also breaks on
        # U+2028/U+2029/U+0085/form-feed, which are LEGAL unescaped inside a JSON
        # string -- using it would wrongly flag a valid record malformed AND shift
        # every later record's physical-line locator. A trailing "\r" is the CR of a
        # CRLF ending and is stripped so the record text is exactly the line's bytes.
        file_hash = source.content_hash  # hoisted: hashing the file once, not per record
        documents: list[ParsedDocument] = []
        diagnostics: list[Diagnostic] = []
        for line_number, raw_line in enumerate(text.split("\n"), start=1):
            line = raw_line[:-1] if raw_line.endswith("\r") else raw_line
            if not line.strip():
                continue  # blank line: skipped silently, not a diagnostic
            try:
                parsed: Any = json.loads(line)
            except Exception:
                # JSONDecodeError, plus RecursionError (pathologically deep nesting)
                # and any other failure -- one bad record must never abort its siblings.
                diagnostics.append(
                    Diagnostic(
                        source_path=source.source_path,
                        adapter=self.name,
                        severity=Severity.WARN,
                        code="malformed-record",
                        message=f"could not parse JSON record at line {line_number}; skipped",
                    )
                )
                continue
            record_key = f"L{line_number}"
            document = IndexedDocument(
                doc_id=make_doc_id(source.source_path, record_key),
                source_path=source.source_path,
                artifact_type=ArtifactType.JSONL,
                project=source.project,
                timestamp=source.timestamp,
                content_hash=file_hash,
                body=_flatten_body(parsed),
                record_locator=record_key,
            )
            locator = Locator(
                source_path=source.source_path,
                line_start=line_number,
                line_end=line_number,
                record_key=record_key,
            )
            documents.append(ParsedDocument(document, locator))
        return AdapterResult(documents=tuple(documents), diagnostics=tuple(diagnostics))
