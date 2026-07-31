"""Decision-record adapter: generic YAML-frontmatter Markdown (plan.md Step 3b, §2).

A decision record is a Markdown file with a leading ``---``-fenced frontmatter block.
This adapter indexes it through a FORMAT CONTRACT only -- the frontmatter keys pinned
in Paper Trail's plan (frozen at its Step 1):
``id``, ``title``, ``status``, ``review_date``. It surfaces those pinned keys as
searchable text plus a locator, TOLERATES any unknown keys (they are still indexed),
and FALLS BACK to plain-Markdown indexing of the whole file when no frontmatter
parses. There is NO build or import dependency on Paper Trail: any producer emitting
the same format is indexed identically, and the absence of Paper Trail simply means
zero decision hits -- never an error.

YAML approach (plan.md Step 3b + §10 stdlib-only): a MINIMAL stdlib parser reads the
constrained ``key: value`` frontmatter block. Decision frontmatter is flat scalars
(the pinned keys are all scalars), so no full YAML engine -- and no PyYAML dependency
-- is required; the project stays stdlib-only. Nested/list values (e.g. Paper Trail's
optional ``evidence``/``supersedes``) are simply not surfaced as keys by the minimal
parser, which is acceptable because only the four pinned scalar keys are load-bearing
for retrieval and the whole Markdown body is indexed regardless.

Corruption: a binary file is diagnosed + skipped. A frontmatter fence that opens but
never closes ("truncated"), or that closes but yields no parseable key, emits a WARN
diagnostic (path + stable ``code`` only, never file content) and STILL indexes the
file as plain Markdown -- it is never dropped. Output is deterministic (pinned keys in
fixed order, remaining keys sorted).
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
    decode_text,
)

# The Paper Trail format-contract keys, surfaced first (in this fixed order) so the
# rendered searchable body is deterministic across runs.
_PINNED_KEYS: tuple[str, ...] = ("id", "title", "status", "review_date")

# The decision signature used for content-based selection: a file whose frontmatter
# carries all of these is treated as a decision record wherever it lives.
_SIGNATURE_KEYS: frozenset[str] = frozenset({"id", "title", "status"})

# A frontmatter key must be an identifier-ish token; this rejects YAML list items
# (`- foo: bar`) and nested-mapping lines the minimal parser cannot represent, so
# they are tolerated (ignored) rather than indexed as junk keys.
_KEY_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_.-]*$")

# How many leading bytes to inspect when deciding ownership. A frontmatter block is
# tiny and always sits at the very top, so a small head peek is sufficient and avoids
# decoding an entire file during adapter selection.
_HEAD_PEEK_BYTES = 8192


def _strip_fence(line: str) -> str:
    return line.rstrip("\r").strip()


def _parse_frontmatter_block(block_lines: list[str]) -> dict[str, str]:
    """Parse the constrained ``key: value`` frontmatter block into a flat dict.

    Blank lines, ``#`` comments, non-``key: value`` lines, and non-identifier keys are
    tolerated (skipped). The first occurrence of a key wins (deterministic).
    """
    parsed: dict[str, str] = {}
    for raw_line in block_lines:
        line = raw_line.rstrip("\r")
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or ":" not in line:
            continue
        # Only UNINDENTED (column-0) lines are top-level keys. An INDENTED line is a
        # value nested inside another key's map/list -- surfacing it would let a nested
        # `status:`/`id:`/`title:` masquerade as the pinned TOP-LEVEL key (wrong value
        # rendered AND a false content-signature match). The minimal parser cannot
        # represent nesting, so such lines are tolerated (ignored) -- exactly what this
        # module's docstring promises for nested/list values.
        if line[:1].isspace():
            continue
        key, _, value = line.partition(":")
        key = key.strip()  # trims only trailing space now (leading indent already gated)
        if not _KEY_RE.match(key) or key in parsed:
            continue
        parsed[key] = value.strip().strip('"').strip("'")
    return parsed


def _split_frontmatter(text: str) -> tuple[dict[str, str] | None, str, str]:
    """Split ``text`` into ``(frontmatter | None, body, status)``.

    ``status`` is one of:

    * ``absent``  -- no leading ``---`` fence; ``frontmatter`` is ``None``, ``body`` is
      the whole text.
    * ``ok``      -- fence opened and closed and yielded >=1 key.
    * ``empty``   -- fence opened and closed but yielded no parseable key.
    * ``truncated`` -- fence opened but never closed; ``frontmatter`` is ``None``.
    """
    lines = text.split("\n")
    if not lines or _strip_fence(lines[0]) != "---":
        return (None, text, "absent")
    for index in range(1, len(lines)):
        if _strip_fence(lines[index]) == "---":
            frontmatter = _parse_frontmatter_block(lines[1:index])
            body = "\n".join(lines[index + 1 :])
            return (frontmatter, body, "ok" if frontmatter else "empty")
    return (None, text, "truncated")


def _render_frontmatter(frontmatter: dict[str, str]) -> str:
    """Render frontmatter as deterministic searchable text: pinned keys, then the rest sorted."""
    rendered: list[str] = [
        f"{key}: {frontmatter[key]}" for key in _PINNED_KEYS if key in frontmatter
    ]
    rendered.extend(
        f"{key}: {frontmatter[key]}" for key in sorted(frontmatter) if key not in _PINNED_KEYS
    )
    return "\n".join(rendered)


class DecisionAdapter(Adapter):
    """Generic YAML-frontmatter Markdown decision-record adapter (format contract only)."""

    name: ClassVar[str] = "decision"

    def handles(self, source: SourceFile) -> bool:
        path = PurePosixPath(source.source_path)
        if path.suffix.lower() not in TEXT_EXTENSIONS:
            return False
        # Path signal: Paper Trail stores records under `<root>/decisions/`
        # (case-insensitive, consistent with the case-insensitive file-name signals).
        if "decisions" in {part.lower() for part in path.parts[:-1]}:
            return True
        # Content signal: a file carrying the decision frontmatter signature is a
        # decision record wherever it lives (producer-agnostic).
        return self._has_decision_signature(source.raw)

    def _has_decision_signature(self, raw: bytes) -> bool:
        if _looks_binary(raw):
            return False
        frontmatter, _, status = _split_frontmatter(decode_text(raw[:_HEAD_PEEK_BYTES]))
        return status == "ok" and frontmatter is not None and frontmatter.keys() >= _SIGNATURE_KEYS

    def _document(self, source: SourceFile, body: str, line_count: int) -> ParsedDocument:
        document = IndexedDocument(
            doc_id=make_doc_id(source.source_path),
            source_path=source.source_path,
            artifact_type=ArtifactType.DECISION,
            project=source.project,
            timestamp=source.timestamp,
            content_hash=source.content_hash,
            body=body,
            record_locator=None,
        )
        # A whole-file locator; the frontmatter sits at line 1, so this opens on it.
        locator = Locator(source_path=source.source_path, line_start=1, line_end=line_count)
        return ParsedDocument(document, locator)

    def _fallback(self, source: SourceFile, code: str | None) -> AdapterResult:
        """Index the whole file as plain Markdown; optionally attach a WARN diagnostic."""
        text = decode_text(source.raw)
        line_count = max(1, len(_split_physical_lines(text)))
        document = self._document(source, text, line_count)
        if code is None:
            return AdapterResult(documents=(document,))
        diagnostic = Diagnostic(
            source_path=source.source_path,
            adapter=self.name,
            severity=Severity.WARN,
            code=code,
            message="frontmatter did not parse; indexed as plain Markdown",
        )
        return AdapterResult(documents=(document,), diagnostics=(diagnostic,))

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
        frontmatter, body, status = _split_frontmatter(text)
        if status == "absent":
            return self._fallback(source, code=None)  # a plain Markdown file, no diagnostic
        if status == "truncated":
            return self._fallback(source, code="frontmatter-not-closed")
        if status == "empty" or frontmatter is None:
            return self._fallback(source, code="unparseable-frontmatter")
        searchable = _render_frontmatter(frontmatter)
        if body.strip():
            searchable = f"{searchable}\n\n{body}"
        line_count = max(1, len(_split_physical_lines(text)))
        return AdapterResult(documents=(self._document(source, searchable, line_count),))
