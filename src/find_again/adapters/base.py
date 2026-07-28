"""The single adapter-to-document contract (plan.md §5 adapters/ role; §8 small adapter contract).

An adapter turns ONE source file -- its root-relative POSIX path, raw bytes, and
the resolved project + timestamp context (:class:`SourceFile`) -- into zero or
more :class:`ParsedDocument` values (an :class:`~find_again.models.IndexedDocument`
paired with its openable :class:`~find_again.models.Locator`) plus zero or more
:class:`~find_again.models.Diagnostic` values, and NEVER raises on bad input.

The contract (plan.md §5 stable locators + diagnostics, §6 Identifiers, §8):

* :meth:`Adapter.handles` decides -- from the file's extension / artifact family
  -- whether this adapter owns the file.
* :meth:`Adapter.parse` yields documents with STABLE locators: an openable
  ``path:line`` for whole-file text, ``path::L<n>`` (a record key) for a record
  inside a multi-document JSON/JSONL file. ``doc_id`` and ``content_hash`` are
  composed with the shared :mod:`find_again.models` helpers -- ``content_hash`` is
  taken over the raw file bytes (file-level; plan.md §6), so every record of a
  JSONL file shares the file's hash.
* A CORRUPT input -- a malformed JSON document, a bad JSONL line -- yields a
  :class:`~find_again.models.Diagnostic` (``severity`` warn/error, a stable
  ``code``, and a ``message`` naming only the path + code, NEVER file content) and
  is SKIPPED. Siblings still index: other files, and -- crucially -- the other
  records of the same JSONL file. ``parse`` returns diagnostics instead of
  raising, so one bad record never aborts its siblings.

Nothing here touches the database or performs exclusion (that is
:mod:`find_again.config`); adapters are pure ``bytes -> documents + diagnostics``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import ClassVar

from ..models import (
    Diagnostic,
    IndexedDocument,
    Locator,
    content_hash,
    normalize_source_path,
)

__all__ = [
    "TEXT_EXTENSIONS",
    "Adapter",
    "AdapterResult",
    "ParsedDocument",
    "SourceFile",
    "decode_text",
    "select_adapter",
]

# The Markdown/plain-text extension family, defined ONCE here so the generic
# Markdown adapter (text.py) and every structured-family adapter (structured.py,
# decision.py) share a single source of truth rather than re-declaring it and
# drifting (dev/.claude/rules/code-quality.md -- one source of truth for
# data-shape constants).
TEXT_EXTENSIONS = frozenset({".md", ".markdown", ".mdown", ".mkd", ".txt", ".text"})


def _suffix(source_path: str) -> str:
    """Lower-cased file extension (including the dot) of a POSIX source path."""
    return PurePosixPath(source_path).suffix.lower()


def decode_text(raw: bytes) -> str:
    """Decode file bytes to text for indexing -- lenient, NEVER raises.

    A UTF-8 BOM is stripped; otherwise strict UTF-8 is tried, falling back to
    ``latin-1`` (which maps every byte 1:1 and never fails). The BOM branch ALSO
    falls back to ``latin-1`` if the post-BOM bytes are not valid UTF-8, so an
    undecodable BOM'd file (or a head-peek slice truncated mid-multibyte-char) can
    never make this raise. Adapters that must *reject* rather than mojibake-index
    such input rely on :func:`_looks_binary` (which flags both a NUL byte and a
    BOM-announced-but-undecodable file) to diagnose + skip it before calling here.
    """
    if raw.startswith(b"\xef\xbb\xbf"):
        try:
            return raw.decode("utf-8-sig")
        except UnicodeDecodeError:
            return raw.decode("latin-1")
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return raw.decode("latin-1")


def _looks_binary(raw: bytes) -> bool:
    """True when bytes should be diagnosed + skipped rather than decoded as text.

    Two cheap, reliable signals:

    * a NUL byte -- the classic marker that bytes are not text; and
    * a leading UTF-8 BOM whose bytes do NOT decode as UTF-8 -- a file that
      *announces* UTF-8 but isn't decodable is genuinely corrupt/undecodable (not
      merely latin-1 text), so it is flagged too. This closes a never-raise hole:
      without it, a BOM + invalid-UTF-8 (NUL-free) file slips past the NUL check and
      ``decode_text``'s BOM branch would be the one to (previously) raise.

    Shared by every text-decoding adapter so genuinely-binary or falsely-BOM'd input
    mislabeled with a text/JSON extension is diagnosed and skipped rather than
    mojibake-indexed (``decode_text``'s ``latin-1`` fallback never fails, so a decode
    error alone can't be relied on to catch it).
    """
    if b"\x00" in raw:
        return True
    if raw.startswith(b"\xef\xbb\xbf"):
        try:
            raw.decode("utf-8-sig")
        except UnicodeDecodeError:
            return True
    return False


def _split_physical_lines(text: str) -> list[str]:
    """Split ``text`` on PHYSICAL newlines (LF / CRLF) only -- never the Unicode ones.

    Unlike ``str.splitlines()``, this does NOT break on U+2028 / U+2029 / U+0085 /
    form-feed -- all legal, unescaped, inside prose and inside a heading's section
    body. Splitting on those would shift every later section's ``path:line`` locator
    away from the physical line a human sees when opening the file (the same drift
    Step 3a's JSONL reader avoids by splitting on physical ``\\n`` only).

    A trailing CR (the CR of a CRLF pair) is stripped from each line, and a single
    trailing newline does NOT yield a phantom empty final line (matching
    ``str.splitlines()``), so section line counts stay stable.
    """
    if text.endswith("\n"):
        text = text[:-1]  # a lone trailing newline is not an extra empty line
    return [line[:-1] if line.endswith("\r") else line for line in text.split("\n")]


@dataclass(frozen=True, slots=True)
class SourceFile:
    """One file's bytes plus the resolved context an adapter needs.

    ``source_path`` is the root-relative POSIX path (the ``doc_id`` base, plan.md
    §6). ``raw`` are the exact file bytes ``content_hash`` is taken over.
    ``project`` and ``timestamp`` (ISO 8601 UTC) are resolved by the indexer
    (Step 4) and passed straight through onto every document.
    """

    source_path: str
    raw: bytes
    project: str
    timestamp: str

    @property
    def content_hash(self) -> str:
        """SHA-256 hex over the raw file bytes (plan.md §6 Identifiers).

        The bare ``content_hash`` here resolves to the module-level helper imported
        from :mod:`find_again.models`, not to this property (which is reached only
        via ``self``).
        """
        return content_hash(self.raw)

    @classmethod
    def from_path(
        cls,
        path: Path | str,
        *,
        source_path: str,
        project: str,
        timestamp: str,
    ) -> SourceFile:
        """Read ``path``'s bytes into a :class:`SourceFile` (``source_path`` normalized)."""
        return cls(
            source_path=normalize_source_path(source_path),
            raw=Path(path).read_bytes(),
            project=project,
            timestamp=timestamp,
        )


@dataclass(frozen=True, slots=True)
class ParsedDocument:
    """An :class:`IndexedDocument` paired with its openable :class:`Locator`.

    The locator carries the line span / record key the storage row does not (the
    DB persists ``record_locator`` only); search (Step 5) renders it as the
    openable ``path:line`` form.
    """

    document: IndexedDocument
    locator: Locator


@dataclass(frozen=True, slots=True)
class AdapterResult:
    """What one :meth:`Adapter.parse` call produced: documents and diagnostics.

    Both default to empty. A whole-file parse failure yields no documents and one
    diagnostic; a JSONL file yields one document per good record plus one
    diagnostic per bad record. Being a frozen dataclass, two results from the same
    input compare equal -- the determinism guarantee, assertable directly.
    """

    documents: tuple[ParsedDocument, ...] = ()
    diagnostics: tuple[Diagnostic, ...] = ()


class Adapter(ABC):
    """The single adapter-to-document contract (plan.md §8 small adapter contract).

    A concrete adapter sets :attr:`name` (its stable ``Diagnostic.adapter`` label),
    decides which files it :meth:`handles`, and turns each into an
    :class:`AdapterResult` via :meth:`parse` without ever raising on bad input.
    """

    name: ClassVar[str]

    @abstractmethod
    def handles(self, source: SourceFile) -> bool:
        """Return ``True`` if this adapter owns ``source`` (by extension / family)."""

    @abstractmethod
    def parse(self, source: SourceFile) -> AdapterResult:
        """Parse ``source`` into documents + diagnostics; never raise on bad input."""


def select_adapter(source: SourceFile, adapters: Sequence[Adapter]) -> Adapter | None:
    """Return the first adapter in ``adapters`` that handles ``source``, else ``None``."""
    for adapter in adapters:
        if adapter.handles(source):
            return adapter
    return None
