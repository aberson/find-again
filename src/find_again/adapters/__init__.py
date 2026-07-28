"""Adapter package: the single adapter-to-document contract + the generic adapters.

Step 3a delivers the contract (:mod:`find_again.adapters.base`) and the two
GENERIC adapters -- Markdown/text (:mod:`find_again.adapters.text`) and JSON/JSONL
(:mod:`find_again.adapters.json_lines`). The structured artifact-family adapters
(plan, handoff, decision, incident, memory) build on this same contract in
Step 3b, and the indexer (Step 4) drives dispatch via :func:`select_adapter`.
"""

from __future__ import annotations

from .base import (
    Adapter,
    AdapterResult,
    ParsedDocument,
    SourceFile,
    decode_text,
    select_adapter,
)
from .json_lines import JsonAdapter
from .text import MarkdownAdapter

# The generic adapters, in dispatch order. Their handled extensions do not
# overlap, so order is not load-bearing today; Step 3b prepends more specific
# structured-family adapters ahead of these generic fallbacks.
GENERIC_ADAPTERS: tuple[Adapter, ...] = (MarkdownAdapter(), JsonAdapter())


def select_generic_adapter(source: SourceFile) -> Adapter | None:
    """Return the generic adapter that handles ``source`` (Markdown/text or JSON), else ``None``."""
    return select_adapter(source, GENERIC_ADAPTERS)


__all__ = [
    "GENERIC_ADAPTERS",
    "Adapter",
    "AdapterResult",
    "JsonAdapter",
    "MarkdownAdapter",
    "ParsedDocument",
    "SourceFile",
    "decode_text",
    "select_adapter",
    "select_generic_adapter",
]
