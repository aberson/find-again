"""Adapter package: the single adapter-to-document contract + the family adapters.

Step 3a delivers the contract (:mod:`find_again.adapters.base`) and the two GENERIC
adapters -- Markdown/text (:mod:`find_again.adapters.text`) and JSON/JSONL
(:mod:`find_again.adapters.json_lines`). Step 3b adds the structured artifact-family
adapters on the SAME contract: plan, handoff/session, incident, memory
(:mod:`find_again.adapters.structured`) and the frontmatter-aware decision-record
adapter (:mod:`find_again.adapters.decision`).

Selection (plan.md Step 3b "route a file to the right family ... generic adapters as
the fallback", §8 small adapter contract): :data:`ALL_ADAPTERS` lists the specific
structured families FIRST and the generic Markdown/JSON adapters LAST, so a
``plan.md`` / ``current.md`` / ``feedback_*.md`` / decision record routes to its
family while an ordinary ``notes.md`` falls through to the generic Markdown adapter.
:func:`select_adapter_for` returns the owning adapter (the indexer, Step 4, drives
dispatch through it). Structured predicates are name/path/frontmatter based and do not
overlap the generic extension fallback.
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
from .decision import DecisionAdapter
from .json_lines import JsonAdapter
from .structured import HandoffAdapter, IncidentAdapter, MemoryAdapter, PlanAdapter
from .text import MarkdownAdapter

# The generic adapters, in dispatch order. Their handled extensions do not overlap,
# so order among them is not load-bearing; they are the fallback tail of ALL_ADAPTERS.
GENERIC_ADAPTERS: tuple[Adapter, ...] = (MarkdownAdapter(), JsonAdapter())

# The structured artifact-family adapters (Step 3b). Predicates are name/path/
# frontmatter based and mutually specific. Memory precedes Incident so a memory
# file whose name happens to contain "incident" (e.g. `feedback_incident_*.md`)
# routes to its specific `feedback_` / `memory/` signal rather than Incident's
# broader "incident"-in-name substring match.
STRUCTURED_ADAPTERS: tuple[Adapter, ...] = (
    PlanAdapter(),
    HandoffAdapter(),
    MemoryAdapter(),
    IncidentAdapter(),
    DecisionAdapter(),
)

# Full dispatch order: specific structured families first, generic adapters last.
ALL_ADAPTERS: tuple[Adapter, ...] = STRUCTURED_ADAPTERS + GENERIC_ADAPTERS


def select_generic_adapter(source: SourceFile) -> Adapter | None:
    """Return the generic adapter that handles ``source`` (Markdown/text or JSON), else ``None``."""
    return select_adapter(source, GENERIC_ADAPTERS)


def select_adapter_for(source: SourceFile) -> Adapter | None:
    """Return the owning adapter for ``source`` (structured family, else generic), else ``None``."""
    return select_adapter(source, ALL_ADAPTERS)


__all__ = [
    "ALL_ADAPTERS",
    "GENERIC_ADAPTERS",
    "STRUCTURED_ADAPTERS",
    "Adapter",
    "AdapterResult",
    "DecisionAdapter",
    "HandoffAdapter",
    "IncidentAdapter",
    "JsonAdapter",
    "MarkdownAdapter",
    "MemoryAdapter",
    "ParsedDocument",
    "PlanAdapter",
    "SourceFile",
    "decode_text",
    "select_adapter",
    "select_adapter_for",
    "select_generic_adapter",
]
