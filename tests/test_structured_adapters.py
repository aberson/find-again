"""Tests for the structured artifact-family adapters (plan.md Step 3b, §5, §8).

Covers the done-when conditions + required fixtures for EACH family:

* plan / handoff / incident / memory / decision each emit searchable text and an
  openable ``path:line`` (or ``path::L<n>`` record) locator,
* the plan adapter is section-aware (``### Step N`` / ``## Phase`` -> a locator at
  that line),
* the decision adapter surfaces the pinned Paper Trail format-contract keys, TOLERATES
  unknown keys, and FALLS BACK to plain-Markdown indexing when frontmatter is absent
  or unparseable,
* a CORRUPT input (binary, bad frontmatter) diagnoses (path + code, NO content) and is
  SKIPPED / fallen-back WITHOUT aborting siblings,
* selection routes each file to the right family, generic adapters as the fallback,
* output is deterministic and diagnostics never carry file content.
"""

from __future__ import annotations

from find_again.adapters import (
    DecisionAdapter,
    HandoffAdapter,
    IncidentAdapter,
    JsonAdapter,
    MarkdownAdapter,
    MemoryAdapter,
    PlanAdapter,
    SourceFile,
    decode_text,
    select_adapter_for,
)
from find_again.models import ArtifactType, content_hash

_PROJECT = "find-again"
_TS = "2026-07-27T00:00:00Z"


def _source(source_path: str, raw: bytes) -> SourceFile:
    return SourceFile(source_path=source_path, raw=raw, project=_PROJECT, timestamp=_TS)


# --------------------------------------------------------------------------- #
# Selection routing (plan.md Step 3b: route to the right family, generic fallback)
# --------------------------------------------------------------------------- #
def test_selection_routes_each_family_and_falls_back_to_generic() -> None:
    cases = [
        ("plans/plan.md", b"# Plan\n", PlanAdapter),
        ("docs/master_plan.md", b"# Plan\n", PlanAdapter),
        (".claude/task-state/current.md", b"# State\n", HandoffAdapter),
        (".claude/task-state/handoff-prompt.md", b"# H\n", HandoffAdapter),
        (".claude/task-state/sessions/abc.md", b"# S\n", HandoffAdapter),
        ("docs/incident-2026.md", b"# Incident\n", IncidentAdapter),
        ("incidents/db-outage.md", b"# Outage\n", IncidentAdapter),
        ("memory/MEMORY.md", b"# Memory\n", MemoryAdapter),
        ("memory/feedback_thing.md", b"# FB\n", MemoryAdapter),
        ("decisions/2026-07-25-x.md", b"# D\n", DecisionAdapter),
        # Generic fallbacks -- no structured family claims these.
        ("docs/notes.md", b"# Notes\n", MarkdownAdapter),
        ("data/config.json", b"{}", JsonAdapter),
        ("runs/log.jsonl", b"{}\n", JsonAdapter),
    ]
    for path, raw, expected in cases:
        adapter = select_adapter_for(_source(path, raw))
        assert isinstance(adapter, expected), f"{path} -> {type(adapter).__name__}"


def test_selection_routes_decision_by_frontmatter_signature_anywhere() -> None:
    # A file with the decision frontmatter signature routes to DecisionAdapter even
    # outside a decisions/ directory (producer-agnostic content signal).
    raw = b"---\nid: 2026-07-25-x\ntitle: T\nstatus: active\n---\nbody\n"
    assert isinstance(select_adapter_for(_source("notes/whatever.md", raw)), DecisionAdapter)
    # But a plain markdown file with an unrelated leading '---' does NOT.
    plain = b"---\nlayout: post\ndate: today\n---\nbody\n"
    assert isinstance(select_adapter_for(_source("notes/whatever.md", plain)), MarkdownAdapter)


def test_selection_returns_none_for_unhandled_extension() -> None:
    assert select_adapter_for(_source("image.png", b"\x89PNG")) is None


# --------------------------------------------------------------------------- #
# PLAN adapter -- section-aware locators
# --------------------------------------------------------------------------- #
def test_plan_sections_emit_locators_at_headings() -> None:
    raw = (
        b"# Seed Plan\n"  # 1
        b"\n"  # 2
        b"intro preamble text\n"  # 3
        b"## Phase A\n"  # 4
        b"phase body\n"  # 5
        b"### Step 1\n"  # 6
        b"step one about token usage levers\n"  # 7
        b"### Step 2\n"  # 8
        b"step two body\n"  # 9
    )
    src = _source("plans/plan.md", raw)
    adapter = PlanAdapter()
    assert adapter.handles(src)

    result = adapter.parse(src)
    assert result.diagnostics == ()
    docs = [p.document for p in result.documents]

    # Preamble + three sections (Phase A, Step 1, Step 2), each its own document.
    assert [d.record_locator for d in docs] == ["L1", "L4", "L6", "L8"]
    assert [d.doc_id for d in docs] == [
        "plans/plan.md::L1",
        "plans/plan.md::L4",
        "plans/plan.md::L6",
        "plans/plan.md::L8",
    ]
    assert all(d.artifact_type is ArtifactType.PLAN for d in docs)
    # Every section shares the file-level content hash.
    assert {d.content_hash for d in docs} == {content_hash(raw)}

    # A match on "token usage levers" lands on Step 1's section, locator at its heading.
    step1 = next(p for p in result.documents if p.document.record_locator == "L6")
    assert "token usage levers" in step1.document.body
    assert step1.locator.rendered() == "plans/plan.md:6-7"

    # The preamble carries the title + intro.
    preamble = result.documents[0].document
    assert "Seed Plan" in preamble.body
    assert "intro preamble text" in preamble.body


def test_plan_ignores_headings_inside_code_fences() -> None:
    # A '#' inside a fenced code block is a shell comment, not a section boundary.
    raw = (
        b"## Real Section\n"  # 1
        b"```bash\n"  # 2
        b"## not a heading\n"  # 3
        b"echo hi\n"  # 4
        b"```\n"  # 5
        b"tail\n"  # 6
    )
    result = PlanAdapter().parse(_source("plans/plan.md", raw))
    # Only ONE real section -> one document spanning the whole file.
    assert len(result.documents) == 1
    assert result.documents[0].document.record_locator == "L1"
    assert "not a heading" in result.documents[0].document.body  # still searchable, not a boundary


def test_plan_without_headings_falls_back_to_whole_file() -> None:
    raw = b"just a title line and prose, no level-2 or level-3 headings\n"
    result = PlanAdapter().parse(_source("plans/plan.md", raw))
    assert len(result.documents) == 1
    doc = result.documents[0].document
    assert doc.record_locator is None  # whole-file document
    assert result.documents[0].locator.rendered() == "plans/plan.md:1"


def test_plan_section_body_with_unicode_line_separator_keeps_later_locators_correct() -> None:
    # U+2028 LINE SEPARATOR is legal, unescaped, inside a section body. Splitting on
    # physical newlines only (NOT str.splitlines()) keeps it inside its line, so a
    # later section location is not shifted. The OLD splitlines() split the body
    # line in two and pushed "## Section B" from line 3 to line 4.
    body_line = "alpha body\u2028still one physical line"  # U+2028 mid-line-2
    raw = (
        "## Section A\n" + body_line + "\n## Section B\nbeta body about token usage levers\n"
    ).encode("utf-8")
    result = PlanAdapter().parse(_source("plans/plan.md", raw))
    assert result.diagnostics == ()
    docs = [p.document for p in result.documents]

    # Two sections, at their TRUE physical lines 1 and 3 (not 1 and 4).
    assert [d.record_locator for d in docs] == ["L1", "L3"]
    section_b = next(p for p in result.documents if p.document.record_locator == "L3")
    assert section_b.locator.rendered() == "plans/plan.md:3-4"
    assert "token usage levers" in section_b.document.body
    # The U+2028 stays inside Section A body (one physical line, not two).
    section_a = result.documents[0].document
    assert body_line in section_a.body


def test_plan_binary_content_diagnosed_and_skipped() -> None:
    result = PlanAdapter().parse(_source("plans/plan.md", b"text\x00\x00binary"))
    assert result.documents == ()
    assert len(result.diagnostics) == 1
    diag = result.diagnostics[0]
    assert diag.code == "binary-content"
    assert diag.adapter == "plan"
    assert "\x00" not in diag.message


def test_bom_with_invalid_utf8_is_diagnosed_not_raised_and_siblings_index() -> None:
    # A UTF-8 BOM followed by invalid, NUL-free bytes: the file ANNOUNCES UTF-8 via
    # the BOM but the bytes do not decode. Because it is NUL-free it slips past the
    # plain binary check, and decode_text's BOM branch used to raise UnicodeDecodeError
    # -- which would escape parse() and abort every sibling in the indexer's loop.
    # It must instead be diagnosed (binary-content) + skipped, never raising.
    bad = b"\xef\xbb\xbf\xff\xfe not valid utf-8 but carries no NUL byte"
    result = PlanAdapter().parse(_source("plans/plan.md", bad))  # must NOT raise
    assert result.documents == ()
    assert len(result.diagnostics) == 1
    assert result.diagnostics[0].code == "binary-content"
    assert result.diagnostics[0].adapter == "plan"

    # A sibling good plan parsed right after still indexes normally.
    good = PlanAdapter().parse(_source("other/plan.md", b"## Section\ngood content\n"))
    assert len(good.documents) == 1 and good.diagnostics == ()


def test_decode_text_never_raises_on_bom_plus_invalid_utf8() -> None:
    # Direct base-contract check: decode_text is documented to NEVER raise. A BOM'd
    # but undecodable input falls back to latin-1 (mojibake) rather than raising; the
    # adapters' _looks_binary gate is what turns that into a diagnose+skip.
    decoded = decode_text(b"\xef\xbb\xbf\xff\xfe tail")  # must not raise
    assert isinstance(decoded, str)


# --------------------------------------------------------------------------- #
# HANDOFF / INCIDENT / MEMORY families
# --------------------------------------------------------------------------- #
def test_handoff_current_md_is_section_aware() -> None:
    raw = b"# Session\n\n## Task\ndoing the thing\n## Next\nrun the gate\n"
    src = _source(".claude/task-state/current.md", raw)
    result = HandoffAdapter().parse(src)
    docs = [p.document for p in result.documents]
    assert all(d.artifact_type is ArtifactType.HANDOFF for d in docs)
    bodies = "\n".join(d.body for d in docs)
    assert "doing the thing" in bodies and "run the gate" in bodies
    # Openable locators for each section.
    assert all(
        p.locator.rendered().startswith(".claude/task-state/current.md:") for p in result.documents
    )


def test_incident_note_emits_searchable_text_and_locator() -> None:
    raw = b"# Incident: DB outage\n\n## Impact\n12 eval games flagged crashed\n"
    src = _source("docs/incidents/incident-db.md", raw)
    result = IncidentAdapter().parse(src)
    assert result.diagnostics == ()
    bodies = "\n".join(p.document.body for p in result.documents)
    assert "12 eval games flagged crashed" in bodies
    assert all(p.document.artifact_type is ArtifactType.INCIDENT for p in result.documents)
    assert result.documents[0].locator.rendered().startswith("docs/incidents/incident-db.md:")


def test_memory_feedback_file_indexed_with_memory_type() -> None:
    raw = b"# feedback\n\nnever dump secret file contents\n"
    src = _source("memory/feedback_never_dump.md", raw)
    result = MemoryAdapter().parse(src)
    parsed = result.documents[0]
    doc = parsed.document
    assert doc.artifact_type is ArtifactType.MEMORY
    assert "never dump secret file contents" in doc.body
    assert doc.content_hash == content_hash(raw)
    # Openable path:line locator (no section headings -> whole-file span, 3 lines).
    assert parsed.locator.rendered() == "memory/feedback_never_dump.md:1-3"


def test_memory_precedes_incident_for_feedback_incident_file() -> None:
    # A `feedback_incident_*.md` file is claimed by BOTH MemoryAdapter (feedback_
    # prefix) and IncidentAdapter ("incident" in name). STRUCTURED_ADAPTERS lists
    # Memory before Incident, so its specific memory signal wins the routing.
    raw = b"# feedback\n\nincident post-mortem note\n"
    adapter = select_adapter_for(_source("docs/feedback_incident_db_outage.md", raw))
    assert isinstance(adapter, MemoryAdapter)


# --------------------------------------------------------------------------- #
# DECISION adapter -- frontmatter format contract + fallback
# --------------------------------------------------------------------------- #
def test_decision_frontmatter_keys_and_body_searchable_with_locator() -> None:
    raw = (
        b"---\n"
        b"id: 2026-07-25-markdown-authoritative\n"
        b"title: Markdown is authoritative\n"
        b"status: active\n"
        b"review_date: 2026-10-25\n"
        b"owner: abero\n"  # unknown-to-pinned key -> tolerated + still searchable
        b"---\n"
        b"\n"
        b"The decision body explains the rationale.\n"
    )
    src = _source("decisions/2026-07-25-markdown-authoritative.md", raw)
    adapter = DecisionAdapter()
    assert adapter.handles(src)

    result = adapter.parse(src)
    assert result.diagnostics == ()
    assert len(result.documents) == 1
    doc = result.documents[0].document
    assert doc.artifact_type is ArtifactType.DECISION

    # All four pinned keys are searchable...
    assert "id: 2026-07-25-markdown-authoritative" in doc.body
    assert "title: Markdown is authoritative" in doc.body
    assert "status: active" in doc.body
    assert "review_date: 2026-10-25" in doc.body
    # ...the unknown key is tolerated and still searchable...
    assert "owner: abero" in doc.body
    # ...and the Markdown body is indexed too.
    assert "explains the rationale" in doc.body

    # Openable whole-file locator (frontmatter sits at line 1).
    assert (
        result.documents[0]
        .locator.rendered()
        .startswith("decisions/2026-07-25-markdown-authoritative.md:1")
    )


def test_decision_absent_frontmatter_falls_back_to_plain_markdown_no_diagnostic() -> None:
    # A file under decisions/ with NO frontmatter routes here (path signal) and is
    # indexed as plain Markdown -- no error, no diagnostic.
    raw = b"# Just a note\n\nplain markdown decision content here\n"
    result = DecisionAdapter().parse(_source("decisions/plain-note.md", raw))
    assert result.diagnostics == ()
    assert len(result.documents) == 1
    doc = result.documents[0].document
    assert doc.artifact_type is ArtifactType.DECISION
    assert "plain markdown decision content here" in doc.body


def test_decision_unparseable_frontmatter_warns_and_still_indexes() -> None:
    # A '---' fence that closes but yields no key:value line -> WARN + plain fallback.
    raw = b"---\nthis is not valid key value yaml at all\n---\nbody text stays searchable\n"
    result = DecisionAdapter().parse(_source("decisions/broken.md", raw))
    assert len(result.documents) == 1  # still indexed (fallback), never dropped
    assert "body text stays searchable" in result.documents[0].document.body
    assert len(result.diagnostics) == 1
    assert result.diagnostics[0].code == "unparseable-frontmatter"
    assert result.diagnostics[0].adapter == "decision"


def test_decision_truncated_frontmatter_warns_and_still_indexes() -> None:
    # An opening fence that never closes -> truncated -> WARN + plain fallback.
    raw = b"---\nid: 2026-07-25-x\ntitle: T\nno closing fence ever appears\nmore body\n"
    result = DecisionAdapter().parse(_source("decisions/truncated.md", raw))
    assert len(result.documents) == 1
    assert result.diagnostics[0].code == "frontmatter-not-closed"


def test_decision_bad_frontmatter_diagnostic_never_leaks_content() -> None:
    # A planted secret token in a corrupt-frontmatter file must never appear in the
    # diagnostic message (it may appear in the indexed body -- that is indexing, not a
    # diagnostic).
    leak = "SUPERSECRETVALUE12345"
    raw = (f"---\n{leak} not a key value line\n---\nbody\n").encode()
    result = DecisionAdapter().parse(_source("decisions/leak.md", raw))
    assert len(result.diagnostics) == 1
    assert leak not in result.diagnostics[0].message


def test_decision_indented_pinned_key_does_not_masquerade_as_top_level() -> None:
    # A nested/indented `status:` (a value inside another key's map) must NOT be taken
    # as the pinned TOP-LEVEL status. The minimal parser only accepts column-0 keys, so
    # the nested one is ignored and the real top-level `status:` wins -- no wrong value
    # rendered and no false content-signature match. (Old code `key.strip()`ped the
    # indentation and let the nested line masquerade as the pinned key.)
    raw = (
        b"---\n"
        b"config:\n"  # a key whose value is a nested map
        b"  status: nested-should-be-ignored\n"  # INDENTED -> not the pinned status
        b"  id: nested-id\n"  # INDENTED -> not the pinned id
        b"id: real-id\n"
        b"title: Real Decision\n"
        b"status: active\n"  # the genuine top-level status
        b"---\n"
        b"body\n"
    )
    src = _source("decisions/nested.md", raw)
    result = DecisionAdapter().parse(src)
    assert result.diagnostics == ()
    doc = result.documents[0].document
    # The real top-level values are surfaced...
    assert "status: active" in doc.body
    assert "id: real-id" in doc.body
    # ...and the nested lines never masquerade as the pinned keys.
    assert "nested-should-be-ignored" not in doc.body
    assert "nested-id" not in doc.body

    # A file whose ONLY id/title/status are indented (nested) does NOT satisfy the
    # content signature -- so it is not falsely claimed as a decision record.
    nested_only = _source(
        "notes/whatever.md",
        b"---\nwrapper:\n  id: x\n  title: T\n  status: active\n---\nbody\n",
    )
    assert not DecisionAdapter()._has_decision_signature(nested_only.raw)


def test_decision_binary_content_diagnosed_and_skipped() -> None:
    # A binary blob mislabeled with a text extension under decisions/ -> diagnosed +
    # skipped (no documents), like the prose families -- parse() never raises.
    result = DecisionAdapter().parse(_source("decisions/blob.md", b"lead\x00\x00binary"))
    assert result.documents == ()
    assert len(result.diagnostics) == 1
    diag = result.diagnostics[0]
    assert diag.code == "binary-content"
    assert diag.adapter == "decision"
    assert "\x00" not in diag.message


# --------------------------------------------------------------------------- #
# Corrupt-without-aborting-siblings + determinism (plan.md Step 3b done-when)
# --------------------------------------------------------------------------- #
def test_corrupt_input_does_not_raise_or_abort_siblings() -> None:
    # A binary "plan" is diagnosed + skipped; parse() returns rather than raising, so a
    # sibling good plan parsed right after still indexes normally.
    adapter = PlanAdapter()
    corrupt = adapter.parse(_source("plans/plan.md", b"\x00\x00\x00binary"))
    assert corrupt.documents == () and corrupt.diagnostics[0].code == "binary-content"

    good = adapter.parse(_source("other/plan.md", b"## Section\ngood content\n"))
    assert len(good.documents) == 1 and good.diagnostics == ()


def test_structured_and_decision_output_is_deterministic() -> None:
    plan_raw = b"# T\n\n## A\nalpha\n### Step 1\nbeta\n"
    plan_src = _source("plans/plan.md", plan_raw)
    assert PlanAdapter().parse(plan_src) == PlanAdapter().parse(plan_src)

    dec_raw = b"---\nid: x\ntitle: T\nstatus: active\nzeta: 1\nalpha: 2\n---\nbody\n"
    dec_src = _source("decisions/x.md", dec_raw)
    assert DecisionAdapter().parse(dec_src) == DecisionAdapter().parse(dec_src)
