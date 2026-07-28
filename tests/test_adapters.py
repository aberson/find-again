"""Tests for the adapter contract and the two generic adapters (plan.md Step 3a, §5, §8).

Covers the done-when conditions and required fixtures:

* a Markdown/text file emits searchable text + an openable ``path:line`` locator,
* a JSON file emits searchable text + a (whole-file) locator,
* a JSONL file emits one document per record with ``path::L<n>`` locators,
* a CORRUPT input (malformed JSON, a bad JSONL line among good ones) emits a
  diagnostic (path + code, NO content) and does NOT abort the sibling records/files,
* ``content_hash`` (over raw bytes) + ``doc_id`` are correct, and output is
  deterministic.
"""

from __future__ import annotations

from pathlib import Path

from find_again.adapters import (
    JsonAdapter,
    MarkdownAdapter,
    SourceFile,
    select_generic_adapter,
)
from find_again.models import ArtifactType, content_hash

_PROJECT = "find-again"
_TS = "2026-07-27T00:00:00Z"


def _source(source_path: str, raw: bytes) -> SourceFile:
    return SourceFile(source_path=source_path, raw=raw, project=_PROJECT, timestamp=_TS)


# --------------------------------------------------------------------------- #
# Markdown / text adapter
# --------------------------------------------------------------------------- #
def test_markdown_indexes_whole_file_with_openable_path_line_locator() -> None:
    raw = b"# Notes\n\nremember to never dump secret file contents\n"
    src = _source("docs/notes.md", raw)
    adapter = MarkdownAdapter()
    assert adapter.handles(src)

    result = adapter.parse(src)
    assert result.diagnostics == ()
    assert len(result.documents) == 1

    parsed = result.documents[0]
    doc = parsed.document
    assert doc.doc_id == "docs/notes.md"  # root-relative POSIX path, no record locator
    assert doc.source_path == "docs/notes.md"
    assert doc.record_locator is None
    assert doc.artifact_type is ArtifactType.MARKDOWN
    assert doc.project == _PROJECT
    assert doc.timestamp == _TS
    assert doc.content_hash == content_hash(raw)  # over raw bytes
    assert "never dump secret file contents" in doc.body  # whole-file searchable text

    # Openable path:line locator spanning the whole file (3 lines).
    assert parsed.locator.rendered() == "docs/notes.md:1-3"


def test_markdown_single_line_locator_and_txt_extension() -> None:
    src = _source("a.txt", b"single line only")
    adapter = MarkdownAdapter()
    assert adapter.handles(src)
    parsed = adapter.parse(src).documents[0]
    assert parsed.locator.rendered() == "a.txt:1"


def test_markdown_rejects_non_text_extension() -> None:
    assert not MarkdownAdapter().handles(_source("image.png", b"\x89PNG"))


def test_markdown_binary_content_is_diagnosed_and_skipped() -> None:
    raw = b"leading text\x00\x00trailing"
    result = MarkdownAdapter().parse(_source("weird.md", raw))
    assert result.documents == ()  # skipped, not indexed as mojibake
    assert len(result.diagnostics) == 1
    diag = result.diagnostics[0]
    assert diag.code == "binary-content"
    assert diag.source_path == "weird.md"
    assert diag.adapter == "markdown"
    assert "\x00" not in diag.message  # message never carries file bytes


def test_markdown_content_hash_over_raw_bytes_not_decoded_body() -> None:
    # A UTF-8 BOM is stripped from the searchable body but IS part of the raw bytes
    # the file-level content_hash is taken over.
    raw = b"\xef\xbb\xbfheader with bom"
    doc = MarkdownAdapter().parse(_source("bom.md", raw)).documents[0].document
    assert doc.body == "header with bom"
    assert doc.content_hash == content_hash(raw)


# --------------------------------------------------------------------------- #
# JSON adapter (single whole-file document)
# --------------------------------------------------------------------------- #
def test_json_emits_searchable_text_and_whole_file_locator() -> None:
    raw = b'{\n  "title": "token usage levers",\n  "count": 42,\n  "active": true\n}\n'
    src = _source("data/config.json", raw)
    adapter = JsonAdapter()
    assert adapter.handles(src)

    result = adapter.parse(src)
    assert result.diagnostics == ()
    assert len(result.documents) == 1

    parsed = result.documents[0]
    doc = parsed.document
    assert doc.doc_id == "data/config.json"  # single document -> no ::record
    assert doc.record_locator is None
    assert doc.artifact_type is ArtifactType.JSON
    assert doc.content_hash == content_hash(raw)

    # Keys AND scalar values are searchable.
    assert "token usage levers" in doc.body
    assert "count: 42" in doc.body
    assert "active: true" in doc.body

    # Whole-file line locator, openable.
    assert parsed.locator.source_path == "data/config.json"
    assert parsed.locator.line_start == 1


def test_json_flatten_is_deterministic_and_key_sorted() -> None:
    src = _source("x.json", b'{"zebra": 1, "apple": 2}')
    body1 = JsonAdapter().parse(src).documents[0].document.body
    body2 = JsonAdapter().parse(src).documents[0].document.body
    assert body1 == body2  # deterministic
    assert body1.index("apple") < body1.index("zebra")  # stable, key-sorted ordering


def test_json_nested_values_are_searchable() -> None:
    raw = b'{"outer": {"inner": "deepvalue"}, "list": ["a", "b"]}'
    doc = JsonAdapter().parse(_source("n.json", raw)).documents[0].document
    assert "deepvalue" in doc.body
    assert "outer.inner: deepvalue" in doc.body
    assert "list[0]: a" in doc.body


def test_json_malformed_document_is_diagnosed_with_no_documents() -> None:
    result = JsonAdapter().parse(_source("broken.json", b'{"unterminated": '))
    assert result.documents == ()
    assert len(result.diagnostics) == 1
    diag = result.diagnostics[0]
    assert diag.code == "malformed-json"
    assert diag.source_path == "broken.json"
    assert diag.adapter == "json"


def test_json_deeply_nested_input_diagnosed_and_parse_does_not_raise() -> None:
    # ~100 KB of unbalanced brackets makes json.loads raise RecursionError, which is
    # NOT a JSONDecodeError. parse() must swallow it (diagnostic + skip), never raise.
    hostile = b"[" * 100_000  # within any size limit; a stack-buster for json.loads
    result = JsonAdapter().parse(_source("hostile.json", hostile))  # must not raise
    assert result.documents == ()
    assert len(result.diagnostics) == 1
    assert result.diagnostics[0].code == "malformed-json"
    assert result.diagnostics[0].source_path == "hostile.json"

    # A sibling good file still indexes -- the hostile file did not abort the run.
    good = JsonAdapter().parse(_source("good.json", b'{"ok": 1}'))
    assert len(good.documents) == 1


def test_json_binary_content_is_diagnosed_and_skipped() -> None:
    # A binary blob mislabeled .json (decode_text's latin-1 fallback never errors, so
    # it would otherwise mojibake-index) -> diagnosed + skipped, like the text adapter.
    result = JsonAdapter().parse(_source("blob.json", b"\x00\x01\x02not json"))
    assert result.documents == ()
    assert len(result.diagnostics) == 1
    diag = result.diagnostics[0]
    assert diag.code == "binary-content"
    assert diag.adapter == "json"
    assert "\x00" not in diag.message


# --------------------------------------------------------------------------- #
# JSONL adapter (one document per record)
# --------------------------------------------------------------------------- #
def test_jsonl_emits_one_document_per_record_with_line_locators() -> None:
    raw = b'{"event": "start"}\n{"event": "stop"}\n{"event": "resume"}\n'
    src = _source("runs/telemetry.jsonl", raw)
    adapter = JsonAdapter()
    assert adapter.handles(src)

    result = adapter.parse(src)
    assert result.diagnostics == ()
    assert len(result.documents) == 3

    docs = [p.document for p in result.documents]
    assert [d.doc_id for d in docs] == [
        "runs/telemetry.jsonl::L1",
        "runs/telemetry.jsonl::L2",
        "runs/telemetry.jsonl::L3",
    ]
    assert [d.record_locator for d in docs] == ["L1", "L2", "L3"]
    assert all(d.artifact_type is ArtifactType.JSONL for d in docs)

    # content_hash is file-level: every record shares the whole-file hash.
    assert {d.content_hash for d in docs} == {content_hash(raw)}

    # Openable path:line locators per record.
    assert result.documents[0].locator.rendered() == "runs/telemetry.jsonl:1"
    assert result.documents[1].locator.rendered() == "runs/telemetry.jsonl:2"
    assert "resume" in docs[2].body


def test_jsonl_malformed_line_is_diagnosed_and_siblings_still_index() -> None:
    raw = b'{"ok": 1}\nnot json at all\n{"ok": 3}\n'
    result = JsonAdapter().parse(_source("runs/bad.jsonl", raw))

    # The two good sibling records still index at their true line positions.
    docs = [p.document for p in result.documents]
    assert [d.doc_id for d in docs] == ["runs/bad.jsonl::L1", "runs/bad.jsonl::L3"]

    # Exactly one diagnostic, for the bad middle line, naming the path + line only.
    assert len(result.diagnostics) == 1
    diag = result.diagnostics[0]
    assert diag.code == "malformed-record"
    assert diag.source_path == "runs/bad.jsonl"
    assert "line 2" in diag.message


def test_jsonl_blank_lines_skipped_silently_and_line_numbers_are_true_positions() -> None:
    raw = b'{"a": 1}\n\n   \n{"b": 2}\n'
    result = JsonAdapter().parse(_source("sparse.jsonl", raw))
    assert result.diagnostics == ()
    assert len(result.documents) == 2
    # Physical lines 2 and 3 are blank; the second record is at line 4.
    assert [p.document.record_locator for p in result.documents] == ["L1", "L4"]


def test_jsonl_deeply_nested_record_diagnosed_and_sibling_records_still_index() -> None:
    # A pathologically deep record (json.loads -> RecursionError) sits between two
    # good records. parse() must not raise; the good siblings still index at their
    # true physical-line positions and only the hostile record is diagnosed.
    hostile = b"[" * 100_000
    raw = b'{"ok": 1}\n' + hostile + b'\n{"ok": 3}\n'
    result = JsonAdapter().parse(_source("runs/hostile.jsonl", raw))  # must not raise

    docs = [p.document for p in result.documents]
    assert [d.doc_id for d in docs] == ["runs/hostile.jsonl::L1", "runs/hostile.jsonl::L3"]
    assert len(result.diagnostics) == 1
    assert result.diagnostics[0].code == "malformed-record"
    assert "line 2" in result.diagnostics[0].message


def test_jsonl_record_with_unicode_line_separator_stays_one_record() -> None:
    # U+2028 LINE SEPARATOR is legal unescaped inside a JSON string. Splitting on
    # physical newlines only keeps such a record whole; the OLD splitlines() split it
    # in two (wrong diagnostic) and shifted every later record's ::L<n> locator.
    first = '{"note": "before\u2028after"}'
    raw = (first + "\n" + '{"next": 2}' + "\n").encode("utf-8")
    result = JsonAdapter().parse(_source("u.jsonl", raw))

    assert result.diagnostics == ()
    docs = [p.document for p in result.documents]
    assert len(docs) == 2
    # The U+2028-bearing record is ONE valid record at physical line 1.
    assert docs[0].record_locator == "L1"
    assert "before\u2028after" in docs[0].body
    # The NEXT record's physical-line locator is line 2 (not shifted to 3).
    assert docs[1].record_locator == "L2"
    assert result.documents[1].locator.rendered() == "u.jsonl:2"


# --------------------------------------------------------------------------- #
# Diagnostics never leak file content (safety contract, plan.md §5 Diagnostic)
# --------------------------------------------------------------------------- #
def test_diagnostic_messages_never_contain_file_content() -> None:
    leak = "SUPERSECRETVALUE12345"

    # Malformed JSONL record carrying the token.
    jsonl = ('{"good": 1}\n{"broken": ' + leak + "\n").encode("utf-8")
    result = JsonAdapter().parse(_source("leak.jsonl", jsonl))
    assert len(result.documents) == 1  # the good sibling survived
    assert len(result.diagnostics) == 1
    assert leak not in result.diagnostics[0].message

    # Malformed whole-file JSON carrying the token.
    bad_json = ('{"broken": ' + leak).encode("utf-8")
    diag = JsonAdapter().parse(_source("leak.json", bad_json)).diagnostics[0]
    assert leak not in diag.message


# --------------------------------------------------------------------------- #
# Contract-level dispatch + determinism
# --------------------------------------------------------------------------- #
def test_select_generic_adapter_dispatches_by_extension() -> None:
    assert isinstance(select_generic_adapter(_source("a.md", b"x")), MarkdownAdapter)
    assert isinstance(select_generic_adapter(_source("a.txt", b"x")), MarkdownAdapter)
    assert isinstance(select_generic_adapter(_source("a.json", b"{}")), JsonAdapter)
    assert isinstance(select_generic_adapter(_source("a.jsonl", b"{}")), JsonAdapter)
    assert isinstance(select_generic_adapter(_source("a.ndjson", b"{}")), JsonAdapter)
    assert select_generic_adapter(_source("a.png", b"x")) is None


def test_parse_result_is_deterministic_across_runs() -> None:
    # Whole-result equality (frozen dataclasses) is the strongest determinism check.
    raw = b'{"k": [3, 1, 2], "z": {"b": 2, "a": 1}}'
    src = _source("d.json", raw)
    assert JsonAdapter().parse(src) == JsonAdapter().parse(src)


def test_source_file_from_path_reads_bytes(tmp_path: Path) -> None:
    file_path = tmp_path / "note.md"
    file_path.write_bytes(b"hello world content")
    src = SourceFile.from_path(
        file_path, source_path="docs/note.md", project=_PROJECT, timestamp=_TS
    )
    assert src.raw == b"hello world content"
    assert src.content_hash == content_hash(b"hello world content")

    doc = MarkdownAdapter().parse(src).documents[0].document
    assert doc.doc_id == "docs/note.md"
    assert doc.body == "hello world content"
