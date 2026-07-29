"""Tests for the ``find-again`` CLI ``index`` and ``status`` verbs (plan.md Step 4, §10).

Drives the real entry point (:func:`find_again.cli.main`) through ``--root`` so no
git repo is needed, and asserts counts, JSON shape, diagnostics reporting, and exit
codes (0 success, 2 config/root failure).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from find_again.cli import main

SECRET_TOKEN = "AKIAIOSFODNN7EXAMPLE"  # noqa: S105 -- test fixture, not a real credential


def _write(root: Path, rel: str, text: str) -> None:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _project(root: Path) -> Path:
    """A configured project root with a find-again.toml and one indexable file."""
    _write(root, "find-again.toml", 'schema_version = 1\nroots = ["docs"]\n')
    _write(root, "docs/note.md", "unique_cli_marker content\n")
    return root


# --------------------------------------------------------------------------- #
# index
# --------------------------------------------------------------------------- #
def test_index_text_reports_counts(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _project(tmp_path)
    code = main(["--root", str(tmp_path), "index", "--no-git-ignore"])
    assert code == 0
    out = capsys.readouterr().out
    assert "new:       1" in out
    assert "documents: 1 total" in out


def test_index_json_shape(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _project(tmp_path)
    code = main(["--root", str(tmp_path), "--json", "index", "--no-git-ignore"])
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["indexed"] == 1
    assert payload["documents"] == 1
    assert payload["diagnostics"] == []
    assert payload["refreshed_at"].endswith("Z")


def test_index_then_status_reports_age_and_count(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _project(tmp_path)
    assert main(["--root", str(tmp_path), "index", "--no-git-ignore"]) == 0
    capsys.readouterr()  # drain

    assert main(["--root", str(tmp_path), "status"]) == 0
    out = capsys.readouterr().out
    assert "documents: 1" in out
    assert "last refreshed:" in out
    assert "ago" in out
    assert "docs" in out  # roots listed


def test_status_json_before_index_reports_never(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _write(tmp_path, "find-again.toml", 'roots = ["docs"]\n')
    code = main(["--root", str(tmp_path), "--json", "status"])
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["documents"] == 0
    assert payload["last_refreshed"] is None
    assert payload["age_seconds"] is None


# --------------------------------------------------------------------------- #
# Diagnostics surface through index + status
# --------------------------------------------------------------------------- #
def test_secret_file_reported_but_not_indexed(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _write(tmp_path, "find-again.toml", 'roots = ["docs"]\n')
    _write(tmp_path, "docs/note.md", "clean content\n")
    _write(tmp_path, "docs/creds.md", f"key {SECRET_TOKEN}\n")

    assert main(["--root", str(tmp_path), "--json", "index", "--no-git-ignore"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["documents"] == 1  # only the clean file
    codes = {d["code"] for d in payload["diagnostics"]}
    assert "secret-content" in codes
    # The secret text never appears in any reported diagnostic.
    assert SECRET_TOKEN not in json.dumps(payload["diagnostics"])

    # status re-reports the persisted diagnostic without re-indexing.
    assert main(["--root", str(tmp_path), "--json", "status"]) == 0
    status_payload = json.loads(capsys.readouterr().out)
    assert any(d["code"] == "secret-content" for d in status_payload["diagnostics"])


# --------------------------------------------------------------------------- #
# Exit codes
# --------------------------------------------------------------------------- #
def test_missing_config_is_exit_2(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    # A directory with no find-again.toml -> explicit config error, exit 2.
    code = main(["--root", str(tmp_path), "index", "--no-git-ignore"])
    assert code == 2
    assert "find-again:" in capsys.readouterr().err


def test_missing_root_override_is_exit_2(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = main(["--root", str(tmp_path / "does-not-exist"), "status"])
    assert code == 2
    assert capsys.readouterr().err


def test_no_command_prints_help_exit_2(capsys: pytest.CaptureFixture[str]) -> None:
    code = main([])
    assert code == 2
    assert "usage:" in capsys.readouterr().out.lower()


def test_search_still_stubbed_exit_2(capsys: pytest.CaptureFixture[str]) -> None:
    code = main(["search", "anything"])
    assert code == 2
    assert "not implemented" in capsys.readouterr().err


# --------------------------------------------------------------------------- #
# Finding 3: a corrupt index DB reports a clean error (exit 1), never a traceback
# --------------------------------------------------------------------------- #
def test_corrupt_index_db_reports_clean_error_exit_1(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _project(tmp_path)
    # Plant a non-SQLite file where the index DB lives: opening it raises a raw
    # sqlite3 error that must be reported cleanly, not leaked as a traceback.
    db_path = tmp_path / ".find-again" / "index.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    db_path.write_bytes(b"this is definitely not a sqlite database\n" * 8)

    code = main(["--root", str(tmp_path), "index", "--no-git-ignore"])
    assert code == 1
    err = capsys.readouterr().err
    assert err.startswith("find-again:")
    assert "Traceback" not in err


# --------------------------------------------------------------------------- #
# Finding 6: a newer-schema index DB reports a clean error at exit 2 (usage/config
# class, matching the documented exit-code contract for "schema newer than build")
# --------------------------------------------------------------------------- #
def test_newer_index_db_reports_clean_error_exit_2(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    import sqlite3

    from find_again.db import Database, default_db_path, latest_migration_version

    _project(tmp_path)
    # Create a valid index DB, then bump its user_version past this build's support.
    Database.open_root(tmp_path).close()
    db_path = default_db_path(tmp_path)
    raw = sqlite3.connect(str(db_path), isolation_level=None)
    raw.execute(f"PRAGMA user_version = {latest_migration_version() + 5}")
    raw.close()

    code = main(["--root", str(tmp_path), "status"])
    assert code == 2
    err = capsys.readouterr().err
    assert err.startswith("find-again:")
    assert "newer" in err.lower()
