"""Tests for root resolution and config loading (plan.md §6 Configuration discovery)."""

from __future__ import annotations

from pathlib import Path

import pytest

from find_again.config import (
    CONFIG_FILENAME,
    SCHEMA_VERSION,
    Config,
    ConfigError,
    RootResolutionError,
    SchemaVersionError,
    load_config,
    resolve_root,
)


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


# --------------------------------------------------------------------------- #
# Root resolution
# --------------------------------------------------------------------------- #
def test_resolve_root_override_wins(tmp_path: Path) -> None:
    target = tmp_path / "explicit"
    target.mkdir()
    assert resolve_root(tmp_path, str(target)) == target.resolve()


def test_resolve_root_override_missing_is_error(tmp_path: Path) -> None:
    with pytest.raises(RootResolutionError):
        resolve_root(tmp_path, str(tmp_path / "does-not-exist"))


def test_resolve_root_finds_enclosing_git_root(tmp_path: Path) -> None:
    (tmp_path / ".git").mkdir()
    nested = tmp_path / "a" / "b"
    nested.mkdir(parents=True)
    assert resolve_root(nested, None) == tmp_path.resolve()


def test_resolve_root_git_dir_may_be_a_file(tmp_path: Path) -> None:
    # Worktrees use a `.git` *file*, not a directory.
    (tmp_path / ".git").write_text("gitdir: /somewhere/else\n", encoding="utf-8")
    assert resolve_root(tmp_path, None) == tmp_path.resolve()


def test_resolve_root_no_repo_no_override_is_explicit_error(tmp_path: Path) -> None:
    # tmp_path lives under the OS temp dir, which is not a git repository.
    with pytest.raises(RootResolutionError):
        resolve_root(tmp_path, None)


# --------------------------------------------------------------------------- #
# Config loading
# --------------------------------------------------------------------------- #
def test_load_config_missing_file_is_explicit_error(tmp_path: Path) -> None:
    with pytest.raises(ConfigError):
        load_config(tmp_path)


def test_load_config_parses_fields(tmp_path: Path) -> None:
    _write(
        tmp_path / CONFIG_FILENAME,
        """
        schema_version = 1
        roots = ["docs", ".claude/task-state"]
        exclude = ["**/*.env", "**/secrets*"]
        max_file_kb = 256
        """,
    )
    config = load_config(tmp_path)
    assert isinstance(config, Config)
    assert config.roots == ("docs", ".claude/task-state")
    assert config.exclude == ("**/*.env", "**/secrets*")
    assert config.max_file_kb == 256
    assert config.schema_version == 1


def test_load_config_defaults_when_optional_fields_absent(tmp_path: Path) -> None:
    _write(tmp_path / CONFIG_FILENAME, 'roots = ["docs"]\n')
    config = load_config(tmp_path)
    assert config.roots == ("docs",)
    assert config.exclude == ()
    assert config.max_file_kb == 512
    assert config.schema_version == SCHEMA_VERSION


def test_load_config_tolerates_older_schema_version(tmp_path: Path) -> None:
    _write(tmp_path / CONFIG_FILENAME, 'schema_version = 0\nroots = ["docs"]\n')
    config = load_config(tmp_path)
    assert config.schema_version == 0


def test_load_config_refuses_newer_schema_version(tmp_path: Path) -> None:
    _write(
        tmp_path / CONFIG_FILENAME,
        f'schema_version = {SCHEMA_VERSION + 1}\nroots = ["docs"]\n',
    )
    with pytest.raises(SchemaVersionError):
        load_config(tmp_path)


def test_load_config_rejects_bad_types(tmp_path: Path) -> None:
    _write(tmp_path / CONFIG_FILENAME, 'roots = "docs"\n')
    with pytest.raises(ConfigError):
        load_config(tmp_path)


def test_load_config_rejects_nonpositive_max_file_kb(tmp_path: Path) -> None:
    _write(tmp_path / CONFIG_FILENAME, 'roots = ["docs"]\nmax_file_kb = 0\n')
    with pytest.raises(ConfigError):
        load_config(tmp_path)


def test_load_config_malformed_toml_is_clean_error(tmp_path: Path) -> None:
    # A syntactically broken TOML must surface as a clean ConfigError, never an
    # unhandled TOMLDecodeError crash.
    _write(tmp_path / CONFIG_FILENAME, 'roots = ["docs"\nmax_file_kb = \n')
    with pytest.raises(ConfigError):
        load_config(tmp_path)
