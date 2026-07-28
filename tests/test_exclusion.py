"""Security-critical exclusion tests (plan.md §6 Explicit roots; Step 1 done-when).

The load-bearing assertion is :func:`test_secret_text_reaches_neither_index_nor_diagnostic`:
a file whose *content* matches a secret pattern is skipped whole, and the secret
text reaches neither the to-be-indexed content set nor any diagnostic message.

Secret-shaped fixtures are generated at runtime under ``tmp_path`` -- never
committed -- so no real-looking credential lands in the repository.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from find_again.config import (
    Config,
    ExclusionDecision,
    evaluate_file,
    git_ignored,
    scan_secret_content,
)
from find_again.models import Severity

# The canonical AWS documentation example key (safe, widely allowlisted),
# assembled in pieces so the 20-char literal never appears in source.
_AWS_EXAMPLE_KEY = "AKIA" + "IOSFODNN7" + "EXAMPLE"
_PRIVATE_KEY_HEADER = "-----BEGIN " + "PRIVATE KEY-----"
_HIGH_ENTROPY_TOKEN = "aB3dE5fG7hJ9kM1nP2qR4sT6uV8wX0yZbC4dF6gH8jK2"
# A unique benign marker used to prove the secret file's body never leaks.
_BODY_MARKER = "UNIQUE_SECRET_BODY_MARKER_7f3a91"

# Secret-shaped fixtures for the broadened content scan. Every value is assembled
# from fragments so no contiguous credential-looking literal appears in source
# (avoids tripping push protection); none is a real credential.
_PG_CONN_SECRET = "postgres://" + "appuser" + ":" + "s3cr3tP" + "assw0rd" + "@" + "db.host:5432/app"
_STRIPE_KEY = "sk_" + "live_" + "0123456789abcdefABCDEF"
_GITHUB_TOKEN = "ghp_" + "A1b2C3d4E5f6G7h8I9j0K1l2M3n4O5p6Q7r8"
_SLACK_TOKEN = "xoxb-" + "12345678901-ABCDEFabcdef"
_SLACK_APP_TOKEN = "xapp-" + "1-A1B2C3D4E5F6"
_GOOGLE_KEY = "AIza" + "SyABCDEFGHIJKLMNOPQRSTUVWXYZ0123456"
_JWT_TOKEN = "eyJ" + "abcdef" + "." + "eyJzdWIi" + "." + "SflKxwRJSMe"
_GOCSPX_SECRET = "GOCSPX-" + "abcABC123_-defGHI"
_SENDGRID_KEY = "SG." + "abcABC123defGHI45" + "." + "abcABC123defGHI45"
_GITLAB_PAT = "glpat-" + "abcABC123_-defGHIJ"
_OPENAI_KEY = "sk-" + "abcABC0123456789defGHIJ"
_NPM_TOKEN = "npm_" + "A1b2C3d4E5f6G7h8I9j0K1l2M3n4O5p6Q7r8"
_PYPI_TOKEN = "pypi-" + "abcABC123_-defGHIJ"
_TWILIO_SID = "SK" + "0123456789abcdef0123456789abcdef"
_ASSIGNED_QUOTED = 'api_key = "' + "s3cr3tV4lue123" + '"'
_ASSIGNED_PADDED_B64 = 'secret: "' + "aGVsbG8gd29ybGQh" + '=="'
_ASSIGNED_SHORT = 'password = "' + "s3" + "cr3t" + '"'
# Benign high-ish-entropy strings that MUST NOT over-skip a legit file
# (over-skip is silent data loss): a 64-char content hash, a UUID, and a prose
# line that merely mentions a secret-shaped key without assigning a value.
_BENIGN_CONTENT_HASH = "a1b2c3d4" * 8
_BENIGN_UUID = "550e8400-e29b-41d4-a716-446655440000"
_BENIGN_PROSE_KEY = "The access_key: documented in the runbook, rotate quarterly."


def _config(root: Path, *, exclude: tuple[str, ...] = (), max_file_kb: int = 512) -> Config:
    return Config(
        root=root,
        roots=("docs",),
        exclude=exclude,
        max_file_kb=max_file_kb,
        schema_version=1,
    )


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


# --------------------------------------------------------------------------- #
# Layer 0 -- outside configured roots
# --------------------------------------------------------------------------- #
def test_outside_roots_excluded_with_diagnostic(tmp_path: Path) -> None:
    config = _config(tmp_path)
    outside = _write(tmp_path / "other" / "note.md", "hello\n")
    decision = evaluate_file(config, outside)
    assert decision.include is False
    assert decision.diagnostic is not None
    assert decision.diagnostic.code == "outside-roots"


# --------------------------------------------------------------------------- #
# Layer 1 -- path-glob deny patterns (built-in secret globs + operator excludes)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "relpath",
    [
        "docs/config.env",
        "docs/.env",
        "docs/secrets.yaml",
        "docs/app-secret.txt",
        "docs/server.pem",
        "docs/id_rsa",
        "docs/.ssh/known_hosts",
    ],
)
def test_secret_path_globs_excluded(tmp_path: Path, relpath: str) -> None:
    config = _config(tmp_path)
    target = _write(tmp_path / relpath, "irrelevant content\n")
    decision = evaluate_file(config, target)
    assert decision.include is False
    assert decision.diagnostic is not None
    assert decision.diagnostic.code == "secret-path-glob"


def test_operator_exclude_glob_excluded(tmp_path: Path) -> None:
    config = _config(tmp_path, exclude=("docs/drafts/**",))
    target = _write(tmp_path / "docs" / "drafts" / "wip.md", "work in progress\n")
    decision = evaluate_file(config, target)
    assert decision.include is False
    assert decision.diagnostic is not None
    assert decision.diagnostic.code == "excluded-glob"


def test_path_glob_denies_before_content_is_read(tmp_path: Path) -> None:
    # A `.env` whose bytes cannot be read must still be excluded by the path
    # layer (which runs before any read). We pass a path that does not exist.
    config = _config(tmp_path)
    missing_env = tmp_path / "docs" / "secret.env"
    decision = evaluate_file(config, missing_env)
    assert decision.include is False
    assert decision.diagnostic is not None
    assert decision.diagnostic.code == "secret-path-glob"


# --------------------------------------------------------------------------- #
# Layer 2 -- git-ignored
# --------------------------------------------------------------------------- #
def test_git_ignored_flag_excludes(tmp_path: Path) -> None:
    config = _config(tmp_path)
    target = _write(tmp_path / "docs" / "generated.md", "derived\n")
    decision = evaluate_file(config, target, is_git_ignored=True)
    assert decision.include is False
    assert decision.diagnostic is not None
    assert decision.diagnostic.code == "git-ignored"


@pytest.mark.skipif(shutil.which("git") is None, reason="git not available")
def test_git_ignored_helper_reports_ignored_paths(tmp_path: Path) -> None:
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)  # noqa: S603, S607
    _write(tmp_path / ".gitignore", "docs/*.log\n")
    _write(tmp_path / "docs" / "run.log", "noise\n")
    _write(tmp_path / "docs" / "keep.md", "kept\n")
    ignored = git_ignored(tmp_path, ["docs/run.log", "docs/keep.md"])
    assert "docs/run.log" in ignored
    assert "docs/keep.md" not in ignored


# --------------------------------------------------------------------------- #
# Layer 3 -- oversized
# --------------------------------------------------------------------------- #
def test_oversized_file_excluded(tmp_path: Path) -> None:
    config = _config(tmp_path, max_file_kb=1)
    target = _write(tmp_path / "docs" / "big.md", "x" * 4096)  # 4 KB > 1 KB
    decision = evaluate_file(config, target)
    assert decision.include is False
    assert decision.diagnostic is not None
    assert decision.diagnostic.code == "oversized"


def test_oversized_check_honors_provided_content(tmp_path: Path) -> None:
    config = _config(tmp_path, max_file_kb=1)
    target = tmp_path / "docs" / "big.md"
    decision = evaluate_file(config, target, content=b"y" * 4096)
    assert decision.include is False
    assert decision.diagnostic is not None
    assert decision.diagnostic.code == "oversized"


# --------------------------------------------------------------------------- #
# Layer 4 -- content-based secret scan
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("payload", "expected_id"),
    [
        (_AWS_EXAMPLE_KEY, "aws-access-key-id"),
        (_PRIVATE_KEY_HEADER, "private-key-header"),
        (_HIGH_ENTROPY_TOKEN, "high-entropy-token"),
    ],
)
def test_scan_secret_content_returns_pattern_id_not_text(payload: str, expected_id: str) -> None:
    body = f"some prose\n{payload}\nmore prose\n"
    result = scan_secret_content(body)
    assert result == expected_id
    # The returned value is an id, never the matched span.
    assert payload not in result


def test_clean_content_is_indexed(tmp_path: Path) -> None:
    config = _config(tmp_path)
    target = _write(
        tmp_path / "docs" / "notes.md",
        "# Notes\n\nJust ordinary prose about the token-usage levers.\n",
    )
    decision = evaluate_file(config, target)
    assert decision.include is True
    assert decision.diagnostic is None


def test_content_secret_skips_whole_file_with_path_and_id_only(tmp_path: Path) -> None:
    config = _config(tmp_path)
    # Innocuously named file (passes the path-glob layer) that hides a secret.
    target = _write(
        tmp_path / "docs" / "onboarding.md",
        f"# Onboarding\n\nExample deploy key:\n{_PRIVATE_KEY_HEADER}\n{_BODY_MARKER}\n",
    )
    decision = evaluate_file(config, target)
    assert decision.include is False  # whole file skipped, never redact-and-index
    assert decision.diagnostic is not None
    diag = decision.diagnostic
    assert diag.code == "secret-content"
    assert diag.adapter == "secret-scan"
    assert diag.severity is Severity.ERROR
    # Names the path + pattern id only.
    assert diag.source_path == "docs/onboarding.md"
    assert diag.message == "content matched secret pattern id=private-key-header"
    # Neither the secret header nor the file body marker appears in the message.
    assert _PRIVATE_KEY_HEADER not in diag.message
    assert _BODY_MARKER not in diag.message


# --------------------------------------------------------------------------- #
# DONE-WHEN (load-bearing): secret text reaches neither the index nor a diagnostic
# --------------------------------------------------------------------------- #
def _simulate_index(config: Config, files: list[Path]) -> tuple[list[str], list[str]]:
    """Mimic the future indexer: collect indexed bodies + all diagnostic strings.

    Returns ``(indexed_bodies, diagnostic_strings)``. Only files the gate admits
    contribute a body; excluded files contribute a diagnostic. The diagnostic
    strings flatten every operator-visible field (message, code, source_path,
    adapter) so the leak assertion covers the whole diagnostic, not just message.
    """
    indexed_bodies: list[str] = []
    diagnostic_strings: list[str] = []
    for path in files:
        decision: ExclusionDecision = evaluate_file(config, path)
        if decision.include:
            indexed_bodies.append(path.read_text(encoding="utf-8"))
        else:
            assert decision.diagnostic is not None
            diag = decision.diagnostic
            diagnostic_strings.append(
                " ".join([diag.message, diag.code, diag.source_path, diag.adapter])
            )
    return indexed_bodies, diagnostic_strings


def test_secret_text_reaches_neither_index_nor_diagnostic(tmp_path: Path) -> None:
    config = _config(tmp_path)

    clean = _write(
        tmp_path / "docs" / "clean.md",
        "# Clean\n\nOrdinary indexable prose.\n",
    )
    # A content-scan secret in an innocuously named file.
    content_secret = _write(
        tmp_path / "docs" / "leaky.md",
        f"# Leaky\n\n{_AWS_EXAMPLE_KEY}\n{_HIGH_ENTROPY_TOKEN}\n{_BODY_MARKER}\n",
    )
    # A path-glob secret whose bytes also contain a marker.
    path_secret = _write(
        tmp_path / "docs" / "prod.env",
        f"AWS_SECRET={_HIGH_ENTROPY_TOKEN}\n{_BODY_MARKER}\n",
    )

    bodies, diagnostics = _simulate_index(config, [clean, content_secret, path_secret])

    # The clean file is indexed; both secret files are excluded.
    assert len(bodies) == 1
    assert "Ordinary indexable prose" in bodies[0]

    haystack_index = "\n".join(bodies)
    haystack_diag = "\n".join(diagnostics)

    for needle in (_AWS_EXAMPLE_KEY, _HIGH_ENTROPY_TOKEN, _BODY_MARKER):
        # Secret / secret-adjacent text never enters the to-be-indexed content.
        assert needle not in haystack_index
        # ...and never appears in any diagnostic string either.
        assert needle not in haystack_diag

    # Diagnostics still name the paths (paths are allowed; content is not).
    assert "docs/leaky.md" in haystack_diag
    assert "docs/prod.env" in haystack_diag


# --------------------------------------------------------------------------- #
# Finding 1/6 -- every content secret shape returns its id (never the span)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("payload", "expected_id"),
    [
        (_PG_CONN_SECRET, "connection-string-credential"),
        (_STRIPE_KEY, "vendor-api-token"),
        (_GOCSPX_SECRET, "vendor-api-token"),
        (_SENDGRID_KEY, "vendor-api-token"),
        (_GITLAB_PAT, "vendor-api-token"),
        (_OPENAI_KEY, "vendor-api-token"),
        (_NPM_TOKEN, "vendor-api-token"),
        (_PYPI_TOKEN, "vendor-api-token"),
        (_TWILIO_SID, "vendor-api-token"),
        (_GITHUB_TOKEN, "github-token"),
        (_SLACK_TOKEN, "slack-token"),
        (_SLACK_APP_TOKEN, "slack-token"),
        (_GOOGLE_KEY, "google-api-key"),
        (_JWT_TOKEN, "json-web-token"),
        (_ASSIGNED_QUOTED, "assigned-credential"),
        (_ASSIGNED_PADDED_B64, "assigned-credential"),
        (_ASSIGNED_SHORT, "assigned-credential"),
    ],
)
def test_new_content_shapes_return_pattern_id_not_text(payload: str, expected_id: str) -> None:
    body = f"# doc\n\nvalue: {payload}\ntrailing prose\n"
    result = scan_secret_content(body)
    assert result == expected_id
    # The id is a stable label -- the matched span never appears in it.
    assert result not in payload


@pytest.mark.parametrize("benign", [_BENIGN_CONTENT_HASH, _BENIGN_UUID, _BENIGN_PROSE_KEY])
def test_benign_high_entropy_strings_do_not_over_skip(benign: str) -> None:
    # Over-skipping a legit file is silent data loss; these must NOT match.
    body = f"# notes\n\ncontent digest {benign} recorded for reconciliation.\n"
    assert scan_secret_content(body) is None


# --------------------------------------------------------------------------- #
# Finding 1 -- each high-value shape is skipped WHOLE (secret in neither the
# to-be-indexed body nor the diagnostic); path-glob shapes short-circuit read.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("relpath", "payload", "expected_code"),
    [
        ("docs/conn.md", _PG_CONN_SECRET, "secret-content"),
        ("docs/pay.md", _STRIPE_KEY, "secret-content"),
        ("docs/app.md", _ASSIGNED_QUOTED, "secret-content"),
        ("docs/tokens.md", _GITHUB_TOKEN, "secret-content"),
    ],
)
def test_secret_shape_skips_whole_file(
    tmp_path: Path, relpath: str, payload: str, expected_code: str
) -> None:
    config = _config(tmp_path)
    target = _write(
        tmp_path / relpath,
        f"# innocuous title\n\nembedded: {payload}\n{_BODY_MARKER}\n",
    )
    decision = evaluate_file(config, target)
    assert decision.include is False  # whole file dropped, never redact-and-index
    assert decision.diagnostic is not None
    diag = decision.diagnostic
    assert diag.code == expected_code
    flattened = " ".join([diag.message, diag.code, diag.source_path, diag.adapter])
    # Neither the secret payload nor the body marker leaks into the diagnostic.
    assert payload not in flattened
    assert _BODY_MARKER not in flattened
    # The path is still named (path is allowed; content is not).
    assert relpath in flattened


# --------------------------------------------------------------------------- #
# Finding 2 -- deny globs are case-insensitive (uppercase/mixed-case secret
# filenames must not bypass on a case-insensitive filesystem).
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "relpath",
    [
        "docs/PROD.ENV",
        "docs/.ENV",
        "docs/ID_RSA",
        "docs/SECRETS.yaml",
        "docs/Server.PEM",
        "docs/App-Secret.txt",
    ],
)
def test_uppercase_secret_filenames_still_denied(tmp_path: Path, relpath: str) -> None:
    config = _config(tmp_path)
    # Path-glob runs before any read, so the file need not exist.
    target = tmp_path / relpath
    decision = evaluate_file(config, target)
    assert decision.include is False
    assert decision.diagnostic is not None
    assert decision.diagnostic.code == "secret-path-glob"


# --------------------------------------------------------------------------- #
# Finding 4 -- git_ignored resolves unicode-named ignored paths (quotePath=false)
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(shutil.which("git") is None, reason="git not available")
def test_git_ignored_helper_reports_unicode_path(tmp_path: Path) -> None:
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)  # noqa: S603, S607
    _write(tmp_path / ".gitignore", "docs/*.log\n")
    unicode_rel = "docs/café-résumé.log"
    _write(tmp_path / unicode_rel, "noise\n")
    ignored = git_ignored(tmp_path, [unicode_rel])
    # Without `-c core.quotePath=false` git octal-quotes the name and the match is
    # silently lost; with it, the path comes back verbatim.
    assert unicode_rel in ignored


# --------------------------------------------------------------------------- #
# Finding 5 -- fail-closed on non-UTF-8 content and on read errors
# --------------------------------------------------------------------------- #
def test_utf16_secret_is_caught_not_mangled(tmp_path: Path) -> None:
    config = _config(tmp_path)
    target = tmp_path / "docs" / "utf16.md"
    # A UTF-16 (BOM) file whose bytes hold an AWS key. errors="replace" would
    # mangle it and index the file; the BOM-aware decode catches the secret.
    content = ("intro line\n" + _AWS_EXAMPLE_KEY + "\n").encode("utf-16")
    decision = evaluate_file(config, target, content=content)
    assert decision.include is False
    assert decision.diagnostic is not None
    assert decision.diagnostic.code == "secret-content"
    assert _AWS_EXAMPLE_KEY not in decision.diagnostic.message


def test_undecodable_content_fails_closed(tmp_path: Path) -> None:
    config = _config(tmp_path)
    target = tmp_path / "docs" / "blob.dat"
    # NUL-bearing, not valid UTF-16 (odd length) -> cannot be cleanly decoded ->
    # skipped rather than scanned as garbage.
    decision = evaluate_file(config, target, content=b"\x00\x01\x02")
    assert decision.include is False
    assert decision.diagnostic is not None
    assert decision.diagnostic.code == "undecodable-content"


def test_unreadable_file_fails_closed(tmp_path: Path) -> None:
    config = _config(tmp_path)
    # A directory passes the size stat but cannot be read as bytes -> the read
    # branch must fail closed (skip-with-diagnostic), never index.
    target = tmp_path / "docs" / "adir"
    target.mkdir(parents=True)
    decision = evaluate_file(config, target)
    assert decision.include is False
    assert decision.diagnostic is not None
    assert decision.diagnostic.code == "read-error"
    assert decision.diagnostic.severity is Severity.ERROR
