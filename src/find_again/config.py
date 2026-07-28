"""Explicit-root configuration and the two-layer secret-exclusion policy.

This module owns the safety boundary. Every file considered for indexing passes
through :func:`evaluate_file`, which decides *whether the file may be indexed at
all* and, when it may not, returns a :class:`~find_again.models.Diagnostic` that
names only the path and a stable pattern/code identifier — never file content
and never a matched secret span.

Exclusion layers, in the order they are applied (plan.md §6 Explicit roots):

1. Outside configured roots.
2. Path-glob deny patterns  -- built-in secret globs (``**/*.env``,
   ``**/secrets*``, private-key/keystore shapes) plus operator ``exclude`` globs.
   Applied *before content is read*.
3. Git-ignored files.
4. Oversized files (``> max_file_kb``). Checked from stat size before the bytes
   are read.
5. Content-based secret scan -- common token-shape regexes plus a conservative
   high-entropy detector. A hit **skips the entire file** (never
   redact-and-index) and emits a diagnostic naming path + pattern id only.

The built-in secret path-globs and content patterns are a *code invariant*: they
apply even if the operator's ``exclude`` list is empty, so the boundary does not
depend on configuration being correct (dev/.claude/rules/security.md — pair
unsafe configs with startup safety checks; documentation is not a control).
"""

from __future__ import annotations

import functools
import math
import re
import shutil
import subprocess
import tomllib
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from .models import Diagnostic, Severity

__all__ = [
    "CONFIG_FILENAME",
    "DEFAULT_MAX_FILE_KB",
    "SCHEMA_VERSION",
    "SECRET_CONTENT_PATTERNS",
    "SECRET_PATH_GLOBS",
    "Config",
    "ConfigError",
    "ExclusionDecision",
    "RootResolutionError",
    "SchemaVersionError",
    "SecretPattern",
    "evaluate_file",
    "git_ignored",
    "load_config",
    "resolve_root",
    "scan_secret_content",
]

SCHEMA_VERSION = 1
CONFIG_FILENAME = "find-again.toml"
DEFAULT_MAX_FILE_KB = 512


# --------------------------------------------------------------------------- #
# Errors
# --------------------------------------------------------------------------- #
class ConfigError(Exception):
    """Base class for configuration and root-resolution failures."""


class RootResolutionError(ConfigError):
    """The target root could not be resolved (no git repo and no ``--root``)."""


class SchemaVersionError(ConfigError):
    """The config declares a ``schema_version`` newer than this build supports."""


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class Config:
    """Resolved configuration for one target root.

    ``roots`` and ``exclude`` are the operator's explicit include roots and deny
    globs; ``max_file_kb`` is the size ceiling. ``schema_version`` is the value
    read from the file (older values are tolerated).
    """

    root: Path
    roots: tuple[str, ...]
    exclude: tuple[str, ...]
    max_file_kb: int
    schema_version: int


# --------------------------------------------------------------------------- #
# Root resolution (plan.md §6 Configuration discovery)
# --------------------------------------------------------------------------- #
def _find_git_root(start: Path) -> Path | None:
    """Walk up from ``start`` returning the first dir containing ``.git``.

    ``.git`` may be a directory (normal clone) or a file (worktree / submodule);
    ``exists()`` covers both.
    """
    for candidate in (start, *start.parents):
        if (candidate / ".git").exists():
            return candidate
    return None


def resolve_root(cwd: Path, root_override: str | None) -> Path:
    """Resolve the target root: enclosing git root, or ``--root`` override.

    No git repository and no ``--root`` is an explicit error -- never a heuristic
    fallback and never an implicit crawl (plan.md §6).
    """
    if root_override is not None:
        resolved = Path(root_override).expanduser().resolve()
        if not resolved.is_dir():
            raise RootResolutionError(
                f"--root path does not exist or is not a directory: {resolved}"
            )
        return resolved

    git_root = _find_git_root(cwd.resolve())
    if git_root is None:
        raise RootResolutionError(
            "not inside a git repository and no --root provided; "
            "pass --root <path> (no heuristic fallback, no implicit crawl)"
        )
    return git_root


def _as_str_tuple(value: object, field: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ConfigError(f"{field} must be a list of strings")
    return tuple(value)


def load_config(root: Path) -> Config:
    """Load ``find-again.toml`` from ``root`` (plan.md §6 Configuration discovery).

    Tolerates older ``schema_version`` values; refuses newer ones with an
    explicit :class:`SchemaVersionError`. Absence of the file is an explicit
    :class:`ConfigError` (never an implicit default crawl).
    """
    path = root / CONFIG_FILENAME
    if not path.is_file():
        raise ConfigError(
            f"no {CONFIG_FILENAME} found at {root}; "
            "configuration must name searchable roots before indexing"
        )
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise ConfigError(f"could not read {path}: {exc}") from exc

    schema_version = data.get("schema_version", SCHEMA_VERSION)
    if not isinstance(schema_version, int) or isinstance(schema_version, bool):
        raise ConfigError("schema_version must be an integer")
    if schema_version > SCHEMA_VERSION:
        raise SchemaVersionError(
            f"config schema_version {schema_version} is newer than supported "
            f"{SCHEMA_VERSION}; upgrade find-again"
        )

    roots = _as_str_tuple(data.get("roots", []), "roots")
    exclude = _as_str_tuple(data.get("exclude", []), "exclude")

    max_file_kb = data.get("max_file_kb", DEFAULT_MAX_FILE_KB)
    if not isinstance(max_file_kb, int) or isinstance(max_file_kb, bool) or max_file_kb <= 0:
        raise ConfigError("max_file_kb must be a positive integer")

    return Config(
        root=root,
        roots=roots,
        exclude=exclude,
        max_file_kb=max_file_kb,
        schema_version=schema_version,
    )


# --------------------------------------------------------------------------- #
# Glob matching (gitignore-ish ``**`` semantics on POSIX-relative paths)
# --------------------------------------------------------------------------- #
@functools.lru_cache(maxsize=512)
def _compile_glob(pattern: str, ignore_case: bool) -> re.Pattern[str]:
    """Translate a glob to a regex with gitignore-style ``**`` handling.

    ``*`` matches within a path segment, ``?`` one non-separator char, ``**``
    matches across separators, and a leading ``**/`` matches zero or more leading
    directories (so ``**/*.env`` matches both ``a/b.env`` and a top-level
    ``b.env``).

    ``ignore_case`` compiles the pattern case-insensitively. The deny layer sets
    it so that on a case-insensitive filesystem an uppercase/mixed-case secret
    filename (``PROD.ENV``, ``ID_RSA``, ``SECRETS.yaml``) cannot bypass a
    lowercase deny glob.
    """
    out: list[str] = ["(?s:"]
    i = 0
    n = len(pattern)
    while i < n:
        char = pattern[i]
        if char == "*":
            if i + 1 < n and pattern[i + 1] == "*":
                i += 2
                if i < n and pattern[i] == "/":
                    i += 1
                    out.append("(?:.*/)?")  # **/ -> zero or more directories
                else:
                    out.append(".*")  # ** -> anything, including separators
                continue
            out.append("[^/]*")
        elif char == "?":
            out.append("[^/]")
        elif char == "/":
            out.append("/")
        else:
            out.append(re.escape(char))
        i += 1
    out.append(r")\Z")
    flags = re.IGNORECASE if ignore_case else 0
    return re.compile("".join(out), flags)


def _matches_glob(rel_posix: str, pattern: str, *, ignore_case: bool = True) -> bool:
    return _compile_glob(pattern, ignore_case).match(rel_posix) is not None


# --------------------------------------------------------------------------- #
# Layer 1 -- path-glob deny patterns (secret globs are a built-in invariant)
# --------------------------------------------------------------------------- #
SECRET_PATH_GLOBS: tuple[str, ...] = (
    "**/*.env",
    "**/.env",
    "**/.env.*",
    "**/secrets*",
    "**/*secret*",
    "**/*.pem",
    "**/*.key",
    "**/*.pfx",
    "**/*.p12",
    "**/*.keystore",
    "**/id_rsa*",
    "**/id_dsa*",
    "**/id_ecdsa*",
    "**/id_ed25519*",
    "**/credentials*",
    "**/.aws/**",
    "**/.ssh/**",
    "**/.netrc",
    "**/*.pgpass",
)


def _match_deny_glob(rel_posix: str, exclude: Sequence[str]) -> tuple[str, str] | None:
    """Return ``(code, pattern)`` for the first matching deny glob, else ``None``.

    Built-in secret globs are checked first so a secret file gets the
    ``secret-path-glob`` code even when the operator also listed it in
    ``exclude``.
    """
    for pattern in SECRET_PATH_GLOBS:
        if _matches_glob(rel_posix, pattern):
            return ("secret-path-glob", pattern)
    for pattern in exclude:
        if _matches_glob(rel_posix, pattern):
            return ("excluded-glob", pattern)
    return None


# --------------------------------------------------------------------------- #
# Layer 2 -- content-based secret scan
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class SecretPattern:
    """A named content secret pattern. ``id`` is the stable diagnostic label."""

    id: str
    regex: re.Pattern[str]


# Best-effort framing (dev/.claude/rules/security.md): no content scanner catches
# EVERYTHING -- path-globs (Layer 1) and explicit roots (Layer 0) are the primary
# defense; this content scan is defense-in-depth. It therefore aims to catch the
# COMMON high-value credential shapes (vendor-prefixed API tokens, connection-string
# passwords, PEM keys, keyword-assigned secrets, high-entropy blobs) and, when a
# shape is even plausibly present, to FAIL CLOSED (skip the whole file) rather than
# risk indexing a live secret. Over-matching here costs one un-indexed file; a miss
# costs a leaked credential -- so the patterns lean toward skipping.
SECRET_CONTENT_PATTERNS: tuple[SecretPattern, ...] = (
    # AWS access key id (AKIA/ASIA + 16 upper/digits).
    SecretPattern(
        "aws-access-key-id",
        re.compile(r"(?<![A-Z0-9])(?:AKIA|ASIA)[0-9A-Z]{16}(?![0-9A-Z])"),
    ),
    # PEM private-key header (RSA/EC/DSA/OpenSSH/PGP/plain).
    SecretPattern(
        "private-key-header",
        re.compile(r"-----BEGIN (?:[A-Z0-9]+ )*PRIVATE KEY-----"),
    ),
    # Credentials embedded in a URI authority: `scheme://user:secret@host`
    # (postgres://, mysql://, mongodb://, redis://, amqp://, https://, ...). The
    # password between the `:` and the `@` would otherwise be indexed verbatim.
    SecretPattern(
        "connection-string-credential",
        re.compile(r"[a-zA-Z][a-zA-Z0-9+.\-]*://[^\s:/@]+:[^\s/@]+@"),
    ),
    # GitHub personal-access / app tokens (ghp_/gho_/ghu_/ghs_/ghr_).
    SecretPattern(
        "github-token",
        re.compile(r"\bgh[pousr]_[A-Za-z0-9]{36,}\b"),
    ),
    # Slack tokens: bot/user/app-level (xoxb-/xoxp-/xoxa-/xoxr-/xoxs-/xapp-).
    SecretPattern(
        "slack-token",
        re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}\b|\bxapp-[A-Za-z0-9-]{10,}\b"),
    ),
    # Google API key.
    SecretPattern(
        "google-api-key",
        re.compile(r"\bAIza[0-9A-Za-z_\-]{35}\b"),
    ),
    # Short fixed-prefix vendor tokens that evade the 40-char entropy floor:
    # Stripe (sk_live_/sk_test_/rk_live_), Google OAuth (GOCSPX-), SendGrid (SG.),
    # Twilio API-key SID (SK + 32 hex), GitLab PAT (glpat-), OpenAI (sk-), npm
    # (npm_), PyPI (pypi-).
    SecretPattern(
        "vendor-api-token",
        re.compile(
            r"\b(?:"
            r"[sr]k_(?:live|test)_[A-Za-z0-9]{10,}"  # Stripe secret/restricted
            r"|GOCSPX-[A-Za-z0-9_\-]{10,}"  # Google OAuth client secret
            r"|SG\.[A-Za-z0-9_\-]{16,}\.[A-Za-z0-9_\-]{16,}"  # SendGrid
            r"|SK[0-9a-fA-F]{32}"  # Twilio API-key SID
            r"|glpat-[A-Za-z0-9_\-]{16,}"  # GitLab PAT
            r"|sk-[A-Za-z0-9_\-]{20,}"  # OpenAI
            r"|npm_[A-Za-z0-9]{36,}"  # npm
            r"|pypi-[A-Za-z0-9_\-]{16,}"  # PyPI
            r")"
        ),
    ),
    # JSON Web Token (three base64url segments).
    SecretPattern(
        "json-web-token",
        re.compile(r"\beyJ[A-Za-z0-9_\-]{6,}\.[A-Za-z0-9_\-]{6,}\.[A-Za-z0-9_\-]{6,}\b"),
    ),
    # Credential assigned to a secret-shaped key. Catches assigned credentials
    # even when the value is not high-entropy (`password=hunter2`), including
    # base64 padding (`...==`) and short quoted values. A quoted value is a strong
    # signal (accepted from 6 chars); an unquoted value must additionally carry a
    # digit/symbol so a prose word (`access_key: documented`) does not over-skip.
    SecretPattern(
        "assigned-credential",
        re.compile(
            r"(?i)(?:api[_-]?key|apikey|secret|token|password|passwd|pwd"
            r"|access[_-]?key|private[_-]?key|auth[_-]?token|client[_-]?secret)"
            r"\s*[:=]\s*(?:"
            r"['\"][A-Za-z0-9+/=_.\-]{6,}['\"]"  # quoted -> any credential token >=6
            r"|(?=[A-Za-z0-9+/=_.\-]*[0-9+/=.\-])"  # unquoted -> must have digit/symbol
            r"[A-Za-z0-9+/=_.\-]{8,}"
            r")"
        ),
    ),
)

# Contiguous token-shaped run used by the high-entropy fallback.
_TOKEN_RE = re.compile(r"[A-Za-z0-9+/=_\-]{40,}")
_HIGH_ENTROPY_BITS = 4.2


def _shannon_entropy(text: str) -> float:
    if not text:
        return 0.0
    counts = Counter(text)
    length = len(text)
    return -sum((c / length) * math.log2(c / length) for c in counts.values())


def _has_high_entropy_token(text: str) -> bool:
    """True if ``text`` holds a long, mixed-class, high-entropy token.

    Requires length >= 40, Shannon entropy >= 4.2 bits/char, and all three of
    lower/upper/digit present. This catches random API secrets while skipping
    prose (contains spaces, never forms a 40-char run), lowercase hex hashes
    (two char classes), and UUIDs (low entropy).
    """
    for match in _TOKEN_RE.finditer(text):
        token = match.group(0)
        has_lower = any(c.islower() for c in token)
        has_upper = any(c.isupper() for c in token)
        has_digit = any(c.isdigit() for c in token)
        if has_lower and has_upper and has_digit and _shannon_entropy(token) >= _HIGH_ENTROPY_BITS:
            return True
    return False


def scan_secret_content(text: str) -> str | None:
    """Return the id of the first secret pattern matched by ``text``, else ``None``.

    The return value is a stable *pattern id* — never any matched span. Callers
    put this id (not the text) into diagnostics.
    """
    for pattern in SECRET_CONTENT_PATTERNS:
        if pattern.regex.search(text):
            return pattern.id
    if _has_high_entropy_token(text):
        return "high-entropy-token"
    return None


def _decode_for_scan(data: bytes) -> str | None:
    """Decode ``data`` into text the secret scan can search, or ``None`` to fail closed.

    ``bytes.decode(errors="replace")`` is deliberately avoided: it mangles a
    UTF-16 or otherwise non-UTF-8 file into replacement characters, so an embedded
    secret would evade every ASCII-shaped pattern and the file would be indexed.

    Strategy, biased to fail closed:

    * A UTF-16/UTF-8 **BOM** is decoded with that codec; a BOM that then fails to
      decode returns ``None`` (skip the file rather than scan garbage).
    * A no-BOM buffer containing **NUL bytes** is almost certainly UTF-16 or
      binary (whose secrets a single-byte scan would miss between the NULs), so it
      is decoded as UTF-16 or, failing that, returns ``None``.
    * Otherwise the buffer is single-byte text: strict UTF-8, else ``latin-1``
      (which never fails and maps each byte 1:1, preserving ASCII secret shapes in
      cp1252/latin-1 files). ``latin-1`` is only reached for NUL-free buffers, so
      ASCII patterns survive intact.
    """
    if data[:2] in (b"\xff\xfe", b"\xfe\xff"):
        try:
            return data.decode("utf-16")
        except UnicodeDecodeError:
            return None
    if data.startswith(b"\xef\xbb\xbf"):
        try:
            return data.decode("utf-8-sig")
        except UnicodeDecodeError:
            return None
    if b"\x00" in data:
        try:
            return data.decode("utf-16")
        except (UnicodeDecodeError, ValueError):
            return None
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return data.decode("latin-1")


# --------------------------------------------------------------------------- #
# Root-relative helpers
# --------------------------------------------------------------------------- #
def _rel_posix(root: Path, path: Path) -> str:
    """Root-relative POSIX path, or the absolute POSIX path if outside root."""
    resolved = path.resolve()
    try:
        return resolved.relative_to(root.resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()


def _within_roots(rel_posix: str, roots: Sequence[str]) -> bool:
    """True if ``rel_posix`` is inside one of the configured include roots."""
    for raw in roots:
        normalized = raw.replace("\\", "/").strip("/")
        if normalized in ("", "."):
            return True  # "." means the whole tree
        if rel_posix == normalized or rel_posix.startswith(normalized + "/"):
            return True
    return False


# --------------------------------------------------------------------------- #
# Git-ignore probe (used by the indexer in a later step)
# --------------------------------------------------------------------------- #
# Upper bound on the git check-ignore probe. A hung git must not hang the tool;
# on timeout we treat git as unavailable (fail-safe -- the path-glob and content
# layers still run, so this is not fail-open on secrets).
_GIT_CHECK_IGNORE_TIMEOUT_S = 30.0


def git_ignored(root: Path, rel_paths: Sequence[str]) -> set[str]:
    """Return the subset of ``rel_paths`` that git ignores under ``root``.

    Uses ``git check-ignore --stdin``. Returns an empty set when git is
    unavailable, times out, or errors (the caller still applies path-glob and
    content exclusion, so a missing git is fail-safe, not fail-open on secrets).

    ``-c core.quotePath=false`` disables git's default octal-quoting of non-ASCII
    bytes, so a unicode-named ignored path comes back verbatim and matches the
    input path (otherwise git emits ``"docs/caf\\303\\251.log"`` and the match is
    silently lost -- a false-negative on the safety boundary).
    """
    paths = [p for p in rel_paths if p]
    if not paths:
        return set()
    git = shutil.which("git")
    if git is None:
        return set()
    # Bytes I/O, not text mode: on Windows ``text=True`` rewrites ``\n`` -> ``\r\n``
    # on stdin, so git would receive ``docs/run.log\r`` and the trailing CR would
    # break the ignore match (a silent false-negative on the safety boundary).
    try:
        proc = subprocess.run(  # noqa: S603 -- fixed argv, no shell
            [git, "-c", "core.quotePath=false", "-C", str(root), "check-ignore", "--stdin"],
            input="\n".join(paths).encode("utf-8"),
            capture_output=True,
            check=False,
            timeout=_GIT_CHECK_IGNORE_TIMEOUT_S,
        )
    except (subprocess.TimeoutExpired, OSError):
        return set()
    if proc.returncode not in (0, 1):
        return set()
    stdout = proc.stdout.decode("utf-8", errors="replace")
    return {line.strip().replace("\\", "/") for line in stdout.splitlines() if line.strip()}


# --------------------------------------------------------------------------- #
# The exclusion gate
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class ExclusionDecision:
    """Result of :func:`evaluate_file`.

    ``include`` is ``True`` only when every layer passed. When ``include`` is
    ``False`` a ``diagnostic`` is always present, and its ``message`` names the
    path + a code/pattern id only — never file content.
    """

    include: bool
    diagnostic: Diagnostic | None


def _excluded(
    *,
    adapter: str,
    severity: Severity,
    code: str,
    source_path: str,
    message: str,
) -> ExclusionDecision:
    return ExclusionDecision(
        include=False,
        diagnostic=Diagnostic(
            source_path=source_path,
            adapter=adapter,
            severity=severity,
            code=code,
            message=message,
        ),
    )


def evaluate_file(
    config: Config,
    path: Path,
    *,
    content: bytes | None = None,
    is_git_ignored: bool = False,
) -> ExclusionDecision:
    """Decide whether ``path`` may be indexed, applying all exclusion layers.

    ``content`` may be passed to reuse bytes the indexer already read for
    hashing; otherwise the file is read here (and only after the cheap
    path/size layers have passed, so oversized and denied files are never read).
    ``is_git_ignored`` is supplied by the indexer (see :func:`git_ignored`).

    On any exclusion the returned diagnostic names the path and a stable
    code/pattern id only — never file bytes and never a matched secret span.
    """
    rel = _rel_posix(config.root, path)

    # Layer 0: outside configured roots.
    if not _within_roots(rel, config.roots):
        return _excluded(
            adapter="exclusion",
            severity=Severity.WARN,
            code="outside-roots",
            source_path=rel,
            message="path is outside the configured roots",
        )

    # Layer 1: path-glob deny (secret globs first). Applied before content is read.
    glob_hit = _match_deny_glob(rel, config.exclude)
    if glob_hit is not None:
        code, pattern = glob_hit
        return _excluded(
            adapter="exclusion",
            severity=Severity.WARN,
            code=code,
            source_path=rel,
            message=f"path matches deny glob {pattern!r}",
        )

    # Layer 2: git-ignored.
    if is_git_ignored:
        return _excluded(
            adapter="exclusion",
            severity=Severity.WARN,
            code="git-ignored",
            source_path=rel,
            message="path is git-ignored",
        )

    limit_bytes = config.max_file_kb * 1024

    # Layer 3: oversized (checked from stat before reading, when possible).
    if content is None:
        try:
            size = path.stat().st_size
        except OSError as exc:
            return _excluded(
                adapter="exclusion",
                severity=Severity.ERROR,
                code="read-error",
                source_path=rel,
                message=f"could not stat file: {exc.strerror or 'unknown error'}",
            )
        if size > limit_bytes:
            return _excluded(
                adapter="exclusion",
                severity=Severity.WARN,
                code="oversized",
                source_path=rel,
                message=f"file size {size} bytes exceeds max_file_kb={config.max_file_kb}",
            )
        try:
            content = path.read_bytes()
        except OSError as exc:
            return _excluded(
                adapter="exclusion",
                severity=Severity.ERROR,
                code="read-error",
                source_path=rel,
                message=f"could not read file: {exc.strerror or 'unknown error'}",
            )
    elif len(content) > limit_bytes:
        return _excluded(
            adapter="exclusion",
            severity=Severity.WARN,
            code="oversized",
            source_path=rel,
            message=f"file size {len(content)} bytes exceeds max_file_kb={config.max_file_kb}",
        )

    # Layer 4: content-based secret scan. A hit skips the ENTIRE file.
    # First get scannable text; a file we cannot cleanly decode (UTF-16/binary)
    # is skipped rather than scanned as garbage -- fail closed, never index a
    # mangled-but-secret-bearing file.
    scan_text = _decode_for_scan(content)
    if scan_text is None:
        return _excluded(
            adapter="secret-scan",
            severity=Severity.WARN,
            code="undecodable-content",
            source_path=rel,
            message="file could not be decoded for secret scanning; skipped (fail-closed)",
        )
    pattern_id = scan_secret_content(scan_text)
    if pattern_id is not None:
        return _excluded(
            adapter="secret-scan",
            severity=Severity.ERROR,
            code="secret-content",
            source_path=rel,
            # pattern_id is a stable label, NOT the matched text.
            message=f"content matched secret pattern id={pattern_id}",
        )

    return ExclusionDecision(include=True, diagnostic=None)
