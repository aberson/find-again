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
    "evaluate_content",
    "evaluate_file",
    "evaluate_path",
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

# Contiguous run considered by the generic high-entropy fallback. Path and
# identifier separators (``/ \ - _ .``) are INCLUDED in the class so a full path
# or a dotted / kebab / snake identifier is captured as ONE run and then rejected
# by its (low) entropy below -- rather than silently split into <40-char pieces.
# Keeping ``/`` and ``\`` in the class is also security-load-bearing: a
# standard-base64 secret embeds ``/`` (its alphabet is ``A-Za-z0-9+/``), so a run
# that stopped at ``/`` would split the secret into two sub-40-char fragments that
# each evade this floor. As one run, the secret's high entropy is measured intact.
_TOKEN_RE = re.compile(r"[A-Za-z0-9+/=_.\\-]{40,}")

# Word-separators that structure readable identifiers (kebab / snake / dotted). A
# random secret carries few of these; a readable identifier is dominated by them.
_IDENTIFIER_SEPARATORS = ("-", "_", ".")

# Raised entropy floor (was 4.2). Genuinely-random mixed-class tokens sit well
# above 5 bits/char; the long readable path/identifier strings that fill
# dev-memory docs sit around 4.3-4.4, so 4.6 excludes those without losing real
# random secrets.
_HIGH_ENTROPY_BITS = 4.6
# A candidate must carry at least this many NON-separator chars -- enough random
# body to be a plausible secret rather than a hyphenated phrase. A conservative
# floor: with length >= 40 and separators a minority it is already satisfied, but
# it pins the "must have a real random body" invariant explicitly.
_MIN_RANDOM_CHARS = 32
# ...and separators must stay a MINORITY. Above this share the run reads as
# separated words (a structured identifier), not a random token. Measured against
# real dev-memory slugs (14-16% separators) vs. base64url randoms (~3-7%).
_MAX_SEPARATOR_SHARE = 0.10

# Nested-path guard (fa6 follow-up). The entropy floor alone clears SHALLOW readable
# paths (~4.2-4.5 bits/char) but a DEEPLY-nested mixed-case path
# (``.../CurrentVersion/Uninstall/...``, a timestamped ``.judge-motion`` dir) can edge
# to ~4.6-4.8 and would then be mis-flagged. So a run that reads as a nested path --
# several ``/``/``\``-delimited segments that each contain a real lowercase word -- is
# skipped. A standard-base64 secret bearing a stray ``/`` does NOT qualify: its
# slash-delimited chunks alternate case with digits and essentially never hold a
# 4-letter lowercase word in >= 3 separate segments (measured false-negative rate on
# 500k random base64 secrets: ~0.03-0.07%, and 0% for base64url). This keys on
# path *vocabulary*, not a fragile entropy cutoff, so it does not reopen the
# base64-with-slash hole the removed slash-skip left.
_PATH_SEGMENT_RE = re.compile(r"[/\\]")
_READABLE_WORD_RE = re.compile(r"[a-z]{4,}")
_MIN_READABLE_PATH_SEGMENTS = 3


def _reads_as_nested_path(run: str) -> bool:
    """True if ``run`` reads as a nested filesystem path rather than a random token.

    Splits on ``/``/``\\`` and counts segments holding a >= 4-char lowercase word run.
    ``>= _MIN_READABLE_PATH_SEGMENTS`` such segments (e.g. ``docs`` / ``reviews`` /
    ``utility``) => a path; a slash-bearing base64 secret has none across that many
    segments. Deliberately keys on readable path *words*, not entropy: a raw
    entropy cutoff can't tell a deeply-nested mixed-case path from a random blob.
    """
    readable = sum(1 for seg in _PATH_SEGMENT_RE.split(run) if _READABLE_WORD_RE.search(seg))
    return readable >= _MIN_READABLE_PATH_SEGMENTS


def _shannon_entropy(text: str) -> float:
    if not text:
        return 0.0
    counts = Counter(text)
    length = len(text)
    return -sum((c / length) * math.log2(c / length) for c in counts.values())


def _has_high_entropy_token(text: str) -> bool:
    r"""True if ``text`` holds a long, mixed-class, GENUINELY-RANDOM high-entropy run.

    Conservative by design. The specific vendor / connection-string / PEM / AWS /
    JWT / assigned-credential patterns above are the real secret detectors; this
    generic fallback exists only to catch an un-prefixed random blob, and must NOT
    fire on the long path/identifier strings that fill dev-memory docs (e.g. a
    56-char ``example-project/.../reference_note`` path slug or an 85-char
    ``example--nested--slug/.../reference_note`` reference). Over-skipping those silently
    drops the whole file from the index -- a real retrieval gap -- which is why the
    *shape* of a random secret is required here, not merely high entropy.

    Crucially there is NO blanket ``/``/``\`` path-skip: a run containing a slash is
    scored like any other, because a standard-base64 secret (whose alphabet includes
    ``/``) would otherwise slip in un-scanned -- a false-NEGATIVE worse than the
    path-drop such a skip prevents (a secret indexed > a doc dropped). Keeping
    ``/``/``\`` INSIDE :data:`_TOKEN_RE` (below) so the whole path/secret is one run,
    then discriminating structurally, is what closes that hole. Readable paths are
    excluded two ways instead: SHALLOW ones fall under the entropy floor
    (~4.2-4.5 bits/char), and DEEPLY-nested mixed-case ones that edge over it are
    caught by :func:`_reads_as_nested_path` (they read as several ``/``-delimited
    word segments; a random secret does not).

    A run qualifies as a secret only when ALL hold:

    * length >= 40 (via :data:`_TOKEN_RE`);
    * **not separator-dominated** -- ``-`` ``_`` ``.`` are a minority (<= 10%), so a
      kebab / snake / dotted identifier is excluded;
    * carries a substantial random body -- at least ``_MIN_RANDOM_CHARS``
      non-separator chars;
    * **does not read as a nested path** -- fewer than
      ``_MIN_READABLE_PATH_SEGMENTS`` ``/``/``\``-delimited word segments;
    * mixes all three of lower / upper / digit; and
    * Shannon entropy >= ``_HIGH_ENTROPY_BITS`` bits/char.

    This still catches random API secrets (including standard-base64 blobs bearing
    ``/``) while skipping prose (spaces break the run), lowercase hex hashes (one
    case class), UUIDs (too short / low entropy), and -- via the entropy floor plus
    the nested-path guard -- long readable path/identifier strings.
    """
    for match in _TOKEN_RE.finditer(text):
        run = match.group(0)
        separators = sum(run.count(sep) for sep in _IDENTIFIER_SEPARATORS)
        # Separator-dominated -> a readable structured identifier, not a secret.
        if separators / len(run) > _MAX_SEPARATOR_SHARE:
            continue
        # Must have a substantial random (non-separator) body.
        if len(run) - separators < _MIN_RANDOM_CHARS:
            continue
        # Nested path of readable words (deeply-nested paths can edge over the
        # entropy floor) -> not a secret. A slash-bearing base64 blob does not.
        if _reads_as_nested_path(run):
            continue
        has_lower = any(c.islower() for c in run)
        has_upper = any(c.isupper() for c in run)
        has_digit = any(c.isdigit() for c in run)
        if has_lower and has_upper and has_digit and _shannon_entropy(run) >= _HIGH_ENTROPY_BITS:
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


def evaluate_path(config: Config, path: Path, *, is_git_ignored: bool = False) -> ExclusionDecision:
    """Apply the path-only exclusion layers (0-2), touching neither stat nor bytes.

    Layers, in order: outside configured roots, path-glob deny (built-in secret
    globs then operator ``exclude``), and git-ignored. Each needs only the path
    string, so the indexer runs them BEFORE reading a file -- a path-denied secret
    file's bytes are never loaded into memory. ``include=True`` means only that
    these cheap layers passed; the size (stat) ceiling and the content secret scan
    (:func:`evaluate_content`) still apply before a file may be indexed.
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

    return ExclusionDecision(include=True, diagnostic=None)


def evaluate_content(config: Config, path: Path, content: bytes) -> ExclusionDecision:
    """Apply the content-dependent layers (3 size, 4 secret scan) to already-read bytes.

    ``content`` is the file's bytes, already read by the caller after
    :func:`evaluate_path` passed. Applies the size ceiling by length, then the
    secret scan: a file that cannot be cleanly decoded (UTF-16/binary) is skipped
    (fail-closed) rather than scanned as garbage, and any secret-pattern hit skips
    the ENTIRE file (never redact-and-index). ``include=True`` means every layer
    passed and these EXACT bytes may be indexed.

    On exclusion the returned diagnostic names the path + a stable code/pattern id
    only -- never file bytes and never a matched secret span.
    """
    rel = _rel_posix(config.root, path)

    # Layer 3: oversized (by the length of the bytes handed in).
    if len(content) > config.max_file_kb * 1024:
        return _excluded(
            adapter="exclusion",
            severity=Severity.WARN,
            code="oversized",
            source_path=rel,
            message=f"file size {len(content)} bytes exceeds max_file_kb={config.max_file_kb}",
        )

    # Layer 4: content-based secret scan. A hit skips the ENTIRE file. First get
    # scannable text; a file we cannot cleanly decode (UTF-16/binary) is skipped
    # rather than scanned as garbage -- fail closed, never index a mangled-but-
    # secret-bearing file.
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


def evaluate_file(
    config: Config,
    path: Path,
    *,
    content: bytes | None = None,
    is_git_ignored: bool = False,
) -> ExclusionDecision:
    """Decide whether ``path`` may be indexed, applying all exclusion layers.

    Composes :func:`evaluate_path` (cheap path-only layers, no read) with
    :func:`evaluate_content` (size + secret scan). ``content`` may be passed to
    reuse bytes the caller already read; otherwise the file is read here, and only
    after the path layers pass and its stat size is within ``max_file_kb`` -- so a
    path-denied or oversized file is never read. ``is_git_ignored`` is supplied by
    the indexer (see :func:`git_ignored`).

    On any exclusion the returned diagnostic names the path and a stable
    code/pattern id only -- never file bytes and never a matched secret span.
    """
    path_decision = evaluate_path(config, path, is_git_ignored=is_git_ignored)
    if not path_decision.include:
        return path_decision

    if content is None:
        rel = _rel_posix(config.root, path)
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
        if size > config.max_file_kb * 1024:
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

    return evaluate_content(config, path, content)
