"""
Detection of real plaintext secrets inside a Kubernetes manifest. Combines
regex on sensitive key names, Shannon entropy on values, connection-string
and PEM patterns, and a placeholder allowlist.

Critical requirement: Kubernetes environment variables show up as
{name: X, value: Y} (or {name: X, valueFrom: ...}) pairs inside a list. A
naive walker that tests the literal dict key ("name", "value") against the
sensitive-name regex never finds anything, because the semantic key lives in
the *value* of the "name" field, not in the dict key itself. The walker below
recognizes that pattern explicitly and uses X (the value of "name") as the
semantic key when evaluating Y.

Second requirement: a credential can also be smuggled inside a longer
command-line string (a CLI flag, a basic-auth URL, a bearer token) rather
than sitting under its own key. The generic leaf walk deliberately excludes
any value containing a space/slash/colon (to avoid flagging ordinary shell
commands and URLs), which makes it blind to that shape by design. A separate,
narrowly-scoped scanner (`find_cli_embedded_secrets`) looks specifically
inside `command`/`args` lists for those known-bad substrings.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

SENSITIVE_KEY_RE = re.compile(
    r"(password|passwd|pwd|secret|token|api[_-]?key|apikey|access[_-]?key|"
    r"private[_-]?key|client[_-]?secret|auth|credential|conn(ection)?[_-]?str|"
    r"dsn|passphrase)",
    re.IGNORECASE,
)

# Keys whose value is always a reference/structure, never a literal secret --
# must not be treated as a "semantic key" even if they match the regex above.
REFERENCE_KEY_RE = re.compile(
    r"(secretKeyRef|configMapKeyRef|valueFrom|secretName|secretRef)$",
    re.IGNORECASE,
)

# Suffixes indicating the key points to a reference (resource name, file
# path, URL, header name) rather than the literal secret value -- e.g.
# TLS_SECRET_NAME, PASSWORD_FILE, GOOGLE_APPLICATION_CREDENTIALS (a path).
# Takes precedence over SENSITIVE_KEY_RE even when the key also matches it.
NON_SECRET_KEY_SUFFIX_RE = re.compile(
    r"(_|-)?(name|names|path|file|dir|directory|url|uri|namespace|type|realm|"
    r"mode|enabled|disabled|ref|reference|header|headers)$",
    re.IGNORECASE,
)

# Structural Kubernetes fields whose value is never a secret, even when the
# text has high entropy (resource names, images, apiVersion, public domain
# hosts often have entropy close to that of a real secret).
STRUCTURAL_KEYS = frozenset(
    {"name", "image", "apiVersion", "kind", "namespace", "uid", "resourceVersion", "generation", "selfLink"}
)

PEM_RE = re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----")

CONN_STRING_RE = re.compile(
    r"^[a-zA-Z][a-zA-Z0-9+.-]*://[^:@/\s]+:[^:@/\s]+@[^/\s]+",
)

PLACEHOLDER_RE = re.compile(
    r"^(changeme|change[_-]?me|xxx+|example|sample|placeholder|"
    r"your[_-]?\w+[_-]?here|todo|fixme|dummy|fake|test|n/?a|none|null|"
    r"redacted|masked|true|false|yes|no|base64[ _-]?encoded[ _-]?\w*|\*+)$",
    re.IGNORECASE,
)

VAR_REF_RE = re.compile(r"^\$\{.*\}$|^\$\(.*\)$|^\$[a-zA-Z0-9_]*\(.*\)$")
ANGLE_PLACEHOLDER_RE = re.compile(r"^<.*>$")
TEMPLATE_RE = re.compile(r"\{\{.*\}\}")
URL_RE = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", re.IGNORECASE)
DOMAIN_LIKE_RE = re.compile(r"^[a-zA-Z0-9]([a-zA-Z0-9-]*[a-zA-Z0-9])?(\.[a-zA-Z0-9]([a-zA-Z0-9-]*[a-zA-Z0-9])?){1,}$")
HEX_RE = re.compile(r"^[0-9a-fA-F]+$")
_HASH_LENGTHS = frozenset({32, 40, 64})
# Kubernetes-style identifier (resource name, label, cloud region): lowercase
# letters, digits and hyphens only. Real secrets (base64, hex, API keys)
# almost always have an uppercase letter, "_", "+" or "/", or are pure hex
# (already handled by HEX_RE) -- an all-lowercase hyphenated string is an
# identifier, not a random secret, even with relatively high entropy.
KEBAB_IDENTIFIER_RE = re.compile(r"^[a-z0-9]+(-[a-z0-9]+)*$")
# Command-line flags ("--timeout=60s") and regex/JSON-embedded values
# (brackets, braces, backslash) are never secrets.
CODE_LIKE_RE = re.compile(r"[\\^$\[\]{}|<>]")

# Key->field pairs that carry the "semantic name" of a sibling value. Used to
# recognize {name: X, value: Y} and similar variants.
_NAME_FIELD_CANDIDATES = ("name", "key")
_VALUE_FIELD_CANDIDATES = ("value",)

MIN_ENTROPY_BITS = 3.2
MIN_LEN_FOR_ENTROPY = 12
MAX_LEN_FOR_ENTROPY = 128

# Credential embedded as a CLI flag, e.g. "--password=hunter2" or "--token X".
CLI_FLAG_CRED_RE = re.compile(
    r"--?(password|passwd|pwd|token|api[-_]?key|apikey|secret|access[-_]?key|"
    r"auth[-_]?token|client[-_]?secret)[= ]+(?P<val>[^\s'\"]+)",
    re.IGNORECASE,
)
# Basic-auth credential embedded in a URL, e.g. "curl https://admin:hunter2@host/...".
BASIC_AUTH_URL_RE = re.compile(r"(?P<val>[a-zA-Z0-9._%+-]+:[^\s@/]+)@[a-zA-Z0-9.-]+")
# Bearer token embedded in a command, e.g. "-H 'Authorization: Bearer eyJhbGc...'".
BEARER_TOKEN_RE = re.compile(r"Bearer\s+(?P<val>[A-Za-z0-9._~+/=-]{8,})", re.IGNORECASE)

_CLI_EMBEDDED_PATTERNS = (
    (CLI_FLAG_CRED_RE, "credential passed as a CLI flag"),
    (BASIC_AUTH_URL_RE, "basic-auth credential embedded in a URL"),
    (BEARER_TOKEN_RE, "bearer token embedded in a command"),
)


def shannon_entropy(s: str) -> float:
    """Shannon entropy in bits per character."""
    if not s:
        return 0.0
    freq: dict[str, int] = {}
    for ch in s:
        freq[ch] = freq.get(ch, 0) + 1
    length = len(s)
    entropy = 0.0
    for count in freq.values():
        p = count / length
        entropy -= p * math.log2(p)
    return entropy


def is_placeholder(value: str) -> bool:
    v = value.strip()
    if not v:
        return True
    if PLACEHOLDER_RE.match(v):
        return True
    if VAR_REF_RE.match(v):
        return True
    if ANGLE_PLACEHOLDER_RE.match(v):
        return True
    if TEMPLATE_RE.search(v):
        return True
    return False


@dataclass
class SecretHit:
    path: str
    key: str
    value: str
    reason: str


def _looks_like_secret_value(key: str, value: str) -> str | None:
    """Returns the detection reason, or None if the value looks safe."""
    if not isinstance(value, str) or not value:
        return None
    if is_placeholder(value):
        return None
    if PEM_RE.search(value):
        return "PEM private key"
    if CONN_STRING_RE.match(value):
        return "connection string with embedded credentials"

    key_str = str(key)
    if key_str in STRUCTURAL_KEYS:
        return None

    if SENSITIVE_KEY_RE.search(key_str) and not NON_SECRET_KEY_SUFFIX_RE.search(key_str):
        if len(value) >= 4 and not URL_RE.match(value) and not value.startswith("/"):
            return f"value under sensitive key '{key_str}'"
        return None

    if MIN_LEN_FOR_ENTROPY <= len(value) <= MAX_LEN_FOR_ENTROPY and not any(
        (
            " " in value,
            "\t" in value,
            "\n" in value,
            "/" in value,
            ":" in value,
            value.startswith("-"),
            URL_RE.match(value),
            DOMAIN_LIKE_RE.match(value),
            KEBAB_IDENTIFIER_RE.match(value),
            CODE_LIKE_RE.search(value),
            HEX_RE.match(value) and len(value) in _HASH_LENGTHS,
        )
    ):
        if shannon_entropy(value) >= MIN_ENTROPY_BITS and re.search(r"[A-Za-z]", value) and re.search(
            r"[0-9]", value
        ):
            return "high entropy (likely random secret)"
    return None


def _dict_env_semantic_key(d: dict) -> str | None:
    """If `d` is a {name: X, value: Y} pair, returns X (the semantic key).
    Returns None if `d` doesn't follow that shape."""
    name_field = None
    for cand in _NAME_FIELD_CANDIDATES:
        if cand in d and isinstance(d[cand], str):
            name_field = d[cand]
            break
    if name_field is None:
        return None
    has_value = any(cand in d for cand in _VALUE_FIELD_CANDIDATES)
    if not has_value:
        return None
    return name_field


def walk_leaves(doc, path: str = ""):
    """Walks a document (dict/list/scalar) and yields (path, key, value) for
    every string leaf, resolving the correct semantic key even inside the
    {name, value} pairs typical of Kubernetes env vars."""
    if isinstance(doc, dict):
        semantic_key = _dict_env_semantic_key(doc)
        if semantic_key is not None and isinstance(doc.get("value"), str):
            yield (f"{path}/value", semantic_key, doc["value"])
            for k, v in doc.items():
                if k == "value":
                    continue
                yield from walk_leaves(v, f"{path}/{k}")
            return
        for k, v in doc.items():
            if REFERENCE_KEY_RE.search(str(k)):
                continue
            yield from walk_leaves(v, f"{path}/{k}")
    elif isinstance(doc, list):
        for i, item in enumerate(doc):
            yield from walk_leaves(item, f"{path}/{i}")
    elif isinstance(doc, str):
        key = path.rsplit("/", 1)[-1] if path else ""
        yield (path, key, doc)


def _find_cli_embedded_secret(value: str) -> tuple[str, str] | None:
    """Searches for a credential-shaped substring embedded in a longer
    command-like string (CLI flag, basic-auth URL, bearer token). Returns
    (reason, matched_credential) or None."""
    for pattern, reason in _CLI_EMBEDDED_PATTERNS:
        m = pattern.search(value)
        if m:
            candidate = m.group("val")
            if is_placeholder(candidate):
                continue
            return reason, candidate
    return None


def find_cli_embedded_secrets(doc) -> list[SecretHit]:
    """Dedicated scan of `command` and `args` list values (anywhere in the
    document) for credentials embedded in longer command strings. Complements
    find_secrets, which is blind to this shape by design: any value
    containing a space/slash/colon is excluded from the generic leaf
    heuristic precisely to avoid flagging ordinary CLI flags and URLs."""
    hits: list[SecretHit] = []

    def _walk(node, path: str) -> None:
        if isinstance(node, dict):
            for k, v in node.items():
                if k in ("command", "args") and isinstance(v, list):
                    for i, item in enumerate(v):
                        if not isinstance(item, str):
                            continue
                        found = _find_cli_embedded_secret(item)
                        if found:
                            reason, candidate = found
                            hits.append(SecretHit(path=f"{path}/{k}/{i}", key=k, value=candidate, reason=reason))
                else:
                    _walk(v, f"{path}/{k}")
        elif isinstance(node, list):
            for i, item in enumerate(node):
                _walk(item, f"{path}/{i}")

    _walk(doc, "")
    return hits


def find_secrets(doc) -> list[SecretHit]:
    """Scans a Kubernetes document and returns the plaintext secrets found.
    `doc` is the already-loaded dict (yaml.safe_load of a single document)."""
    hits: list[SecretHit] = []
    for path, key, value in walk_leaves(doc):
        reason = _looks_like_secret_value(key, value)
        if reason:
            hits.append(SecretHit(path=path, key=key, value=value, reason=reason))
    hits.extend(find_cli_embedded_secrets(doc))
    return hits
