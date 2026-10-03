"""
One mutator per KSEC rule. Each mutator receives a document already in
canonical form (clean, with no finding for that rule -- checked via a
precondition) and returns the document with the defect injected, along with
findings/patch/new_resources DERIVED from the mutation itself. The patch is
always built to be the exact inverse of what was injected: applying `patch`
to `mutated_doc` must reproduce `canonical` byte for byte (verified in
build.py for 100% of the dataset).

Findings are never hand-written: after mutating, we call the matching
detector from detect.py on the mutated document. That guarantees the finding
and the mutation never diverge -- the same logic that validates the dataset
is the one that generates the label.
"""

from __future__ import annotations

import collections
import copy
import functools
import random
import re
import string
from dataclasses import dataclass, field

import jsonpatch

from dataset.detect import (
    detect_all,
    detect_ksec001,
    detect_ksec002,
    detect_ksec003,
    detect_ksec004,
    detect_ksec005,
    detect_ksec006,
    detect_ksec007,
    detect_ksec008,
    detect_ksec009,
    detect_ksec010,
    detect_ksec011,
    detect_ksec006_services,
    detect_ksec012,
)
from dataset.k8s import (
    NON_HTTP_IMAGE_PORTS,
    PROBE_FIELDS,
    RBAC_BINDING_KINDS,
    RBAC_ROLE_KINDS,
    SENSITIVE_HOST_PATHS,
    declared_volume_names,
    get_container_ports,
    get_pod_spec,
    get_selector_match_labels,
    get_service_selector,
    get_template_labels,
    image_basename,
    is_init_container_path,
    is_unpinned_image,
    iter_containers,
    label_selector_matches,
    parse_quantity,
    pod_label_sets,
    required_env_for_container,
    split_image,
)
from dataset import names
from dataset.scanning import CONN_STRING_RE, NON_SECRET_KEY_SUFFIX_RE, SENSITIVE_KEY_RE
from dataset.schema import RULE_IDS, Finding, PatchOp, escape_json_pointer_token


@dataclass
class MutationResult:
    mutated_doc: dict
    canonical: dict
    findings: list[Finding]
    patch: list[PatchOp]
    new_resources: list[str] = field(default_factory=list)


@dataclass
class FileMutationResult:
    """MutationResult for a mutation that spans a whole multi-document file:
    one list entry per document, patch ops carrying their own doc index."""

    mutated_docs: list[dict]
    canonical_docs: list[dict]
    findings: list[Finding]
    patch: list[PatchOp]
    new_resources: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Generic inverse-patch helpers
# ---------------------------------------------------------------------------


def _set_field_with_patch(
    mutated_owner: dict,
    canonical_owner: dict,
    path_prefix: str,
    field_parts: list[str],
    new_value,
    doc_index: int,
) -> PatchOp:
    """Sets `new_value` on mutated_owner at the position given by
    field_parts (creating intermediate dicts as needed) and returns the
    minimal PatchOp that undoes exactly that change: if the whole path
    already existed in canonical_owner, a "replace" with the old value;
    otherwise a "remove" at the first ancestor that had to be created (avoids
    leaving behind an orphan empty dict that didn't exist in the canonical
    form)."""
    cur_pre = canonical_owner
    existing_depth = 0
    for part in field_parts:
        if isinstance(cur_pre, dict) and part in cur_pre:
            cur_pre = cur_pre[part]
            existing_depth += 1
        else:
            break

    cur = mutated_owner
    for part in field_parts[:-1]:
        cur = cur.setdefault(part, {})
    cur[field_parts[-1]] = new_value

    if existing_depth == len(field_parts):
        pointer = path_prefix + "/" + "/".join(field_parts)
        return PatchOp(doc_index, "replace", pointer, copy.deepcopy(cur_pre))

    pointer = path_prefix + "/" + "/".join(field_parts[: existing_depth + 1])
    return PatchOp(doc_index, "remove", pointer)


def _append_list_item_with_patch(
    mutated_owner: dict,
    canonical_owner: dict,
    list_key: str,
    item,
    path_prefix: str,
    doc_index: int,
) -> PatchOp:
    """Appends `item` to mutated_owner's `list_key` list and returns the
    PatchOp that undoes the append: remove by index if the list already
    existed in the canonical form, or remove the whole key if it didn't."""
    existed = isinstance(canonical_owner.get(list_key), list)
    lst = mutated_owner.setdefault(list_key, [])
    new_index = len(lst)
    lst.append(item)
    if existed:
        return PatchOp(doc_index, "remove", f"{path_prefix}/{list_key}/{new_index}")
    return PatchOp(doc_index, "remove", f"{path_prefix}/{list_key}")


# ---------------------------------------------------------------------------
# KSEC-001 -- plaintext credential
# ---------------------------------------------------------------------------

# Base pool of injectable variable names for the env-var variant. Deliberately
# mixes naming conventions (SCREAMING_SNAKE_CASE, kebab-case, camelCase) since
# real corpus data uses all three -- a model trained only on one convention
# risks learning "flag these literal strings" instead of the intended
# "flag identifiers containing a credential-shaped word". At build time,
# dataset/build.py extends this with names actually harvested (never values)
# from real corpus documents dropped for containing a genuine secret -- see
# dataset/build.py's harvest_credential_key_names.
FAKE_SECRET_VAR_NAMES = (
    "DB_PASSWORD",
    "API_TOKEN",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_ACCESS_KEY_ID",
    "REDIS_PASSWORD",
    "JWT_SECRET",
    "SMTP_PASSWORD",
    "STRIPE_API_KEY",
    "ADMIN_PASSWORD",
    "MYSQL_ROOT_PASSWORD",
    "MYSQL_PASSWORD",
    "POSTGRES_PASSWORD",
    "MONGO_PASSWORD",
    "MONGO_ROOT_PASSWORD",
    "RABBITMQ_PASSWORD",
    "MQTT_PASSWORD",
    "GITHUB_TOKEN",
    "GITLAB_TOKEN",
    "NPM_TOKEN",
    "DOCKER_PASSWORD",
    "SLACK_WEBHOOK_TOKEN",
    "ENCRYPTION_SECRET_KEY",
    "PRIVATE_KEY",
    "CLIENT_SECRET",
    "OAUTH_TOKEN",
    "OAUTH_CLIENT_SECRET",
    "WEBHOOK_SECRET",
    "ENCRYPTION_PASSPHRASE",
    "GRAFANA_ADMIN_PASSWORD",
    "MINIO_ACCESS_KEY",
    "MINIO_SECRET_KEY",
    "ELASTIC_PASSWORD",
    "AZURE_CLIENT_SECRET",
    "GCP_SERVICE_ACCOUNT_CREDENTIALS",
    "DATABASE_PASSWORD",
    "SESSION_SECRET",
    "COOKIE_SECRET",
    "SIGNING_SECRET",
    "VAULT_TOKEN",
    "mysql-root-password",
    "admin-password",
    "db-password",
    "redis-password",
    "wordpress-db-password",
    "grafana-admin-password",
    "minio-secret-key",
    "tls-private-key",
    "dbPassword",
    "apiToken",
    "adminPassword",
    "clientSecret",
    "encryptionSecretKey",
)

_SECRET_CHARSET = string.ascii_letters + string.digits

# Templates for injecting a credential into a container's command/args,
# rather than as a dedicated env var. Covers the pattern where a secret is
# smuggled inside a longer shell invocation instead of its own key -- a shape
# the env-var variant below never produces and scanning.py's generic leaf
# walk can't see (see find_cli_embedded_secrets for why).
_CLI_INJECTION_TEMPLATES = (
    lambda fake: f"--password={fake}",
    lambda fake: f"--api-key={fake}",
    lambda fake: f"--token={fake}",
    lambda fake: f"curl https://admin:{fake}@internal.example.com/report",
    # Split form: the flag and its value as two separate list elements
    # (e.g. orion's `-dbpwd 123456789`). Password/API-key flags only --
    # scanning.py doesn't flag hyphenated names after --secret/--token.
    lambda fake: ["--password", fake],
    lambda fake: ["--db-password", fake],
    lambda fake: ["-dbpwd", fake],
    lambda fake: ["--api-key", fake],
)

# Probability of attempting the env-var injection first (vs. command/args).
_ENV_VARIANT_PROBABILITY = 0.7

# --- Realistic fake-secret-value shapes ---------------------------------
#
# A model trained only on random_alnum's fixed-shape output (letters+digits,
# always 20 chars) can learn "flag this exact literal shape" instead of the
# intended "flag anything assigned to a credential-shaped key" -- confirmed
# in practice: a real manifest's password ("mypassowrd 123" -- unquoted,
# space, misspelled word + digits) was missed by a model trained this way,
# while a differently-shaped fake password during manual testing was caught
# fine. These pools add format diversity (word-based, keyboard walks,
# well-known-weak shapes, messy/typo'd, connection-string-embedded) without
# using any real leaked-credential data -- see README.md's design decisions
# for why a real breach wordlist (e.g. rockyou.txt) was deliberately not
# used: real people's actual leaked passwords aren't necessary here, only
# format diversity is, which is a much smaller and uncomplicated thing to
# synthesize from scratch. None of these words/phrases are drawn from any
# breach corpus.
_COMMON_WEAK_WORDS = (
    "summer", "winter", "spring", "autumn", "dragon", "tiger", "shadow",
    "phoenix", "falcon", "admin", "welcome", "sunshine", "football",
    "baseball", "monkey", "cookie", "hunter", "master", "ninja", "wizard",
    "eagle", "thunder", "matrix", "rocket", "silver", "golden",
)

# Well-known keyboard-walk strings -- a distinct, generic weak-password
# shape (adjacent-key sequences), not tied to any breach corpus.
_KEYBOARD_WALKS = ("qwerty", "asdfgh", "zxcvbn", "1qaz2wsx", "qazwsx", "1q2w3e4r")

# Generic weak-password *shapes*, illustrative only -- verified below (see
# test_mutate.py) to never exactly equal (case-insensitive, whole-string) any
# scanning.PLACEHOLDER_RE alternative, since that would make find_secrets
# skip it as a placeholder instead of flagging it.
_KNOWN_WEAK_PASSWORDS = (
    "Passw0rd!", "Welcome1", "Admin123", "Qwerty123", "Sunshine1",
    "Football1", "Monkey123", "Dragon99",
)

# Symbols safe to append even inside a CLI flag/basic-auth URL value -- "@"
# is deliberately excluded there (it would terminate BASIC_AUTH_URL_RE's
# capture early in scanning.py). The env-variant path has no such
# constraint (detection there doesn't regex-parse the value at all).
_SYMBOLS_CLI_SAFE = "!#$%^&*"
_SYMBOLS = _SYMBOLS_CLI_SAFE + "@"

_CONN_SCHEMES = ("postgres", "mysql", "mongodb", "redis", "amqp")
_CONN_USERS = ("admin", "root", "app", "user")
_CONN_HOSTS = ("db", "database", "localhost", "db-service", "mongodb-service")
_CONN_PORTS = (5432, 3306, 27017, 6379, 5672)
_CONN_DBS = ("app", "mydb", "prod", "data")


def _random_alnum_value(rng: random.Random, length: int | None = None) -> str:
    length = length if length is not None else rng.randint(12, 32)
    return "".join(rng.choices(_SECRET_CHARSET, k=length))


def _word_based_value(rng: random.Random, cli_safe: bool) -> str:
    word = rng.choice(_COMMON_WEAK_WORDS)
    word = word.capitalize() if rng.random() < 0.5 else word
    digits = "".join(rng.choices(string.digits, k=rng.randint(1, 4)))
    symbols = _SYMBOLS_CLI_SAFE if cli_safe else _SYMBOLS
    suffix = rng.choice(symbols) if rng.random() < 0.4 else ""
    return f"{word}{digits}{suffix}"


def _keyboard_walk_value(rng: random.Random) -> str:
    base = rng.choice(_KEYBOARD_WALKS)
    return base if rng.random() < 0.5 else base + str(rng.randint(0, 99))


def _known_weak_value(rng: random.Random) -> str:
    return rng.choice(_KNOWN_WEAK_PASSWORDS)


def _messy_value(rng: random.Random) -> str:
    """Mimics a real-world messy credential value: a (possibly typo'd) word
    plus digits, separated by a space -- e.g. the exact shape of a real
    manifest's "mypassowrd 123". Only used where the caller doesn't need the
    value to survive being embedded inside a regex-parsed CLI flag/URL."""
    word = rng.choice(_COMMON_WEAK_WORDS)
    if rng.random() < 0.5:
        word = _typo(word, rng)
    digits = "".join(rng.choices(string.digits, k=rng.randint(1, 3)))
    return f"{word} {digits}"


def _connection_string_value(rng: random.Random) -> str:
    """A full connection-string value (scheme://user:pass@host:port/db),
    matching scanning.CONN_STRING_RE independent of the env var's key name --
    real manifests routinely put this shape under keys like DATABASE_URL or
    DB_CONNECTION, which mutate.py never generated as a plain env value
    before (the only near-equivalent was one of four CLI-injection
    templates, a different and much rarer code path). The password
    component reuses the cli_safe pool so it can never contain ':'/'@'/'/'
    itself, keeping the whole value matching CONN_STRING_RE from position 0."""
    scheme = rng.choice(_CONN_SCHEMES)
    user = rng.choice(_CONN_USERS)
    host = rng.choice(_CONN_HOSTS)
    port = rng.choice(_CONN_PORTS)
    db = rng.choice(_CONN_DBS)
    password = _fake_secret_value(rng, cli_safe=True)
    return f"{scheme}://{user}:{password}@{host}:{port}/{db}"


def _fake_secret_value(rng: random.Random, length: int | None = None, cli_safe: bool = False) -> str:
    """Generates a synthetic value shaped like a secret. Never a real
    credential -- just a plausible literal for training pattern detection.

    cli_safe=True restricts to shapes with no whitespace/'@'//'/quotes, for
    values that get embedded inside a longer CLI flag or basic-auth URL
    string and matched by a regex there (see scanning.py's CLI_FLAG_CRED_RE/
    BASIC_AUTH_URL_RE) -- a space or stray '@' would truncate that capture.
    cli_safe=False (the env-var variant's case) has no such constraint, so it
    additionally allows the space-containing "messy" and connection-string
    styles."""
    styles = [
        lambda: _random_alnum_value(rng, length),
        lambda: _word_based_value(rng, cli_safe),
        lambda: _keyboard_walk_value(rng),
        lambda: _known_weak_value(rng),
    ]
    if not cli_safe:
        styles.append(lambda: _messy_value(rng))
        styles.append(lambda: _connection_string_value(rng))
    return rng.choice(styles)()


def _mutate_ksec001_env(
    canonical_doc: dict, rng: random.Random, doc_index: int, candidate_names=None
) -> MutationResult | None:
    pod_spec, prefix = get_pod_spec(canonical_doc)
    if pod_spec is None:
        return None
    containers = list(iter_containers(pod_spec, prefix))
    if not containers:
        return None

    container_idx = rng.randrange(len(containers))
    cpath, orig_container = containers[container_idx]

    name_pool = candidate_names if candidate_names else FAKE_SECRET_VAR_NAMES
    existing_names = {e.get("name") for e in orig_container.get("env", []) if isinstance(e, dict)}
    # Composition guards against KSEC-011 (see multi_mutate.py), which removes
    # a required env entry and restores it by index:
    #  - never inject a var that satisfies the requirement (a plaintext
    #    POSTGRES_PASSWORD, or a second accepted var KSEC-011 would collapse
    #    away) -- either would erase a finding;
    #  - never append to an env list with a pending KSEC-011 removal -- the
    #    restoring insertion would shift this entry's index out from under
    #    this mutator's own patch.
    applicable = required_env_for_container(cpath, orig_container, pod_spec)
    requirements = applicable[1] if applicable else ()
    blocked = frozenset().union(*(r.accepted for r in requirements))
    if any(not (existing_names & r.accepted) for r in requirements):
        return None
    candidates = [n for n in name_pool if n not in existing_names and n not in blocked]
    if not candidates:
        return None
    var_name = rng.choice(candidates)
    secret_key = var_name.lower().replace("_", "-")
    secret_resource_name = f"{orig_container.get('name', 'app')}-secrets"

    # The "real" canonical form for this pair already references an external
    # Secret (which doesn't exist yet) via secretKeyRef -- that's what
    # detect_ksec001 considers clean. The input canonical rarely already has
    # this pair, so we build that form here: it's the round-trip target.
    hardened_entry = {
        "name": var_name,
        "valueFrom": {"secretKeyRef": {"name": secret_resource_name, "key": secret_key}},
    }
    # Sometimes preceded by its plaintext username (DB_USER: admin before
    # DB_PASSWORD) -- not a finding, present in the target too. Without
    # this, a model trained on v5 skipped every credential that came after a
    # username (scenarios/10-mongodb.yaml, 3-mysql.yaml), a pairing real
    # manifests almost always have but the training data never showed.
    companion = _companion_username(var_name, existing_names | blocked, rng)
    new_entries = [companion, hardened_entry] if companion else [hardened_entry]

    # Inserted at a random position rather than always appended, but never
    # before an env index an existing finding points at (an earlier step of
    # a multi-defect composition), which would shift it out from under that
    # step's own patch.
    orig_env = orig_container.get("env") if isinstance(orig_container.get("env"), list) else []
    env_prefix = f"{cpath}/env/"
    taken = [
        int(f.path[len(env_prefix):].split("/")[0])
        for f in detect_all(canonical_doc, doc_index)
        if f.path.startswith(env_prefix) and f.path[len(env_prefix):].split("/")[0].isdigit()
    ]
    lowest = min(max(taken) + 1 if taken else 0, len(orig_env))
    insert_at = rng.randint(lowest, len(orig_env))
    env_index = insert_at + len(new_entries) - 1

    canonical_with_ref = copy.deepcopy(canonical_doc)
    c_pod_spec, _ = get_pod_spec(canonical_with_ref)
    c_containers = list(iter_containers(c_pod_spec, prefix))
    _, c_container = c_containers[container_idx]
    c_container.setdefault("env", [])[insert_at:insert_at] = copy.deepcopy(new_entries)

    mutated_doc = copy.deepcopy(canonical_doc)
    m_pod_spec, _ = get_pod_spec(mutated_doc)
    m_containers = list(iter_containers(m_pod_spec, prefix))
    _, m_container = m_containers[container_idx]
    plaintext = {"name": var_name, "value": _fake_secret_value(rng)}
    m_container.setdefault("env", [])[insert_at:insert_at] = copy.deepcopy(new_entries[:-1]) + [plaintext]

    env_path = f"{cpath}/env/{env_index}"
    patch = [PatchOp(doc_index, "replace", env_path, hardened_entry)]

    existing = detect_ksec001(canonical_doc, doc_index)
    findings = [f for f in detect_ksec001(mutated_doc, doc_index) if f not in existing]
    if len(findings) != 1:
        return None

    new_resources = [_placeholder_secret_yaml(canonical_doc, secret_resource_name, secret_key)]

    return MutationResult(mutated_doc, canonical_with_ref, findings, patch, new_resources)


_PASSWORD_SUFFIX_RE = re.compile(r"^(?P<stem>.*?)(?P<sep>[_-]?)(?P<word>password|passwd|pwd|pass)$", re.IGNORECASE)
_USERNAME_VALUES = ("admin", "root", "app", "user", "service", "postgres", "mongo")
_COMPANION_USERNAME_PROBABILITY = 0.5


def _companion_username(var_name: str, taken: set, rng: random.Random) -> dict | None:
    """The username variable that usually accompanies a password variable
    (DB_PASSWORD -> DB_USER / DB_USERNAME, mysqlPass -> mysqlUser), or None
    (half the time, for names without a password-like suffix, or if that
    name is already in use)."""
    m = _PASSWORD_SUFFIX_RE.match(var_name)
    if not m or not m["stem"] or rng.random() >= _COMPANION_USERNAME_PROBABILITY:
        return None
    word = rng.choice(("user", "username"))
    if m["word"].isupper():
        word = word.upper()
    elif m["word"][:1].isupper():
        word = word.capitalize()
    name = f"{m['stem']}{m['sep']}{word}"
    if name in taken or _is_credential_name(name):
        return None
    return {"name": name, "value": rng.choice(_USERNAME_VALUES)}


def _placeholder_secret_yaml(doc: dict, secret_name: str, key: str) -> str:
    """The companion Secret a secretKeyRef-based fix references. The value is
    always a placeholder: the real credential is never known here."""
    namespace = (doc.get("metadata") or {}).get("namespace")
    lines = ["apiVersion: v1", "kind: Secret", "metadata:", f"  name: {secret_name}"]
    if namespace:
        lines.append(f"  namespace: {namespace}")
    lines += ["type: Opaque", "stringData:", f'  {key}: "<REPLACE_WITH_SECRET_VALUE>"']
    return "\n".join(lines)


def _mutate_ksec001_command(canonical_doc: dict, rng: random.Random, doc_index: int) -> MutationResult | None:
    """Injects a credential embedded in a container's command/args, e.g. a
    `--password=...` flag or a basic-auth URL passed to curl. The fix here is
    simply removing the offending arg (unlike the env variant, this doesn't
    attempt to rewire the command to read from a Secret-backed env var --
    that would require rewriting the invocation itself, out of scope for a
    minimal patch)."""
    pod_spec, prefix = get_pod_spec(canonical_doc)
    if pod_spec is None:
        return None
    containers = list(iter_containers(pod_spec, prefix))
    if not containers:
        return None

    container_idx = rng.randrange(len(containers))
    cpath, orig_container = containers[container_idx]

    if isinstance(orig_container.get("args"), list):
        list_key = "args"
    elif isinstance(orig_container.get("command"), list):
        list_key = "command"
    else:
        list_key = "args"

    mutated_doc = copy.deepcopy(canonical_doc)
    m_pod_spec, _ = get_pod_spec(mutated_doc)
    _, m_container = list(iter_containers(m_pod_spec, prefix))[container_idx]

    fake = _fake_secret_value(rng, cli_safe=True)
    template = rng.choice(_CLI_INJECTION_TEMPLATES)
    injected = template(fake)
    items = injected if isinstance(injected, list) else [injected]

    # Inserted at a random position, not only appended: real manifests carry
    # the credential mid-list (scenarios/1-orion.yaml's `-dbpwd 123456789`
    # sits between other flags), and a model that only ever saw it at the end
    # missed it. Never before an index an existing finding in this list points
    # at (an earlier step of a multi-defect composition) -- that would shift
    # the element out from under that step's own patch. A `command` keeps its
    # executable at index 0.
    list_existed = isinstance(orig_container.get(list_key), list)
    current = orig_container.get(list_key) if list_existed else []
    list_prefix = f"{cpath}/{list_key}/"
    taken = [
        int(f.path[len(list_prefix):].split("/")[0])
        for f in detect_all(canonical_doc, doc_index)
        if f.path.startswith(list_prefix) and f.path[len(list_prefix):].split("/")[0].isdigit()
    ]
    lowest = max(taken) + 1 if taken else (1 if list_key == "command" else 0)
    insert_at = rng.randint(min(lowest, len(current)), len(current))
    m_container.setdefault(list_key, [])[insert_at:insert_at] = items
    # Removing the inserted index once per item undoes the whole insertion
    # (each removal shifts the next item into that index); if the list itself
    # was new, a single "remove the key" op covers every item.
    if list_existed:
        patch = [PatchOp(doc_index, "remove", f"{list_prefix}{insert_at}") for _ in items]
    else:
        patch = [PatchOp(doc_index, "remove", f"{cpath}/{list_key}")]

    # A client-style arg (e.g. `curl ...`) as the first arg turns a database
    # server container into what KSEC-011 treats as a client job, silently
    # erasing (or hiding) its finding -- an injection must never flip another
    # rule's applicability.
    if required_env_for_container(cpath, orig_container, pod_spec) != required_env_for_container(
        cpath, m_container, m_pod_spec
    ):
        return None

    existing = detect_ksec001(canonical_doc, doc_index)
    findings = [f for f in detect_ksec001(mutated_doc, doc_index) if f not in existing]
    assert findings, "KSEC-001 command mutation produced no finding"

    return MutationResult(mutated_doc, canonical_doc, findings, patch, [])


def _is_credential_name(name) -> bool:
    return isinstance(name, str) and bool(SENSITIVE_KEY_RE.search(name)) and not NON_SECRET_KEY_SUFFIX_RE.search(name)


def _env_entries(canonical_doc: dict):
    """(container index, cpath, env index, entry) for every env entry of every container."""
    pod_spec, prefix = get_pod_spec(canonical_doc)
    if pod_spec is None:
        return
    for c_idx, (cpath, container) in enumerate(iter_containers(pod_spec, prefix)):
        env = container.get("env")
        if isinstance(env, list):
            for e_idx, entry in enumerate(env):
                if isinstance(entry, dict):
                    yield c_idx, cpath, e_idx, entry


def _env_entry(doc: dict, c_idx: int, e_idx: int) -> dict:
    pod_spec, prefix = get_pod_spec(doc)
    _, container = list(iter_containers(pod_spec, prefix))[c_idx]
    return container["env"][e_idx]


def _mutate_ksec001_env_in_place(canonical_doc: dict, rng: random.Random, doc_index: int) -> MutationResult | None:
    """Turns a credential variable the manifest ALREADY has (read from a
    Secret via secretKeyRef) into a plaintext literal, in place -- the app's
    own variable name at its real position in the env list, which is what a
    real leak looks like (scenarios/10-mongodb.yaml's
    MONGO_INITDB_ROOT_PASSWORD), unlike the appended fake variable of
    _mutate_ksec001_env. The fix restores the original reference: the
    Secret already exists, so no new resource."""
    existing = detect_ksec001(canonical_doc, doc_index)
    candidates = [
        (c_idx, cpath, e_idx, entry)
        for c_idx, cpath, e_idx, entry in _env_entries(canonical_doc)
        if _is_credential_name(entry.get("name"))
        and isinstance((entry.get("valueFrom") or {}).get("secretKeyRef"), dict)
    ]
    if not candidates:
        return None
    c_idx, cpath, e_idx, entry = rng.choice(candidates)
    mutated_doc = copy.deepcopy(canonical_doc)
    pod_spec, prefix = get_pod_spec(mutated_doc)
    _, m_container = list(iter_containers(pod_spec, prefix))[c_idx]
    m_container["env"][e_idx] = {"name": entry["name"], "value": _fake_secret_value(rng)}
    findings = [f for f in detect_ksec001(mutated_doc, doc_index) if f not in existing]
    if len(findings) != 1:
        return None
    patch = [PatchOp(doc_index, "replace", f"{cpath}/env/{e_idx}", copy.deepcopy(entry))]
    return MutationResult(mutated_doc, canonical_doc, findings, patch, [])


_CREDENTIAL_LESS_URL_RE = re.compile(
    r"^(?P<scheme>mongodb(\+srv)?|postgres(ql)?|mysql|mariadb|redis|rediss|amqps?|nats|mssql)://(?P<rest>[^\s@/]+(/\S*)?)$"
)
_URL_USERS = ("admin", "root", "app", "user", "service")


def _mutate_ksec001_url(canonical_doc: dict, rng: random.Random, doc_index: int) -> MutationResult | None:
    """Embeds credentials into a database/broker URL the manifest already
    has (mongodb://db:27017 -> mongodb://admin:<pw>@db:27017), as in
    scenarios/10-mongodb.yaml's MONGODB_URL. Fixed forward like
    _mutate_ksec001_env: the whole URL moves into a Secret referenced by
    secretKeyRef, plus the placeholder Secret."""
    existing = detect_ksec001(canonical_doc, doc_index)
    candidates = [
        (c_idx, cpath, e_idx, entry, m)
        for c_idx, cpath, e_idx, entry in _env_entries(canonical_doc)
        if isinstance(entry.get("name"), str)
        and isinstance(entry.get("value"), str)
        and (m := _CREDENTIAL_LESS_URL_RE.match(entry["value"]))
    ]
    if not candidates:
        return None
    c_idx, cpath, e_idx, entry, m = rng.choice(candidates)
    for _ in range(5):
        password = _fake_secret_value(rng, cli_safe=True)
        url = f"{m['scheme']}://{rng.choice(_URL_USERS)}:{password}@{m['rest']}"
        if CONN_STRING_RE.match(url):
            break
    else:
        return None

    pod_spec, prefix = get_pod_spec(canonical_doc)
    _, orig_container = list(iter_containers(pod_spec, prefix))[c_idx]
    secret_name = f"{orig_container.get('name', 'app')}-secrets"
    secret_key = entry["name"].lower().replace("_", "-")
    hardened = {"name": entry["name"], "valueFrom": {"secretKeyRef": {"name": secret_name, "key": secret_key}}}

    canonical = copy.deepcopy(canonical_doc)
    c_pod_spec, _ = get_pod_spec(canonical)
    list(iter_containers(c_pod_spec, prefix))[c_idx][1]["env"][e_idx] = hardened
    mutated_doc = copy.deepcopy(canonical_doc)
    m_pod_spec, _ = get_pod_spec(mutated_doc)
    list(iter_containers(m_pod_spec, prefix))[c_idx][1]["env"][e_idx] = {"name": entry["name"], "value": url}

    findings = [f for f in detect_ksec001(mutated_doc, doc_index) if f not in existing]
    if len(findings) != 1:
        return None
    patch = [PatchOp(doc_index, "replace", f"{cpath}/env/{e_idx}", hardened)]
    return MutationResult(mutated_doc, canonical, findings, patch, [_placeholder_secret_yaml(canonical_doc, secret_name, secret_key)])


def mutate_ksec001(
    canonical_doc: dict, rng: random.Random, doc_index: int = 0, candidate_names=None
) -> MutationResult | None:
    """`candidate_names`, if given, overrides FAKE_SECRET_VAR_NAMES for the
    env-var variant only -- used by dataset/build.py to inject a much wider,
    partly corpus-harvested pool of realistic variable names instead of the
    fixed built-in list."""

    def call_env():
        # Three env shapes, in random order until one applies: a new
        # variable appended, an existing secretKeyRef variable leaked in
        # place, or credentials embedded in an existing URL.
        shapes = [
            lambda: _mutate_ksec001_env(canonical_doc, rng, doc_index, candidate_names=candidate_names),
            lambda: _mutate_ksec001_env_in_place(canonical_doc, rng, doc_index),
            lambda: _mutate_ksec001_url(canonical_doc, rng, doc_index),
        ]
        rng.shuffle(shapes)
        for shape in shapes:
            result = shape()
            if result is not None:
                return result
        return None

    def call_command():
        return _mutate_ksec001_command(canonical_doc, rng, doc_index)

    if rng.random() < _ENV_VARIANT_PROBABILITY:
        primary, fallback = call_env, call_command
    else:
        primary, fallback = call_command, call_env

    result = primary()
    if result is not None:
        return result
    return fallback()


# ---------------------------------------------------------------------------
# KSEC-002 -- insecure securityContext
# ---------------------------------------------------------------------------

_KSEC002_SUBCASES = ("privileged", "run_as_root", "allow_priv_esc", "capabilities")
_CAPABILITIES_POOL = ("SYS_ADMIN", "NET_ADMIN", "SYS_PTRACE", "ALL")


def mutate_ksec002(canonical_doc: dict, rng: random.Random, doc_index: int = 0) -> MutationResult | None:
    assert not detect_ksec002(canonical_doc, doc_index), "canonical already has a KSEC-002 finding"

    pod_spec, prefix = get_pod_spec(canonical_doc)
    if pod_spec is None:
        return None
    containers = list(iter_containers(pod_spec, prefix))
    if not containers:
        return None

    container_idx = rng.randrange(len(containers))
    cpath, orig_container = containers[container_idx]

    mutated_doc = copy.deepcopy(canonical_doc)
    m_pod_spec, _ = get_pod_spec(mutated_doc)
    m_containers = list(iter_containers(m_pod_spec, prefix))
    _, m_container = m_containers[container_idx]

    subcase = rng.choice(_KSEC002_SUBCASES)

    if subcase == "privileged":
        field_parts, new_value = ["securityContext", "privileged"], True
    elif subcase == "run_as_root":
        field_parts, new_value = ["securityContext", "runAsUser"], 0
    elif subcase == "allow_priv_esc":
        field_parts, new_value = ["securityContext", "allowPrivilegeEscalation"], True
    else:
        field_parts = ["securityContext", "capabilities", "add"]
        new_value = [rng.choice(_CAPABILITIES_POOL)]

    patch_op = _set_field_with_patch(m_container, orig_container, cpath, field_parts, new_value, doc_index)

    findings = detect_ksec002(mutated_doc, doc_index)
    assert findings, "KSEC-002 mutation produced no finding"

    return MutationResult(mutated_doc, canonical_doc, findings, [patch_op], [])


# ---------------------------------------------------------------------------
# KSEC-003 -- host access
# ---------------------------------------------------------------------------

_HOST_NAMESPACE_FIELDS = ("hostNetwork", "hostPID", "hostIPC")
_INJECTABLE_SENSITIVE_PATHS = tuple(p for p in SENSITIVE_HOST_PATHS if p != "/")


def mutate_ksec003(canonical_doc: dict, rng: random.Random, doc_index: int = 0) -> MutationResult | None:
    assert not detect_ksec003(canonical_doc, doc_index), "canonical already has a KSEC-003 finding"

    pod_spec, prefix = get_pod_spec(canonical_doc)
    if pod_spec is None:
        return None

    mutated_doc = copy.deepcopy(canonical_doc)
    m_pod_spec, _ = get_pod_spec(mutated_doc)

    subcase = rng.choice(("host_namespace", "host_path"))
    patch_ops = []

    if subcase == "host_namespace":
        host_field = rng.choice(_HOST_NAMESPACE_FIELDS)
        patch_ops.append(_set_field_with_patch(m_pod_spec, pod_spec, prefix, [host_field], True, doc_index))
    else:
        containers = list(iter_containers(pod_spec, prefix))
        if not containers:
            return None
        container_idx = rng.randrange(len(containers))
        container_cpath, orig_container = containers[container_idx]
        _, m_container = list(iter_containers(m_pod_spec, prefix))[container_idx]

        host_path = rng.choice(_INJECTABLE_SENSITIVE_PATHS)
        volume_name = "host-mount"
        patch_ops.append(
            _append_list_item_with_patch(
                m_pod_spec,
                pod_spec,
                "volumes",
                {"name": volume_name, "hostPath": {"path": host_path}},
                prefix,
                doc_index,
            )
        )
        patch_ops.append(
            _append_list_item_with_patch(
                m_container,
                orig_container,
                "volumeMounts",
                {"name": volume_name, "mountPath": host_path},
                container_cpath,
                doc_index,
            )
        )

    findings = detect_ksec003(mutated_doc, doc_index)
    assert findings, "KSEC-003 mutation produced no finding"

    return MutationResult(mutated_doc, canonical_doc, findings, patch_ops, [])


# ---------------------------------------------------------------------------
# KSEC-004 -- permissive RBAC
# ---------------------------------------------------------------------------


def mutate_ksec004(canonical_doc: dict, rng: random.Random, doc_index: int = 0) -> MutationResult | None:
    assert not detect_ksec004(canonical_doc, doc_index), "canonical already has a KSEC-004 finding"

    kind = canonical_doc.get("kind")
    mutated_doc = copy.deepcopy(canonical_doc)

    if kind in RBAC_ROLE_KINDS:
        rules = canonical_doc.get("rules")
        if isinstance(rules, list) and rules:
            rule_idx = rng.randrange(len(rules))
            field = rng.choice(("apiGroups", "resources", "verbs"))
            patch_op = _set_field_with_patch(
                mutated_doc["rules"][rule_idx], rules[rule_idx], f"/rules/{rule_idx}", [field], ["*"], doc_index
            )
        else:
            new_rule = {"apiGroups": ["*"], "resources": ["*"], "verbs": ["*"]}
            patch_op = _append_list_item_with_patch(mutated_doc, canonical_doc, "rules", new_rule, "", doc_index)

    elif kind in RBAC_BINDING_KINDS:
        role_ref = canonical_doc.get("roleRef")
        if not isinstance(role_ref, dict) or "name" not in role_ref:
            return None
        patch_op = _set_field_with_patch(
            mutated_doc, canonical_doc, "", ["roleRef", "name"], "cluster-admin", doc_index
        )
    else:
        return None

    findings = detect_ksec004(mutated_doc, doc_index)
    assert findings, "KSEC-004 mutation produced no finding"

    return MutationResult(mutated_doc, canonical_doc, findings, [patch_op], [])


# ---------------------------------------------------------------------------
# KSEC-005 -- unpinned image
# ---------------------------------------------------------------------------


def _is_conventionally_tagged(image: str) -> bool:
    """True if the image uses the repo:tag format (not pinned by @sha256
    digest, which would need a different mutation strategy)."""
    if "@" in image:
        return False
    tail = image[image.rfind("/") + 1 :]
    return ":" in tail and tail.rsplit(":", 1)[1] != "latest"


def mutate_ksec005(canonical_doc: dict, rng: random.Random, doc_index: int = 0) -> MutationResult | None:
    """Unpins one still-pinned image. Repeatable within a manifest: the
    candidates are only containers whose image is still conventionally
    tagged, so a second call in a multi-defect composition unpins a
    different container -- a manifest with two untagged images (an app plus
    a `busybox` init container, as in scenarios/7-elasticsearch.yaml) is
    common in practice, and a model that only ever saw one per manifest
    stopped after reporting the first."""
    existing = detect_ksec005(canonical_doc, doc_index)

    pod_spec, prefix = get_pod_spec(canonical_doc)
    if pod_spec is None:
        return None
    containers = [
        (p, c)
        for p, c in iter_containers(pod_spec, prefix)
        if isinstance(c.get("image"), str) and _is_conventionally_tagged(c["image"])
    ]
    if not containers:
        return None

    container_idx = rng.randrange(len(containers))
    cpath, orig_container = containers[container_idx]

    mutated_doc = copy.deepcopy(canonical_doc)
    m_pod_spec, _ = get_pod_spec(mutated_doc)
    m_containers = [
        (p, c)
        for p, c in iter_containers(m_pod_spec, prefix)
        if isinstance(c.get("image"), str) and _is_conventionally_tagged(c["image"])
    ]
    _, m_container = m_containers[container_idx]

    repo, _tag = split_image(orig_container["image"])
    new_image = repo if rng.random() < 0.5 else f"{repo}:latest"
    if not is_unpinned_image(new_image):
        # Malformed original tags (e.g. a stray "repo:sha256:<hash>" typo'd
        # digest instead of "repo@sha256:<hash>") can leave a substring in
        # `repo` that still looks like a valid non-latest tag after
        # stripping -- ":latest" always works regardless of what `repo` is.
        new_image = f"{repo}:latest"

    patch_op = _set_field_with_patch(m_container, orig_container, cpath, ["image"], new_image, doc_index)

    findings = [f for f in detect_ksec005(mutated_doc, doc_index) if f not in existing]
    assert len(findings) == 1, "KSEC-005 mutation should add exactly one finding"

    return MutationResult(mutated_doc, canonical_doc, findings, [patch_op], [])


# ---------------------------------------------------------------------------
# KSEC-006..011 -- semantic/configuration-correctness checks
#
# Unlike 001-005, normalize.py does NOT fix or drop documents for these
# rules, so a real corpus document could already exhibit the bug in the
# wild (e.g. an example YAML in a repo that was never actually applied).
# These mutators therefore use a soft `if detect(...): return None` skip
# instead of a hard `assert` precondition -- an assert here would crash the
# whole build on a messy-but-real document instead of just skipping it.
# ---------------------------------------------------------------------------


def mutate_ksec006(canonical_doc: dict, rng: random.Random, doc_index: int = 0) -> MutationResult | None:
    if detect_ksec006(canonical_doc, doc_index):
        return None

    match_labels, sel_prefix = get_selector_match_labels(canonical_doc)
    if not match_labels:
        return None
    template_labels, _ = get_template_labels(canonical_doc)
    if not template_labels:
        return None

    key = rng.choice(list(match_labels.keys()))
    old_value = match_labels[key]
    new_value = f"{old_value}-x{rng.randrange(1000)}"

    mutated_doc = copy.deepcopy(canonical_doc)
    m_match_labels, _ = get_selector_match_labels(mutated_doc)
    m_match_labels[key] = new_value

    patch_op = PatchOp(doc_index, "replace", f"{sel_prefix}/{escape_json_pointer_token(key)}", old_value)

    findings = detect_ksec006(mutated_doc, doc_index)
    assert findings, "KSEC-006 mutation produced no finding"

    return MutationResult(mutated_doc, canonical_doc, findings, [patch_op], [])


def mutate_ksec007(canonical_doc: dict, rng: random.Random, doc_index: int = 0) -> MutationResult | None:
    if detect_ksec007(canonical_doc, doc_index):
        return None

    pod_spec, prefix = get_pod_spec(canonical_doc)
    if pod_spec is None:
        return None

    containers = list(iter_containers(pod_spec, prefix))
    candidates = []
    for idx, (cpath, container) in enumerate(containers):
        port_numbers, port_names = get_container_ports(container)
        if not port_numbers and not port_names:
            continue
        for probe_field in PROBE_FIELDS:
            probe = container.get(probe_field)
            if not isinstance(probe, dict):
                continue
            for check_field in ("httpGet", "tcpSocket"):
                check = probe.get(check_field)
                if not isinstance(check, dict):
                    continue
                port = check.get("port")
                if isinstance(port, bool):
                    continue
                if isinstance(port, int) and port in port_numbers:
                    candidates.append((idx, cpath, probe_field, check_field, port, "int"))
                elif isinstance(port, str) and port in port_names:
                    candidates.append((idx, cpath, probe_field, check_field, port, "str"))
    if not candidates:
        return None

    idx, cpath, probe_field, check_field, old_port, port_kind = rng.choice(candidates)

    mutated_doc = copy.deepcopy(canonical_doc)
    m_pod_spec, _ = get_pod_spec(mutated_doc)
    _, m_container = list(iter_containers(m_pod_spec, prefix))[idx]

    if port_kind == "int":
        port_numbers, _ = get_container_ports(m_container)
        offsets = [1, 2, 3, 5, 7, 11, 13]
        rng.shuffle(offsets)
        new_port = next((old_port + off for off in offsets if (old_port + off) not in port_numbers), old_port + 9999)
    else:
        new_port = old_port + "x"

    m_container[probe_field][check_field]["port"] = new_port

    path = f"{cpath}/{probe_field}/{check_field}/port"
    patch_op = PatchOp(doc_index, "replace", path, old_port)

    findings = detect_ksec007(mutated_doc, doc_index)
    assert findings, "KSEC-007 mutation produced no finding"

    return MutationResult(mutated_doc, canonical_doc, findings, [patch_op], [])


_QUANTITY_LITERAL_RE = re.compile(r"^([0-9]*\.?[0-9]+)([A-Za-z]*)$")


def _bump_quantity(quantity_str, factor: float) -> str:
    """Scales a Kubernetes quantity literal (e.g. "500m", "1Gi") by `factor`,
    preserving its suffix. Falls back to a large plain number if the literal
    can't be parsed (e.g. it used exponential notation)."""
    m = _QUANTITY_LITERAL_RE.match(str(quantity_str).strip())
    if not m:
        return "999999"
    num_str, suffix = m.groups()
    new_num = float(num_str) * factor
    new_num_str = str(int(new_num)) if new_num == int(new_num) else f"{new_num:.3f}".rstrip("0").rstrip(".")
    return f"{new_num_str}{suffix}"


def mutate_ksec008(canonical_doc: dict, rng: random.Random, doc_index: int = 0) -> MutationResult | None:
    if detect_ksec008(canonical_doc, doc_index):
        return None

    pod_spec, prefix = get_pod_spec(canonical_doc)
    if pod_spec is None:
        return None

    containers = list(iter_containers(pod_spec, prefix))
    candidates = []
    for idx, (cpath, container) in enumerate(containers):
        resources = container.get("resources")
        if not isinstance(resources, dict):
            continue
        requests = resources.get("requests")
        limits = resources.get("limits")
        if not isinstance(requests, dict) or not isinstance(limits, dict):
            continue
        for resource_name in ("cpu", "memory"):
            req_val = requests.get(resource_name)
            lim_val = limits.get(resource_name)
            if req_val is None or lim_val is None:
                continue
            if parse_quantity(req_val) is None or parse_quantity(lim_val) is None:
                continue
            candidates.append((idx, cpath, resource_name, req_val, lim_val))
    if not candidates:
        return None

    idx, cpath, resource_name, old_req_val, lim_val = rng.choice(candidates)

    mutated_doc = copy.deepcopy(canonical_doc)
    m_pod_spec, _ = get_pod_spec(mutated_doc)
    _, m_container = list(iter_containers(m_pod_spec, prefix))[idx]

    new_req_val = _bump_quantity(lim_val, 2.0)  # double the LIMIT value: always > limit, regardless of the old request
    m_container["resources"]["requests"][resource_name] = new_req_val

    path = f"{cpath}/resources/requests/{resource_name}"
    patch_op = PatchOp(doc_index, "replace", path, old_req_val)

    findings = detect_ksec008(mutated_doc, doc_index)
    assert findings, "KSEC-008 mutation produced no finding"

    return MutationResult(mutated_doc, canonical_doc, findings, [patch_op], [])


def _typo(name: str, rng: random.Random) -> str:
    """Applies one small, human-plausible edit (transpose/delete/duplicate a
    character) to `name`. Represents the kind of copy-paste/fat-finger
    mistake that produces a reference which looks right but isn't."""
    if len(name) < 2:
        return name + rng.choice(string.ascii_lowercase)
    op = rng.choice(("transpose", "delete", "duplicate"))
    if op == "transpose":
        i = rng.randrange(len(name) - 1)
        chars = list(name)
        chars[i], chars[i + 1] = chars[i + 1], chars[i]
        return "".join(chars)
    if op == "delete":
        i = rng.randrange(len(name))
        return name[:i] + name[i + 1 :]
    i = rng.randrange(len(name))
    return name[:i] + name[i] + name[i:]


_SERVICE_SUFFIXES = ("-app", "-svc", "-server", "-api")


def _mismatched_selector_value(value: str, rng: random.Random) -> str:
    """A wrong-but-plausible Service selector value: usually a one-character
    typo (orionld -> orionlds, selenium-hub -> sellenium-hub), otherwise a
    dropped or extra name segment (mongodb-app -> mongodb) -- the two shapes
    seen in the hand-written scenarios."""
    if rng.random() < 0.6:
        return _typo(value, rng)
    if "-" in value:
        return value.rsplit("-", 1)[0]
    return value + rng.choice(_SERVICE_SUFFIXES)


def mutate_ksec006_service(docs: list[dict], rng: random.Random) -> FileMutationResult | None:
    """KSEC-006 across documents: breaks a Service's selector so it no longer
    matches the one workload in the file it selected. The fix is the inverse
    `replace` on the Service -- the workload's labels are the source of
    truth, as the Service is what routes to them."""
    if detect_ksec006_services(docs):
        return None
    workloads = pod_label_sets(docs)
    candidates = []
    for i, doc in enumerate(docs):
        selector = get_service_selector(doc)
        if selector is None:
            continue
        if sum(label_selector_matches(selector, labels) for _, labels in workloads) == 1:
            candidates.append(i)
    if not candidates:
        return None

    service_index = rng.choice(candidates)
    selector = get_service_selector(docs[service_index])
    key = rng.choice(sorted(selector, key=str))
    old_value = selector[key]
    for _ in range(5):
        new_value = _mismatched_selector_value(str(old_value), rng)
        broken = {**selector, key: new_value}
        if new_value != str(old_value) and not any(label_selector_matches(broken, labels) for _, labels in workloads):
            break
    else:
        return None

    mutated_docs = copy.deepcopy(docs)
    mutated_docs[service_index]["spec"]["selector"][key] = new_value
    patch = [PatchOp(service_index, "replace", f"/spec/selector/{escape_json_pointer_token(str(key))}", old_value)]

    findings = detect_ksec006_services(mutated_docs)
    assert len(findings) == 1, "KSEC-006 Service mutation should produce exactly one finding"

    return FileMutationResult(mutated_docs, copy.deepcopy(docs), findings, patch, [])


def mutate_ksec009(canonical_doc: dict, rng: random.Random, doc_index: int = 0) -> MutationResult | None:
    if detect_ksec009(canonical_doc, doc_index):
        return None

    pod_spec, prefix = get_pod_spec(canonical_doc)
    if pod_spec is None:
        return None

    volume_names = declared_volume_names(canonical_doc, pod_spec)
    if not volume_names:
        return None

    containers = list(iter_containers(pod_spec, prefix))
    candidates = []
    for idx, (cpath, container) in enumerate(containers):
        mounts = container.get("volumeMounts")
        if not isinstance(mounts, list):
            continue
        for mi, mount in enumerate(mounts):
            if isinstance(mount, dict) and isinstance(mount.get("name"), str) and mount["name"] in volume_names:
                candidates.append((idx, cpath, mi, mount["name"]))
    if not candidates:
        return None

    idx, cpath, mi, old_name = rng.choice(candidates)
    typo_name = _typo(old_name, rng)
    attempts = 0
    while (typo_name == old_name or typo_name in volume_names) and attempts < 5:
        typo_name = _typo(old_name, rng)
        attempts += 1
    if typo_name == old_name or typo_name in volume_names:
        return None

    mutated_doc = copy.deepcopy(canonical_doc)
    m_pod_spec, _ = get_pod_spec(mutated_doc)
    _, m_container = list(iter_containers(m_pod_spec, prefix))[idx]
    m_container["volumeMounts"][mi]["name"] = typo_name

    path = f"{cpath}/volumeMounts/{mi}/name"
    patch_op = PatchOp(doc_index, "replace", path, old_name)

    findings = detect_ksec009(mutated_doc, doc_index)
    assert findings, "KSEC-009 mutation produced no finding"

    return MutationResult(mutated_doc, canonical_doc, findings, [patch_op], [])


_HTTP_PROBE_PATHS = ("/", "/health", "/healthz", "/ready", "/status")


def _probe_port_is_consistent(port, port_numbers: set[int], port_names: set[str]) -> bool:
    """True if detect_ksec007 would NOT flag this probe port -- the mirror of
    its conditions. KSEC-010 keeps the port untouched, so starting from a
    consistent one guarantees it never creates or hides a KSEC-007 finding."""
    if isinstance(port, bool):
        return False
    if isinstance(port, int):
        return not port_numbers or port in port_numbers
    if isinstance(port, str):
        return not port_names or port in port_names
    return False


def mutate_ksec010(canonical_doc: dict, rng: random.Random, doc_index: int = 0) -> MutationResult | None:
    """Turns a non-HTTP server's tcpSocket/exec probe into an httpGet probe.
    If the target container has no probe at all, the round-trip target
    ("canonical") gains a tcpSocket probe on the server's default port -- the
    same fixed-forward trick as KSEC-001's env variant -- so the fix taught is
    always "use tcpSocket/exec", never "delete the probe"."""
    if detect_ksec010(canonical_doc, doc_index):
        return None

    pod_spec, prefix = get_pod_spec(canonical_doc)
    if pod_spec is None:
        return None

    candidates = []  # (container idx, cpath, probe field, original check field or None, port)
    for idx, (cpath, container) in enumerate(iter_containers(pod_spec, prefix)):
        if is_init_container_path(cpath):
            continue
        default_port = NON_HTTP_IMAGE_PORTS.get(image_basename(container.get("image")))
        if default_port is None:
            continue
        port_numbers, port_names = get_container_ports(container)
        default_port_ok = not port_numbers or default_port in port_numbers
        for probe_field in PROBE_FIELDS:
            probe = container.get(probe_field)
            if isinstance(probe, dict):
                tcp = probe.get("tcpSocket")
                if isinstance(tcp, dict) and _probe_port_is_consistent(tcp.get("port"), port_numbers, port_names):
                    candidates.append((idx, cpath, probe_field, "tcpSocket", tcp["port"]))
                elif isinstance(probe.get("exec"), dict) and default_port_ok:
                    candidates.append((idx, cpath, probe_field, "exec", default_port))
            elif probe is None and probe_field != "startupProbe" and default_port_ok:
                candidates.append((idx, cpath, probe_field, None, default_port))
    if not candidates:
        return None

    idx, cpath, probe_field, check_field, port = rng.choice(candidates)
    http_get = {"path": rng.choice(_HTTP_PROBE_PATHS), "port": port}

    if check_field is None:
        canonical = copy.deepcopy(canonical_doc)
        c_pod_spec, _ = get_pod_spec(canonical)
        _, c_container = list(iter_containers(c_pod_spec, prefix))[idx]
        c_container[probe_field] = {"tcpSocket": {"port": port}}
        check_field, restored_check = "tcpSocket", {"port": port}
    else:
        canonical = canonical_doc
        _, orig_container = list(iter_containers(pod_spec, prefix))[idx]
        restored_check = copy.deepcopy(orig_container[probe_field][check_field])

    mutated_doc = copy.deepcopy(canonical)
    m_pod_spec, _ = get_pod_spec(mutated_doc)
    _, m_container = list(iter_containers(m_pod_spec, prefix))[idx]
    del m_container[probe_field][check_field]
    m_container[probe_field]["httpGet"] = http_get

    probe_path = f"{cpath}/{probe_field}"
    patch = [
        PatchOp(doc_index, "remove", f"{probe_path}/httpGet"),
        PatchOp(doc_index, "add", f"{probe_path}/{check_field}", restored_check),
    ]

    findings = detect_ksec010(mutated_doc, doc_index)
    assert len(findings) == 1, "KSEC-010 mutation should produce exactly one finding"

    return MutationResult(mutated_doc, canonical, findings, patch, [])


def _keeps_env_entry_verbatim(entry: dict) -> bool:
    """An entry that already satisfies the requirement without a plaintext
    password -- a Secret/ConfigMap reference, a *_FILE path, or a random
    root password -- is restored as-is by KSEC-011's fix."""
    name = str(entry.get("name", ""))
    return "valueFrom" in entry or name.endswith("_FILE") or "RANDOM" in name


def mutate_ksec011(canonical_doc: dict, rng: random.Random, doc_index: int = 0) -> MutationResult | None:
    """Removes an env var a database image requires (see
    k8s.IMAGE_ENV_CONTRACTS). A secret requirement's fix restores it as a
    secretKeyRef plus a companion placeholder Secret (the same shape as
    KSEC-001's externalization) -- unless the original entry was already a
    reference/*_FILE/random-password, which is restored verbatim. Plaintext
    values or "trust"/allow-empty settings in the source doc are replaced by
    the secretKeyRef form in the round-trip target, so the model is never
    taught to re-add a plaintext or insecure setting. A literal requirement
    (ACCEPT_EULA) is restored as its plain value."""
    if detect_ksec011(canonical_doc, doc_index):
        return None

    pod_spec, prefix = get_pod_spec(canonical_doc)
    if pod_spec is None:
        return None

    # An entry already carrying a plaintext credential (KSEC-001's in-place
    # variant leaks e.g. POSTGRES_PASSWORD itself) is never removed here:
    # that would erase the KSEC-001 finding along with it.
    leaked = {f.path for f in detect_ksec001(canonical_doc, doc_index)}
    candidates = []  # (container idx, cpath, container name, requirement, satisfying env indices)
    for idx, (cpath, container) in enumerate(iter_containers(pod_spec, prefix)):
        applicable = required_env_for_container(cpath, container, pod_spec)
        env = container.get("env")
        if applicable is None or not isinstance(env, list):
            continue
        for requirement in applicable[1]:
            satisfying = [i for i, e in enumerate(env) if isinstance(e, dict) and e.get("name") in requirement.accepted]
            if satisfying and not any(f"{cpath}/env/{i}/value" in leaked for i in satisfying):
                candidates.append((idx, cpath, container.get("name") or "app", requirement, satisfying))
    if not candidates:
        return None

    idx, cpath, container_name, requirement, satisfying = rng.choice(candidates)
    _, orig_container = list(iter_containers(pod_spec, prefix))[idx]
    orig_env = orig_container["env"]
    primary = requirement.primary

    kept = next((orig_env[i] for i in satisfying if _keeps_env_entry_verbatim(orig_env[i])), None)
    if requirement.literal is not None:
        restored_entry = {"name": primary, "value": requirement.literal}
        new_resources = []
    elif kept is not None:
        restored_entry = copy.deepcopy(kept)
        new_resources = []
    else:
        secret_name = f"{container_name}-secrets"
        secret_key = primary.lower().replace("_", "-")
        restored_entry = {"name": primary, "valueFrom": {"secretKeyRef": {"name": secret_name, "key": secret_key}}}
        new_resources = [_placeholder_secret_yaml(canonical_doc, secret_name, secret_key)]

    # Every satisfying entry collapses into restored_entry at the first one's
    # position -- satisfying is sorted, so nothing before it moves.
    insert_at = satisfying[0]
    remaining = [e for i, e in enumerate(orig_env) if i not in satisfying]

    canonical = copy.deepcopy(canonical_doc)
    c_pod_spec, _ = get_pod_spec(canonical)
    _, c_container = list(iter_containers(c_pod_spec, prefix))[idx]
    c_container["env"] = copy.deepcopy(remaining[:insert_at] + [restored_entry] + remaining[insert_at:])

    mutated_doc = copy.deepcopy(canonical_doc)
    m_pod_spec, _ = get_pod_spec(mutated_doc)
    _, m_container = list(iter_containers(m_pod_spec, prefix))[idx]
    if remaining:
        m_container["env"] = copy.deepcopy(remaining)
        patch = [PatchOp(doc_index, "add", f"{cpath}/env/{insert_at}", restored_entry)]
    else:
        del m_container["env"]
        patch = [PatchOp(doc_index, "add", f"{cpath}/env", [restored_entry])]

    findings = detect_ksec011(mutated_doc, doc_index)
    assert len(findings) == 1, "KSEC-011 mutation should produce exactly one finding"

    return MutationResult(mutated_doc, canonical, findings, patch, new_resources)


def _misspell(token: str, rng: random.Random) -> str:
    """One human-plausible slip: _typo's transpose/delete/duplicate, or a
    substituted character (a digit for another digit -- python3 -> python5)."""
    if rng.random() < 0.6:
        return _typo(token, rng)
    i = rng.randrange(len(token))
    pool = string.digits if token[i].isdigit() else string.ascii_lowercase
    return token[:i] + rng.choice([c for c in pool if c != token[i].lower()]) + token[i + 1 :]


def mutate_ksec012(canonical_doc: dict, rng: random.Random, doc_index: int = 0) -> MutationResult | None:
    """Misspells a name the corpus treats as very common: a container's
    command binary (python3 -> python5) or the first directory of a path in
    command/args/env (/home -> /hom). For a very common deep path
    (/var/run/secrets/kubernetes.io/...), the first directory may instead be
    replaced by a longer word-like variant (/varia/run/secrets/...), the
    shape of scenarios/7-elasticsearch.yaml's /variavel/. The fix restores
    the original string."""
    if detect_ksec012(canonical_doc, doc_index):
        return None
    pod_spec, prefix = get_pod_spec(canonical_doc)
    if pod_spec is None:
        return None
    vocab = names.load_vocab()
    candidates = []  # (pointer, original string, token, kind)
    for cpath, container in iter_containers(pod_spec, prefix):
        binary = names.command_binary(container)
        if binary and len(binary) >= names.MIN_BINARY_LEN and vocab["binaries"].get(binary, 0) >= names.COMMON_MIN:
            candidates.append((f"{cpath}/command/0", container["command"][0], binary, "binary"))
        for pointer, text in names.container_strings(cpath, container):
            for path in names.paths_in(text):
                first = path.strip("/").split("/")[0]
                if len(first) >= names.MIN_LEN and vocab["path_segments"].get(first, 0) >= names.COMMON_MIN:
                    candidates.append((pointer, text, path, "path"))
    if not candidates:
        return None

    pointer, text, token, kind = rng.choice(candidates)
    for _ in range(8):
        if kind == "binary":
            new_token = _misspell(token, rng)
            new_text = text[: len(text) - len(token)] + new_token
        else:
            parts = token.strip("/").split("/")
            tail = "/".join(parts[1 : names.MAX_PREFIX_DEPTH])
            deep_common = any(tail.startswith(t) for t in vocab["common_tails"] if t.count("/") >= 2)
            if deep_common and rng.random() < 0.4:
                new_first = parts[0] + "".join(rng.choice(string.ascii_lowercase) for _ in range(rng.randint(2, 5)))
            else:
                new_first = _misspell(parts[0], rng)
            new_token = "/" + "/".join([new_first, *parts[1:]])
            new_text = text.replace(token, new_token, 1)
        if new_text == text:
            continue
        mutated_doc = jsonpatch.apply_patch(canonical_doc, [{"op": "replace", "path": pointer, "value": new_text}])
        findings = detect_ksec012(mutated_doc, doc_index)
        if len(findings) == 1 and findings[0].path == pointer:
            return MutationResult(mutated_doc, canonical_doc, findings, [PatchOp(doc_index, "replace", pointer, text)], [])
    return None


def _other_rule_counts(doc: dict, rule_id: str, doc_index: int) -> collections.Counter:
    return collections.Counter(
        f.rule_id for f in detect_all(doc, doc_index) if f.rule_id in RULE_IDS and f.rule_id != rule_id
    )


def _preserving_other_rules(rule_id: str, mutator):
    """Enforces, for every registered mutator, the invariant labels depend
    on: injecting one rule's defect must never create, erase or hide another
    active rule's finding. A label only lists the injected rule's findings
    (single-defect), or asserts one finding per injection (multi-defect), so
    a side effect on another rule means a wrong label. Rules interact through
    shared fields -- found in practice: KSEC-001 injecting a plaintext
    POSTGRES_PASSWORD erased a KSEC-011 finding; KSEC-009 renaming a
    replica's data mount made KSEC-011 start applying -- so this is checked
    after every mutation instead of trusting each mutator to anticipate every
    other rule. Compared against the mutator's own round-trip target, not its
    input: KSEC-001/010/011 legitimately fix forward in that target."""

    @functools.wraps(mutator)
    def wrapped(canonical_doc: dict, rng: random.Random, doc_index: int = 0, **kwargs):
        result = mutator(canonical_doc, rng, doc_index, **kwargs)
        if result is None:
            return None
        if _other_rule_counts(result.mutated_doc, rule_id, doc_index) != _other_rule_counts(
            result.canonical, rule_id, doc_index
        ):
            return None
        return result

    return wrapped


# The registry dataset/build.py reads. To disable a rule from generation,
# remove it here AND from dataset/schema.py's RULES (see the note there).
MUTATORS = {
    rule_id: _preserving_other_rules(rule_id, mutator)
    for rule_id, mutator in (
        ("KSEC-001", mutate_ksec001),
        ("KSEC-002", mutate_ksec002),
        ("KSEC-003", mutate_ksec003),
        ("KSEC-004", mutate_ksec004),
        ("KSEC-005", mutate_ksec005),
        ("KSEC-006", mutate_ksec006),
        ("KSEC-007", mutate_ksec007),
        ("KSEC-008", mutate_ksec008),
        ("KSEC-009", mutate_ksec009),
        ("KSEC-010", mutate_ksec010),
        ("KSEC-011", mutate_ksec011),
        ("KSEC-012", mutate_ksec012),
    )
}
