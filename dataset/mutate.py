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

import copy
import random
import re
import string
from dataclasses import dataclass, field

from dataset.detect import (
    detect_ksec001,
    detect_ksec002,
    detect_ksec003,
    detect_ksec004,
    detect_ksec005,
    detect_ksec006,
    detect_ksec007,
    detect_ksec008,
    detect_ksec009,
)
from dataset.k8s import (
    PROBE_FIELDS,
    RBAC_BINDING_KINDS,
    RBAC_ROLE_KINDS,
    SENSITIVE_HOST_PATHS,
    get_container_ports,
    get_pod_spec,
    get_selector_match_labels,
    get_template_labels,
    is_unpinned_image,
    iter_containers,
    parse_quantity,
    split_image,
)
from dataset.schema import Finding, PatchOp, escape_json_pointer_token


@dataclass
class MutationResult:
    mutated_doc: dict
    canonical: dict
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
)

# Probability of attempting the env-var injection first (vs. command/args).
_ENV_VARIANT_PROBABILITY = 0.7


def _fake_secret_value(rng: random.Random, length: int = 20) -> str:
    """Generates a synthetic value shaped like a secret. Never a real
    credential -- just a plausible literal for training pattern detection."""
    return "".join(rng.choices(_SECRET_CHARSET, k=length))


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
    candidates = [n for n in name_pool if n not in existing_names]
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
    canonical_with_ref = copy.deepcopy(canonical_doc)
    c_pod_spec, _ = get_pod_spec(canonical_with_ref)
    c_containers = list(iter_containers(c_pod_spec, prefix))
    _, c_container = c_containers[container_idx]
    c_env = c_container.setdefault("env", [])
    env_index = len(c_env)
    c_env.append(hardened_entry)

    mutated_doc = copy.deepcopy(canonical_doc)
    m_pod_spec, _ = get_pod_spec(mutated_doc)
    m_containers = list(iter_containers(m_pod_spec, prefix))
    _, m_container = m_containers[container_idx]
    m_env = m_container.setdefault("env", [])
    assert len(m_env) == env_index
    m_env.append({"name": var_name, "value": _fake_secret_value(rng)})

    env_path = f"{cpath}/env/{env_index}"
    patch = [PatchOp(doc_index, "replace", env_path, hardened_entry)]

    findings = detect_ksec001(mutated_doc, doc_index)
    assert findings, "KSEC-001 env mutation produced no finding"

    namespace = canonical_doc.get("metadata", {}).get("namespace")
    secret_yaml_lines = [
        "apiVersion: v1",
        "kind: Secret",
        "metadata:",
        f"  name: {secret_resource_name}",
    ]
    if namespace:
        secret_yaml_lines.append(f"  namespace: {namespace}")
    secret_yaml_lines += [
        "type: Opaque",
        "stringData:",
        f'  {secret_key}: "<REPLACE_WITH_SECRET_VALUE>"',
    ]
    new_resources = ["\n".join(secret_yaml_lines)]

    return MutationResult(mutated_doc, canonical_with_ref, findings, patch, new_resources)


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

    fake = _fake_secret_value(rng)
    template = rng.choice(_CLI_INJECTION_TEMPLATES)
    injected_string = template(fake)

    patch_op = _append_list_item_with_patch(m_container, orig_container, list_key, injected_string, cpath, doc_index)

    findings = detect_ksec001(mutated_doc, doc_index)
    assert findings, "KSEC-001 command mutation produced no finding"

    return MutationResult(mutated_doc, canonical_doc, findings, [patch_op], [])


def mutate_ksec001(
    canonical_doc: dict, rng: random.Random, doc_index: int = 0, candidate_names=None
) -> MutationResult | None:
    """`candidate_names`, if given, overrides FAKE_SECRET_VAR_NAMES for the
    env-var variant only -- used by dataset/build.py to inject a much wider,
    partly corpus-harvested pool of realistic variable names instead of the
    fixed built-in list."""

    def call_env():
        return _mutate_ksec001_env(canonical_doc, rng, doc_index, candidate_names=candidate_names)

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
    assert not detect_ksec005(canonical_doc, doc_index), "canonical already has a KSEC-005 finding"

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

    findings = detect_ksec005(mutated_doc, doc_index)
    assert findings, "KSEC-005 mutation produced no finding"

    return MutationResult(mutated_doc, canonical_doc, findings, [patch_op], [])


# ---------------------------------------------------------------------------
# KSEC-006..009 -- semantic/configuration-correctness checks
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


def mutate_ksec009(canonical_doc: dict, rng: random.Random, doc_index: int = 0) -> MutationResult | None:
    if detect_ksec009(canonical_doc, doc_index):
        return None

    pod_spec, prefix = get_pod_spec(canonical_doc)
    if pod_spec is None:
        return None

    volumes = pod_spec.get("volumes")
    volume_names = {v.get("name") for v in volumes if isinstance(v, dict)} if isinstance(volumes, list) else set()
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


# KSEC-003, KSEC-004 and KSEC-009 are implemented and tested above (see
# mutate_ksec003/004/009) but deliberately left out of this registry, so
# dataset/build.py never injects or labels examples for them -- see the note
# next to dataset/schema.py's RULES dict. Add them back here (and to RULES)
# to re-enable.
MUTATORS = {
    "KSEC-001": mutate_ksec001,
    "KSEC-002": mutate_ksec002,
    "KSEC-005": mutate_ksec005,
    "KSEC-006": mutate_ksec006,
    "KSEC-007": mutate_ksec007,
    "KSEC-008": mutate_ksec008,
}
