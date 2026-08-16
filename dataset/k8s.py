"""
Kubernetes manifest navigation helpers, shared by detect.py, normalize.py and
mutate.py. The main job here is resolving, for any kind with a pod template,
where the PodSpec actually lives -- especially the CronJob case, which nests
the PodSpec four levels below spec (spec.jobTemplate.spec.template.spec).
"""

from __future__ import annotations

POD_TEMPLATE_KINDS = frozenset(
    {"Pod", "Deployment", "StatefulSet", "DaemonSet", "ReplicaSet", "Job", "CronJob"}
)
RBAC_ROLE_KINDS = frozenset({"Role", "ClusterRole"})
RBAC_BINDING_KINDS = frozenset({"RoleBinding", "ClusterRoleBinding"})
# Kinds where spec.selector.matchLabels must select the pod template's own
# labels (spec.template.metadata.labels), in the same document. Job/CronJob
# are excluded: their selector is normally auto-populated/immutable rather
# than hand-written, so a mismatch there isn't the same kind of human error.
WORKLOAD_SELECTOR_KINDS = frozenset({"Deployment", "StatefulSet", "DaemonSet", "ReplicaSet"})

SENSITIVE_HOST_PATHS = (
    "/",
    "/etc",
    "/var/run/docker.sock",
    "/proc",
    "/root",
    "/var/lib/kubelet",
    "/boot",
    "/sys",
    "/home",
)

DEFAULT_PINNED_TAG = "1.0.0"


def is_sensitive_host_path(path) -> bool:
    if not isinstance(path, str) or not path:
        return False
    p = path.rstrip("/") or "/"
    for sensitive in SENSITIVE_HOST_PATHS:
        s = sensitive.rstrip("/") or "/"
        if p == s or p.startswith(s + "/"):
            return True
    return False


def is_unpinned_image(image: str) -> bool:
    if "@" in image:  # digest-pinned
        return False
    tail = image[image.rfind("/") + 1 :]
    if ":" not in tail:
        return True
    return tail.rsplit(":", 1)[1] == "latest"


def split_image(image: str) -> tuple[str, str | None]:
    """Splits `image` into (repository, tag). tag is None if absent."""
    tail = image[image.rfind("/") + 1 :]
    if ":" not in tail:
        return image, None
    repo_len = len(image) - len(tail)
    name, tag = tail.rsplit(":", 1)
    return image[:repo_len] + name, tag


def get_pod_spec(doc) -> tuple[dict | None, str]:
    """Returns (pod_spec_dict, json_pointer_prefix) for the document's kind,
    or (None, "") if the kind has no PodSpec or the structure is missing."""
    if not isinstance(doc, dict):
        return None, ""
    kind = doc.get("kind")
    spec = doc.get("spec")

    if kind == "Pod":
        if isinstance(spec, dict):
            return spec, "/spec"
        return None, ""

    if kind in ("Deployment", "StatefulSet", "DaemonSet", "ReplicaSet", "Job"):
        if isinstance(spec, dict) and isinstance(spec.get("template"), dict):
            tspec = spec["template"].get("spec")
            if isinstance(tspec, dict):
                return tspec, "/spec/template/spec"
        return None, ""

    if kind == "CronJob":
        if isinstance(spec, dict) and isinstance(spec.get("jobTemplate"), dict):
            jspec = spec["jobTemplate"].get("spec")
            if isinstance(jspec, dict) and isinstance(jspec.get("template"), dict):
                tspec = jspec["template"].get("spec")
                if isinstance(tspec, dict):
                    return tspec, "/spec/jobTemplate/spec/template/spec"
        return None, ""

    return None, ""


def iter_containers(pod_spec: dict, prefix: str):
    """Iterates containers and initContainers of a PodSpec, yielding
    (container_json_pointer_prefix, container_dict)."""
    for list_key in ("containers", "initContainers"):
        containers = pod_spec.get(list_key)
        if not isinstance(containers, list):
            continue
        for i, container in enumerate(containers):
            if isinstance(container, dict):
                yield f"{prefix}/{list_key}/{i}", container


def get_selector_match_labels(doc) -> tuple[dict | None, str]:
    """Returns (matchLabels_dict, json_pointer_prefix) for
    spec.selector.matchLabels, or (None, "") if absent/not applicable."""
    if not isinstance(doc, dict) or doc.get("kind") not in WORKLOAD_SELECTOR_KINDS:
        return None, ""
    spec = doc.get("spec")
    if not isinstance(spec, dict):
        return None, ""
    selector = spec.get("selector")
    if not isinstance(selector, dict):
        return None, ""
    match_labels = selector.get("matchLabels")
    if not isinstance(match_labels, dict):
        return None, ""
    return match_labels, "/spec/selector/matchLabels"


def get_template_labels(doc) -> tuple[dict | None, str]:
    """Returns (labels_dict, json_pointer_prefix) for the pod template's own
    metadata.labels (spec.template.metadata.labels), or (None, "") if
    absent/not applicable."""
    if not isinstance(doc, dict) or doc.get("kind") not in WORKLOAD_SELECTOR_KINDS:
        return None, ""
    spec = doc.get("spec")
    if not isinstance(spec, dict) or not isinstance(spec.get("template"), dict):
        return None, ""
    metadata = spec["template"].get("metadata")
    if not isinstance(metadata, dict):
        return None, ""
    labels = metadata.get("labels")
    if not isinstance(labels, dict):
        return None, ""
    return labels, "/spec/template/metadata/labels"


def get_container_ports(container: dict) -> tuple[set[int], set[str]]:
    """Returns (declared_port_numbers, declared_port_names) for a container's
    spec.containers[].ports list."""
    numbers: set[int] = set()
    names: set[str] = set()
    ports = container.get("ports")
    if not isinstance(ports, list):
        return numbers, names
    for port in ports:
        if not isinstance(port, dict):
            continue
        if isinstance(port.get("containerPort"), int):
            numbers.add(port["containerPort"])
        if isinstance(port.get("name"), str):
            names.add(port["name"])
    return numbers, names


PROBE_FIELDS = ("livenessProbe", "readinessProbe", "startupProbe")

_QUANTITY_BINARY_SUFFIXES = {"Ki": 2**10, "Mi": 2**20, "Gi": 2**30, "Ti": 2**40, "Pi": 2**50, "Ei": 2**60}
_QUANTITY_DECIMAL_SUFFIXES = {"n": 1e-9, "u": 1e-6, "m": 1e-3, "k": 1e3, "M": 1e6, "G": 1e9, "T": 1e12, "P": 1e15, "E": 1e18}


def parse_quantity(value) -> float | None:
    """Parses a Kubernetes resource quantity (e.g. "500m", "1Gi", "2") into
    a float in base units (cores for cpu, bytes for memory). Returns None if
    the value can't be parsed."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if not isinstance(value, str) or not value.strip():
        return None
    s = value.strip()
    for suffix, mult in _QUANTITY_BINARY_SUFFIXES.items():
        if s.endswith(suffix):
            try:
                return float(s[: -len(suffix)]) * mult
            except ValueError:
                return None
    for suffix, mult in _QUANTITY_DECIMAL_SUFFIXES.items():
        if s.endswith(suffix):
            try:
                return float(s[: -len(suffix)]) * mult
            except ValueError:
                return None
    try:
        return float(s)
    except ValueError:
        return None
