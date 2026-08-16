"""
Canonical form (gold target) of a manifest: applies exactly the fixes rules
KSEC-002..005 require, deterministically, and nothing beyond that. Real
secrets (KSEC-001) are NOT fixed here -- documents with a real secret are
dropped before normalize is even called (see build.py), because attempting
to "externalize" a real value would mean keeping that value in
memory/in the pipeline longer than necessary.

Deliberate policy: a field is only touched if it's already in an explicitly
insecure state (e.g. privileged=true, runAsUser=0). The absence of a
hardening field (no securityContext, no explicit allowPrivilegeEscalation) is
treated as compliant, not as a finding -- otherwise the vast majority of the
real corpus would be "dirty" before mutation even happens.

Rules that can't be fixed in place (RBAC wildcards that would empty out every
rule, binding to cluster-admin) make the whole document get dropped: returns
None.
"""

from __future__ import annotations

import copy

from dataset.k8s import (
    DEFAULT_PINNED_TAG,
    RBAC_BINDING_KINDS,
    RBAC_ROLE_KINDS,
    get_pod_spec,
    is_sensitive_host_path,
    is_unpinned_image,
    iter_containers,
    split_image,
)


def _cleanup_empty_dict(parent: dict, key: str) -> None:
    if key in parent and isinstance(parent[key], dict) and not parent[key]:
        del parent[key]


def _harden_container_security_context(sc: dict) -> None:
    if sc.get("privileged") is True:
        del sc["privileged"]
    if sc.get("runAsUser") == 0:
        del sc["runAsUser"]
    if sc.get("allowPrivilegeEscalation") is True:
        sc["allowPrivilegeEscalation"] = False
    caps = sc.get("capabilities")
    if isinstance(caps, dict) and caps.get("add"):
        del caps["add"]
        _cleanup_empty_dict(sc, "capabilities")


def _harden_pod_spec(pod_spec: dict) -> None:
    for field in ("hostNetwork", "hostPID", "hostIPC"):
        if pod_spec.get(field) is True:
            del pod_spec[field]

    pod_sc = pod_spec.get("securityContext")
    if isinstance(pod_sc, dict):
        if pod_sc.get("runAsUser") == 0:
            del pod_sc["runAsUser"]
        _cleanup_empty_dict(pod_spec, "securityContext")

    volumes = pod_spec.get("volumes")
    if isinstance(volumes, list):
        removed_names = set()
        kept_volumes = []
        for volume in volumes:
            if (
                isinstance(volume, dict)
                and isinstance(volume.get("hostPath"), dict)
                and is_sensitive_host_path(volume["hostPath"].get("path"))
            ):
                removed_names.add(volume.get("name"))
                continue
            kept_volumes.append(volume)
        pod_spec["volumes"] = kept_volumes
        if not pod_spec["volumes"]:
            del pod_spec["volumes"]
        if removed_names:
            _prune_volume_mounts(pod_spec, removed_names)

    for _cpath, container in iter_containers(pod_spec, ""):
        sc = container.get("securityContext")
        if isinstance(sc, dict):
            _harden_container_security_context(sc)
            _cleanup_empty_dict(container, "securityContext")

        image = container.get("image")
        if isinstance(image, str) and image and is_unpinned_image(image):
            repo, _tag = split_image(image)
            container["image"] = f"{repo}:{DEFAULT_PINNED_TAG}"


def _prune_volume_mounts(pod_spec: dict, removed_names: set) -> None:
    for _cpath, container in iter_containers(pod_spec, ""):
        mounts = container.get("volumeMounts")
        if not isinstance(mounts, list):
            continue
        container["volumeMounts"] = [
            m for m in mounts if not (isinstance(m, dict) and m.get("name") in removed_names)
        ]
        if not container["volumeMounts"]:
            del container["volumeMounts"]


def _harden_rbac_role(doc: dict) -> dict | None:
    rules = doc.get("rules")
    if not isinstance(rules, list):
        return doc
    kept_rules = []
    for rule in rules:
        if not isinstance(rule, dict):
            kept_rules.append(rule)
            continue
        is_wildcard = any(
            isinstance(rule.get(field), list) and "*" in rule[field]
            for field in ("apiGroups", "resources", "verbs")
        )
        if is_wildcard:
            continue
        kept_rules.append(rule)
    if not kept_rules:
        return None  # no way to infer a less-permissive rule: drop it
    doc["rules"] = kept_rules
    return doc


def _harden_rbac_binding(doc: dict) -> dict | None:
    role_ref = doc.get("roleRef")
    if isinstance(role_ref, dict) and role_ref.get("name") == "cluster-admin":
        return None  # no "correct" roleRef to infer: drop it
    return doc


def normalize_document(doc: dict) -> dict | None:
    """Takes an already-loaded document (dict) and returns its canonical
    form, or None if the document can't be safely fixed and should be
    dropped from the base corpus."""
    if not isinstance(doc, dict):
        return None

    doc = copy.deepcopy(doc)
    kind = doc.get("kind")

    pod_spec, _prefix = get_pod_spec(doc)
    if pod_spec is not None:
        _harden_pod_spec(pod_spec)

    if kind in RBAC_ROLE_KINDS:
        doc = _harden_rbac_role(doc)
    elif kind in RBAC_BINDING_KINDS:
        doc = _harden_rbac_binding(doc)

    return doc
