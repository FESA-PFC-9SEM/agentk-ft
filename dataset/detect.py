"""
Structural, read-only detectors for the KSEC rules. This module is used in
three different places in the pipeline: to drop corpus manifests that already
violate KSEC-001 (a real secret), as a precondition for the mutators in
mutate.py (guarantees the target field is clean before injecting the
defect), and as a post-normalize assertion (guarantees normalize.py actually
produced a canonical form with no structural findings). Writing the logic
once here avoids three divergent copies of the same rule.

KSEC-006..009 are semantic/configuration-correctness checks, not security
checks in the strict sense. Unlike 001-005, normalize.py does NOT guarantee
the corpus is clean for them (see mutate.py for why), so they're kept out of
detect_structural -- that function specifically backs the post-normalize
assertion in build.py and would break it if extended here.
"""

from __future__ import annotations

from dataset import scanning
from dataset.k8s import (
    PROBE_FIELDS,
    RBAC_BINDING_KINDS,
    RBAC_ROLE_KINDS,
    get_container_ports,
    get_pod_spec,
    get_selector_match_labels,
    get_template_labels,
    is_sensitive_host_path,
    is_unpinned_image,
    iter_containers,
    parse_quantity,
)
from dataset.schema import Finding, escape_json_pointer_token, mask_evidence


def detect_ksec001(doc, doc_index: int = 0) -> list[Finding]:
    findings = []
    for hit in scanning.find_secrets(doc):
        findings.append(
            Finding(
                rule_id="KSEC-001",
                severity="critical",
                doc=doc_index,
                path=hit.path,
                message=f"Plaintext credential: {hit.reason}",
                evidence=mask_evidence(hit.value),
            )
        )
    return findings


def detect_ksec002(doc, doc_index: int = 0) -> list[Finding]:
    findings = []
    pod_spec, prefix = get_pod_spec(doc)
    if pod_spec is None:
        return findings

    pod_sc = pod_spec.get("securityContext")
    if isinstance(pod_sc, dict) and pod_sc.get("runAsUser") == 0:
        findings.append(
            Finding(
                "KSEC-002",
                "high",
                doc_index,
                f"{prefix}/securityContext/runAsUser",
                "Pod is configured to run as root (uid 0)",
                mask_evidence("0"),
            )
        )

    for cpath, container in iter_containers(pod_spec, prefix):
        sc = container.get("securityContext")
        if not isinstance(sc, dict):
            continue
        if sc.get("privileged") is True:
            findings.append(
                Finding(
                    "KSEC-002",
                    "critical",
                    doc_index,
                    f"{cpath}/securityContext/privileged",
                    "Container runs in privileged mode",
                    mask_evidence("true"),
                )
            )
        if sc.get("runAsUser") == 0:
            findings.append(
                Finding(
                    "KSEC-002",
                    "high",
                    doc_index,
                    f"{cpath}/securityContext/runAsUser",
                    "Container is configured to run as root (uid 0)",
                    mask_evidence("0"),
                )
            )
        if sc.get("allowPrivilegeEscalation") is True:
            findings.append(
                Finding(
                    "KSEC-002",
                    "medium",
                    doc_index,
                    f"{cpath}/securityContext/allowPrivilegeEscalation",
                    "allowPrivilegeEscalation is explicitly enabled",
                    mask_evidence("true"),
                )
            )
        caps = sc.get("capabilities")
        if isinstance(caps, dict) and caps.get("add"):
            findings.append(
                Finding(
                    "KSEC-002",
                    "high",
                    doc_index,
                    f"{cpath}/securityContext/capabilities/add",
                    f"Added capabilities: {caps['add']}",
                    mask_evidence(str(caps["add"])),
                )
            )
    return findings


def detect_ksec003(doc, doc_index: int = 0) -> list[Finding]:
    findings = []
    pod_spec, prefix = get_pod_spec(doc)
    if pod_spec is None:
        return findings

    for field in ("hostNetwork", "hostPID", "hostIPC"):
        if pod_spec.get(field) is True:
            findings.append(
                Finding(
                    "KSEC-003",
                    "high",
                    doc_index,
                    f"{prefix}/{field}",
                    f"{field} is enabled: the Pod shares the host namespace",
                    mask_evidence("true"),
                )
            )

    volumes = pod_spec.get("volumes")
    if isinstance(volumes, list):
        for i, volume in enumerate(volumes):
            if not isinstance(volume, dict):
                continue
            host_path = volume.get("hostPath")
            if isinstance(host_path, dict) and is_sensitive_host_path(host_path.get("path")):
                findings.append(
                    Finding(
                        "KSEC-003",
                        "critical",
                        doc_index,
                        f"{prefix}/volumes/{i}/hostPath/path",
                        "hostPath volume points to a sensitive host path",
                        mask_evidence(host_path.get("path", "")),
                    )
                )
    return findings


def detect_ksec004(doc, doc_index: int = 0) -> list[Finding]:
    findings = []
    if not isinstance(doc, dict):
        return findings
    kind = doc.get("kind")

    if kind in RBAC_ROLE_KINDS:
        rules = doc.get("rules")
        if isinstance(rules, list):
            for i, rule in enumerate(rules):
                if not isinstance(rule, dict):
                    continue
                for field in ("apiGroups", "resources", "verbs"):
                    values = rule.get(field)
                    if isinstance(values, list) and "*" in values:
                        findings.append(
                            Finding(
                                "KSEC-004",
                                "high",
                                doc_index,
                                f"/rules/{i}/{field}",
                                f"RBAC rule uses a wildcard in '{field}'",
                                mask_evidence("*"),
                            )
                        )

    elif kind in RBAC_BINDING_KINDS:
        role_ref = doc.get("roleRef")
        if isinstance(role_ref, dict) and role_ref.get("name") == "cluster-admin":
            findings.append(
                Finding(
                    "KSEC-004",
                    "critical",
                    doc_index,
                    "/roleRef/name",
                    "Binding grants the cluster-admin role",
                    mask_evidence("cluster-admin"),
                )
            )
    return findings


def detect_ksec005(doc, doc_index: int = 0) -> list[Finding]:
    findings = []
    pod_spec, prefix = get_pod_spec(doc)
    if pod_spec is None:
        return findings
    for cpath, container in iter_containers(pod_spec, prefix):
        image = container.get("image")
        if isinstance(image, str) and image and is_unpinned_image(image):
            findings.append(
                Finding(
                    "KSEC-005",
                    "medium",
                    doc_index,
                    f"{cpath}/image",
                    f"Container image has no pinned tag: {image}",
                    mask_evidence(image),
                )
            )
    return findings


def detect_ksec006(doc, doc_index: int = 0) -> list[Finding]:
    findings = []
    match_labels, sel_prefix = get_selector_match_labels(doc)
    if match_labels is None:
        return findings
    template_labels, _ = get_template_labels(doc)
    if template_labels is None:
        return findings
    for key, value in match_labels.items():
        if template_labels.get(key) != value:
            findings.append(
                Finding(
                    "KSEC-006",
                    "high",
                    doc_index,
                    f"{sel_prefix}/{escape_json_pointer_token(key)}",
                    f"Selector label '{key}' does not match the pod template's labels",
                    mask_evidence(str(value)),
                )
            )
    return findings


def detect_ksec007(doc, doc_index: int = 0) -> list[Finding]:
    findings = []
    pod_spec, prefix = get_pod_spec(doc)
    if pod_spec is None:
        return findings
    for cpath, container in iter_containers(pod_spec, prefix):
        port_numbers, port_names = get_container_ports(container)
        for probe_field in PROBE_FIELDS:
            probe = container.get(probe_field)
            if not isinstance(probe, dict):
                continue
            for check_field in ("httpGet", "tcpSocket"):
                check = probe.get(check_field)
                if not isinstance(check, dict):
                    continue
                port = check.get("port")
                path = f"{cpath}/{probe_field}/{check_field}/port"
                if isinstance(port, bool):
                    continue
                if isinstance(port, int) and port_numbers and port not in port_numbers:
                    findings.append(
                        Finding(
                            "KSEC-007",
                            "medium",
                            doc_index,
                            path,
                            f"{probe_field} targets port {port}, which is not declared in the container's ports",
                            mask_evidence(str(port)),
                        )
                    )
                elif isinstance(port, str) and port_names and port not in port_names:
                    findings.append(
                        Finding(
                            "KSEC-007",
                            "medium",
                            doc_index,
                            path,
                            f"{probe_field} targets named port '{port}', which is not declared in the container's ports",
                            mask_evidence(port),
                        )
                    )
    return findings


def detect_ksec008(doc, doc_index: int = 0) -> list[Finding]:
    findings = []
    pod_spec, prefix = get_pod_spec(doc)
    if pod_spec is None:
        return findings
    for cpath, container in iter_containers(pod_spec, prefix):
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
            req_num = parse_quantity(req_val)
            lim_num = parse_quantity(lim_val)
            if req_num is None or lim_num is None:
                continue
            if req_num > lim_num:
                findings.append(
                    Finding(
                        "KSEC-008",
                        "medium",
                        doc_index,
                        f"{cpath}/resources/requests/{resource_name}",
                        f"resources.requests.{resource_name} ({req_val}) exceeds "
                        f"resources.limits.{resource_name} ({lim_val})",
                        mask_evidence(str(req_val)),
                    )
                )
    return findings


def detect_ksec009(doc, doc_index: int = 0) -> list[Finding]:
    findings = []
    pod_spec, prefix = get_pod_spec(doc)
    if pod_spec is None:
        return findings
    volumes = pod_spec.get("volumes")
    volume_names = {v.get("name") for v in volumes if isinstance(v, dict)} if isinstance(volumes, list) else set()
    for cpath, container in iter_containers(pod_spec, prefix):
        mounts = container.get("volumeMounts")
        if not isinstance(mounts, list):
            continue
        for i, mount in enumerate(mounts):
            if not isinstance(mount, dict):
                continue
            name = mount.get("name")
            if isinstance(name, str) and name not in volume_names:
                findings.append(
                    Finding(
                        "KSEC-009",
                        "high",
                        doc_index,
                        f"{cpath}/volumeMounts/{i}/name",
                        f"volumeMount references undefined volume '{name}'",
                        mask_evidence(name),
                    )
                )
    return findings


_STRUCTURAL_DETECTORS = (detect_ksec002, detect_ksec003, detect_ksec004, detect_ksec005)
_SEMANTIC_DETECTORS = (detect_ksec006, detect_ksec007, detect_ksec008, detect_ksec009)


def detect_structural(doc, doc_index: int = 0) -> list[Finding]:
    """Findings for rules 002-005 only (no secret scan). Used by the
    post-normalize assertion and by the preconditions of mutators 002-005."""
    findings: list[Finding] = []
    for detector in _STRUCTURAL_DETECTORS:
        findings.extend(detector(doc, doc_index))
    return findings


def detect_semantic(doc, doc_index: int = 0) -> list[Finding]:
    """Findings for rules 006-009 only. Kept separate from detect_structural
    since normalize.py does not guarantee the corpus is clean for these."""
    findings: list[Finding] = []
    for detector in _SEMANTIC_DETECTORS:
        findings.extend(detector(doc, doc_index))
    return findings


def detect_all(doc, doc_index: int = 0) -> list[Finding]:
    return detect_ksec001(doc, doc_index) + detect_structural(doc, doc_index) + detect_semantic(doc, doc_index)
