"""
Structural, read-only detectors for the KSEC rules. This module is used in
three different places in the pipeline: to drop corpus manifests that already
violate KSEC-001 (a real secret), as a precondition for the mutators in
mutate.py (guarantees the target field is clean before injecting the
defect), and as a post-normalize assertion (guarantees normalize.py actually
produced a canonical form with no structural findings). Writing the logic
once here avoids three divergent copies of the same rule.

KSEC-006..011 are semantic/configuration-correctness checks, not security
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
    get_service_selector,
    label_selector_matches,
    pod_label_sets,
    NON_HTTP_IMAGE_PORTS,
    declared_volume_names,
    get_template_labels,
    image_basename,
    is_sensitive_host_path,
    is_unpinned_image,
    iter_containers,
    parse_quantity,
    required_env_for_container,
)
from dataset.names import command_binary, container_strings, misspelled_binary, misspelled_path, paths_in
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
    volume_names = declared_volume_names(doc, pod_spec)
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


def detect_ksec010(doc, doc_index: int = 0) -> list[Finding]:
    """httpGet probe on a container whose image is a non-HTTP server
    (postgres, redis, ...). Any httpGet there is wrong regardless of port:
    the container only runs that server, so nothing answers HTTP."""
    findings = []
    pod_spec, prefix = get_pod_spec(doc)
    if pod_spec is None:
        return findings
    for cpath, container in iter_containers(pod_spec, prefix):
        name = image_basename(container.get("image"))
        if name not in NON_HTTP_IMAGE_PORTS:
            continue
        for probe_field in PROBE_FIELDS:
            probe = container.get(probe_field)
            if not isinstance(probe, dict) or not isinstance(probe.get("httpGet"), dict):
                continue
            # A failing liveness/startup probe restarts the container forever;
            # a failing readiness probe "only" keeps it out of Service endpoints.
            severity = "medium" if probe_field == "readinessProbe" else "high"
            findings.append(
                Finding(
                    "KSEC-010",
                    severity,
                    doc_index,
                    f"{cpath}/{probe_field}/httpGet",
                    f"{probe_field} uses httpGet against a {name} container, which does not speak HTTP -- "
                    "the probe can never succeed (use tcpSocket or exec)",
                    mask_evidence(container["image"]),
                )
            )
    return findings


def detect_ksec011(doc, doc_index: int = 0) -> list[Finding]:
    """Database server container missing an env var its entrypoint requires
    (see k8s.IMAGE_ENV_CONTRACTS) -- the container exits at startup. One
    finding per unmet requirement (mssql can miss both ACCEPT_EULA and the
    SA password)."""
    findings = []
    pod_spec, prefix = get_pod_spec(doc)
    if pod_spec is None:
        return findings
    for cpath, container in iter_containers(pod_spec, prefix):
        applicable = required_env_for_container(cpath, container, pod_spec)
        if applicable is None:
            continue
        label, requirements = applicable
        env = container.get("env")
        env_names = {e.get("name") for e in env if isinstance(e, dict)} if isinstance(env, list) else set()
        for requirement in requirements:
            if env_names & requirement.accepted:
                continue
            findings.append(
                Finding(
                    "KSEC-011",
                    "high",
                    doc_index,
                    f"{cpath}/env",
                    f"{label} image requires {requirement.primary} (or an equivalent) to be set -- "
                    "the container will exit at startup",
                    mask_evidence(container["image"]),
                )
            )
    return findings


def detect_ksec012(doc, doc_index: int = 0) -> list[Finding]:
    """A command binary or file path that's a rare one-edit variant of a very
    common one in the corpus (python5 for python3, /hom/ for /home/), or a
    very common path under a wrong first directory -- see dataset/names.py.
    At most one finding per field."""
    findings = []
    pod_spec, prefix = get_pod_spec(doc)
    if pod_spec is None:
        return findings
    for cpath, container in iter_containers(pod_spec, prefix):
        flagged = set()
        binary = command_binary(container)
        if binary and (fix := misspelled_binary(binary)):
            pointer = f"{cpath}/command/0"
            flagged.add(pointer)
            findings.append(
                Finding(
                    "KSEC-012",
                    "high",
                    doc_index,
                    pointer,
                    f"Command '{binary}' looks like a misspelling of '{fix}' -- the container fails to start",
                    mask_evidence(binary),
                )
            )
        for pointer, text in container_strings(cpath, container):
            if pointer in flagged:
                continue
            for path in paths_in(text):
                fix = misspelled_path(path)
                if fix:
                    findings.append(
                        Finding(
                            "KSEC-012",
                            "high" if "/command/" in pointer else "medium",
                            doc_index,
                            pointer,
                            f"Path '{path}' looks like a misspelling of '{fix}'",
                            mask_evidence(path),
                        )
                    )
                    break
    return findings


_STRUCTURAL_DETECTORS = (detect_ksec002, detect_ksec003, detect_ksec004, detect_ksec005)
_SEMANTIC_DETECTORS = (
    detect_ksec006,
    detect_ksec007,
    detect_ksec008,
    detect_ksec009,
    detect_ksec010,
    detect_ksec011,
    detect_ksec012,
)


def detect_structural(doc, doc_index: int = 0) -> list[Finding]:
    """Findings for rules 002-005 only (no secret scan). Used by the
    post-normalize assertion and by the preconditions of mutators 002-005."""
    findings: list[Finding] = []
    for detector in _STRUCTURAL_DETECTORS:
        findings.extend(detector(doc, doc_index))
    return findings


def detect_semantic(doc, doc_index: int = 0) -> list[Finding]:
    """Findings for rules 006-011 only. Kept separate from detect_structural
    since normalize.py does not guarantee the corpus is clean for these."""
    findings: list[Finding] = []
    for detector in _SEMANTIC_DETECTORS:
        findings.extend(detector(doc, doc_index))
    return findings


def detect_all(doc, doc_index: int = 0) -> list[Finding]:
    return detect_ksec001(doc, doc_index) + detect_structural(doc, doc_index) + detect_semantic(doc, doc_index)


def closest_selected_workload(selector: dict, workloads: list[tuple[int, dict]]) -> tuple[int, dict] | None:
    """Among the workloads whose pod labels carry every key of `selector`,
    the one agreeing on the most values (first on a tie) -- the workload a
    mismatched Service was evidently meant for. None if no workload has all
    the keys: then the Service most likely targets a workload defined in
    another file, which a single file can't be checked against."""
    candidates = [(i, labels) for i, labels in workloads if all(k in labels for k in selector)]
    if not candidates:
        return None
    return max(candidates, key=lambda c: sum(str(c[1][k]) == str(v) for k, v in selector.items()))


def detect_ksec006_services(docs: list) -> list[Finding]:
    """KSEC-006 across documents of one file: a Service whose spec.selector
    matches the pod labels of NO workload in the file, while a workload with
    the same label keys is right there -- routing is broken by a typo'd
    value (orionlds vs orionld). One finding per selector key that differs
    from that workload's labels."""
    findings = []
    workloads = pod_label_sets(docs)
    if not workloads:
        return findings
    for i, doc in enumerate(docs):
        selector = get_service_selector(doc)
        if selector is None or any(label_selector_matches(selector, labels) for _, labels in workloads):
            continue
        target = closest_selected_workload(selector, workloads)
        if target is None:
            continue
        _, labels = target
        for key, value in selector.items():
            if str(labels[key]) != str(value):
                findings.append(
                    Finding(
                        "KSEC-006",
                        "high",
                        i,
                        f"/spec/selector/{escape_json_pointer_token(str(key))}",
                        f"Service selector label '{key}' matches no workload's pod labels in this file",
                        mask_evidence(str(value)),
                    )
                )
    return findings


def detect_file(docs: list) -> list[Finding]:
    """Every finding for a whole (possibly multi-document) file: each
    document's own findings, tagged with its index, plus the checks that
    need more than one document (see detect_ksec006_services)."""
    findings: list[Finding] = []
    for i, doc in enumerate(docs):
        if isinstance(doc, dict):
            findings.extend(detect_all(doc, i))
    findings.extend(detect_ksec006_services([d if isinstance(d, dict) else {} for d in docs]))
    return findings
