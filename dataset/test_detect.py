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


def _pod(container_extra=None, pod_spec_extra=None):
    container = {"name": "app", "image": "myapp:1.2.3"}
    if container_extra:
        container.update(container_extra)
    spec = {"containers": [container]}
    if pod_spec_extra:
        spec.update(pod_spec_extra)
    return {"apiVersion": "v1", "kind": "Pod", "metadata": {"name": "p"}, "spec": spec}


def test_ksec001_clean_pod():
    doc = _pod({"env": [{"name": "SESSION_TIMEOUT", "value": "3600"}]})
    assert detect_ksec001(doc) == []


def test_ksec001_dirty_pod():
    doc = _pod({"env": [{"name": "DB_PASSWORD", "value": "S3cr3tR34lLeak99"}]})
    findings = detect_ksec001(doc)
    assert len(findings) == 1
    assert findings[0].rule_id == "KSEC-001"
    assert findings[0].evidence == "S3cr***"


def test_ksec002_privileged():
    doc = _pod({"securityContext": {"privileged": True}})
    findings = detect_ksec002(doc)
    assert any(f.path.endswith("/privileged") for f in findings)


def test_ksec002_run_as_root():
    doc = _pod({"securityContext": {"runAsUser": 0}})
    findings = detect_ksec002(doc)
    assert any(f.path.endswith("/runAsUser") for f in findings)


def test_ksec002_clean_container_is_not_flagged():
    doc = _pod({"securityContext": {"runAsNonRoot": True}})
    assert detect_ksec002(doc) == []


def test_ksec002_absent_security_context_is_not_flagged():
    doc = _pod()
    assert detect_ksec002(doc) == []


def test_ksec003_host_network():
    doc = _pod(pod_spec_extra={"hostNetwork": True})
    findings = detect_ksec003(doc)
    assert any(f.path.endswith("/hostNetwork") for f in findings)


def test_ksec003_sensitive_hostpath():
    doc = _pod(
        pod_spec_extra={
            "volumes": [{"name": "docker", "hostPath": {"path": "/var/run/docker.sock"}}]
        }
    )
    findings = detect_ksec003(doc)
    assert len(findings) == 1
    assert findings[0].severity == "critical"


def test_ksec003_benign_hostpath_not_flagged():
    doc = _pod(pod_spec_extra={"volumes": [{"name": "data", "hostPath": {"path": "/data/app"}}]})
    assert detect_ksec003(doc) == []


def test_ksec004_wildcard_role():
    doc = {
        "apiVersion": "rbac.authorization.k8s.io/v1",
        "kind": "Role",
        "metadata": {"name": "r"},
        "rules": [{"apiGroups": ["*"], "resources": ["*"], "verbs": ["*"]}],
    }
    findings = detect_ksec004(doc)
    assert len(findings) == 3  # apiGroups, resources, verbs


def test_ksec004_cluster_admin_binding():
    doc = {
        "apiVersion": "rbac.authorization.k8s.io/v1",
        "kind": "ClusterRoleBinding",
        "metadata": {"name": "b"},
        "roleRef": {"kind": "ClusterRole", "name": "cluster-admin", "apiGroup": "rbac.authorization.k8s.io"},
        "subjects": [{"kind": "ServiceAccount", "name": "sa", "namespace": "default"}],
    }
    findings = detect_ksec004(doc)
    assert len(findings) == 1
    assert findings[0].severity == "critical"


def test_ksec004_scoped_role_not_flagged():
    doc = {
        "apiVersion": "rbac.authorization.k8s.io/v1",
        "kind": "Role",
        "metadata": {"name": "r"},
        "rules": [{"apiGroups": [""], "resources": ["pods"], "verbs": ["get", "list"]}],
    }
    assert detect_ksec004(doc) == []


def test_ksec005_latest_tag():
    doc = _pod({"image": "myapp:latest"})
    findings = detect_ksec005(doc)
    assert len(findings) == 1


def test_ksec005_missing_tag():
    doc = _pod({"image": "myapp"})
    findings = detect_ksec005(doc)
    assert len(findings) == 1


def test_ksec005_pinned_tag_not_flagged():
    doc = _pod({"image": "myapp:1.2.3"})
    assert detect_ksec005(doc) == []


def test_ksec005_digest_pinned_not_flagged():
    doc = _pod({"image": "myapp@sha256:" + "a" * 64})
    assert detect_ksec005(doc) == []


def test_ksec005_registry_with_port_and_no_tag():
    doc = _pod({"image": "registry.internal:5000/myapp"})
    findings = detect_ksec005(doc)
    assert len(findings) == 1


def test_ksec005_registry_with_port_and_tag_is_clean():
    doc = _pod({"image": "registry.internal:5000/myapp:1.0.0"})
    assert detect_ksec005(doc) == []


def test_cronjob_nested_pod_spec():
    doc = {
        "apiVersion": "batch/v1",
        "kind": "CronJob",
        "metadata": {"name": "c"},
        "spec": {
            "jobTemplate": {
                "spec": {
                    "template": {
                        "spec": {
                            "containers": [
                                {"name": "app", "image": "myapp:latest", "securityContext": {"privileged": True}}
                            ]
                        }
                    }
                }
            }
        },
    }
    img_findings = detect_ksec005(doc)
    sc_findings = detect_ksec002(doc)
    assert len(img_findings) == 1
    assert img_findings[0].path == "/spec/jobTemplate/spec/template/spec/containers/0/image"
    assert len(sc_findings) == 1


def _deployment(selector_labels, template_labels, container_extra=None):
    container = {"name": "app", "image": "myapp:1.2.3"}
    if container_extra:
        container.update(container_extra)
    return {
        "apiVersion": "apps/v1",
        "kind": "Deployment",
        "metadata": {"name": "d"},
        "spec": {
            "selector": {"matchLabels": selector_labels},
            "template": {
                "metadata": {"labels": template_labels},
                "spec": {"containers": [container]},
            },
        },
    }


def test_ksec006_matching_selector_not_flagged():
    doc = _deployment({"app": "x"}, {"app": "x", "tier": "web"})
    assert detect_ksec006(doc) == []


def test_ksec006_mismatched_selector_flagged():
    doc = _deployment({"app": "x"}, {"app": "y"})
    findings = detect_ksec006(doc)
    assert len(findings) == 1
    assert findings[0].path == "/spec/selector/matchLabels/app"


def test_ksec006_not_applicable_to_pod():
    assert detect_ksec006(_pod()) == []


def test_ksec006_label_key_with_slash_is_escaped_in_path():
    # label keys routinely contain '/' (app.kubernetes.io/name) -- the path
    # must escape it per RFC 6901 or the JSON Pointer is unparseable.
    doc = _deployment({"app.kubernetes.io/name": "x"}, {"app.kubernetes.io/name": "y"})
    findings = detect_ksec006(doc)
    assert len(findings) == 1
    assert findings[0].path == "/spec/selector/matchLabels/app.kubernetes.io~1name"


def test_ksec007_port_mismatch_flagged():
    doc = _pod({"ports": [{"containerPort": 8080}], "livenessProbe": {"httpGet": {"path": "/health", "port": 9090}}})
    findings = detect_ksec007(doc)
    assert len(findings) == 1
    assert findings[0].path == "/spec/containers/0/livenessProbe/httpGet/port"


def test_ksec007_matching_port_not_flagged():
    doc = _pod({"ports": [{"containerPort": 8080}], "livenessProbe": {"httpGet": {"path": "/health", "port": 8080}}})
    assert detect_ksec007(doc) == []


def test_ksec007_named_port_mismatch_flagged():
    doc = _pod(
        {
            "ports": [{"containerPort": 8080, "name": "http"}],
            "readinessProbe": {"tcpSocket": {"port": "grpc"}},
        }
    )
    findings = detect_ksec007(doc)
    assert len(findings) == 1


def test_ksec007_no_declared_ports_not_flagged():
    # sem ports declaradas, nao ha base para comparar -- evita falso positivo
    doc = _pod({"livenessProbe": {"httpGet": {"path": "/health", "port": 9090}}})
    assert detect_ksec007(doc) == []


def test_ksec008_requests_exceed_limits_flagged():
    doc = _pod({"resources": {"requests": {"cpu": "1000m"}, "limits": {"cpu": "500m"}}})
    findings = detect_ksec008(doc)
    assert len(findings) == 1
    assert findings[0].path == "/spec/containers/0/resources/requests/cpu"


def test_ksec008_requests_within_limits_not_flagged():
    doc = _pod({"resources": {"requests": {"cpu": "250m", "memory": "128Mi"}, "limits": {"cpu": "500m", "memory": "256Mi"}}})
    assert detect_ksec008(doc) == []


def test_ksec008_memory_binary_suffix_comparison():
    doc = _pod({"resources": {"requests": {"memory": "2Gi"}, "limits": {"memory": "1024Mi"}}})
    findings = detect_ksec008(doc)
    assert len(findings) == 1


def test_ksec008_missing_limits_not_flagged():
    doc = _pod({"resources": {"requests": {"cpu": "500m"}}})
    assert detect_ksec008(doc) == []


def test_ksec009_dangling_volume_mount_flagged():
    doc = _pod(
        {"volumeMounts": [{"name": "cofnig-volume", "mountPath": "/etc/app"}]},
        pod_spec_extra={"volumes": [{"name": "config-volume", "configMap": {"name": "app-config"}}]},
    )
    findings = detect_ksec009(doc)
    assert len(findings) == 1
    assert findings[0].path == "/spec/containers/0/volumeMounts/0/name"


def test_ksec009_matching_volume_mount_not_flagged():
    doc = _pod(
        {"volumeMounts": [{"name": "config-volume", "mountPath": "/etc/app"}]},
        pod_spec_extra={"volumes": [{"name": "config-volume", "configMap": {"name": "app-config"}}]},
    )
    assert detect_ksec009(doc) == []
