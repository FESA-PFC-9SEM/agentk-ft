import copy

from dataset.detect import detect_structural
from dataset.normalize import normalize_document


def test_hardens_privileged_and_root_and_caps():
    doc = {
        "apiVersion": "v1",
        "kind": "Pod",
        "metadata": {"name": "p"},
        "spec": {
            "hostNetwork": True,
            "containers": [
                {
                    "name": "app",
                    "image": "myapp",
                    "securityContext": {
                        "privileged": True,
                        "runAsUser": 0,
                        "allowPrivilegeEscalation": True,
                        "capabilities": {"add": ["SYS_ADMIN"]},
                    },
                }
            ],
        },
    }
    canonical = normalize_document(doc)
    assert canonical is not None
    assert detect_structural(canonical) == []
    sc = canonical["spec"]["containers"][0]["securityContext"]
    assert "privileged" not in sc
    assert "runAsUser" not in sc
    assert sc["allowPrivilegeEscalation"] is False
    assert "capabilities" not in sc
    assert "hostNetwork" not in canonical["spec"]
    assert canonical["spec"]["containers"][0]["image"] == "myapp:1.0.0"


def test_absent_security_context_untouched():
    doc = {
        "apiVersion": "v1",
        "kind": "Pod",
        "metadata": {"name": "p"},
        "spec": {"containers": [{"name": "app", "image": "myapp:1.0.0"}]},
    }
    canonical = normalize_document(doc)
    assert "securityContext" not in canonical["spec"]["containers"][0]
    assert canonical == doc  # nothing should change


def test_sensitive_hostpath_volume_and_mount_removed():
    doc = {
        "apiVersion": "v1",
        "kind": "Pod",
        "metadata": {"name": "p"},
        "spec": {
            "containers": [
                {
                    "name": "app",
                    "image": "myapp:1.0.0",
                    "volumeMounts": [{"name": "docker", "mountPath": "/var/run/docker.sock"}],
                }
            ],
            "volumes": [{"name": "docker", "hostPath": {"path": "/var/run/docker.sock"}}],
        },
    }
    canonical = normalize_document(doc)
    assert "volumes" not in canonical["spec"]
    assert "volumeMounts" not in canonical["spec"]["containers"][0]
    assert detect_structural(canonical) == []


def test_benign_hostpath_kept():
    doc = {
        "apiVersion": "v1",
        "kind": "Pod",
        "metadata": {"name": "p"},
        "spec": {
            "containers": [{"name": "app", "image": "myapp:1.0.0"}],
            "volumes": [{"name": "data", "hostPath": {"path": "/data/app"}}],
        },
    }
    canonical = normalize_document(doc)
    assert canonical["spec"]["volumes"][0]["hostPath"]["path"] == "/data/app"


def test_cronjob_nested_hardened():
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
                                {
                                    "name": "app",
                                    "image": "myapp:latest",
                                    "securityContext": {"privileged": True},
                                }
                            ]
                        }
                    }
                }
            }
        },
    }
    canonical = normalize_document(doc)
    assert detect_structural(canonical) == []
    tspec = canonical["spec"]["jobTemplate"]["spec"]["template"]["spec"]
    assert tspec["containers"][0]["image"] == "myapp:1.0.0"


def test_rbac_wildcard_rule_stripped():
    doc = {
        "apiVersion": "rbac.authorization.k8s.io/v1",
        "kind": "Role",
        "metadata": {"name": "r"},
        "rules": [
            {"apiGroups": ["*"], "resources": ["*"], "verbs": ["*"]},
            {"apiGroups": [""], "resources": ["pods"], "verbs": ["get"]},
        ],
    }
    canonical = normalize_document(doc)
    assert canonical is not None
    assert len(canonical["rules"]) == 1
    assert detect_structural(canonical) == []


def test_rbac_role_dropped_when_all_rules_wildcard():
    doc = {
        "apiVersion": "rbac.authorization.k8s.io/v1",
        "kind": "Role",
        "metadata": {"name": "r"},
        "rules": [{"apiGroups": ["*"], "resources": ["*"], "verbs": ["*"]}],
    }
    assert normalize_document(doc) is None


def test_cluster_admin_binding_dropped():
    doc = {
        "apiVersion": "rbac.authorization.k8s.io/v1",
        "kind": "ClusterRoleBinding",
        "metadata": {"name": "b"},
        "roleRef": {"kind": "ClusterRole", "name": "cluster-admin", "apiGroup": "rbac.authorization.k8s.io"},
        "subjects": [{"kind": "ServiceAccount", "name": "sa", "namespace": "default"}],
    }
    assert normalize_document(doc) is None


def test_deterministic_and_idempotent():
    doc = {
        "apiVersion": "v1",
        "kind": "Pod",
        "metadata": {"name": "p"},
        "spec": {
            "containers": [
                {"name": "app", "image": "myapp", "securityContext": {"privileged": True}}
            ]
        },
    }
    once = normalize_document(copy.deepcopy(doc))
    twice = normalize_document(copy.deepcopy(doc))
    assert once == twice
    assert normalize_document(copy.deepcopy(once)) == once


def test_does_not_mutate_input():
    doc = {
        "apiVersion": "v1",
        "kind": "Pod",
        "metadata": {"name": "p"},
        "spec": {
            "containers": [
                {"name": "app", "image": "myapp", "securityContext": {"privileged": True}}
            ]
        },
    }
    original = copy.deepcopy(doc)
    normalize_document(doc)
    assert doc == original
