from demo.reset_cluster import clean


def test_owned_agent_created_and_cluster_managed_objects_are_skipped():
    assert clean({"apiVersion": "v1", "kind": "Pod", "metadata": {"name": "web-1", "ownerReferences": [{"kind": "ReplicaSet"}]}}) is None
    assert clean({"apiVersion": "v1", "kind": "ConfigMap", "metadata": {"name": "x", "labels": {"app.kubernetes.io/managed-by": "cluster-agent"}}}) is None
    assert clean({"apiVersion": "v1", "kind": "Service", "metadata": {"name": "kubernetes"}}) is None
    assert clean({"apiVersion": "v1", "kind": "ConfigMap", "metadata": {"name": "kube-root-ca.crt"}}) is None


def test_runtime_fields_are_stripped():
    service = {
        "apiVersion": "v1", "kind": "Service",
        "metadata": {"name": "web", "namespace": "demo", "uid": "u", "resourceVersion": "9",
                     "annotations": {"kubectl.kubernetes.io/last-applied-configuration": "{}", "team": "a"}},
        "spec": {"clusterIP": "10.0.0.1", "clusterIPs": ["10.0.0.1"], "ports": [{"port": 80}], "selector": {"app": "web"}},
        "status": {"loadBalancer": {}},
    }
    assert clean(service) == {
        "apiVersion": "v1", "kind": "Service", "metadata": {"name": "web", "annotations": {"team": "a"}},
        "spec": {"ports": [{"port": 80}], "selector": {"app": "web"}},
    }


def test_bare_pod_drops_node_and_injected_token_volume():
    pod = {
        "apiVersion": "v1", "kind": "Pod", "metadata": {"name": "mysql"},
        "spec": {
            "nodeName": "minikube",
            "containers": [{"name": "db", "image": "mysql", "volumeMounts": [{"name": "kube-api-access-abcde", "mountPath": "/var/run/secrets"}, {"name": "data", "mountPath": "/var/lib/mysql"}]}],
            "volumes": [{"name": "kube-api-access-abcde", "projected": {}}, {"name": "data", "emptyDir": {}}],
        },
    }
    spec = clean(pod)["spec"]
    assert "nodeName" not in spec
    assert spec["volumes"] == [{"name": "data", "emptyDir": {}}]
    assert spec["containers"][0]["volumeMounts"] == [{"name": "data", "mountPath": "/var/lib/mysql"}]


def test_job_drops_generated_selector_and_labels():
    job = {
        "apiVersion": "batch/v1", "kind": "Job", "metadata": {"name": "j", "labels": {"job-name": "j", "app": "x"}},
        "spec": {"selector": {"matchLabels": {"controller-uid": "1"}},
                 "template": {"metadata": {"labels": {"controller-uid": "1", "job-name": "j", "app": "x"}}, "spec": {"containers": []}}},
    }
    out = clean(job)
    assert "selector" not in out["spec"]
    assert out["spec"]["template"]["metadata"]["labels"] == {"app": "x"}
    assert out["metadata"]["labels"] == {"app": "x"}
