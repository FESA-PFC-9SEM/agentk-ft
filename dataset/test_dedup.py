from dataset.dedup import dedup, skeleton_hash


def _pod(name, namespace, image="myapp:1.0.0"):
    return {
        "apiVersion": "v1",
        "kind": "Pod",
        "metadata": {"name": name, "namespace": namespace, "labels": {"app": name}},
        "spec": {"containers": [{"name": "app", "image": image}]},
    }


def test_same_shape_different_names_collide():
    a = _pod("service-a", "team-x")
    b = _pod("service-b", "team-y")
    assert skeleton_hash(a) == skeleton_hash(b)


def test_different_kind_does_not_collide():
    pod = _pod("service-a", "team-x")
    deploy = {
        "apiVersion": "apps/v1",
        "kind": "Deployment",
        "metadata": {"name": "service-a"},
        "spec": {"template": {"spec": {"containers": [{"name": "app", "image": "myapp:1.0.0"}]}}},
    }
    assert skeleton_hash(pod) != skeleton_hash(deploy)


def test_different_container_count_does_not_collide():
    a = _pod("s", "ns")
    b = _pod("s", "ns")
    b["spec"]["containers"].append({"name": "sidecar", "image": "sidecar:1.0.0"})
    assert skeleton_hash(a) != skeleton_hash(b)


def test_dedup_reports_survival_rate():
    docs = [_pod("a", "ns"), _pod("b", "ns"), _pod("c", "other-ns")]
    kept, rate = dedup(docs)
    assert kept == [0]
    assert rate == 1 / 3
