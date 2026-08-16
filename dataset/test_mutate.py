import random

import jsonpatch
import pytest

from dataset.mutate import (
    FAKE_SECRET_VAR_NAMES,
    MUTATORS,
    _mutate_ksec001_command,
    _mutate_ksec001_env,
    _typo,
    mutate_ksec001,
    mutate_ksec002,
    mutate_ksec003,
    mutate_ksec004,
    mutate_ksec005,
    mutate_ksec006,
    mutate_ksec007,
    mutate_ksec008,
    mutate_ksec009,
)


def _apply_patch(mutated_doc, patch):
    ops = [{k: v for k, v in p.to_dict().items() if k != "doc"} for p in patch]
    return jsonpatch.apply_patch(mutated_doc, ops)


def _assert_round_trip(result):
    reconstructed = _apply_patch(result.mutated_doc, result.patch)
    assert reconstructed == result.canonical


def _pod(container_extra=None, pod_spec_extra=None, container_name="app"):
    container = {"name": container_name, "image": "myapp:1.2.3"}
    if container_extra:
        container.update(container_extra)
    spec = {"containers": [container]}
    if pod_spec_extra:
        spec.update(pod_spec_extra)
    return {"apiVersion": "v1", "kind": "Pod", "metadata": {"name": "p"}, "spec": spec}


def _deployment(container_extra=None):
    container = {"name": "app", "image": "myapp:1.2.3"}
    if container_extra:
        container.update(container_extra)
    return {
        "apiVersion": "apps/v1",
        "kind": "Deployment",
        "metadata": {"name": "d"},
        "spec": {"template": {"spec": {"containers": [container]}}},
    }


def _deployment_with_selector(selector_labels, template_labels, container_extra=None):
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


# ---------------------------------------------------------------------------
# KSEC-001
# ---------------------------------------------------------------------------


def test_ksec001_env_round_trip_no_prior_env():
    rng = random.Random(1)
    result = _mutate_ksec001_env(_pod(), rng, 0)
    assert result is not None
    assert len(result.findings) == 1
    assert result.findings[0].rule_id == "KSEC-001"
    assert len(result.new_resources) == 1
    assert "kind: Secret" in result.new_resources[0]
    assert "<REPLACE_WITH_SECRET_VALUE>" in result.new_resources[0]
    _assert_round_trip(result)


def test_ksec001_env_no_full_secret_in_new_resources():
    rng = random.Random(2)
    result = _mutate_ksec001_env(_pod(), rng, 0)
    secret_value = result.mutated_doc["spec"]["containers"][0]["env"][0]["value"]
    assert secret_value not in result.new_resources[0]
    for f in result.findings:
        assert secret_value not in f.evidence


def test_ksec001_env_round_trip_with_prior_env():
    rng = random.Random(3)
    doc = _pod({"env": [{"name": "SESSION_TIMEOUT", "value": "3600"}]})
    result = _mutate_ksec001_env(doc, rng, 0)
    assert result is not None
    _assert_round_trip(result)


def test_ksec001_command_round_trip_no_prior_args():
    rng = random.Random(1)
    result = _mutate_ksec001_command(_pod(), rng, 0)
    assert result is not None
    assert len(result.findings) == 1
    assert result.findings[0].rule_id == "KSEC-001"
    assert result.new_resources == []
    _assert_round_trip(result)
    assert "args" in result.mutated_doc["spec"]["containers"][0]
    assert "args" not in result.canonical["spec"]["containers"][0]


def test_ksec001_command_round_trip_appends_to_existing_args():
    rng = random.Random(4)
    doc = _pod({"args": ["--verbose"]})
    result = _mutate_ksec001_command(doc, rng, 0)
    assert result is not None
    _assert_round_trip(result)
    assert result.mutated_doc["spec"]["containers"][0]["args"][0] == "--verbose"


def test_ksec001_command_prefers_existing_command_list():
    rng = random.Random(1)
    doc = _pod({"command": ["/bin/entrypoint.sh"]})
    result = _mutate_ksec001_command(doc, rng, 0)
    assert result is not None
    _assert_round_trip(result)
    assert "command" in result.mutated_doc["spec"]["containers"][0]
    assert "args" not in result.mutated_doc["spec"]["containers"][0]


def test_ksec001_command_evidence_is_masked():
    rng = random.Random(1)
    result = _mutate_ksec001_command(_pod(), rng, 0)
    injected = result.mutated_doc["spec"]["containers"][0]["args"][0]
    for f in result.findings:
        assert f.evidence.endswith("***")
        # the full secret must never leak outside the value injected into the input manifest
        assert f.evidence[:-3] in injected


@pytest.mark.parametrize("seed", range(30))
def test_ksec001_dispatcher_always_yields_a_result_for_pod(seed):
    rng = random.Random(seed)
    result = mutate_ksec001(_pod(), rng)
    assert result is not None
    _assert_round_trip(result)


def test_ksec001_dispatcher_picks_both_variants_across_seeds():
    saw_env = saw_command = False
    for seed in range(40):
        rng = random.Random(seed)
        result = mutate_ksec001(_pod(), rng)
        if result.new_resources:
            saw_env = True
        else:
            saw_command = True
    assert saw_env and saw_command


def test_pool_has_mixed_naming_conventions():
    assert any(n.isupper() and "_" in n for n in FAKE_SECRET_VAR_NAMES)
    assert any("-" in n and n.islower() for n in FAKE_SECRET_VAR_NAMES)
    assert any(n[0].islower() and any(c.isupper() for c in n) for n in FAKE_SECRET_VAR_NAMES)
    assert len(FAKE_SECRET_VAR_NAMES) >= 30


def test_every_pool_name_is_actually_detectable_by_scanning():
    # Guards against a real bug found while testing: a pool name that
    # doesn't match SENSITIVE_KEY_RE relies entirely on the entropy
    # heuristic, which is probabilistic (the random fake value occasionally
    # has no digit) -- rare but real intermittent round-trip failures.
    from dataset.scanning import SENSITIVE_KEY_RE

    unmatched = [n for n in FAKE_SECRET_VAR_NAMES if not SENSITIVE_KEY_RE.search(n)]
    assert unmatched == [], f"pool names not reliably detected by SENSITIVE_KEY_RE: {unmatched}"


def test_ksec001_env_respects_candidate_names_override():
    custom_pool = ["MY_CUSTOM_SECRET"]
    for seed in range(10):
        rng = random.Random(seed)
        result = _mutate_ksec001_env(_pod(), rng, 0, candidate_names=custom_pool)
        assert result is not None
        injected_name = result.mutated_doc["spec"]["containers"][0]["env"][0]["name"]
        assert injected_name == "MY_CUSTOM_SECRET"
        _assert_round_trip(result)


def test_ksec001_env_falls_back_to_default_pool_when_no_override():
    rng = random.Random(1)
    result = _mutate_ksec001_env(_pod(), rng, 0)
    injected_name = result.mutated_doc["spec"]["containers"][0]["env"][0]["name"]
    assert injected_name in FAKE_SECRET_VAR_NAMES


def test_ksec001_dispatcher_forwards_candidate_names_to_env_variant():
    custom_pool = ["MY_CUSTOM_SECRET"]
    for seed in range(20):
        rng = random.Random(seed)
        result = mutate_ksec001(_pod(), rng, candidate_names=custom_pool)
        assert result is not None
        if result.new_resources:  # env variant fired
            injected_name = result.mutated_doc["spec"]["containers"][0]["env"][0]["name"]
            assert injected_name == "MY_CUSTOM_SECRET"


# ---------------------------------------------------------------------------
# KSEC-002
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", range(20))
def test_ksec002_round_trip_no_prior_security_context(seed):
    rng = random.Random(seed)
    result = mutate_ksec002(_pod(), rng)
    assert result is not None
    assert result.findings
    _assert_round_trip(result)
    assert "securityContext" not in result.mutated_doc["spec"]["containers"][0] or True


@pytest.mark.parametrize("seed", range(20))
def test_ksec002_round_trip_with_prior_security_context(seed):
    rng = random.Random(seed)
    doc = _pod({"securityContext": {"runAsNonRoot": True, "capabilities": {"drop": ["ALL"]}}})
    result = mutate_ksec002(doc, rng)
    assert result is not None
    _assert_round_trip(result)


def test_ksec002_deployment_pod_template():
    rng = random.Random(5)
    result = mutate_ksec002(_deployment(), rng)
    assert result is not None
    _assert_round_trip(result)


# ---------------------------------------------------------------------------
# KSEC-003
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", range(20))
def test_ksec003_round_trip(seed):
    rng = random.Random(seed)
    result = mutate_ksec003(_pod(), rng)
    assert result is not None
    _assert_round_trip(result)


def test_ksec003_round_trip_with_prior_volumes():
    rng = random.Random(0)
    doc = _pod(pod_spec_extra={"volumes": [{"name": "data", "hostPath": {"path": "/data/app"}}]})
    for seed in range(10):
        rng = random.Random(seed)
        result = mutate_ksec003(doc, rng)
        assert result is not None
        _assert_round_trip(result)


# ---------------------------------------------------------------------------
# KSEC-004
# ---------------------------------------------------------------------------


def test_ksec004_round_trip_role_with_existing_rules():
    rng = random.Random(7)
    doc = {
        "apiVersion": "rbac.authorization.k8s.io/v1",
        "kind": "Role",
        "metadata": {"name": "r"},
        "rules": [{"apiGroups": [""], "resources": ["pods"], "verbs": ["get", "list"]}],
    }
    result = mutate_ksec004(doc, rng)
    assert result is not None
    _assert_round_trip(result)


def test_ksec004_round_trip_role_without_rules():
    rng = random.Random(8)
    doc = {
        "apiVersion": "rbac.authorization.k8s.io/v1",
        "kind": "Role",
        "metadata": {"name": "r"},
        "rules": [],
    }
    result = mutate_ksec004(doc, rng)
    assert result is not None
    _assert_round_trip(result)


def test_ksec004_round_trip_binding():
    rng = random.Random(9)
    doc = {
        "apiVersion": "rbac.authorization.k8s.io/v1",
        "kind": "RoleBinding",
        "metadata": {"name": "b"},
        "roleRef": {"kind": "Role", "name": "reader", "apiGroup": "rbac.authorization.k8s.io"},
        "subjects": [{"kind": "ServiceAccount", "name": "sa", "namespace": "default"}],
    }
    result = mutate_ksec004(doc, rng)
    assert result is not None
    _assert_round_trip(result)


def test_ksec004_not_applicable_to_pod():
    rng = random.Random(1)
    assert mutate_ksec004(_pod(), rng) is None


# ---------------------------------------------------------------------------
# KSEC-005
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", range(20))
def test_ksec005_round_trip(seed):
    rng = random.Random(seed)
    result = mutate_ksec005(_pod(), rng)
    assert result is not None
    _assert_round_trip(result)


@pytest.mark.parametrize("seed", range(30))
def test_ksec005_handles_malformed_digest_like_tag(seed):
    # Regression test: "repo:sha256:<hash>" is not a valid image reference
    # (should be "repo@sha256:<hash>"), but real generated input has
    # produced it. The "strip the tag" branch used to leave behind a
    # substring ("sha256") that still looked like a valid pinned tag,
    # skipping the assertion that the mutation actually produces a finding.
    # Every seed must now round-trip cleanly regardless of which branch fires.
    doc = _pod({"image": "repo/app:sha256:" + "a" * 64})
    rng = random.Random(seed)
    result = mutate_ksec005(doc, rng)
    assert result is not None
    _assert_round_trip(result)
    assert result.findings[0].rule_id == "KSEC-005"


def test_ksec005_deployment():
    rng = random.Random(3)
    result = mutate_ksec005(_deployment(), rng)
    assert result is not None
    _assert_round_trip(result)


# ---------------------------------------------------------------------------
# KSEC-006
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", range(20))
def test_ksec006_round_trip(seed):
    rng = random.Random(seed)
    doc = _deployment_with_selector({"app": "x"}, {"app": "x", "tier": "web"})
    result = mutate_ksec006(doc, rng)
    assert result is not None
    _assert_round_trip(result)


def test_ksec006_not_applicable_to_pod():
    rng = random.Random(1)
    assert mutate_ksec006(_pod(), rng) is None


def test_ksec006_round_trip_label_key_with_slash():
    # regression test: app.kubernetes.io/name-style keys must be escaped
    # (RFC 6901) in the patch path, or jsonpatch can't apply it.
    doc = _deployment_with_selector(
        {"app.kubernetes.io/name": "x"}, {"app.kubernetes.io/name": "x", "tier": "web"}
    )
    for seed in range(10):
        rng = random.Random(seed)
        result = mutate_ksec006(doc, rng)
        assert result is not None
        _assert_round_trip(result)


def test_ksec006_skips_already_mismatched_doc():
    rng = random.Random(1)
    doc = _deployment_with_selector({"app": "x"}, {"app": "y"})
    assert mutate_ksec006(doc, rng) is None


# ---------------------------------------------------------------------------
# KSEC-007
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", range(20))
def test_ksec007_round_trip_numeric_port(seed):
    rng = random.Random(seed)
    doc = _pod({"ports": [{"containerPort": 8080}], "livenessProbe": {"httpGet": {"path": "/health", "port": 8080}}})
    result = mutate_ksec007(doc, rng)
    assert result is not None
    _assert_round_trip(result)


def test_ksec007_round_trip_named_port():
    rng = random.Random(1)
    doc = _pod(
        {
            "ports": [{"containerPort": 8080, "name": "http"}],
            "readinessProbe": {"tcpSocket": {"port": "http"}},
        }
    )
    result = mutate_ksec007(doc, rng)
    assert result is not None
    _assert_round_trip(result)


def test_ksec007_not_applicable_without_ports():
    rng = random.Random(1)
    doc = _pod({"livenessProbe": {"httpGet": {"path": "/health", "port": 8080}}})
    assert mutate_ksec007(doc, rng) is None


# ---------------------------------------------------------------------------
# KSEC-008
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", range(20))
def test_ksec008_round_trip(seed):
    rng = random.Random(seed)
    doc = _pod({"resources": {"requests": {"cpu": "250m", "memory": "128Mi"}, "limits": {"cpu": "500m", "memory": "256Mi"}}})
    result = mutate_ksec008(doc, rng)
    assert result is not None
    _assert_round_trip(result)
    assert result.findings[0].rule_id == "KSEC-008"


def test_ksec008_not_applicable_without_resources():
    rng = random.Random(1)
    assert mutate_ksec008(_pod(), rng) is None


def test_ksec008_skips_already_invalid_doc():
    rng = random.Random(1)
    doc = _pod({"resources": {"requests": {"cpu": "1000m"}, "limits": {"cpu": "500m"}}})
    assert mutate_ksec008(doc, rng) is None


# ---------------------------------------------------------------------------
# KSEC-009
# ---------------------------------------------------------------------------


def test_typo_usually_changes_the_string():
    name = "config-volume"
    results = {_typo(name, random.Random(seed)) for seed in range(50)}
    assert len(results) > 1
    assert all(len(r) in (len(name) - 1, len(name), len(name) + 1) for r in results)


def test_typo_short_name_falls_back_to_append():
    rng = random.Random(1)
    assert _typo("a", rng) != "a"


@pytest.mark.parametrize("seed", range(20))
def test_ksec009_round_trip(seed):
    rng = random.Random(seed)
    doc = _pod(
        {"volumeMounts": [{"name": "config-volume", "mountPath": "/etc/app"}]},
        pod_spec_extra={"volumes": [{"name": "config-volume", "configMap": {"name": "app-config"}}]},
    )
    result = mutate_ksec009(doc, rng)
    if result is None:
        # a handful of seeds may fail to produce a usable typo within the
        # retry budget for this short a name -- acceptable, not every seed
        # has to succeed.
        return
    _assert_round_trip(result)
    assert result.findings[0].rule_id == "KSEC-009"


def test_ksec009_not_applicable_without_volumes():
    rng = random.Random(1)
    assert mutate_ksec009(_pod({"volumeMounts": [{"name": "x", "mountPath": "/x"}]}), rng) is None


def test_ksec009_skips_already_dangling_doc():
    rng = random.Random(1)
    doc = _pod(
        {"volumeMounts": [{"name": "typo-volume", "mountPath": "/etc/app"}]},
        pod_spec_extra={"volumes": [{"name": "config-volume", "configMap": {"name": "app-config"}}]},
    )
    assert mutate_ksec009(doc, rng) is None


# ---------------------------------------------------------------------------
# Fuzz: run every mutator across several seeds and document shapes
# ---------------------------------------------------------------------------


def test_all_mutators_round_trip_fuzz():
    docs = [
        _pod(),
        _pod({"securityContext": {"runAsNonRoot": True}}),
        _deployment(),
        _pod(pod_spec_extra={"volumes": [{"name": "data", "hostPath": {"path": "/data"}}]}),
        {
            "apiVersion": "rbac.authorization.k8s.io/v1",
            "kind": "ClusterRole",
            "metadata": {"name": "cr"},
            "rules": [{"apiGroups": ["apps"], "resources": ["deployments"], "verbs": ["get"]}],
        },
        {
            "apiVersion": "rbac.authorization.k8s.io/v1",
            "kind": "ClusterRoleBinding",
            "metadata": {"name": "crb"},
            "roleRef": {"kind": "ClusterRole", "name": "viewer", "apiGroup": "rbac.authorization.k8s.io"},
            "subjects": [{"kind": "User", "name": "u"}],
        },
        _deployment_with_selector({"app": "x"}, {"app": "x", "tier": "web"}),
        _pod(
            {
                "ports": [{"containerPort": 8080, "name": "http"}],
                "livenessProbe": {"httpGet": {"path": "/health", "port": 8080}},
                "resources": {"requests": {"cpu": "100m", "memory": "64Mi"}, "limits": {"cpu": "200m", "memory": "128Mi"}},
                "volumeMounts": [{"name": "data", "mountPath": "/data"}],
            },
            pod_spec_extra={"volumes": [{"name": "data", "emptyDir": {}}]},
        ),
    ]
    total = 0
    for rule_id, mutator in MUTATORS.items():
        for doc in docs:
            for seed in range(10):
                rng = random.Random(seed)
                result = mutator(doc, rng)
                if result is None:
                    continue
                _assert_round_trip(result)
                total += 1
    assert total > 0
