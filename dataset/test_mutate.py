import copy
import random

import jsonpatch
import pytest

from dataset.mutate import (
    FAKE_SECRET_VAR_NAMES,
    MUTATORS,
    _fake_secret_value,
    _mutate_ksec001_command,
    _mutate_ksec001_env,
    _mutate_ksec001_env_in_place,
    _mutate_ksec001_url,
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
    mutate_ksec010,
    mutate_ksec011,
    mutate_ksec006_service,
)
from dataset.detect import detect_all, detect_ksec007, detect_ksec010, detect_ksec011
from dataset.scanning import CONN_STRING_RE, PLACEHOLDER_RE, find_secrets


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


def test_ksec001_command_round_trip_inserts_into_existing_args():
    positions = set()
    for seed in range(40):
        doc = _pod({"args": ["--verbose", "--port", "8080"]})
        result = _mutate_ksec001_command(doc, random.Random(seed), 0)
        assert result is not None
        _assert_round_trip(result)
        args = result.mutated_doc["spec"]["containers"][0]["args"]
        # The original args keep their relative order around the insertion.
        assert [a for a in args if a in ("--verbose", "--port", "8080")] == ["--verbose", "--port", "8080"]
        positions.add(args.index("--verbose"))
    assert len(positions) > 1  # not always appended at the end


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


def test_fake_secret_value_never_matches_placeholder_regardless_of_style():
    for seed in range(500):
        rng = random.Random(seed)
        value = _fake_secret_value(rng, cli_safe=seed % 2 == 0)
        assert len(value) >= 4
        assert not PLACEHOLDER_RE.match(value.strip())


def test_fake_secret_value_cli_safe_has_no_regex_breaking_characters():
    for seed in range(500):
        rng = random.Random(seed)
        value = _fake_secret_value(rng, cli_safe=True)
        assert " " not in value
        assert "@" not in value
        assert "/" not in value
        assert "'" not in value
        assert '"' not in value


def test_fake_secret_value_messy_style_is_reachable_and_still_detected():
    # Regression test for the sql.yaml gap: a space-containing, word-based
    # value must (a) actually be reachable from the env-variant's style pool
    # and (b) still be flagged by scanning.find_secrets once placed under a
    # sensitive key, exactly like the real manifest's "mypassowrd 123".
    found_messy = False
    for seed in range(500):
        rng = random.Random(seed)
        value = _fake_secret_value(rng, cli_safe=False)
        if " " in value and "://" not in value:
            found_messy = True
            doc = _pod({"env": [{"name": "DB_PASSWORD", "value": value}]})
            hits = find_secrets(doc)
            assert any(h.value == value for h in hits)
    assert found_messy


def test_fake_secret_value_connection_string_style_is_reachable_and_detected():
    # Regression test for the "credentials embedded in a URL" gap: a plain
    # env value shaped like scheme://user:pass@host:port/db (e.g. a real
    # DATABASE_URL/DB_CONNECTION) must be reachable and caught by
    # scanning.py's CONN_STRING_RE path, independent of the key name.
    found_conn_string = False
    for seed in range(500):
        rng = random.Random(seed)
        value = _fake_secret_value(rng, cli_safe=False)
        if "://" in value:
            found_conn_string = True
            assert CONN_STRING_RE.match(value)
            doc = _pod({"env": [{"name": "DATABASE_URL", "value": value}]})
            hits = find_secrets(doc)
            assert any(h.reason == "connection string with embedded credentials" for h in hits)
    assert found_conn_string


@pytest.mark.parametrize("seed", range(100))
def test_ksec001_env_round_trip_across_many_seeds_and_styles(seed):
    rng = random.Random(seed)
    result = _mutate_ksec001_env(_pod(), rng, 0)
    assert result is not None
    assert result.findings
    _assert_round_trip(result)


@pytest.mark.parametrize("seed", range(100))
def test_ksec001_command_round_trip_across_many_seeds_and_styles(seed):
    rng = random.Random(seed)
    result = _mutate_ksec001_command(_pod(), rng, 0)
    assert result is not None
    assert result.findings
    _assert_round_trip(result)


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
        _pod(
            {
                "image": "postgres:15",
                "ports": [{"containerPort": 5432, "name": "pg"}],
                "livenessProbe": {"tcpSocket": {"port": "pg"}},
                "env": [
                    {"name": "POSTGRES_DB", "value": "app"},
                    {"name": "POSTGRES_PASSWORD", "valueFrom": {"secretKeyRef": {"name": "pg", "key": "pw"}}},
                ],
            }
        ),
        _pod({"image": "redis:7", "readinessProbe": {"exec": {"command": ["redis-cli", "ping"]}}}),
        _pod({"image": "mysql:8", "env": [{"name": "MYSQL_ALLOW_EMPTY_PASSWORD", "value": "yes"}]}),
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


# ---------------------------------------------------------------------------
# KSEC-010 -- httpGet probe on a non-HTTP server
# ---------------------------------------------------------------------------


def _db_deployment(image, container_extra=None, extra_containers=()):
    container = {"name": "db", "image": image}
    if container_extra:
        container.update(container_extra)
    return {
        "apiVersion": "apps/v1",
        "kind": "Deployment",
        "metadata": {"name": "db", "namespace": "data"},
        "spec": {"template": {"spec": {"containers": [container, *extra_containers]}}},
    }


@pytest.mark.parametrize("seed", range(10))
def test_ksec010_tcpsocket_probe_becomes_httpget_and_round_trips(seed):
    doc = _db_deployment(
        "postgres:15", {"ports": [{"containerPort": 5432}], "livenessProbe": {"tcpSocket": {"port": 5432}}}
    )
    result = mutate_ksec010(doc, random.Random(seed))
    assert result is not None
    _assert_round_trip(result)
    assert len(result.findings) == 1
    assert result.new_resources == []


def test_ksec010_existing_probe_keeps_its_other_fields():
    probe = {"tcpSocket": {"port": 6379}, "initialDelaySeconds": 15, "periodSeconds": 20}
    doc = _db_deployment("redis:7", {"livenessProbe": probe, "readinessProbe": {"tcpSocket": {"port": 6379}}})
    result = mutate_ksec010(doc, random.Random(0))
    _assert_round_trip(result)
    mutated_probe = result.mutated_doc["spec"]["template"]["spec"]["containers"][0][result.patch[0].path.split("/")[-2]]
    assert "tcpSocket" not in mutated_probe and "httpGet" in mutated_probe


def test_ksec010_exec_probe_is_restored_verbatim():
    exec_check = {"command": ["pg_isready", "-U", "postgres"]}
    doc = _db_deployment("postgres:15", {"livenessProbe": {"exec": exec_check}, "readinessProbe": {"exec": exec_check}})
    result = mutate_ksec010(doc, random.Random(0))
    _assert_round_trip(result)
    assert result.patch[1].op == "add" and result.patch[1].value == exec_check


def test_ksec010_container_without_probe_gains_tcpsocket_in_canonical():
    # The fix taught must be "use tcpSocket", never "delete the probe".
    doc = _db_deployment("mongo:6")
    result = mutate_ksec010(doc, random.Random(0))
    assert result is not None
    _assert_round_trip(result)
    container = result.canonical["spec"]["template"]["spec"]["containers"][0]
    probe_field = result.patch[0].path.split("/")[-2]
    assert probe_field in ("livenessProbe", "readinessProbe")
    assert container[probe_field] == {"tcpSocket": {"port": 27017}}
    assert result.patch[1].value == {"port": 27017}


def test_ksec010_never_creates_or_hides_a_ksec007_finding():
    # Probe port consistent with declared ports: stays that way.
    doc = _db_deployment("mysql:8", {"ports": [{"containerPort": 3306, "name": "mysql"}], "livenessProbe": {"tcpSocket": {"port": "mysql"}}})
    for seed in range(10):
        result = mutate_ksec010(doc, random.Random(seed))
        assert detect_ksec007(result.mutated_doc) == []
    # Default port not declared and no usable probe: nothing safe to mutate.
    doc = _db_deployment("mysql:8", {"ports": [{"containerPort": 3307}]})
    assert mutate_ksec010(doc, random.Random(0)) is None


def test_ksec010_not_applicable():
    assert mutate_ksec010(_deployment(), random.Random(0)) is None
    already_bad = _db_deployment("redis:7", {"livenessProbe": {"httpGet": {"path": "/", "port": 6379}}})
    assert mutate_ksec010(already_bad, random.Random(0)) is None


# ---------------------------------------------------------------------------
# KSEC-011 -- missing required env var for the image
# ---------------------------------------------------------------------------


def _env_of(doc):
    return doc["spec"]["template"]["spec"]["containers"][0].get("env")


@pytest.mark.parametrize("seed", range(10))
def test_ksec011_secret_ref_is_removed_and_restored_verbatim(seed):
    env = [
        {"name": "POSTGRES_DB", "value": "app"},
        {"name": "POSTGRES_PASSWORD", "valueFrom": {"secretKeyRef": {"name": "pg", "key": "password"}}},
        {"name": "PGDATA", "value": "/var/lib/postgresql/data/pgdata"},
    ]
    doc = _db_deployment("postgres:15", {"env": env})
    result = mutate_ksec011(doc, random.Random(seed))
    _assert_round_trip(result)
    assert result.canonical == doc
    assert len(result.findings) == 1
    assert result.new_resources == []
    assert [e["name"] for e in _env_of(result.mutated_doc)] == ["POSTGRES_DB", "PGDATA"]
    assert result.patch[0].path == "/spec/template/spec/containers/0/env/1"


def test_ksec011_plaintext_setting_is_fixed_forward_to_a_secret():
    # A placeholder/"trust" setting in the source doc must not be what the
    # model learns to re-add: the round-trip target uses a secretKeyRef.
    env = [{"name": "MYSQL_ALLOW_EMPTY_PASSWORD", "value": "yes"}, {"name": "MYSQL_DATABASE", "value": "app"}]
    doc = _db_deployment("mysql:8", {"env": env})
    result = mutate_ksec011(doc, random.Random(0))
    _assert_round_trip(result)
    restored = _env_of(result.canonical)[0]
    assert restored == {
        "name": "MYSQL_ROOT_PASSWORD",
        "valueFrom": {"secretKeyRef": {"name": "db-secrets", "key": "mysql-root-password"}},
    }
    assert len(result.new_resources) == 1
    assert "kind: Secret" in result.new_resources[0]
    assert "namespace: data" in result.new_resources[0]
    assert "mysql-root-password" in result.new_resources[0]
    assert detect_all(result.canonical) == []


def test_ksec011_removing_the_only_env_entry_drops_the_env_key():
    env = [{"name": "MARIADB_ROOT_PASSWORD", "valueFrom": {"secretKeyRef": {"name": "m", "key": "root"}}}]
    doc = _db_deployment("mariadb:11", {"env": env})
    result = mutate_ksec011(doc, random.Random(0))
    _assert_round_trip(result)
    assert _env_of(result.mutated_doc) is None
    assert result.patch[0].path == "/spec/template/spec/containers/0/env"


def test_ksec011_multiple_satisfying_entries_collapse_into_one():
    env = [
        {"name": "MYSQL_RANDOM_ROOT_PASSWORD", "value": "yes"},
        {"name": "MYSQL_USER", "value": "app"},
        {"name": "MYSQL_ROOT_PASSWORD_FILE", "value": "/run/secrets/root"},
    ]
    doc = _db_deployment("mysql:8", {"env": env})
    result = mutate_ksec011(doc, random.Random(0))
    _assert_round_trip(result)
    assert len(result.findings) == 1
    assert [e["name"] for e in _env_of(result.mutated_doc)] == ["MYSQL_USER"]


@pytest.mark.parametrize("seed", range(10))
def test_ksec011_mssql_removes_one_requirement_at_a_time(seed):
    env = [
        {"name": "ACCEPT_EULA", "value": "Y"},
        {"name": "MSSQL_PID", "value": "Developer"},
        {"name": "MSSQL_SA_PASSWORD", "valueFrom": {"secretKeyRef": {"name": "mssql", "key": "sa"}}},
    ]
    doc = _db_deployment("mcr.microsoft.com/mssql/server:2022-latest", {"env": env})
    result = mutate_ksec011(doc, random.Random(seed))
    _assert_round_trip(result)
    assert result.canonical == doc
    assert len(result.findings) == 1
    assert result.new_resources == []


def test_ksec011_literal_requirement_is_restored_as_a_plain_value():
    env = [
        {"name": "ACCEPT_EULA", "value": "y"},
        {"name": "SA_PASSWORD", "valueFrom": {"secretKeyRef": {"name": "mssql", "key": "sa"}}},
    ]
    doc = _db_deployment("mcr.microsoft.com/mssql/server:2019-latest", {"env": env})
    eula = next(
        r for seed in range(20)
        if "ACCEPT_EULA" in (r := mutate_ksec011(doc, random.Random(seed))).findings[0].message
    )
    _assert_round_trip(eula)
    assert eula.patch[0].value == {"name": "ACCEPT_EULA", "value": "Y"}
    assert eula.new_resources == []


def test_ksec011_bitnami_allow_empty_is_fixed_forward_to_a_secret():
    doc = _db_deployment("bitnami/redis:7.2", {"env": [{"name": "ALLOW_EMPTY_PASSWORD", "value": "yes"}]})
    result = mutate_ksec011(doc, random.Random(0))
    _assert_round_trip(result)
    assert _env_of(result.canonical) == [
        {"name": "REDIS_PASSWORD", "valueFrom": {"secretKeyRef": {"name": "db-secrets", "key": "redis-password"}}}
    ]
    assert len(result.new_resources) == 1


def test_ksec011_not_applicable():
    assert mutate_ksec011(_deployment(), random.Random(0)) is None
    # No satisfying entry to remove (already a real-world KSEC-011 bug).
    assert mutate_ksec011(_db_deployment("postgres:15"), random.Random(0)) is None
    # Requirement not checkable: envFrom may supply it.
    env_from = {"envFrom": [{"secretRef": {"name": "pg"}}], "env": [{"name": "POSTGRES_PASSWORD", "value": "x"}]}
    assert mutate_ksec011(_db_deployment("postgres:15", env_from), random.Random(0)) is None


def test_ksec001_env_never_injects_a_var_that_satisfies_ksec011():
    # Otherwise KSEC-011 would later collapse the injected entry away along
    # with the real one, erasing the KSEC-001 finding.
    secret_ref = {"name": "POSTGRES_PASSWORD", "valueFrom": {"secretKeyRef": {"name": "pg", "key": "pw"}}}
    doc = _db_deployment("postgres:15", {"env": [secret_ref]})
    names = ["POSTGRES_PASSWORD", "POSTGRES_HOST_AUTH_METHOD", "DB_PASSWORD"]
    for seed in range(30):
        result = _mutate_ksec001_env(doc, random.Random(seed), 0, candidate_names=names)
        plaintext = [e["name"] for e in _env_of(result.mutated_doc) if "value" in e and e["name"].endswith("PASSWORD")]
        assert plaintext == ["DB_PASSWORD"]


@pytest.mark.parametrize("seed", range(40))
def test_ksec001_env_inserts_at_random_positions_with_an_optional_username(seed):
    env = [{"name": "TZ", "value": "UTC"}, {"name": "LOG_LEVEL", "value": "info"}]
    doc = _deployment({"env": env})
    result = _mutate_ksec001_env(doc, random.Random(seed), 0, candidate_names=["DB_PASSWORD"])
    _assert_round_trip(result)
    assert len(result.findings) == 1
    names = [e["name"] for e in _env_of(result.mutated_doc)]
    assert [n for n in names if n in ("TZ", "LOG_LEVEL")] == ["TZ", "LOG_LEVEL"]
    if "DB_USER" in names or "DB_USERNAME" in names:
        user = names.index("DB_USER") if "DB_USER" in names else names.index("DB_USERNAME")
        assert names[user + 1] == "DB_PASSWORD"  # username right before its password
        assert _env_of(result.canonical)[user] == _env_of(result.mutated_doc)[user]  # kept in the target


def test_ksec001_env_username_appears_and_position_varies():
    seen_user, positions = False, set()
    for seed in range(60):
        doc = _deployment({"env": [{"name": "TZ", "value": "UTC"}, {"name": "LOG_LEVEL", "value": "info"}]})
        names = [e["name"] for e in _env_of(_mutate_ksec001_env(doc, random.Random(seed), 0, candidate_names=["DB_PASSWORD"]).mutated_doc)]
        seen_user |= any(n.startswith("DB_USER") for n in names)
        positions.add(names.index("DB_PASSWORD"))
    assert seen_user and len(positions) > 1


def test_ksec001_env_leaves_a_container_with_pending_ksec011_alone():
    # KSEC-011's restoring insertion would shift an appended entry's index
    # out from under KSEC-001's own patch when the two are composed.
    doc = _db_deployment("postgres:15", {"env": [{"name": "POSTGRES_DB", "value": "app"}]})
    assert detect_ksec011(doc)
    for seed in range(10):
        assert _mutate_ksec001_env(doc, random.Random(seed), 0) is None


def test_ksec001_command_never_turns_a_db_server_into_a_client_job():
    # A client-style first arg (`curl ...`) would make KSEC-011 stop applying
    # to the container, silently hiding its finding.
    secret_ref = {"name": "POSTGRES_PASSWORD", "valueFrom": {"secretKeyRef": {"name": "pg", "key": "pw"}}}
    doc = _db_deployment("postgres:15", {"env": [secret_ref]})
    for seed in range(40):
        result = _mutate_ksec001_command(doc, random.Random(seed), 0)
        if result is not None:
            args = result.mutated_doc["spec"]["template"]["spec"]["containers"][0]["args"]
            assert args[0].startswith("-")


@pytest.mark.parametrize("seed", range(10))
def test_ksec009_can_target_a_statefulset_claim_template_mount(seed):
    doc = {
        "apiVersion": "apps/v1",
        "kind": "StatefulSet",
        "metadata": {"name": "db"},
        "spec": {
            "template": {"spec": {"containers": [{"name": "db", "image": "myapp:1.2.3", "volumeMounts": [{"name": "pgdata", "mountPath": "/data"}]}]}},
            "volumeClaimTemplates": [{"metadata": {"name": "pgdata"}}],
        },
    }
    result = mutate_ksec009(doc, random.Random(seed))
    assert result is not None
    _assert_round_trip(result)
    assert len(result.findings) == 1


def _replica_postgres():
    # No POSTGRES_PASSWORD is legitimate here: an init container clones the
    # data dir in, so the entrypoint never initializes a new cluster.
    doc = _db_deployment(
        "postgres:15",
        {
            "env": [{"name": "PGDATA", "value": "/var/lib/postgresql/data/pg"}],
            "volumeMounts": [{"name": "data", "mountPath": "/var/lib/postgresql/data"}],
        },
    )
    pod_spec = doc["spec"]["template"]["spec"]
    pod_spec["initContainers"] = [{"name": "clone", "image": "busybox:1.36", "volumeMounts": [{"name": "data", "mountPath": "/mnt"}]}]
    pod_spec["volumes"] = [{"name": "data", "emptyDir": {}}]
    return doc


def test_ksec009_raw_mutation_on_a_replica_would_make_ksec011_apply():
    # Documents why the registry guard exists: typoing the data mount breaks
    # the "data dir is pre-populated" exemption, so KSEC-011 starts firing.
    doc = _replica_postgres()
    assert not detect_ksec011(doc, 0)
    flipped = [
        seed for seed in range(20)
        if (result := mutate_ksec009(doc, random.Random(seed))) is not None and detect_ksec011(result.mutated_doc, 0)
    ]
    assert flipped


@pytest.mark.parametrize("seed", range(20))
def test_registered_mutators_never_change_another_rules_findings(seed):
    doc = _replica_postgres()
    for rule_id, mutator in MUTATORS.items():
        result = mutator(copy.deepcopy(doc), random.Random(seed))
        if result is None:
            continue
        others = lambda d: sorted(f.rule_id for f in detect_all(d, 0) if f.rule_id != rule_id)
        assert others(result.mutated_doc) == others(result.canonical), rule_id


@pytest.mark.parametrize("seed", range(40))
def test_ksec001_command_split_flag_variant_round_trips(seed):
    for container_extra in (None, {"args": ["--verbose"]}):
        result = _mutate_ksec001_command(_pod(container_extra), random.Random(seed), 0)
        if result is None:
            continue
        _assert_round_trip(result)
        assert len(result.findings) == 1


def test_ksec001_command_split_flag_variant_is_generated():
    splits = 0
    for seed in range(60):
        result = _mutate_ksec001_command(_pod({"args": ["--verbose"]}), random.Random(seed), 0)
        if result is not None and "separate CLI flag" in result.findings[0].message:
            splits += 1
            assert len(result.patch) == 2  # one remove per appended element
            _assert_round_trip(result)
    assert splits


def _service_file(service_first=False, selector=None):
    workload = _deployment_with_selector({"app": "web"}, {"app": "web", "tier": "fe"})
    service = {
        "apiVersion": "v1",
        "kind": "Service",
        "metadata": {"name": "web"},
        "spec": {"selector": selector or {"app": "web"}, "ports": [{"port": 80}]},
    }
    return [service, workload] if service_first else [workload, service]


def _apply_file_patch(docs, patch):
    docs = copy.deepcopy(docs)
    for i in range(len(docs)):
        ops = [{k: v for k, v in op.to_dict().items() if k != "doc"} for op in patch if op.doc == i]
        if ops:
            docs[i] = jsonpatch.apply_patch(docs[i], ops)
    return docs


@pytest.mark.parametrize("seed", range(20))
@pytest.mark.parametrize("service_first", [False, True])
def test_ksec006_service_mutation_round_trips_with_one_finding_on_the_service(seed, service_first):
    docs = _service_file(service_first)
    result = mutate_ksec006_service(docs, random.Random(seed))
    service_index = 0 if service_first else 1
    assert result is not None
    assert result.canonical_docs == docs
    assert [(f.rule_id, f.doc) for f in result.findings] == [("KSEC-006", service_index)]
    assert result.patch[0].doc == service_index
    assert _apply_file_patch(result.mutated_docs, result.patch) == docs


def test_ksec006_service_mutation_not_applicable():
    # No Service; a Service selecting no workload; a Service already broken.
    assert mutate_ksec006_service([_service_file()[0]], random.Random(0)) is None
    assert mutate_ksec006_service(_service_file(selector={"component": "db"}), random.Random(0)) is None
    assert mutate_ksec006_service(_service_file(selector={"app": "wbe"}), random.Random(0)) is None


def _env_deployment(env):
    return _db_deployment("mongo:7.0", {"env": env})


@pytest.mark.parametrize("seed", range(15))
def test_ksec001_in_place_leaks_an_existing_secret_ref_and_restores_it(seed):
    ref = {"name": "MONGO_INITDB_ROOT_PASSWORD", "valueFrom": {"secretKeyRef": {"name": "mongo", "key": "pw"}}}
    env = [{"name": "MONGO_INITDB_ROOT_USERNAME", "value": "admin"}, ref, {"name": "TZ", "value": "UTC"}]
    doc = _env_deployment(env)
    result = _mutate_ksec001_env_in_place(doc, random.Random(seed), 0)
    _assert_round_trip(result)
    assert result.canonical == doc and result.new_resources == []
    leaked = _env_of(result.mutated_doc)[1]
    assert leaked["name"] == "MONGO_INITDB_ROOT_PASSWORD" and "value" in leaked  # same name, same position
    assert result.findings[0].path.endswith("/env/1/value")


def test_ksec001_in_place_needs_a_credential_secret_ref():
    not_credential = [{"name": "MONGO_DB", "valueFrom": {"secretKeyRef": {"name": "m", "key": "db"}}}]
    assert _mutate_ksec001_env_in_place(_env_deployment(not_credential), random.Random(0), 0) is None


@pytest.mark.parametrize("seed", range(15))
def test_ksec001_url_embeds_credentials_and_fixes_forward(seed):
    env = [{"name": "MONGODB_URL", "value": "mongodb://mongodb-service:27017/admin"}]
    doc = _env_deployment(env)
    result = _mutate_ksec001_url(doc, random.Random(seed), 0)
    _assert_round_trip(result)
    assert "@mongodb-service:27017/admin" in _env_of(result.mutated_doc)[0]["value"]
    assert "connection string" in result.findings[0].message
    assert _env_of(result.canonical)[0]["valueFrom"]["secretKeyRef"]["key"] == "mongodb-url"
    assert len(result.new_resources) == 1


def test_ksec001_url_skips_urls_that_already_have_credentials_or_no_scheme():
    for value in ("mongodb://u:p@db:27017", "http://web:8080", "db:27017"):
        assert _mutate_ksec001_url(_env_deployment([{"name": "URL", "value": value}]), random.Random(0), 0) is None


def test_ksec005_can_unpin_a_second_image_in_the_same_manifest():
    doc = _deployment({"image": "nginx:1.25"})
    doc["spec"]["template"]["spec"]["initContainers"] = [{"name": "init", "image": "busybox:1.36"}]
    first = mutate_ksec005(doc, random.Random(0))
    second = mutate_ksec005(first.mutated_doc, random.Random(1))
    assert second is not None and len(second.findings) == 1
    assert second.findings[0].path != first.findings[0].path
    assert len(detect_all(second.mutated_doc)) == 2
