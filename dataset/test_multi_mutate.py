import copy
import random

import jsonpatch
import pytest

from dataset.detect import detect_file
from dataset.multi_mutate import mutate_multi_defect, mutate_multi_defect_file, mutate_single_defect_file
from dataset.schema import RULE_IDS


def _apply_patch(mutated_doc, patch):
    ops = [{k: v for k, v in p.to_dict().items() if k != "doc"} for p in patch]
    return jsonpatch.apply_patch(mutated_doc, ops)


def _rich_deployment():
    """A doc every active mutator can apply to: pinned image (KSEC-005), no
    securityContext (KSEC-002), no env vars (KSEC-001), matching
    selector/template labels (KSEC-006), a probe port that matches a
    declared container port (KSEC-007), and requests <= limits (KSEC-008)."""
    return {
        "apiVersion": "apps/v1",
        "kind": "Deployment",
        "metadata": {"name": "d"},
        "spec": {
            "selector": {"matchLabels": {"app": "web"}},
            "template": {
                "metadata": {"labels": {"app": "web"}},
                "spec": {
                    "containers": [
                        {
                            "name": "app",
                            "image": "myapp:1.2.3",
                            "ports": [{"containerPort": 8080}],
                            "livenessProbe": {"httpGet": {"path": "/", "port": 8080}},
                            "resources": {
                                "requests": {"cpu": "100m", "memory": "64Mi"},
                                "limits": {"cpu": "200m", "memory": "128Mi"},
                            },
                        }
                    ]
                },
            },
        },
    }


def test_round_trip_holds_for_a_multi_defect_composition():
    rng = random.Random(1)
    doc = _rich_deployment()

    result = mutate_multi_defect(doc, rng, min_defects=2, max_defects=4)

    assert result is not None
    assert 2 <= len(result.applied_rule_ids) <= 4
    assert _apply_patch(result.mutated_doc, result.patch) == result.canonical


def test_findings_match_applied_rule_ids():
    rng = random.Random(2)
    doc = _rich_deployment()

    result = mutate_multi_defect(doc, rng, min_defects=3, max_defects=3)

    assert result is not None
    assert len(result.applied_rule_ids) == 3
    assert {f.rule_id for f in result.findings} == set(result.applied_rule_ids)
    assert all(f.rule_id in RULE_IDS for f in result.findings)


def test_new_resources_are_accumulated_when_ksec001_env_variant_is_injected():
    # KSEC-001's env-variant mutator emits a companion Secret YAML in
    # new_resources -- composing it with other rules must not silently drop
    # that, or the model would be trained to reference a secretKeyRef whose
    # backing Secret was never actually shown as needing creation.
    doc = _rich_deployment()
    saw_ksec001_with_new_resources = False
    for seed in range(50):
        result = mutate_multi_defect(doc, random.Random(seed), min_defects=2, max_defects=6)
        if result is not None and "KSEC-001" in result.applied_rule_ids and result.new_resources:
            saw_ksec001_with_new_resources = True
            break
    assert saw_ksec001_with_new_resources


def test_rejects_a_base_doc_that_is_already_dirty():
    rng = random.Random(3)
    doc = _rich_deployment()
    doc["spec"]["template"]["spec"]["containers"][0]["securityContext"] = {"privileged": True}

    result = mutate_multi_defect(doc, rng, min_defects=2, max_defects=4)

    assert result is None


def test_returns_none_when_fewer_than_two_mutators_are_applicable():
    rng = random.Random(4)
    # A Service has no PodSpec at all -- none of the active mutators apply.
    doc = {"apiVersion": "v1", "kind": "Service", "metadata": {"name": "s"}, "spec": {}}

    result = mutate_multi_defect(doc, rng, min_defects=2, max_defects=4)

    assert result is None


def test_round_trip_holds_when_ksec001_env_variant_adds_a_new_field():
    # Regression test: KSEC-001's env-variant mutator's own "canonical" isn't
    # just its input doc -- it ADDS a secretKeyRef env entry (the
    # fixed-forward form) that never existed before. A doc with no env at
    # all reliably forces the env-variant (there's nothing to collide with),
    # so composing it with another rule used to silently drop that addition
    # from the overall round-trip target. See dataset/multi_mutate.py's
    # canonical_delta comment for the fix.
    doc = {
        "apiVersion": "apps/v1",
        "kind": "Deployment",
        "metadata": {"name": "d", "labels": {"app": "d"}},
        "spec": {
            "selector": {"matchLabels": {"app": "d"}},
            "template": {
                "metadata": {"labels": {"app": "d"}},
                "spec": {
                    "containers": [
                        {"name": "app", "image": "myapp:1.2.3", "resources": {"limits": {"cpu": "500m"}}}
                    ]
                },
            },
        },
    }

    found_ksec001 = False
    for seed in range(200):
        result = mutate_multi_defect(doc, random.Random(seed), min_defects=2, max_defects=4)
        if result is None or "KSEC-001" not in result.applied_rule_ids:
            continue
        found_ksec001 = True
        assert _apply_patch(result.mutated_doc, result.patch) == result.canonical

    assert found_ksec001


def _pod_with_two_containers():
    """Two containers, each independently eligible for a fresh KSEC-001
    injection (no pre-existing env), each with its own securityContext-free,
    pinned-image, matching-selector setup."""
    return {
        "apiVersion": "apps/v1",
        "kind": "Deployment",
        "metadata": {"name": "d"},
        "spec": {
            "selector": {"matchLabels": {"app": "web"}},
            "template": {
                "metadata": {"labels": {"app": "web"}},
                "spec": {
                    "containers": [
                        {"name": "app", "image": "myapp:1.2.3"},
                        {"name": "sidecar", "image": "sidecar:1.0.0"},
                    ]
                },
            },
        },
    }


def test_same_rule_can_fire_twice_across_multiple_containers():
    # Regression test for the original sql.yaml gap: a manifest can have TWO
    # separate plaintext credentials, not just two different rule types.
    doc = _pod_with_two_containers()
    found_two_ksec001 = False
    for seed in range(200):
        result = mutate_multi_defect(doc, random.Random(seed), min_defects=2, max_defects=6)
        if result is not None and result.applied_rule_ids.count("KSEC-001") >= 2:
            found_two_ksec001 = True
            assert _apply_patch(result.mutated_doc, result.patch) == result.canonical
            assert sum(1 for f in result.findings if f.rule_id == "KSEC-001") >= 2
            break
    assert found_two_ksec001


def test_same_rule_can_fire_twice_on_the_same_single_container():
    # Matches sql.yaml's exact shape: ONE container, two separate
    # credentials (e.g. MYSQL_ROOT_PASSWORD and MYSQL_PASSWORD).
    # _rich_deployment has exactly one container, so any doubled KSEC-001
    # here necessarily lands on that same container.
    doc = _rich_deployment()
    found_two_ksec001 = False
    for seed in range(300):
        result = mutate_multi_defect(doc, random.Random(seed), min_defects=2, max_defects=6)
        if result is not None and result.applied_rule_ids.count("KSEC-001") >= 2:
            found_two_ksec001 = True
            assert sum(1 for f in result.findings if f.rule_id == "KSEC-001") >= 2
            assert _apply_patch(result.mutated_doc, result.patch) == result.canonical
            env = result.mutated_doc["spec"]["template"]["spec"]["containers"][0].get("env")
            if env:  # only meaningful when at least one hit was the env-variant
                names = [e["name"] for e in env]
                assert len(names) == len(set(names))  # distinct credential names, same container
            break
    assert found_two_ksec001


def test_repeat_attempt_on_exhausted_hard_precondition_rule_is_skipped_not_fatal():
    # _rich_deployment has exactly one container -- once KSEC-002 fires once,
    # a second attempt hits mutate_ksec002's "assert not detect_ksec002(...)"
    # precondition. That must be caught and skipped, not blow up the whole
    # composition.
    doc = _rich_deployment()
    for seed in range(200):
        result = mutate_multi_defect(doc, random.Random(seed), min_defects=2, max_defects=6)
        if result is not None:
            assert result.applied_rule_ids.count("KSEC-002") <= 1


def test_multiple_runs_produce_varied_rule_combinations():
    doc = _rich_deployment()
    combos = set()
    for seed in range(30):
        rng = random.Random(seed)
        result = mutate_multi_defect(doc, rng, min_defects=2, max_defects=3)
        if result is not None:
            combos.add(tuple(sorted(result.applied_rule_ids)))

    assert len(combos) > 1


def _db_deployment():
    """Eligible for KSEC-010 and KSEC-011 as well as the generic rules, so
    compositions exercise the new rules alongside the old ones."""
    return {
        "apiVersion": "apps/v1",
        "kind": "Deployment",
        "metadata": {"name": "db"},
        "spec": {
            "selector": {"matchLabels": {"app": "db"}},
            "template": {
                "metadata": {"labels": {"app": "db"}},
                "spec": {
                    "containers": [
                        {
                            "name": "db",
                            "image": "postgres:15",
                            "ports": [{"containerPort": 5432}],
                            "readinessProbe": {"tcpSocket": {"port": 5432}},
                            "env": [
                                {
                                    "name": "POSTGRES_PASSWORD",
                                    "valueFrom": {"secretKeyRef": {"name": "pg", "key": "password"}},
                                }
                            ],
                            "resources": {
                                "requests": {"cpu": "100m", "memory": "64Mi"},
                                "limits": {"cpu": "200m", "memory": "128Mi"},
                            },
                        }
                    ]
                },
            },
        },
    }


def test_new_rules_compose_with_the_others():
    applied = set()
    successes = 0
    for seed in range(200):
        result = mutate_multi_defect(_db_deployment(), random.Random(seed), min_defects=2, max_defects=6)
        if result is None:
            continue
        successes += 1
        applied |= set(result.applied_rule_ids)
        assert _apply_patch(result.mutated_doc, result.patch) == result.canonical
        assert sorted(f.rule_id for f in result.findings) == sorted(result.applied_rule_ids)
    assert successes > 150
    assert {"KSEC-010", "KSEC-011"} <= applied


def _service_for(workload):
    labels = workload["spec"]["template"]["metadata"]["labels"]
    return {"apiVersion": "v1", "kind": "Service", "metadata": {"name": "s"}, "spec": {"selector": dict(labels), "ports": [{"port": 80}]}}


def _files():
    config_map = {"apiVersion": "v1", "kind": "ConfigMap", "metadata": {"name": "cfg"}, "data": {"a": "b"}}
    rich, db = _rich_deployment(), _db_deployment()
    db_pod_spec = db["spec"]["template"]["spec"]
    db_pod_spec["containers"][0]["volumeMounts"] = [{"name": "data", "mountPath": "/var/lib/postgresql/data"}]
    db_pod_spec["volumes"] = [{"name": "data", "emptyDir": {}}]
    db_pod_spec["containers"][0].setdefault("env", []).append({"name": "PGDATA", "value": "/var/lib/postgresql/data/pgdata"})
    return {
        "workload+service": [rich, _service_for(rich)],
        "service+db+configmap": [_service_for(db), db, config_map],
    }


def _apply_file_patch(docs, patch):
    docs = copy.deepcopy(docs)
    for i in range(len(docs)):
        ops = [{k: v for k, v in op.to_dict().items() if k != "doc"} for op in patch if op.doc == i]
        if ops:
            docs[i] = jsonpatch.apply_patch(docs[i], ops)
    return docs


@pytest.mark.parametrize("name", ["workload+service", "service+db+configmap"])
@pytest.mark.parametrize("seed", range(40))
def test_multi_defect_file_round_trips_and_counts_every_finding(name, seed):
    docs = _files()[name]
    result = mutate_multi_defect_file(copy.deepcopy(docs), random.Random(seed), min_defects=2, max_defects=5)
    if result is None:
        return
    assert _apply_file_patch(result.mutated_docs, result.patch) == result.canonical_docs
    assert len([f for f in detect_file(result.mutated_docs) if f.rule_id in RULE_IDS]) == len(result.findings)
    assert [f for f in detect_file(result.canonical_docs) if f.rule_id in RULE_IDS] == []


def test_multi_defect_file_reaches_documents_beyond_index_zero_and_the_service_rule():
    seen_docs, service_findings = set(), 0
    for seed in range(60):
        result = mutate_multi_defect_file(_files()["service+db+configmap"], random.Random(seed))
        if result is None:
            continue
        seen_docs |= {f.doc for f in result.findings}
        service_findings += sum(f.doc == 0 and f.rule_id == "KSEC-006" for f in result.findings)
    assert 1 in seen_docs and service_findings


@pytest.mark.parametrize("rule_id", sorted(RULE_IDS - {"KSEC-004"}))
def test_single_defect_file_injects_exactly_that_rule(rule_id):
    produced = 0
    for seed in range(15):
        for docs in _files().values():
            result = mutate_single_defect_file(copy.deepcopy(docs), random.Random(seed), rule_id)
            if result is None:
                continue
            produced += 1
            assert {f.rule_id for f in result.findings} == {rule_id}
            assert _apply_file_patch(result.mutated_docs, result.patch) == result.canonical_docs
    assert produced


def test_min_defects_one_allows_single_defect_positives():
    counts = set()
    for seed in range(60):
        result = mutate_multi_defect(_rich_deployment(), random.Random(seed), min_defects=1, max_defects=1)
        if result is not None:
            counts.add(len(result.applied_rule_ids))
            assert _apply_patch(result.mutated_doc, result.patch) == result.canonical
    assert counts == {1}


def test_large_defect_counts_round_trip():
    reached = 0
    for seed in range(60):
        result = mutate_multi_defect(_rich_deployment(), random.Random(seed), min_defects=6, max_defects=8)
        if result is None:
            continue
        reached = max(reached, len(result.applied_rule_ids))
        assert _apply_patch(result.mutated_doc, result.patch) == result.canonical
    assert reached >= 6


def test_rule_weights_favour_credentials_and_image_tags():
    import collections

    first_picks = collections.Counter()
    for seed in range(400):
        result = mutate_multi_defect(_rich_deployment(), random.Random(seed), min_defects=1, max_defects=1)
        if result is not None:
            first_picks[result.applied_rule_ids[0]] += 1
    others = [n for rule, n in first_picks.items() if rule not in ("KSEC-001", "KSEC-005")]
    assert first_picks["KSEC-001"] > max(others) and first_picks["KSEC-005"] > max(others)


def _collapsing_db():
    # KSEC-011's fix collapses BOTH satisfying entries into one, shortening
    # env; PGDATA after them is something KSEC-012 fixes by index. Before the
    # per-step round-trip check, KSEC-012 then KSEC-011 left KSEC-012's op
    # pointing past the end of env ("can't replace outside of list").
    doc = _db_deployment()
    container = doc["spec"]["template"]["spec"]["containers"][0]
    container["env"] = [
        {"name": "POSTGRES_HOST_AUTH_METHOD", "value": "md5"},
        {"name": "POSTGRES_PASSWORD_FILE", "value": "/run/secrets/pg"},
        {"name": "TZ", "value": "UTC"},
        {"name": "PGDATA", "value": "/var/lib/postgresql/data/pgdata"},
    ]
    return doc


@pytest.mark.parametrize("seed", range(80))
def test_compositions_that_shorten_a_list_still_round_trip(seed):
    doc = _collapsing_db()
    try:
        result = mutate_multi_defect(copy.deepcopy(doc), random.Random(seed), min_defects=2, max_defects=8)
    except AssertionError:
        return
    if result is None:
        return
    assert _apply_patch(result.mutated_doc, result.patch) == result.canonical


@pytest.mark.parametrize("seed", range(40))
def test_file_compositions_that_shorten_a_list_still_round_trip(seed):
    db = _collapsing_db()
    docs = [db, _service_for(db)]
    result = mutate_multi_defect_file(copy.deepcopy(docs), random.Random(seed), 2, 8)
    if result is None:
        return
    assert _apply_file_patch(result.mutated_docs, result.patch) == result.canonical_docs
