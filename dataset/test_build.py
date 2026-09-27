import argparse
import json
import random

import pandas as pd
import yaml

from dataset.build import Record, _is_harvestable_key_name, build, doc_to_yaml, find_sibling_bundles, load_synthetic_records
from dataset.detect import detect_file


def test_load_synthetic_records_parses_curated_jsonl(tmp_path):
    p = tmp_path / "rbac.curated.jsonl"
    record = {
        "index": 0,
        "mode": "rbac",
        "manifest_yaml": "apiVersion: rbac.authorization.k8s.io/v1\nkind: Role\nmetadata:\n  name: r\nrules: []\n",
    }
    p.write_text(json.dumps(record) + "\n", encoding="utf-8")

    records = load_synthetic_records(tmp_path)
    assert len(records) == 1
    assert records[0].doc["kind"] == "Role"
    assert records[0].repo == "synthetic:rbac:0:0"


def test_load_synthetic_records_skips_non_curated_files(tmp_path):
    p = tmp_path / "rbac.jsonl"  # no .curated suffix -- should not be read
    p.write_text(json.dumps({"manifest_yaml": "kind: Role\napiVersion: v1\n"}) + "\n", encoding="utf-8")
    assert load_synthetic_records(tmp_path) == []


def test_load_synthetic_records_missing_dir_returns_empty(tmp_path):
    assert load_synthetic_records(tmp_path / "does-not-exist") == []


def test_build_survives_malformed_image_without_crashing(tmp_path):
    # Regression test: a malformed image reference (repo:sha256:hash instead
    # of repo@sha256:hash -- something a noisy LLM generation produced in
    # practice) used to crash the whole build via an unguarded assert in
    # mutate_ksec005. Both the mutator itself (now self-healing, see
    # test_mutate.py) and build()'s defensive catch protect against this;
    # here we just confirm the end-to-end run no longer crashes on it.
    malformed_doc = {
        "apiVersion": "v1",
        "kind": "Pod",
        "metadata": {"name": "bad-image-pod"},
        "spec": {
            "containers": [
                {"name": "app", "image": "repo/app:sha256:" + "a" * 64}
            ]
        },
    }

    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    df = pd.DataFrame(
        {
            "content": [json.dumps(malformed_doc)],
            "max_stars_repo_name": ["repoA"],
            "max_stars_repo_path": ["a.yaml"],
        }
    )
    df.to_parquet(corpus_dir / "shard.parquet")

    args = argparse.Namespace(
        corpus_dir=str(corpus_dir),
        synthetic_dir=None,
        output_dir=str(tmp_path / "output"),
        limit=None,
        total=10,
        negative_ratio=0.0,
        train_ratio=1.0,
        val_ratio=0.0,
        seed=1,
    )
    diagnostic = build(args)  # must not raise
    assert diagnostic["usable_canonical_docs"] == 1


def test_build_multi_defect_strategy_emits_multi_finding_examples(tmp_path):
    # End-to-end smoke test for --strategy multi-defect: every emitted
    # positive example must carry 2+ findings, and every patch must still
    # round-trip (build() raises on a round-trip failure, so just not
    # raising already proves that -- we additionally check the finding
    # counts and the diagnostic's defect_count_distribution).
    docs = [
        {
            "apiVersion": "apps/v1",
            "kind": "Deployment",
            "metadata": {"name": f"d{i}"},
            "spec": {
                "selector": {"matchLabels": {"app": f"web{i}"}},
                "template": {
                    "metadata": {"labels": {"app": f"web{i}"}},
                    "spec": {
                        "containers": [
                            {
                                "name": "app",
                                "image": f"myapp{i}:1.2.3",
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
        for i in range(20)
    ]

    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    repos = [f"repo{i}" for i in range(len(docs))]
    df = pd.DataFrame(
        {
            "content": [json.dumps(d) for d in docs],
            "max_stars_repo_name": repos,
            "max_stars_repo_path": [f"{r}.yaml" for r in repos],
        }
    )
    df.to_parquet(corpus_dir / "shard.parquet")

    args = argparse.Namespace(
        corpus_dir=str(corpus_dir),
        synthetic_dir=None,
        strategy="multi-defect",
        min_defects=2,
        max_defects=4,
        output_dir=str(tmp_path / "output-multi-defect"),
        limit=None,
        total=15,
        negative_ratio=0.2,
        train_ratio=1.0,
        val_ratio=0.0,
        seed=1,
    )
    diagnostic = build(args)

    assert diagnostic["strategy"] == "multi-defect"
    assert diagnostic["defect_count_distribution"]
    assert all(count >= 2 for count in diagnostic["defect_count_distribution"])

    path = tmp_path / "output-multi-defect" / "train.jsonl"
    examples = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    positive_examples = [e for e in examples if json.loads(e["messages"][2]["content"])["findings"]]
    assert positive_examples
    for example in positive_examples:
        findings = json.loads(example["messages"][2]["content"])["findings"]
        assert len(findings) >= 2
        assert len({f["rule_id"] for f in findings}) == len(findings)


def test_build_skips_a_mutator_that_raises_assertion_error(tmp_path, monkeypatch):
    # Verifies build()'s defensive catch directly: a mutator raising
    # AssertionError (its own internal invariant failing on some document)
    # must not crash the whole run -- it should be counted and skipped.
    import dataset.build as build_module

    def _always_fails(doc, rng, doc_index=0, candidate_names=None):
        assert False, "synthetic failure for the resilience test"

    monkeypatch.setitem(build_module.MUTATORS, "KSEC-001", _always_fails)

    doc = {
        "apiVersion": "v1",
        "kind": "Pod",
        "metadata": {"name": "p"},
        "spec": {"containers": [{"name": "app", "image": "myapp:1.2.3"}]},
    }
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    df = pd.DataFrame(
        {
            "content": [json.dumps(doc)],
            "max_stars_repo_name": ["repoA"],
            "max_stars_repo_path": ["a.yaml"],
        }
    )
    df.to_parquet(corpus_dir / "shard.parquet")

    args = argparse.Namespace(
        corpus_dir=str(corpus_dir),
        synthetic_dir=None,
        output_dir=str(tmp_path / "output"),
        limit=None,
        total=10,
        negative_ratio=0.0,
        train_ratio=1.0,
        val_ratio=0.0,
        seed=1,
    )
    diagnostic = build(args)  # must not raise
    assert diagnostic["mutation_precondition_failures"] >= 1


def test_is_harvestable_key_name_accepts_sensitive_key_match():
    assert _is_harvestable_key_name("MYSQL_ROOT_PASSWORD", "value under sensitive key 'MYSQL_ROOT_PASSWORD'")


def test_is_harvestable_key_name_rejects_high_entropy_only():
    # high-entropy hits don't require the key itself to look sensitive --
    # it could be a list index or an unrelated field, not worth harvesting.
    assert not _is_harvestable_key_name("0", "high entropy (likely random secret)")
    assert not _is_harvestable_key_name("name", "high entropy (likely random secret)")


def test_is_harvestable_key_name_rejects_non_identifier_strings():
    assert not _is_harvestable_key_name("0", "value under sensitive key '0'")
    assert not _is_harvestable_key_name("a b", "value under sensitive key 'a b'")
    assert not _is_harvestable_key_name("", "value under sensitive key ''")


def test_is_harvestable_key_name_accepts_pem_and_connection_string_keys():
    assert _is_harvestable_key_name("tls.key", "PEM private key")
    assert _is_harvestable_key_name("DATABASE_URL", "connection string with embedded credentials")


def test_build_harvests_real_key_names_and_uses_them_for_ksec001(tmp_path, monkeypatch):
    # A real corpus doc leaking a distinctively-named credential should have
    # that NAME (never the value) show up as an injectable identifier
    # elsewhere in the dataset -- confirms the harvest -> KSEC-001 wiring
    # end to end, not just the filter function in isolation. The base pool
    # is emptied so the harvested name is the *only* env-variant candidate,
    # otherwise it would just be one of 50+ names rng.choice could pick and
    # the test would be flaky by design rather than by bug.
    import dataset.build as build_module

    monkeypatch.setattr(build_module, "FAKE_SECRET_VAR_NAMES", ())

    dirty_doc = {
        "apiVersion": "v1",
        "kind": "Secret",
        "metadata": {"name": "leaky"},
        "stringData": {"MY_DISTINCTIVE_HARVESTED_TOKEN": "S3cr3tR34lLeak99xyz"},
    }
    clean_docs = [
        {
            "apiVersion": "v1",
            "kind": "Pod",
            "metadata": {"name": f"p{i}"},
            "spec": {"containers": [{"name": "app", "image": f"myapp{i}:1.0.0"}]},
        }
        for i in range(30)
    ]

    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    contents = [json.dumps(dirty_doc)] + [json.dumps(d) for d in clean_docs]
    repos = [f"repo{i}" for i in range(len(contents))]
    df = pd.DataFrame(
        {"content": contents, "max_stars_repo_name": repos, "max_stars_repo_path": [f"{r}.yaml" for r in repos]}
    )
    df.to_parquet(corpus_dir / "shard.parquet")

    args = argparse.Namespace(
        corpus_dir=str(corpus_dir),
        synthetic_dir=None,
        output_dir=str(tmp_path / "output"),
        limit=None,
        total=30,
        negative_ratio=0.0,
        train_ratio=1.0,
        val_ratio=0.0,
        seed=1,
    )
    diagnostic = build(args)
    assert diagnostic["harvested_credential_key_names"] >= 1

    found_harvested_name = False
    for split in ("train", "val", "test"):
        path = tmp_path / "output" / f"{split}.jsonl"
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            example = json.loads(line)
            if "MY_DISTINCTIVE_HARVESTED_TOKEN" in example["messages"][1]["content"]:
                found_harvested_name = True
    assert found_harvested_name


def test_build_drops_docs_with_a_preexisting_semantic_finding(tmp_path):
    # normalize.py only guarantees rules 001-005 are clean. A real document
    # already violating a semantic rule (here: postgres without
    # POSTGRES_PASSWORD, KSEC-011) must never be emitted -- as a negative it
    # would be labeled clean despite a real defect.
    broken = {
        "apiVersion": "v1",
        "kind": "Pod",
        "metadata": {"name": "db"},
        "spec": {"containers": [{"name": "db", "image": "postgres:15"}]},
    }
    clean = {
        "apiVersion": "v1",
        "kind": "Pod",
        "metadata": {"name": "web"},
        # A different shape from `broken`, so structural dedup keeps both.
        "spec": {"containers": [{"name": "web", "image": "myapp:1.2.3", "ports": [{"containerPort": 8080}]}]},
    }
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    pd.DataFrame(
        {
            "content": [json.dumps(broken), json.dumps(clean)],
            "max_stars_repo_name": ["repoA", "repoB"],
            "max_stars_repo_path": ["a.yaml", "b.yaml"],
        }
    ).to_parquet(corpus_dir / "shard.parquet")

    args = argparse.Namespace(
        corpus_dir=str(corpus_dir),
        synthetic_dir=None,
        output_dir=str(tmp_path / "output"),
        limit=None,
        total=10,
        negative_ratio=1.0,
        train_ratio=1.0,
        val_ratio=0.0,
        seed=1,
    )
    diagnostic = build(args)
    assert diagnostic["dropped_preexisting_semantic_finding"] == 1
    assert diagnostic["usable_canonical_docs"] == 1

    emitted = [json.loads(line) for line in (tmp_path / "output" / "train.jsonl").read_text().splitlines()]
    assert emitted
    assert all("postgres" not in ex["messages"][1]["content"] for ex in emitted)


def _workload(name, app):
    return {
        "apiVersion": "apps/v1",
        "kind": "Deployment",
        "metadata": {"name": name},
        "spec": {
            "selector": {"matchLabels": {"app": app}},
            "template": {
                "metadata": {"labels": {"app": app}},
                "spec": {
                    "containers": [
                        {
                            "name": "app",
                            "image": f"{name}:1.2.3",
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


def _service(app, name="svc"):
    return {"apiVersion": "v1", "kind": "Service", "metadata": {"name": name}, "spec": {"selector": {"app": app}, "ports": [{"port": 80}]}}


def test_find_sibling_bundles_pairs_a_service_with_the_one_workload_it_selects():
    records = [
        Record(_workload("web", "web"), "repoA", "k8s/deployment.yaml"),
        Record(_workload("db", "db"), "repoA", "k8s/db.yaml"),
        Record(_service("web"), "repoA", "k8s/service.yaml"),
        Record({"apiVersion": "v1", "kind": "ConfigMap", "metadata": {"name": "c"}, "data": {}}, "repoA", "k8s/cm.yaml"),
        # Same repo, different directory: never paired with k8s/.
        Record(_service("db"), "repoA", "other/service.yaml"),
        # Selects nothing in its directory.
        Record(_service("ghost"), "repoA", "k8s/ghost.yaml"),
        # Synthetic records have no sibling files.
        Record(_service("web"), "synthetic:base:0:0", "base.curated.jsonl"),
    ]
    for seed in range(20):
        bundles = find_sibling_bundles(records, random.Random(seed))
        assert len(bundles) == 1
        kinds = [d["kind"] for d in bundles[0].docs]
        assert sorted(k for k in kinds if k != "ConfigMap") == ["Deployment", "Service"]
        assert bundles[0].docs[kinds.index("Deployment")]["metadata"]["name"] == "web"
        assert bundles[0].repo == "repoA"


def test_doc_to_yaml_joins_a_list_as_a_multi_document_file():
    docs = [_service("web"), _workload("web", "web")]
    text = doc_to_yaml(docs)
    assert list(yaml.safe_load_all(text)) == docs
    assert doc_to_yaml(docs[0]) == yaml.safe_dump(docs[0], sort_keys=False, default_flow_style=False)


def _sibling_corpus(tmp_path, n=12):
    rows = []
    for i in range(n):
        repo = f"repo{i}"
        # Distinct shapes per repo so structural dedup keeps every bundle.
        workload = _workload(f"app{i}", f"app{i}")
        workload["spec"]["template"]["spec"]["containers"][0]["ports"] += [{"containerPort": 9000 + j} for j in range(i)]
        rows.append((workload, repo, "deploy/deployment.yaml"))
        rows.append((_service(f"app{i}"), repo, "deploy/service.yaml"))
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir(parents=True)
    pd.DataFrame(
        {
            "content": [json.dumps(d) for d, _, _ in rows],
            "max_stars_repo_name": [r for _, r, _ in rows],
            "max_stars_repo_path": [p for _, _, p in rows],
        }
    ).to_parquet(corpus_dir / "shard.parquet")
    return corpus_dir


def _read_examples(output_dir):
    return [json.loads(line) for line in (output_dir / "train.jsonl").read_text(encoding="utf-8").splitlines()]


def test_build_emits_multi_document_examples_that_round_trip(tmp_path):
    for strategy in ("single-defect", "multi-defect"):
        output_dir = tmp_path / strategy
        args = argparse.Namespace(
            corpus_dir=str(_sibling_corpus(tmp_path / strategy)),
            synthetic_dir=None,
            strategy=strategy,
            min_defects=2,
            max_defects=4,
            output_dir=str(output_dir),
            limit=None,
            total=20,
            negative_ratio=0.25,
            multi_doc_ratio=0.5,
            train_ratio=1.0,
            val_ratio=0.0,
            seed=3,
        )
        diagnostic = build(args)
        assert diagnostic["usable_bundles"] == 12
        assert diagnostic["multi_doc_examples"]["positive"] > 0
        assert diagnostic["multi_doc_examples"]["negative"] > 0
        assert diagnostic["docs_per_example"].get(2)

        multi = [e for e in _read_examples(output_dir) if "\n---\n" in e["messages"][1]["content"]]
        assert multi
        for example in multi:
            docs = list(yaml.safe_load_all(example["messages"][1]["content"]))
            response = json.loads(example["messages"][2]["content"])
            found = sorted((f.rule_id, f.doc, f.path) for f in detect_file(docs))
            assert found == sorted((f["rule_id"], f["doc"], f["path"]) for f in response["findings"])


def test_build_without_multi_doc_ratio_emits_single_documents_only(tmp_path):
    args = argparse.Namespace(
        corpus_dir=str(_sibling_corpus(tmp_path)),
        synthetic_dir=None,
        output_dir=str(tmp_path / "output"),
        limit=None,
        total=10,
        negative_ratio=0.3,
        train_ratio=1.0,
        val_ratio=0.0,
        seed=3,
    )
    diagnostic = build(args)
    assert "usable_bundles" not in diagnostic
    assert all("\n---\n" not in e["messages"][1]["content"] for e in _read_examples(tmp_path / "output"))
