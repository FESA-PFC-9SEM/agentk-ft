import argparse
import json

import pandas as pd

from dataset.build import _is_harvestable_key_name, build, load_synthetic_records


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
