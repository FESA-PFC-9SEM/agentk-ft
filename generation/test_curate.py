import json
import shutil

import pytest

from generation.curate import curate_record, looks_suspicious_naive, parse_docs, run_kubeconform

HAS_KUBECONFORM = shutil.which("kubeconform") is not None

CLEAN_POD_YAML = """\
apiVersion: v1
kind: Pod
metadata:
  name: app
spec:
  containers:
  - name: app
    image: myapp:1.2.3
"""

DIRTY_POD_YAML = """\
apiVersion: v1
kind: Pod
metadata:
  name: app
spec:
  containers:
  - name: app
    image: myapp:1.2.3
    env:
    - name: DB_PASSWORD
      value: S3cr3tR34lLeak99
"""

HARD_NEGATIVE_YAML = """\
apiVersion: v1
kind: Pod
metadata:
  name: app
spec:
  containers:
  - name: app
    image: myapp:1.2.3
    env:
    - name: API_TOKEN_HEADER
      value: "X-Api-Token"
    - name: SESSION_TIMEOUT
      value: "3600"
"""

NOT_A_MANIFEST = "this is not a manifest, it's just a paragraph of prose explaining something."


def test_parse_docs_valid():
    docs = parse_docs(CLEAN_POD_YAML)
    assert docs is not None
    assert docs[0]["kind"] == "Pod"


def test_parse_docs_invalid_yaml():
    assert parse_docs("not: [valid: yaml") is None


def test_parse_docs_prose_rejected():
    assert parse_docs(NOT_A_MANIFEST) is None


def test_looks_suspicious_naive_true_for_sensitive_key():
    docs = parse_docs(DIRTY_POD_YAML)
    assert looks_suspicious_naive(docs)


def test_looks_suspicious_naive_false_for_boring_manifest():
    docs = parse_docs(CLEAN_POD_YAML)
    assert not looks_suspicious_naive(docs)


def test_curate_record_rejects_invalid_yaml():
    record = {"manifest_yaml": NOT_A_MANIFEST, "mode": "base"}
    ok, reason = curate_record(record, set(), "kubeconform", "1.29.0", skip_kubeconform=True, corpus_hashes=None)
    assert not ok
    assert reason == "invalid_yaml"


def test_curate_record_rejects_real_secret_in_base_mode():
    record = {"manifest_yaml": DIRTY_POD_YAML, "mode": "base"}
    ok, reason = curate_record(record, set(), "kubeconform", "1.29.0", skip_kubeconform=True, corpus_hashes=None)
    assert not ok
    assert reason == "real_secret_detected"


def test_curate_record_accepts_clean_base_manifest():
    record = {"manifest_yaml": CLEAN_POD_YAML, "mode": "base"}
    ok, reason = curate_record(record, set(), "kubeconform", "1.29.0", skip_kubeconform=True, corpus_hashes=None)
    assert ok
    assert reason == "ok"


def test_curate_record_rejects_duplicate_within_batch():
    record = {"manifest_yaml": CLEAN_POD_YAML, "mode": "base"}
    seen = set()
    ok1, _ = curate_record(record, seen, "kubeconform", "1.29.0", skip_kubeconform=True, corpus_hashes=None)
    ok2, reason2 = curate_record(record, seen, "kubeconform", "1.29.0", skip_kubeconform=True, corpus_hashes=None)
    assert ok1
    assert not ok2
    assert reason2 == "structural_duplicate_in_batch"


def test_curate_record_hard_negative_accepted_when_suspicious_but_clean():
    record = {"manifest_yaml": HARD_NEGATIVE_YAML, "mode": "hard-negative"}
    ok, reason = curate_record(record, set(), "kubeconform", "1.29.0", skip_kubeconform=True, corpus_hashes=None)
    assert ok
    assert reason == "ok"


def test_curate_record_hard_negative_rejected_when_not_suspicious():
    record = {"manifest_yaml": CLEAN_POD_YAML, "mode": "hard-negative"}
    ok, reason = curate_record(record, set(), "kubeconform", "1.29.0", skip_kubeconform=True, corpus_hashes=None)
    assert not ok
    assert reason == "hard_negative_not_suspicious"


def test_curate_record_hard_negative_still_rejects_real_secret():
    record = {"manifest_yaml": DIRTY_POD_YAML, "mode": "hard-negative"}
    ok, reason = curate_record(record, set(), "kubeconform", "1.29.0", skip_kubeconform=True, corpus_hashes=None)
    assert not ok
    assert reason == "real_secret_detected"


def test_curate_record_rejects_corpus_duplicate():
    from dataset.dedup import skeleton_hash

    docs = parse_docs(CLEAN_POD_YAML)
    corpus_hashes = {skeleton_hash(docs[0])}
    record = {"manifest_yaml": CLEAN_POD_YAML, "mode": "base"}
    ok, reason = curate_record(record, set(), "kubeconform", "1.29.0", skip_kubeconform=True, corpus_hashes=corpus_hashes)
    assert not ok
    assert reason == "structural_duplicate_in_corpus"


@pytest.mark.skipif(not HAS_KUBECONFORM, reason="kubeconform not installed in this environment")
def test_kubeconform_accepts_valid_pod():
    ok, reason = run_kubeconform(CLEAN_POD_YAML, "kubeconform", "1.29.0")
    assert ok, reason


@pytest.mark.skipif(not HAS_KUBECONFORM, reason="kubeconform not installed in this environment")
def test_kubeconform_rejects_malformed_resource():
    bad_yaml = "apiVersion: v1\nkind: Pod\nmetadata:\n  name: p\nspec:\n  containers: \"not-a-list\"\n"
    ok, _reason = run_kubeconform(bad_yaml, "kubeconform", "1.29.0")
    assert not ok
