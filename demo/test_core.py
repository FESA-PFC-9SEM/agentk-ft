from demo.core import apply_patch_to_manifest, patch_ops_for_finding, unmatched_patch_ops

MANIFEST = """\
apiVersion: v1
kind: Pod
spec:
  containers:
  - image: x:latest
    env:
    - name: A
      value: a
    - name: PW
      value: secret
"""


def test_apply_patch_to_manifest_selective():
    ops = [{"doc": 0, "op": "replace", "path": "/spec/containers/0/image", "value": "x:1.0"}]
    patched, error = apply_patch_to_manifest(MANIFEST, ops)
    assert error is None
    assert "x:1.0" in patched
    assert "value: secret" in patched  # untouched


def test_apply_patch_to_manifest_multi_doc_routes_by_doc_index():
    two_docs = "apiVersion: v1\nkind: A\nspec:\n  x: 1\n---\napiVersion: v1\nkind: B\nspec:\n  y: 1\n"
    ops = [{"doc": 1, "op": "replace", "path": "/spec/y", "value": 2}]
    patched, error = apply_patch_to_manifest(two_docs, ops)
    assert error is None
    assert "x: 1" in patched  # doc 0 untouched
    assert "y: 2" in patched  # doc 1 patched


def test_apply_patch_to_manifest_reports_error_instead_of_raising():
    # A patch path that doesn't exist in the actual document -- e.g. a
    # cross-document KSEC-006 fix the model was never trained to emit
    # correctly (see README.md's known limitations) -- must come back as an
    # (None, error) pair, never as an uncaught exception into the caller
    # (this crashed the Streamlit app before core.py caught Exception broadly
    # instead of just jsonpatch.JsonPatchException).
    ops = [{"doc": 0, "op": "replace", "path": "/spec/containers/0/nonexistent/matchLabels", "value": {}}]
    patched, error = apply_patch_to_manifest(MANIFEST, ops)
    assert patched is None
    assert error is not None


def test_apply_patch_to_manifest_bad_yaml_reports_error():
    patched, error = apply_patch_to_manifest("not: valid: yaml: [", [])
    assert patched is None
    assert error is not None


def test_patch_ops_for_finding_matches_ancestor_path():
    finding = {"doc": 0, "path": "/spec/containers/0/env/1/value"}
    patch = [{"doc": 0, "op": "replace", "path": "/spec/containers/0/env/1", "value": {}}]
    assert patch_ops_for_finding(finding, patch) == patch


def test_patch_ops_for_finding_ignores_other_docs():
    finding = {"doc": 0, "path": "/spec/x"}
    patch = [{"doc": 1, "op": "replace", "path": "/spec/x", "value": 1}]
    assert patch_ops_for_finding(finding, patch) == []


def test_unmatched_patch_ops_returns_ops_with_no_finding():
    findings = [{"doc": 0, "path": "/spec/a"}]
    patch = [
        {"doc": 0, "op": "replace", "path": "/spec/a", "value": 1},
        {"doc": 0, "op": "replace", "path": "/spec/b", "value": 2},
    ]
    assert unmatched_patch_ops(findings, patch) == [patch[1]]
