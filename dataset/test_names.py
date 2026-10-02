import random

import pytest

from dataset.detect import detect_ksec012
from dataset.mutate import mutate_ksec012
from dataset.names import is_style_variant, misspelled_binary, misspelled_path, one_edit_apart, paths_in


@pytest.mark.parametrize(
    "a,b,expected",
    [
        ("python5", "python3", True),  # substitution
        ("hom", "home", True),  # deletion
        ("pyhton", "python", True),  # adjacent swap
        ("homee", "home", True),  # duplication
        ("python", "python", False),
        ("variavel", "var", False),
        ("ab", "ba", True),
    ],
)
def test_one_edit_apart(a, b, expected):
    assert one_edit_apart(a, b) is expected


def test_style_variants_are_not_typos():
    assert is_style_variant("target1", "target")
    assert is_style_variant("work_dir", "work-dir")
    assert is_style_variant("containers", "container")
    assert not is_style_variant("python5", "python3")


def test_scenario_misspellings_are_caught_with_their_fix():
    assert misspelled_binary("python5") == "python3"
    assert misspelled_path("/hom/auto-reload-nginx.sh") == "/home/auto-reload-nginx.sh"
    assert (
        misspelled_path("/variavel/run/secrets/kubernetes.io/serviceaccount/ca.crt")
        == "/var/run/secrets/kubernetes.io/serviceaccount/ca.crt"
    )


@pytest.mark.parametrize("binary", ["python3", "python", "nginx", "ssh", "npx", "gcp", "my-custom-tool"])
def test_known_short_or_unrelated_binaries_are_not_flagged(binary):
    assert misspelled_binary(binary) is None


@pytest.mark.parametrize(
    "path",
    ["/home/app", "/var/lib/data", "/run/secrets/kubernetes.io/serviceaccount/token", "/target1", "/work_dir/x", "/app/main.py"],
)
def test_valid_or_style_variant_paths_are_not_flagged(path):
    assert misspelled_path(path) is None


def test_paths_in_finds_absolute_paths_inside_strings():
    assert paths_in("--config=/etc/app/conf.yaml --log /var/log/x") == ["/etc/app/conf.yaml", "/var/log/x"]
    assert paths_in("http://example.com/api") == []


def _pod(command=None, env=None):
    container = {"name": "c", "image": "app:1.0"}
    if command:
        container["command"] = command
    if env:
        container["env"] = env
    return {"apiVersion": "v1", "kind": "Pod", "metadata": {"name": "p"}, "spec": {"containers": [container]}}


def test_detect_ksec012_flags_binary_and_env_path_once_per_field():
    findings = detect_ksec012(
        _pod(["python5", "-c", "x"], [{"name": "CA", "value": "/variavel/run/secrets/kubernetes.io/serviceaccount/ca.crt"}])
    )
    assert [(f.path, f.severity) for f in findings] == [
        ("/spec/containers/0/command/0", "high"),
        ("/spec/containers/0/env/0/value", "medium"),
    ]
    assert detect_ksec012(_pod(["python3", "-c", "x"], [{"name": "CA", "value": "/var/run/secrets/x"}])) == []


@pytest.mark.parametrize("seed", range(25))
def test_mutate_ksec012_round_trips_with_one_finding(seed):
    import jsonpatch

    doc = _pod(["python3", "/app/main.py"], [{"name": "DATA_DIR", "value": "/var/lib/app"}])
    result = mutate_ksec012(doc, random.Random(seed))
    assert result is not None
    assert len(result.findings) == 1
    ops = [{k: v for k, v in op.to_dict().items() if k != "doc"} for op in result.patch]
    assert jsonpatch.apply_patch(result.mutated_doc, ops) == doc


def test_mutate_ksec012_not_applicable_without_common_names():
    assert mutate_ksec012(_pod(["my-custom-tool"]), random.Random(0)) is None
