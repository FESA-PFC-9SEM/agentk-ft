import argparse
import json

import pytest

from finetune.infer import load_manifest_and_ground_truth

CLEAN_RESPONSE = {"findings": [], "patch": [], "new_resources": [], "notes": []}


def _args(**overrides):
    base = dict(manifest=None, stdin=False, test_file=None, index=0)
    base.update(overrides)
    return argparse.Namespace(**base)


def test_load_from_manifest_file(tmp_path):
    manifest_path = tmp_path / "pod.yaml"
    manifest_path.write_text("kind: Pod\n", encoding="utf-8")

    text, ground_truth = load_manifest_and_ground_truth(_args(manifest=str(manifest_path)))

    assert text == "kind: Pod\n"
    assert ground_truth is None


def test_load_from_test_file_returns_ground_truth(tmp_path):
    example = {
        "messages": [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "kind: Pod\n"},
            {"role": "assistant", "content": json.dumps(CLEAN_RESPONSE)},
        ]
    }
    test_file = tmp_path / "test.jsonl"
    test_file.write_text(json.dumps(example) + "\n", encoding="utf-8")

    text, ground_truth = load_manifest_and_ground_truth(_args(test_file=str(test_file), index=0))

    assert text == "kind: Pod\n"
    assert ground_truth == CLEAN_RESPONSE


def test_load_from_test_file_index_out_of_range(tmp_path):
    test_file = tmp_path / "test.jsonl"
    test_file.write_text("", encoding="utf-8")

    with pytest.raises(IndexError):
        load_manifest_and_ground_truth(_args(test_file=str(test_file), index=0))


def test_load_requires_one_source():
    with pytest.raises(ValueError):
        load_manifest_and_ground_truth(_args())


def test_load_from_test_file_picks_correct_index(tmp_path):
    examples = [
        {
            "messages": [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": f"kind: Pod{i}\n"},
                {"role": "assistant", "content": json.dumps(CLEAN_RESPONSE)},
            ]
        }
        for i in range(3)
    ]
    test_file = tmp_path / "test.jsonl"
    test_file.write_text("\n".join(json.dumps(e) for e in examples) + "\n", encoding="utf-8")

    text, _ = load_manifest_and_ground_truth(_args(test_file=str(test_file), index=2))
    assert text == "kind: Pod2\n"
