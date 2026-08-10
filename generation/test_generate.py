import json
import random
from pathlib import Path

from generation.generate import (
    build_prompt,
    load_completed_indices,
    looks_like_manifest,
    strip_fences,
)
from generation.seeds import sample_seed


def test_strip_fences_removes_markdown_code_block():
    text = "```yaml\napiVersion: v1\nkind: Pod\n```"
    assert strip_fences(text) == "apiVersion: v1\nkind: Pod"


def test_strip_fences_noop_on_plain_yaml():
    text = "apiVersion: v1\nkind: Pod"
    assert strip_fences(text) == text


def test_looks_like_manifest_true_for_valid_yaml():
    assert looks_like_manifest("apiVersion: v1\nkind: Pod\nmetadata:\n  name: p")


def test_looks_like_manifest_false_for_prose():
    assert not looks_like_manifest("Here is the manifest you asked for:")


def test_looks_like_manifest_false_for_yaml_without_kind():
    assert not looks_like_manifest("foo: bar\nbaz: 1")


def test_build_prompt_embeds_constraints():
    seed = sample_seed(random.Random(1), mode="base")
    prompt = build_prompt(seed)
    assert seed.domain in prompt
    assert seed.service_name in prompt
    assert "```" not in prompt.split("Respond with ONLY")[0]


def test_hard_negative_prompt_differs_from_base():
    seed_base = sample_seed(random.Random(1), mode="base")
    seed_hn = sample_seed(random.Random(1), mode="hard-negative")
    assert "secretKeyRef" in build_prompt(seed_hn)
    assert "secretKeyRef" not in build_prompt(seed_base)


def test_load_completed_indices_empty_when_missing(tmp_path):
    assert load_completed_indices(tmp_path / "missing.jsonl") == set()


def test_load_completed_indices_reads_existing(tmp_path):
    p = tmp_path / "out.jsonl"
    p.write_text('{"index": 0}\n{"index": 3}\n', encoding="utf-8")
    assert load_completed_indices(p) == {0, 3}


def test_load_completed_indices_skips_malformed_lines(tmp_path):
    p = tmp_path / "out.jsonl"
    p.write_text('{"index": 0}\nnot json\n{"index": 2}\n', encoding="utf-8")
    assert load_completed_indices(p) == {0, 2}
