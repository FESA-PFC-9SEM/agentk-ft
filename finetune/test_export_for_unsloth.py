import argparse
import json

from finetune.export_for_unsloth import (
    char_approx_render_and_count,
    export,
    percentiles,
    process_split,
)


def _example(system="sys", user="user content", assistant='{"findings": []}'):
    return {
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ]
    }


def test_percentiles_basic():
    values = list(range(1, 101))  # 1..100
    p = percentiles(values, ps=(0.5, 0.9, 1.0))
    assert p["p50"] == 51
    assert p["p90"] == 91
    assert p["p100"] == 100


def test_percentiles_empty():
    assert percentiles([]) == {}


def test_char_approx_render_and_count_includes_all_roles():
    messages = [
        {"role": "system", "content": "S"},
        {"role": "user", "content": "U"},
        {"role": "assistant", "content": "A"},
    ]
    text, num_tokens = char_approx_render_and_count(messages)
    assert "S" in text and "U" in text and "A" in text
    assert num_tokens >= 1


def test_process_split_keeps_short_examples(tmp_path):
    input_path = tmp_path / "train.jsonl"
    with input_path.open("w", encoding="utf-8") as fh:
        for _ in range(5):
            fh.write(json.dumps(_example()) + "\n")

    output_path = tmp_path / "out.jsonl"
    stats = process_split(input_path, output_path, tokenizer=None, max_seq_length=4096, use_char_approx=True)

    assert stats["kept"] == 5
    assert stats["dropped"] == 0
    lines = output_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 5
    record = json.loads(lines[0])
    assert set(record.keys()) == {"messages", "text", "num_tokens"}


def test_process_split_drops_oversized_examples(tmp_path):
    input_path = tmp_path / "train.jsonl"
    with input_path.open("w", encoding="utf-8") as fh:
        fh.write(json.dumps(_example(user="short")) + "\n")
        fh.write(json.dumps(_example(user="x" * 100_000)) + "\n")  # ~25000 tokens at chars/4

    output_path = tmp_path / "out.jsonl"
    stats = process_split(input_path, output_path, tokenizer=None, max_seq_length=4096, use_char_approx=True)

    assert stats["kept"] == 1
    assert stats["dropped"] == 1
    assert stats["dropped_token_percentiles"]["p100"] > 4096
    lines = output_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1


def test_process_split_never_truncates_only_drops(tmp_path):
    # A dropped example must not appear in the output at all -- truncating
    # the assistant's JSON turn would corrupt the label, so the only safe
    # behavior is exclusion.
    input_path = tmp_path / "train.jsonl"
    big = _example(user="x" * 100_000)
    with input_path.open("w", encoding="utf-8") as fh:
        fh.write(json.dumps(big) + "\n")

    output_path = tmp_path / "out.jsonl"
    process_split(input_path, output_path, tokenizer=None, max_seq_length=4096, use_char_approx=True)

    assert output_path.read_text(encoding="utf-8") == ""


def test_export_end_to_end_with_char_approx(tmp_path):
    input_dir = tmp_path / "in"
    input_dir.mkdir()
    output_dir = tmp_path / "out"

    for split, count in (("train", 8), ("val", 3), ("test", 2)):
        with (input_dir / f"{split}.jsonl").open("w", encoding="utf-8") as fh:
            for _ in range(count):
                fh.write(json.dumps(_example()) + "\n")

    args = argparse.Namespace(
        input_dir=str(input_dir),
        output_dir=str(output_dir),
        tokenizer="unused",
        max_seq_length=4096,
        char_approx=True,
    )
    summary = export(args)

    assert summary["train"]["kept"] == 8
    assert summary["val"]["kept"] == 3
    assert summary["test"]["kept"] == 2
    assert (output_dir / "export_diagnostic.json").exists()
    assert (output_dir / "train.jsonl").exists()


def test_export_skips_missing_split_files(tmp_path):
    input_dir = tmp_path / "in"
    input_dir.mkdir()
    output_dir = tmp_path / "out"
    with (input_dir / "train.jsonl").open("w", encoding="utf-8") as fh:
        fh.write(json.dumps(_example()) + "\n")
    # no val.jsonl / test.jsonl

    args = argparse.Namespace(
        input_dir=str(input_dir),
        output_dir=str(output_dir),
        tokenizer="unused",
        max_seq_length=4096,
        char_approx=True,
    )
    summary = export(args)

    assert "train" in summary
    assert "val" not in summary
    assert "test" not in summary
