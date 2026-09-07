from finetune.benchmark_throughput import format_row, parse_args, parse_combo


def test_parse_combo():
    assert parse_combo("8x4") == (8, 4)
    assert parse_combo("32x1") == (32, 1)
    assert parse_combo("2X16") == (2, 16)  # case-insensitive


def test_format_row_success():
    row = {
        "batch_size": 8,
        "grad_accum": 4,
        "effective_batch": 32,
        "samples_per_second": 1.234,
        "steps_per_second": 0.0386,
        "peak_vram_gb": 6.71,
        "oom": False,
    }
    text = format_row(row)
    assert "8" in text and "4" in text and "32" in text
    assert "1.234" in text
    assert "6.71" in text


def test_format_row_oom():
    row = {"batch_size": 32, "grad_accum": 1, "effective_batch": 32, "oom": True}
    text = format_row(row)
    assert "OOM" in text


def test_output_defaults_to_an_auto_generated_path_under_finetune_output():
    args = parse_args(["--model", "unsloth/Qwen2.5-Coder-7B-Instruct-bnb-4bit", "--combos", "8x4"])
    assert args.output.startswith("finetune/output/throughput/")
    assert args.output.endswith("_qwen2_5-coder-7b-instruct-bnb-4bit.json")


def test_explicit_output_is_not_overridden():
    args = parse_args(
        ["--model", "unsloth/Qwen2.5-Coder-7B-Instruct-bnb-4bit", "--combos", "8x4", "--output", "custom.json"]
    )
    assert args.output == "custom.json"
