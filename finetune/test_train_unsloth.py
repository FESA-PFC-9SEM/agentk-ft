import pytest

# train_unsloth.py imports unsloth/trl/datasets at module level, which only
# exist in finetune/.venv (Python 3.11), not the project's main venv -- skip
# cleanly there instead of breaking the combined `pytest dataset/ generation/
# finetune/` run. Run this file for real via:
#   finetune/.venv/bin/python -m pytest finetune/test_train_unsloth.py
pytest.importorskip("unsloth")

from finetune.train_unsloth import PRESETS, parse_args


_SEVEN_B = "unsloth/Qwen2.5-Coder-7B-Instruct-bnb-4bit"


def test_default_preset_is_4060ti_and_matches_current_hardcoded_values():
    args = parse_args([])
    assert args.preset == "4060ti"
    assert args.model == _SEVEN_B
    assert args.max_seq_length == 4096
    assert args.batch_size == 4
    assert args.grad_accum == 4
    assert args.eval_batch_size == 1
    assert args.eval_accumulation_steps == 1
    assert args.eval_steps == 100
    assert args.save_steps == 100


def test_a100_preset_applies_recommended_values():
    args = parse_args(["--preset", "a100"])
    assert args.model == _SEVEN_B
    assert args.max_seq_length == 4096
    assert args.batch_size == 16
    assert args.grad_accum == 2
    assert args.eval_batch_size == 8
    assert args.eval_accumulation_steps == 4
    assert args.eval_steps == 50
    assert args.save_steps == 50


def test_l4_preset_applies_recommended_values():
    args = parse_args(["--preset", "l4"])
    assert args.model == _SEVEN_B
    assert args.max_seq_length == 4096
    assert args.batch_size == 8
    assert args.grad_accum == 4
    assert args.eval_batch_size == 4
    assert args.eval_accumulation_steps == 2
    assert args.eval_steps == 50
    assert args.save_steps == 50
    # same effective batch as a100 (32), via more accumulation instead of a
    # larger per-device batch -- less VRAM/compute margin than a100
    assert args.batch_size * args.grad_accum == 32


def test_explicit_flag_overrides_preset():
    args = parse_args(["--preset", "a100", "--batch-size", "32"])
    assert args.batch_size == 32
    # everything else from the a100 preset is untouched
    assert args.grad_accum == 2
    assert args.eval_steps == 50


def test_explicit_model_overrides_preset_model():
    args = parse_args(["--preset", "a100", "--model", "unsloth/Qwen2.5-Coder-3B-Instruct-bnb-4bit"])
    assert args.model == "unsloth/Qwen2.5-Coder-3B-Instruct-bnb-4bit"
    # batch/seq knobs from the preset are untouched even though they were
    # sized for 7B -- overriding --model doesn't retune them automatically
    assert args.batch_size == 16


def test_explicit_flag_overrides_default_preset_too():
    args = parse_args(["--batch-size", "2"])
    assert args.preset == "4060ti"
    assert args.batch_size == 2
    assert args.grad_accum == 4  # unaffected


def test_non_hardware_params_are_not_preset_controlled():
    default_args = parse_args([])
    a100_args = parse_args(["--preset", "a100"])
    assert default_args.r == a100_args.r == 16
    assert default_args.lora_alpha == a100_args.lora_alpha == 32
    assert default_args.lr == a100_args.lr == 2e-4
    assert default_args.epochs == a100_args.epochs == 1


def test_invalid_preset_rejected():
    with pytest.raises(SystemExit):
        parse_args(["--preset", "h100"])


def test_all_preset_dicts_share_the_same_keys():
    key_sets = [set(preset.keys()) for preset in PRESETS.values()]
    assert all(keys == key_sets[0] for keys in key_sets)
