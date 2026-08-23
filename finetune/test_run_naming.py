from datetime import datetime

from finetune.run_naming import generate_run_name, slugify_model


def test_slugify_model_strips_org_prefix_and_lowercases():
    assert slugify_model("unsloth/Qwen2.5-Coder-7B-Instruct-bnb-4bit") == "qwen2_5-coder-7b-instruct-bnb-4bit"


def test_generate_run_name_format():
    when = datetime(2026, 8, 22, 14, 5)
    name = generate_run_name("multi-defect", "unsloth/Qwen2.5-Coder-7B-Instruct-bnb-4bit", when=when)
    assert name == "20260822-1405_multi-defect_qwen2_5-coder-7b-instruct-bnb-4bit"


def test_generate_run_name_defaults_to_now():
    name = generate_run_name("single-defect", "some/model")
    assert name.endswith("_single-defect_model")
    assert len(name.split("_")[0]) == len("20260822-1405")
