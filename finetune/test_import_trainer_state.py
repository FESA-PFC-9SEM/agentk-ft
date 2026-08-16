import argparse
import json

from finetune.import_trainer_state import import_log_history, run, write_metrics_jsonl


def _fake_trainer_state(tmp_path):
    state = {
        "global_step": 3,
        "log_history": [
            {"step": 1, "epoch": 0.33, "loss": 1.0, "learning_rate": 2e-4},
            {"step": 2, "epoch": 0.66, "loss": 0.5, "learning_rate": 1e-4},
            {"step": 2, "epoch": 0.66, "eval_loss": 0.6},
            {"step": 3, "epoch": 1.0, "loss": 0.2, "learning_rate": 0.0},
        ],
    }
    path = tmp_path / "trainer_state.json"
    path.write_text(json.dumps(state), encoding="utf-8")
    return path


def test_import_log_history_reads_all_entries(tmp_path):
    path = _fake_trainer_state(tmp_path)
    records = import_log_history(path)
    assert len(records) == 4
    assert records[0]["loss"] == 1.0
    assert records[2]["eval_loss"] == 0.6


def test_import_log_history_missing_key_returns_empty(tmp_path):
    path = tmp_path / "trainer_state.json"
    path.write_text(json.dumps({"global_step": 0}), encoding="utf-8")
    assert import_log_history(path) == []


def test_write_metrics_jsonl_round_trips(tmp_path):
    records = [{"step": 1, "loss": 1.0}, {"step": 2, "eval_loss": 0.5}]
    out = tmp_path / "out" / "metrics.jsonl"
    write_metrics_jsonl(records, out)

    lines = out.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0]) == records[0]
    assert json.loads(lines[1]) == records[1]


def test_run_end_to_end(tmp_path):
    trainer_state_path = _fake_trainer_state(tmp_path)
    output_path = tmp_path / "metrics.jsonl"

    args = argparse.Namespace(trainer_state=str(trainer_state_path), output=str(output_path))
    summary = run(args)

    assert summary["records_imported"] == 4
    assert output_path.exists()
    assert len(output_path.read_text(encoding="utf-8").splitlines()) == 4


def test_output_is_compatible_with_plot_metrics(tmp_path):
    # the whole point: the converted file must work with the existing
    # plotting tool unchanged.
    from finetune.metrics_logger import load_metrics
    from finetune.plot_metrics import split_series

    trainer_state_path = _fake_trainer_state(tmp_path)
    output_path = tmp_path / "metrics.jsonl"
    run(argparse.Namespace(trainer_state=str(trainer_state_path), output=str(output_path)))

    records = load_metrics(output_path)
    series = split_series(records)
    assert series["loss"] == [(1, 1.0), (2, 0.5), (3, 0.2)]
    assert series["eval_loss"] == [(2, 0.6)]
