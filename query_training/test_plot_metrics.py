import json

from query_training.plot_metrics import load_metrics, split_series


def test_split_series_separates_train_and_eval(tmp_path):
    path = tmp_path / "metrics.jsonl"
    path.write_text(
        "\n".join(
            json.dumps(r)
            for r in [
                {"loss": 1.0, "learning_rate": 2e-4, "grad_norm": 3.0, "step": 10},
                {"eval_loss": 0.9, "step": 10},
                {"loss": 0.7, "learning_rate": 1e-4, "grad_norm": 2.0, "step": 20},
                {"eval_loss": 0.6, "step": 20},
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    series = split_series(load_metrics(path))
    assert series["loss"] == [(10, 1.0), (20, 0.7)]
    assert series["eval_loss"] == [(10, 0.9), (20, 0.6)]
    assert series["learning_rate"] == [(10, 2e-4), (20, 1e-4)]


def test_load_metrics_missing_file_is_empty(tmp_path):
    assert load_metrics(tmp_path / "nope.jsonl") == []


def test_split_series_skips_records_without_step():
    assert split_series([{"loss": 1.0}]) == {}
