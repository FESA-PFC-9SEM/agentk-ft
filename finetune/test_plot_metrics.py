import argparse
import json

from finetune.plot_metrics import plot, split_series, write_csv, write_png


def _records():
    return [
        {"step": 10, "epoch": 0.1, "loss": 1.0, "learning_rate": 2e-4},
        {"step": 20, "epoch": 0.2, "loss": 0.8, "learning_rate": 1.8e-4},
        {"step": 20, "epoch": 0.2, "eval_loss": 0.9},
        {"step": 30, "epoch": 0.3, "loss": 0.6, "learning_rate": 1.6e-4, "grad_norm": 0.5},
        {"step": 40, "epoch": 0.4, "eval_loss": 0.7},
    ]


def test_split_series_separates_metrics_by_key():
    series = split_series(_records())
    assert series["loss"] == [(10, 1.0), (20, 0.8), (30, 0.6)]
    assert series["eval_loss"] == [(20, 0.9), (40, 0.7)]
    assert series["learning_rate"] == [(10, 2e-4), (20, 1.8e-4), (30, 1.6e-4)]
    assert series["grad_norm"] == [(30, 0.5)]


def test_split_series_ignores_records_without_step():
    series = split_series([{"loss": 1.0}])
    assert series == {}


def test_write_csv_contains_all_series(tmp_path):
    series = split_series(_records())
    out = tmp_path / "metrics.csv"
    write_csv(series, out)
    content = out.read_text(encoding="utf-8")
    assert "loss" in content
    assert "eval_loss" in content
    assert "learning_rate" in content


def test_write_png_creates_file(tmp_path):
    series = split_series(_records())
    out = tmp_path / "metrics.png"
    write_png(series, out)
    assert out.exists()
    assert out.stat().st_size > 0


def test_write_png_handles_missing_series(tmp_path):
    # grad_norm absent entirely -- must not crash, just shows "no data"
    series = {"loss": [(1, 1.0)]}
    out = tmp_path / "metrics.png"
    write_png(series, out)
    assert out.exists()


def test_plot_end_to_end(tmp_path):
    metrics_path = tmp_path / "metrics.jsonl"
    with metrics_path.open("w", encoding="utf-8") as fh:
        for r in _records():
            fh.write(json.dumps(r) + "\n")

    args = argparse.Namespace(metrics_file=str(metrics_path), output_dir=str(tmp_path / "out"))
    summary = plot(args)

    assert summary["total_records"] == 5
    assert (tmp_path / "out" / "metrics.csv").exists()
    assert (tmp_path / "out" / "metrics.png").exists()


def test_plot_missing_file_returns_zero(tmp_path):
    args = argparse.Namespace(metrics_file=str(tmp_path / "missing.jsonl"), output_dir=None)
    summary = plot(args)
    assert summary["total_records"] == 0
