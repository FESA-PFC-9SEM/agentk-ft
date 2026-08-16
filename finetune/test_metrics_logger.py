import json

from finetune.metrics_logger import JsonlMetricsLogger, load_metrics


class _FakeState:
    def __init__(self, global_step, epoch):
        self.global_step = global_step
        self.epoch = epoch


def test_on_log_appends_a_record(tmp_path):
    path = tmp_path / "metrics.jsonl"
    logger = JsonlMetricsLogger(path)

    logger.on_log(args=None, state=_FakeState(10, 0.5), control=None, logs={"loss": 0.42, "learning_rate": 1e-4})

    records = load_metrics(path)
    assert len(records) == 1
    assert records[0]["loss"] == 0.42
    assert records[0]["step"] == 10
    assert records[0]["epoch"] == 0.5


def test_on_log_ignores_empty_logs(tmp_path):
    path = tmp_path / "metrics.jsonl"
    logger = JsonlMetricsLogger(path)

    logger.on_log(args=None, state=_FakeState(1, 0.1), control=None, logs=None)
    logger.on_log(args=None, state=_FakeState(2, 0.2), control=None, logs={})

    assert load_metrics(path) == []


def test_on_log_appends_across_multiple_calls(tmp_path):
    path = tmp_path / "metrics.jsonl"
    logger = JsonlMetricsLogger(path)

    logger.on_log(args=None, state=_FakeState(10, 0.1), control=None, logs={"loss": 1.0})
    logger.on_log(args=None, state=_FakeState(20, 0.2), control=None, logs={"eval_loss": 0.9})

    records = load_metrics(path)
    assert len(records) == 2
    assert records[0]["loss"] == 1.0
    assert records[1]["eval_loss"] == 0.9


def test_on_log_creates_parent_dir(tmp_path):
    path = tmp_path / "nested" / "dir" / "metrics.jsonl"
    JsonlMetricsLogger(path)
    assert path.parent.exists()


def test_load_metrics_missing_file_returns_empty_list(tmp_path):
    assert load_metrics(tmp_path / "does-not-exist.jsonl") == []


def test_load_metrics_skips_blank_lines(tmp_path):
    path = tmp_path / "metrics.jsonl"
    path.write_text('{"loss": 1.0, "step": 1}\n\n{"loss": 2.0, "step": 2}\n', encoding="utf-8")
    records = load_metrics(path)
    assert len(records) == 2
