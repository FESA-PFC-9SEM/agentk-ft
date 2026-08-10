import json

from generation.report import compute_metrics, report


def _record(index, kind="Pod", namespace="prod", image="myapp:1.0.0", domain="fintech", stack="Go"):
    manifest = (
        f"apiVersion: v1\nkind: {kind}\nmetadata:\n  name: app-{index}\n  namespace: {namespace}\n"
        f"spec:\n  containers:\n  - name: app\n    image: {image}\n"
    )
    return {
        "index": index,
        "mode": "base",
        "manifest_yaml": manifest,
        "seed": {"domain": domain, "stack": stack, "kind": kind},
    }


def test_compute_metrics_basic_distributions():
    records = [_record(i) for i in range(5)] + [_record(5, image="other:2.0.0")]
    metrics = compute_metrics(records)
    assert metrics["total_records"] == 6
    assert metrics["kind_distribution"]["Pod"] == 6
    assert metrics["namespace_distribution"]["prod"] == 6
    assert metrics["image_distribution"]["myapp"] == 5
    assert metrics["image_distribution"]["other"] == 1
    assert 0.0 <= metrics["structural_uniqueness_ratio"] <= 1.0


def test_compute_metrics_detects_collapse():
    # todos identicos exceto o nome -- deveriam colapsar para 1 esqueleto so.
    records = [_record(i, namespace="default", image="nginx:1.0.0") for i in range(20)]
    metrics = compute_metrics(records)
    assert metrics["structural_uniqueness_ratio"] < 0.2


def test_report_writes_csv_and_png(tmp_path):
    input_path = tmp_path / "records.jsonl"
    with input_path.open("w") as fh:
        for i in range(10):
            fh.write(json.dumps(_record(i)) + "\n")

    class Args:
        input = str(input_path)
        output_dir = str(tmp_path / "out")

    metrics = report(Args())
    assert (tmp_path / "out" / "report.csv").exists()
    assert (tmp_path / "out" / "distributions.png").exists()
    assert metrics["total_records"] == 10
