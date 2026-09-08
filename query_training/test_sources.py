import pandas as pd
import pytest

from query_training.sources import load_devops_kubectl, load_kubectl_command_csv, load_sources


@pytest.fixture
def corpus(tmp_path):
    csv_dir = tmp_path / "kubernetes-kubectl-command-dataset"
    csv_dir.mkdir()
    pd.DataFrame(
        {
            "objective": ["List pods", "Describe a pod"],
            "question": ["List all pods in kube-system", "Describe the pod named web"],
            "command": ["kubectl get pods -n kube-system", "kubectl describe pod web"],
            "chain_of_thought": ["1. use get\n2. -n kube-system", ""],
        }
    ).to_csv(csv_dir / "train.csv", index=False)

    dev_dir = tmp_path / "devops-kubectl-v1" / "data"
    dev_dir.mkdir(parents=True)
    pd.DataFrame(
        {
            "prompt": ["Get the logs of the api pod", "malformed row"],
            "response": [
                "Command: ```bash\nkubectl logs api-pod\n``` What it does: prints logs.",
                "no fence here",
            ],
        }
    ).to_parquet(dev_dir / "train-00000-of-00001.parquet")
    return tmp_path


def test_load_kubectl_command_csv_uses_question_and_cot(corpus):
    records = list(load_kubectl_command_csv(corpus))
    assert [r.instruction for r in records] == ["List all pods in kube-system", "Describe the pod named web"]
    assert records[0].command == "kubectl get pods -n kube-system"
    assert records[0].cot == "1. use get\n2. -n kube-system"
    assert records[1].cot is None  # empty string -> None
    assert all(r.source == "kubectl-command-csv" for r in records)


def test_load_devops_kubectl_parses_fence_and_skips_malformed(corpus):
    records = list(load_devops_kubectl(corpus))
    assert len(records) == 1
    assert records[0].instruction == "Get the logs of the api pod"
    assert records[0].command == "kubectl logs api-pod"
    assert records[0].cot is None
    assert records[0].source == "devops-kubectl"


def test_load_sources_combines_and_rejects_unknown(corpus):
    combined = load_sources(["kubectl-command-csv", "devops-kubectl"], corpus)
    assert len(combined) == 3
    with pytest.raises(KeyError):
        load_sources(["nope"], corpus)
