import json

from dataset.stats import main, render, split_stats


def _example(manifest, findings, patch=(), new_resources=()):
    response = {"findings": findings, "patch": list(patch), "new_resources": list(new_resources), "notes": []}
    return {
        "messages": [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": manifest},
            {"role": "assistant", "content": json.dumps(response)},
        ]
    }


def _finding(rule_id, severity="high"):
    return {"rule_id": rule_id, "severity": severity, "doc": 0, "path": "/spec", "message": "m", "evidence": "abcd***"}


def test_split_stats_counts_rules_severities_and_kinds(tmp_path):
    path = tmp_path / "train.jsonl"
    rows = [
        _example("kind: Pod\n", []),
        _example(
            "kind: Deployment\n---\nkind: Secret\n",
            [_finding("KSEC-001", "critical"), _finding("KSEC-001", "critical"), _finding("KSEC-011")],
            patch=[{"doc": 0, "op": "add", "path": "/spec", "value": 1}],
            new_resources=["kind: Secret"],
        ),
    ]
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    stats = split_stats(path)
    assert stats["examples"] == 2 and stats["negatives"] == 1 and stats["positives"] == 1
    assert stats["findings_per_rule"]["KSEC-001"] == 2
    assert stats["examples_per_rule"]["KSEC-001"] == 1
    assert stats["examples_per_rule"]["KSEC-004"] == 0  # every rule listed, even when absent
    assert stats["severities"] == {"critical": 2, "high": 1}
    assert stats["findings_per_example"] == {0: 1, 3: 1}
    assert stats["patch_ops"] == {"add": 1}
    assert stats["examples_with_new_resources"] == 1
    assert stats["kinds"] == {"Pod": 1, "Deployment": 1, "Secret": 1}
    assert "KSEC-011" in render({"train": stats})


def test_main_writes_json(tmp_path, capsys):
    (tmp_path / "val.jsonl").write_text(json.dumps(_example("kind: Pod\n", [])) + "\n")
    out = tmp_path / "stats.json"
    main([str(tmp_path), "--json", str(out)])
    assert json.loads(out.read_text())["val"]["examples"] == 1
    assert "Overview" in capsys.readouterr().out


def test_split_stats_counts_documents_and_multi_document_findings(tmp_path):
    path = tmp_path / "train.jsonl"
    multi = _example("kind: Service\n---\nkind: Deployment\n", [dict(_finding("KSEC-006"), doc=0), dict(_finding("KSEC-005"), doc=1)])
    single = _example("kind: Pod\n", [_finding("KSEC-005")])
    path.write_text("\n".join(json.dumps(r) for r in (multi, single)) + "\n")
    stats = split_stats(path)
    assert stats["docs_per_example"] == {1: 1, 2: 1}
    assert stats["multi_doc_findings_per_rule"]["KSEC-006"] == 1
    assert stats["multi_doc_findings_per_rule"]["KSEC-005"] == 1
    assert stats["findings_with_nonzero_doc_index"] == 1
    assert "Documents per example" in render({"train": stats})
