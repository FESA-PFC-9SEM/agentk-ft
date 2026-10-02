from finetune.scenario_summary import lenient_parse


def test_lenient_parse_keeps_findings_despite_schema_errors():
    text = '```json\n{"findings": [{"rule_id": "KSEC-001", "doc": 0, "path": "/x", "evidence": "hardc***"}], "patch": []}\n```'
    obj, errors = lenient_parse(text)
    assert errors == []
    assert obj["findings"][0]["rule_id"] == "KSEC-001"
    assert obj["findings"][0]["message"] == ""
    assert obj["notes"] == [] and obj["new_resources"] == []


def test_lenient_parse_drops_malformed_entries_and_rejects_non_json():
    obj, _ = lenient_parse('{"findings": ["oops", {"severity": "high"}], "patch": [{"path": "/a"}, {"op": "remove", "path": "/b"}]}')
    assert obj["findings"] == []
    assert obj["patch"] == [{"op": "remove", "path": "/b"}]
    assert lenient_parse("not json")[0] is None
