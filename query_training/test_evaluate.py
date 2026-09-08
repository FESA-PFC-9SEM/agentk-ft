from query_training.evaluate import aggregate, canonicalise, clean_prediction, command_verb, score_example


def test_clean_prediction_strips_fences_and_prompt():
    assert clean_prediction("```bash\nkubectl get pods\n```") == "kubectl get pods"
    assert clean_prediction("$ kubectl get pods") == "kubectl get pods"
    assert clean_prediction("kubectl get pods\n# explanation follows") == "kubectl get pods"


def test_clean_prediction_extracts_command_from_plan_json():
    assert clean_prediction('{"plan": "use get", "command": "kubectl get pods -A"}') == "kubectl get pods -A"


def test_score_example_plan_command_on_both_sides():
    expected = '{"plan": "x", "command": "kubectl get pods -A"}'
    raw = '{"plan": "list every pod cluster-wide", "command": "kubectl get pods -A"}'
    r = score_example(expected, raw)
    assert r["exact_match"] and r["normalized_match"] and r["verb_match"]


def test_command_verb():
    assert command_verb("kubectl get pods") == "get"
    assert command_verb("not a command") is None


def test_canonicalise_forgives_flag_order():
    assert canonicalise("kubectl get pods -n foo -o wide") == canonicalise("kubectl get -o wide -n foo pods")
    assert canonicalise("kubectl get pods") != canonicalise("kubectl get svc")


def test_score_example_exact_match():
    r = score_example("kubectl get pods -A", "kubectl get pods -A")
    assert r["exact_match"] and r["normalized_match"] and r["valid_kubectl"] and r["verb_match"]


def test_score_example_normalized_only():
    r = score_example("kubectl get pods -n foo -o wide", "kubectl get -n foo -o wide pods")
    assert not r["exact_match"]
    assert r["normalized_match"]
    assert r["verb_match"]


def test_score_example_wrong_verb():
    r = score_example("kubectl get pods", "kubectl describe pods")
    assert not r["normalized_match"]
    assert not r["verb_match"]
    assert r["valid_kubectl"]


def test_score_example_not_kubectl():
    r = score_example("kubectl get pods", "here is the command you want")
    assert not r["valid_kubectl"]
    assert not r["verb_match"]


def test_aggregate_rates():
    results = [
        {"exact_match": True, "normalized_match": True, "valid_kubectl": True, "verb_match": True},
        {"exact_match": False, "normalized_match": True, "valid_kubectl": True, "verb_match": True},
        {"exact_match": False, "normalized_match": False, "valid_kubectl": False, "verb_match": False},
        {"exact_match": False, "normalized_match": False, "valid_kubectl": True, "verb_match": True},
    ]
    summary = aggregate(results)
    assert summary["total"] == 4
    assert summary["exact_match_rate"] == 0.25
    assert summary["normalized_match_rate"] == 0.5
    assert summary["valid_kubectl_rate"] == 0.75
    assert summary["verb_match_rate"] == 0.75


def test_aggregate_empty():
    assert aggregate([]) == {"total": 0}
