import json

from query_training.schema import (
    PLAN_COMMAND_SYSTEM_PROMPT,
    SYSTEM_PROMPT,
    build_messages,
    extract_command,
)


def test_build_messages_command_format():
    m = build_messages("list pods", "kubectl get pods")
    assert [x["role"] for x in m] == ["system", "user", "assistant"]
    assert m[0]["content"] == SYSTEM_PROMPT
    assert m[2]["content"] == "kubectl get pods"


def test_build_messages_plan_command_format():
    m = build_messages("list pods", "kubectl get pods", plan="use the get subcommand")
    assert m[0]["content"] == PLAN_COMMAND_SYSTEM_PROMPT
    obj = json.loads(m[2]["content"])
    assert obj == {"plan": "use the get subcommand", "command": "kubectl get pods"}


def test_extract_command_from_bare_string():
    assert extract_command("kubectl get pods") == "kubectl get pods"


def test_extract_command_from_plan_json():
    payload = json.dumps({"plan": "whatever", "command": "kubectl get pods -A"})
    assert extract_command(payload) == "kubectl get pods -A"


def test_extract_command_from_malformed_json_falls_back():
    assert extract_command('{"plan": "oops"') == '{"plan": "oops"'
    assert extract_command('{"plan": "no command key"}') == '{"plan": "no command key"}'
