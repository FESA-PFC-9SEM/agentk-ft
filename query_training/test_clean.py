from query_training.clean import (
    CleanRow,
    clean_record,
    command_verb,
    dedup_exact,
    normalise_command,
    normalise_instruction,
    resolve_conflicts,
    split_by_group,
)
from query_training.sources import Record

REC_KW = dict(
    min_instruction_chars=10,
    max_instruction_chars=300,
    max_command_chars=256,
    read_only=False,
    require_cot=False,
)


def _rec(instruction, command, cot=None, source="t"):
    return Record(instruction, command, cot, source)


def test_normalise_instruction_collapses_whitespace():
    assert normalise_instruction("  List   all\tpods\n") == "List all pods"


def test_normalise_command_strips_fences_and_collapses():
    assert normalise_command("`kubectl   get pods`") == "kubectl get pods"


def test_normalise_command_joins_line_continuation():
    assert normalise_command("kubectl get pods \\\n  -n kube-system") == "kubectl get pods -n kube-system"


def test_normalise_command_rejects_chained_commands():
    assert normalise_command("kubectl delete pod x && kubectl get pods") is None
    assert normalise_command("kubectl get pods | grep Running") is None
    assert normalise_command("kubectl get pods\nkubectl get svc") is None
    assert normalise_command("kubectl delete pod x ; kubectl get pods") is None


def test_normalise_command_keeps_quoted_punctuation():
    # semicolon / pipe inside quotes is not a shell separator
    assert normalise_command('kubectl exec pod -- nginx -g "daemon off;"') == 'kubectl exec pod -- nginx -g "daemon off;"'


def test_command_verb():
    assert command_verb("kubectl get pods") == "get"
    assert command_verb('KUBE_EDITOR="vim" kubectl edit svc/x') is None
    assert command_verb("helm install x") is None
    assert command_verb("kubectl") is None


def test_clean_record_happy_path():
    r = clean_record(_rec("Show all pods in the cluster", "kubectl get pods -A"), **REC_KW)
    assert r.row == CleanRow("Show all pods in the cluster", "kubectl get pods -A", None, "t")
    assert r.reason is None


def test_clean_record_rejects_short_and_one_word():
    assert clean_record(_rec("pods", "kubectl get pods"), **REC_KW).reason == "instruction_length"
    assert clean_record(_rec("aaaaaaaaaaaaaaa", "kubectl get pods"), **REC_KW).reason == "instruction_too_few_words"


def test_clean_record_rejects_unknown_verb():
    assert clean_record(_rec("Do a barrel roll now", "kubectl barrelroll --loop"), **REC_KW).reason == "unknown_verb"


def test_clean_record_read_only_filter():
    kw = {**REC_KW, "read_only": True}
    assert clean_record(_rec("Delete the nginx pod now", "kubectl delete pod nginx"), **kw).reason == "mutating_verb"
    assert clean_record(_rec("Get the nginx pod now", "kubectl get pod nginx"), **kw).row is not None


def test_clean_record_require_cot():
    kw = {**REC_KW, "require_cot": True}
    assert clean_record(_rec("List pods in default", "kubectl get pods"), **kw).reason == "no_cot"
    ok = clean_record(_rec("List pods in default", "kubectl get pods", cot="use get"), **kw)
    assert ok.row is not None and ok.row.cot == "use get"


def test_dedup_exact_prefers_cot_bearing_row():
    rows = [
        CleanRow("list pods", "kubectl get pods", None, "a"),
        CleanRow("list pods", "kubectl get pods", "use get subcommand", "b"),
        CleanRow("list svc", "kubectl get svc", None, "a"),
    ]
    kept, dropped = dedup_exact(rows)
    assert dropped == 1
    assert [r.command for r in kept] == ["kubectl get pods", "kubectl get svc"]
    assert kept[0].cot == "use get subcommand"


def test_resolve_conflicts_first_keeps_one_per_instruction():
    rows = [
        CleanRow("list pods", "kubectl get pods", None, "a"),
        CleanRow("list pods", "kubectl get pods -A", None, "a"),
        CleanRow("show nodes", "kubectl get nodes", None, "a"),
    ]
    resolved, drops = resolve_conflicts(rows, on_conflict="first")
    assert [r.command for r in resolved] == ["kubectl get pods", "kubectl get nodes"]
    assert drops["conflict_minority"] == 1


def test_resolve_conflicts_most_common_drops_ties():
    rows = [
        CleanRow("a b", "kubectl get pods", None, "s"),
        CleanRow("a b", "kubectl get pods", None, "s"),
        CleanRow("a b", "kubectl get svc", None, "s"),
        CleanRow("c d", "kubectl get x", None, "s"),
        CleanRow("c d", "kubectl get y", None, "s"),
    ]
    resolved, drops = resolve_conflicts(rows, on_conflict="most-common")
    assert any(r.instruction == "a b" and r.command == "kubectl get pods" for r in resolved)
    assert all(r.instruction != "c d" for r in resolved)
    assert drops["conflict_tie"] == 2


def test_split_by_group_no_command_straddles_splits():
    rows = []
    for i in range(200):
        cmd = f"kubectl get x{i % 40}"  # 40 distinct commands, 5 rows each
        rows.append(CleanRow(f"instruction variant {i}", cmd, None, "s"))
    a = split_by_group(rows, key=lambda r: r.command, val_frac=0.1, test_frac=0.1, seed=1)
    b = split_by_group(rows, key=lambda r: r.command, val_frac=0.1, test_frac=0.1, seed=1)
    assert {k: [r._asdict() for r in v] for k, v in a.items()} == {k: [r._asdict() for r in v] for k, v in b.items()}
    cmds = {name: {r.command for r in rs} for name, rs in a.items()}
    assert not (cmds["train"] & cmds["val"])
    assert not (cmds["train"] & cmds["test"])
    assert not (cmds["val"] & cmds["test"])
    assert sum(len(v) for v in a.values()) == 200
