"""
Cleans the raw kubectl-query corpora under query_training/corpus/ into
train/val/test .jsonl splits for SFT.

Sources (see query_training/sources.py for each one's raw schema):
  kubectl-command-csv  -- primary. kubernetes-kubectl-command-dataset/train.csv:
      concrete `question` -> `command`, plus a `chain_of_thought` for every
      row (used by --target-format plan-command).
  devops-kubectl       -- optional extra surface. devops-kubectl-v1/*.parquet;
      command parsed out of a ```bash``` fence, no CoT.

  cli_queries_1/ is intentionally not a source: its command set is identical
  to kubectl-command-csv's, so it would only add duplicates.

Cleaning pipeline (each step's drop count lands in clean_diagnostic.json):
  1. load + normalise whitespace on instruction and command
  2. drop non-kubectl / multi-command / malformed commands
  3. drop rows outside the length bounds
  4. optional --read-only: keep only read-verb commands
  5. --target-format plan-command: drop rows with no chain_of_thought
  6. cross-source exact de-dup on (instruction, command)
  7. resolve one-instruction-many-commands conflicts (--on-conflict)
  8. GROUP split: rows are grouped by their normalised command, and whole
     groups go to one split -- so no command (and none of its phrasings,
     from any source) is split across train/val/test

Usage:
    python -m query_training.clean
    python -m query_training.clean --sources kubectl-command-csv,devops-kubectl
    python -m query_training.clean --target-format plan-command
    python -m query_training.clean --read-only --on-conflict drop
"""

from __future__ import annotations

import argparse
import json
import random
import re
import shlex
from collections import Counter, defaultdict
from pathlib import Path
from typing import Callable, NamedTuple

from query_training.schema import KUBECTL_VERBS, READ_ONLY_VERBS, TARGET_FORMATS, build_messages
from query_training.sources import DEFAULT_SOURCES, Record, load_sources

_WS_RE = re.compile(r"\s+")
_FENCE_RE = re.compile(r"^`+|`+$")
# shell-significant command separators: a space-delimited &&/||/| operator, or
# a `;` that chains another kubectl. Punctuation inside quotes (e.g. `nginx -g
# "daemon off;"`) is deliberately not matched.
_CHAIN_RE = re.compile(r"\s(?:&&|\|\||\|)\s|;\s*kubectl\b")


def normalise_instruction(raw: str) -> str:
    """Collapse internal whitespace, strip ends. Casing is left untouched --
    the request phrasing is the model's input and should look natural.
    `\\s` matches unicode spaces too, so stray NBSPs get folded to plain ones."""
    return _WS_RE.sub(" ", raw).strip()


def normalise_command(raw: str) -> str | None:
    """Return a single-line kubectl command, or None if the row can't be
    salvaged as one: strips surrounding backticks, joins `\\`+newline shell
    continuations, collapses whitespace, and rejects real newlines or a
    chained second command."""
    text = raw.strip()
    text = _FENCE_RE.sub("", text).strip()
    text = re.sub(r"\\\s*\n\s*", " ", text)
    if "\n" in text:
        return None
    if _CHAIN_RE.search(text):
        return None
    text = _WS_RE.sub(" ", text).strip().rstrip("\\").strip()
    return text or None


def command_verb(command: str) -> str | None:
    """First token after `kubectl`, or None if the command doesn't parse or
    doesn't start with a bare `kubectl` (env-var prefixes like
    `KUBE_EDITOR=vim kubectl ...` count as not-bare and are rejected)."""
    try:
        tokens = shlex.split(command)
    except ValueError:
        return None
    if len(tokens) < 2 or tokens[0] != "kubectl":
        return None
    return tokens[1]


class CleanRow(NamedTuple):
    instruction: str
    command: str
    cot: str | None
    source: str


class CleanResult(NamedTuple):
    """Exactly one of `row` / `reason` is set."""

    row: CleanRow | None = None
    reason: str | None = None


REASONS = (
    "instruction_length",
    "instruction_too_few_words",
    "not_single_command",
    "command_length",
    "not_kubectl",
    "unknown_verb",
    "mutating_verb",
    "no_cot",
    "exact_duplicate",
    "conflict_tie",
    "conflict_minority",
)


def clean_record(
    record: Record,
    *,
    min_instruction_chars: int,
    max_instruction_chars: int,
    max_command_chars: int,
    read_only: bool,
    require_cot: bool,
) -> CleanResult:
    """Pure per-row cleaner."""
    instr = normalise_instruction(record.instruction)
    if not (min_instruction_chars <= len(instr) <= max_instruction_chars):
        return CleanResult(reason="instruction_length")
    if len(instr.split()) < 2:
        return CleanResult(reason="instruction_too_few_words")

    cmd = normalise_command(record.command)
    if cmd is None:
        return CleanResult(reason="not_single_command")
    if len(cmd) > max_command_chars:
        return CleanResult(reason="command_length")

    verb = command_verb(cmd)
    if verb is None:
        return CleanResult(reason="not_kubectl")
    if verb not in KUBECTL_VERBS:
        return CleanResult(reason="unknown_verb")
    if read_only and verb not in READ_ONLY_VERBS:
        return CleanResult(reason="mutating_verb")

    cot = record.cot.strip() if record.cot and record.cot.strip() else None
    if require_cot and cot is None:
        return CleanResult(reason="no_cot")

    return CleanResult(row=CleanRow(instr, cmd, cot, record.source))


def dedup_exact(rows: list[CleanRow]) -> tuple[list[CleanRow], int]:
    """Drop rows with an already-seen (instruction, command). A CoT-bearing
    row wins over a CoT-less duplicate."""
    best: dict[tuple[str, str], CleanRow] = {}
    order: list[tuple[str, str]] = []
    dropped = 0
    for row in rows:
        key = (row.instruction, row.command)
        if key not in best:
            best[key] = row
            order.append(key)
        else:
            dropped += 1
            if best[key].cot is None and row.cot is not None:
                best[key] = row
    return [best[k] for k in order], dropped


def resolve_conflicts(rows: list[CleanRow], *, on_conflict: str) -> tuple[list[CleanRow], Counter]:
    """Collapse each normalised instruction to (at most) one command.

    on_conflict:
      - "first":       keep the first-seen command per instruction (default).
      - "most-common": keep the modal command; drop the instruction on a tie.
      - "drop":        drop every instruction that has >1 distinct command.
      - "keep-all":    no-op.
    """
    drops: Counter = Counter()
    if on_conflict == "keep-all":
        return rows, drops

    by_instr: dict[str, list[CleanRow]] = defaultdict(list)
    order: list[str] = []
    for row in rows:
        if row.instruction not in by_instr:
            order.append(row.instruction)
        by_instr[row.instruction].append(row)

    resolved: list[CleanRow] = []
    for instr in order:
        group = by_instr[instr]
        commands = Counter(r.command for r in group)
        if len(commands) == 1:
            resolved.append(group[0])
            continue
        if on_conflict == "drop":
            drops["conflict_minority"] += len(group)
            continue
        if on_conflict == "first":
            resolved.append(group[0])
            drops["conflict_minority"] += len(group) - 1
            continue
        # most-common
        top = commands.most_common()
        best_count = top[0][1]
        if sum(1 for _, n in top if n == best_count) > 1:
            drops["conflict_tie"] += len(group)
            continue
        winner = top[0][0]
        resolved.append(next(r for r in group if r.command == winner))
        drops["conflict_minority"] += len(group) - best_count
    return resolved, drops


def split_by_group(
    rows: list[CleanRow],
    *,
    key: Callable[[CleanRow], str],
    val_frac: float,
    test_frac: float,
    seed: int,
) -> dict[str, list[CleanRow]]:
    """Assign whole groups (keyed by `key`) to one split each, so nothing
    that shares a key straddles splits. Groups are shuffled, then greedily
    filled into test -> val -> train until each hits its target row count."""
    groups: dict[str, list[CleanRow]] = defaultdict(list)
    for row in rows:
        groups[key(row)].append(row)

    keys = list(groups)
    random.Random(seed).shuffle(keys)

    n = len(rows)
    targets = {"test": int(n * test_frac), "val": int(n * val_frac)}
    splits: dict[str, list[CleanRow]] = {"test": [], "val": [], "train": []}
    for k in keys:
        for name in ("test", "val"):
            if len(splits[name]) < targets[name]:
                splits[name].extend(groups[k])
                break
        else:
            splits["train"].extend(groups[k])
    return splits


def build(args: argparse.Namespace) -> dict:
    corpus_dir = Path(args.corpus_dir)
    source_names = [s.strip() for s in args.sources.split(",") if s.strip()]
    raw = load_sources(source_names, corpus_dir)
    require_cot = args.target_format == "plan-command"

    per_source_raw = Counter(r.source for r in raw)
    drops: Counter = Counter()
    kept: list[CleanRow] = []
    for record in raw:
        result = clean_record(
            record,
            min_instruction_chars=args.min_instruction_chars,
            max_instruction_chars=args.max_instruction_chars,
            max_command_chars=args.max_command_chars,
            read_only=args.read_only,
            require_cot=require_cot,
        )
        if result.row is None:
            drops[result.reason] += 1
            continue
        kept.append(result.row)

    deduped, exact_dups = dedup_exact(kept)
    drops["exact_duplicate"] += exact_dups

    resolved, conflict_drops = resolve_conflicts(deduped, on_conflict=args.on_conflict)
    drops.update(conflict_drops)

    splits = split_by_group(
        resolved,
        key=lambda r: r.command,
        val_frac=args.val_frac,
        test_frac=args.test_frac,
        seed=args.seed,
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for split_name, split_rows in splits.items():
        path = out_dir / f"{split_name}.jsonl"
        with path.open("w", encoding="utf-8") as fh:
            for row in split_rows:
                plan = None
                if args.target_format == "plan-command" and row.cot:
                    plan = _WS_RE.sub(" ", row.cot).strip()
                messages = build_messages(row.instruction, row.command, plan=plan)
                fh.write(json.dumps({"messages": messages}, ensure_ascii=False) + "\n")

    verb_hist = Counter(command_verb(r.command) for r in resolved)
    summary = {
        "sources": source_names,
        "target_format": args.target_format,
        "corpus_dir": str(corpus_dir),
        "raw_rows": len(raw),
        "raw_by_source": dict(per_source_raw),
        "kept": len(resolved),
        "kept_fraction": round(len(resolved) / len(raw), 4) if raw else 0.0,
        "kept_with_cot": sum(1 for r in resolved if r.cot is not None),
        "kept_by_source": dict(Counter(r.source for r in resolved)),
        "read_only": args.read_only,
        "on_conflict": args.on_conflict,
        "drops": dict(sorted(drops.items(), key=lambda kv: -kv[1])),
        "split_sizes": {k: len(v) for k, v in splits.items()},
        "distinct_commands": len(set(r.command for r in resolved)),
        "verb_histogram": dict(verb_hist.most_common()),
        "seed": args.seed,
    }
    (out_dir / "clean_diagnostic.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nSplits written to {out_dir}/")
    return summary


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--corpus-dir", default="query_training/corpus")
    parser.add_argument(
        "--sources",
        default=",".join(DEFAULT_SOURCES),
        help="comma-separated: kubectl-command-csv, devops-kubectl",
    )
    parser.add_argument("--output-dir", default="query_training/output")
    parser.add_argument("--target-format", choices=TARGET_FORMATS, default="command")
    parser.add_argument(
        "--on-conflict",
        choices=["first", "most-common", "drop", "keep-all"],
        default="first",
    )
    parser.add_argument("--read-only", action="store_true", help="keep only read-verb commands")
    parser.add_argument("--min-instruction-chars", type=int, default=10)
    parser.add_argument("--max-instruction-chars", type=int, default=300)
    parser.add_argument("--max-command-chars", type=int, default=256)
    parser.add_argument("--val-frac", type=float, default=0.05)
    parser.add_argument("--test-frac", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


def main(argv=None) -> None:
    build(parse_args(argv))


if __name__ == "__main__":
    main()
