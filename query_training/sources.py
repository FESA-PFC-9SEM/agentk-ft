"""
Per-source loaders for the raw kubectl-query corpora under
query_training/corpus/. Each loader normalises its source's own schema into a
stream of Record(instruction, command, cot, source) tuples; clean.py does the
filtering, de-duplication and splitting on top.

Registered sources (see SOURCES):

  kubectl-command-csv   kubernetes-kubectl-command-dataset/train.csv
      Columns: objective, question, command, description, syntax, flags,
      chain_of_thought. `question` is the *concrete* phrasing that matches
      `command` (objective is the generic help text -> not used as the
      instruction). Ships chain_of_thought for every row. This is the primary
      source: its `command` column is a strict superset of the old
      cli-queries `output`.

  devops-kubectl        devops-kubectl-v1/data/*.parquet
      Columns: prompt, response. The command sits in a ```bash ... ``` fence
      inside `response`; the rest of `response` is generic per-command
      boilerplate (already covered by the CSV) and is discarded. No CoT.
      Optional extra surface variety -- heavy objective overlap with the CSV,
      so clean.py's cross-source de-dup does real work here.

cli_queries_1/ is deliberately NOT registered: its command set is
byte-identical to kubectl-command-csv's, so it would only add duplicates.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Callable, Iterator, NamedTuple

_FENCE_RE = re.compile(r"```(?:bash|sh|shell)?\s*(.+?)```", re.DOTALL)


class Record(NamedTuple):
    instruction: str
    command: str
    cot: str | None
    source: str


def _first_existing(root: Path, *candidates: str) -> Path:
    for rel in candidates:
        matches = sorted(root.glob(rel))
        if matches:
            return matches[0]
    raise FileNotFoundError(f"none of {candidates} found under {root}")


def _clean_cell(value) -> str | None:
    import pandas as pd

    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    return text or None


def load_kubectl_command_csv(corpus_dir: Path) -> Iterator[Record]:
    import pandas as pd

    path = _first_existing(corpus_dir, "kubernetes-kubectl-command-dataset/train.csv")
    df = pd.read_csv(path)
    for row in df.itertuples(index=False):
        instruction = _clean_cell(row.question)
        command = _clean_cell(row.command)
        cot = _clean_cell(getattr(row, "chain_of_thought", None))
        if instruction and command:
            yield Record(instruction, command, cot, "kubectl-command-csv")


def load_devops_kubectl(corpus_dir: Path) -> Iterator[Record]:
    import pandas as pd

    paths = sorted((corpus_dir / "devops-kubectl-v1").glob("data/train-*.parquet"))
    if not paths:
        raise FileNotFoundError(f"no devops-kubectl-v1/data/train-*.parquet under {corpus_dir}")
    for path in paths:
        df = pd.read_parquet(path, columns=["prompt", "response"])
        for prompt, response in zip(df["prompt"].astype(str), df["response"].astype(str)):
            match = _FENCE_RE.search(response)
            if not match:
                continue
            command = " ".join(match.group(1).split()).strip()
            instruction = prompt.strip()
            if instruction and command:
                yield Record(instruction, command, None, "devops-kubectl")


SOURCES: dict[str, Callable[[Path], Iterator[Record]]] = {
    "kubectl-command-csv": load_kubectl_command_csv,
    "devops-kubectl": load_devops_kubectl,
}

DEFAULT_SOURCES = ("kubectl-command-csv",)


def load_sources(names: list[str], corpus_dir: Path) -> list[Record]:
    records: list[Record] = []
    for name in names:
        if name not in SOURCES:
            raise KeyError(f"unknown source {name!r}; known: {sorted(SOURCES)}")
        records.extend(SOURCES[name](corpus_dir))
    return records
