"""
Pipeline orchestration: real corpus (parquet) -> dataset.jsonl.

Flow: load shards -> filter to valid manifests (skip Helm templates with
'{{' and YAML that doesn't parse) -> deduplicate structurally -> drop
documents with a real secret -> normalize into canonical form -> inject one
defect per rule (plus a slice of clean negatives) -> validate the round-trip
of every patch (100%, not sampled) -> write train/val/test.jsonl, split by
source repository (never randomly, to avoid leaking near-identical forks
across splits).

Usage:
    python -m dataset.build --limit 1000 --total 300
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import random
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import jsonpatch
import pandas as pd
import yaml

from dataset.dedup import dedup
from dataset.detect import detect_structural
from dataset.mutate import FAKE_SECRET_VAR_NAMES, MUTATORS
from dataset.normalize import normalize_document
from dataset.scanning import find_secrets
from dataset.schema import SYSTEM_PROMPT, Response, validate_response

RULE_IDS = tuple(MUTATORS)

_IDENTIFIER_NAME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_.-]{2,39}$")


def _is_harvestable_key_name(key: str, reason: str) -> bool:
    """True if a SecretHit's key (never its value) looks like a real
    variable/field name worth reusing as an injectable KSEC-001 identifier.
    Excludes high-entropy-only hits, where the key isn't required to look
    sensitive at all and could be a list index or an unrelated field name --
    only hits scanning.py matched on the key itself (sensitive-key regex,
    PEM, connection string) have a key worth harvesting."""
    if reason.startswith("high entropy"):
        return False
    return bool(_IDENTIFIER_NAME_RE.match(key))


@dataclass
class Record:
    doc: dict
    repo: str
    path: str


# ---------------------------------------------------------------------------
# Loading and filtering
# ---------------------------------------------------------------------------


def load_records(corpus_dir: Path, limit: int | None) -> list[Record]:
    """Reads the parquet shards and returns one Record per valid YAML
    document (with 'kind' and 'apiVersion'), skipping Helm templates ('{{')
    and YAML that doesn't parse. `limit`, if given, caps the number of
    parquet ROWS read (not the final number of documents)."""
    records: list[Record] = []
    rows_read = 0
    shard_paths = sorted(corpus_dir.glob("*.parquet"))
    if not shard_paths:
        raise FileNotFoundError(f"no parquet shard found in {corpus_dir}")

    for shard_path in shard_paths:
        df = pd.read_parquet(shard_path, columns=["content", "max_stars_repo_name", "max_stars_repo_path"])
        for _, row in df.iterrows():
            if limit is not None and rows_read >= limit:
                return records
            rows_read += 1
            text = row["content"]
            if not isinstance(text, str) or "{{" in text:
                continue
            try:
                loaded = list(yaml.safe_load_all(text))
            except yaml.YAMLError:
                continue
            for doc in loaded:
                if isinstance(doc, dict) and "kind" in doc and "apiVersion" in doc:
                    records.append(
                        Record(
                            doc=doc,
                            repo=str(row["max_stars_repo_name"]),
                            path=str(row["max_stars_repo_path"]),
                        )
                    )
    return records


def load_synthetic_records(synthetic_dir: Path) -> list[Record]:
    """Reads every *.curated.jsonl from generation/output/ (already filtered
    by curate.py) and returns one Record per document. Each synthetic
    document gets its own unique fake "repository" for split purposes -- it
    already went through structural dedup in curate.py, so there's no risk
    of leaking near-duplicates across train/val/test the way real forks
    would."""
    records: list[Record] = []
    if not synthetic_dir.exists():
        return records
    for jsonl_path in sorted(synthetic_dir.glob("*.curated.jsonl")):
        with jsonl_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                mode = item.get("mode", "synthetic")
                index = item.get("index", 0)
                try:
                    docs = list(yaml.safe_load_all(item.get("manifest_yaml", "")))
                except yaml.YAMLError:
                    continue
                for doc_idx, doc in enumerate(docs):
                    if isinstance(doc, dict) and "kind" in doc and "apiVersion" in doc:
                        records.append(
                            Record(
                                doc=doc,
                                repo=f"synthetic:{mode}:{index}:{doc_idx}",
                                path=jsonl_path.name,
                            )
                        )
    return records


# ---------------------------------------------------------------------------
# Round-trip and serialization
# ---------------------------------------------------------------------------


def assert_round_trip(mutated_doc: dict, patch, canonical: dict, context: str) -> None:
    ops = [{k: v for k, v in p.to_dict().items() if k != "doc"} for p in patch]
    reconstructed = jsonpatch.apply_patch(mutated_doc, ops)
    if reconstructed != canonical:
        raise RuntimeError(f"round-trip failure in {context}: patch does not reproduce the canonical form")


def doc_to_yaml(doc: dict) -> str:
    return yaml.safe_dump(doc, sort_keys=False, default_flow_style=False)


def make_example(doc_for_input: dict, response: Response, repo: str, rule_id: str) -> dict:
    errors = validate_response(response.to_dict())
    if errors:
        raise RuntimeError(f"response fails schema validation ({rule_id}, repo={repo}): {errors}")
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": doc_to_yaml(doc_for_input)},
            {"role": "assistant", "content": json.dumps(response.to_dict(), ensure_ascii=False)},
        ]
    }


# ---------------------------------------------------------------------------
# Split by repository (deterministic, never random)
# ---------------------------------------------------------------------------


def split_bucket(repo: str, train_ratio: float, val_ratio: float) -> str:
    digest = hashlib.sha256(repo.encode("utf-8")).hexdigest()
    bucket = int(digest[:8], 16) / 0xFFFFFFFF
    if bucket < train_ratio:
        return "train"
    if bucket < train_ratio + val_ratio:
        return "val"
    return "test"


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------


def build(args: argparse.Namespace) -> dict:
    rng = random.Random(args.seed)
    corpus_dir = Path(args.corpus_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Reading corpus shards...", file=sys.stderr)
    records = load_records(corpus_dir, args.limit)
    corpus_docs_read = len(records)

    synthetic_docs_read = 0
    if args.synthetic_dir:
        synthetic_records = load_synthetic_records(Path(args.synthetic_dir))
        synthetic_docs_read = len(synthetic_records)
        records.extend(synthetic_records)
        print(f"Synthetic documents merged in: {synthetic_docs_read}", file=sys.stderr)

    docs_read = len(records)
    print(f"Valid documents read (corpus + synthetic): {docs_read}", file=sys.stderr)

    kept_idx, survival_rate = dedup([r.doc for r in records])
    records = [records[i] for i in kept_idx]
    unique_after_dedup = len(records)

    dirty_count = 0
    clean_records = []
    harvested_key_names: set[str] = set()
    for r in records:
        hits = find_secrets(r.doc)
        if hits:
            dirty_count += 1
            for hit in hits:
                if _is_harvestable_key_name(hit.key, hit.reason):
                    harvested_key_names.add(hit.key)
        else:
            clean_records.append(r)

    kind_distribution = collections.Counter(r.doc.get("kind", "?") for r in records)

    canonical_records: list[Record] = []
    dropped_unfixable = 0
    for r in clean_records:
        canonical = normalize_document(r.doc)
        if canonical is None:
            dropped_unfixable += 1
            continue
        if detect_structural(canonical):
            raise RuntimeError(f"bug in normalize.py: doc from {r.repo} is still dirty after normalization")
        canonical_records.append(Record(doc=canonical, repo=r.repo, path=r.path))

    # Union of the curated base pool with names actually seen in real corpus
    # documents (key names only -- never the leaked values, which were
    # already discarded above along with the whole document). This gives
    # KSEC-001 far more realistic lexical/naming-convention diversity than
    # the fixed base pool alone.
    ksec001_candidate_names = sorted(set(FAKE_SECRET_VAR_NAMES) | harvested_key_names)

    diagnostic = {
        "docs_read": docs_read,
        "corpus_docs_read": corpus_docs_read,
        "synthetic_docs_read": synthetic_docs_read,
        "unique_after_dedup": unique_after_dedup,
        "dedup_survival_rate": round(survival_rate, 4),
        "dropped_as_dirty_secret": dirty_count,
        "dropped_as_unfixable_rbac": dropped_unfixable,
        "usable_canonical_docs": len(canonical_records),
        "kind_distribution": dict(kind_distribution.most_common()),
        "harvested_credential_key_names": len(harvested_key_names),
        "ksec001_candidate_name_pool_size": len(ksec001_candidate_names),
    }
    print("\n=== Corpus diagnostic ===", file=sys.stderr)
    print(json.dumps(diagnostic, indent=2, ensure_ascii=False), file=sys.stderr)

    if not canonical_records:
        raise RuntimeError("no usable canonical document: pipeline cannot generate examples")

    # Pool of applicable mutations per rule: tries to mutate every canonical
    # doc once; keeps the result so we don't mutate twice with divergent rng
    # states between the counting phase and the sampling phase.
    pools: dict[str, list[tuple[Record, object]]] = {rid: [] for rid in RULE_IDS}
    mutation_precondition_failures = 0
    order = list(canonical_records)
    rng.shuffle(order)
    for rid in RULE_IDS:
        mutator = MUTATORS[rid]
        for r in order:
            try:
                if rid == "KSEC-001":
                    result = mutator(r.doc, rng, candidate_names=ksec001_candidate_names)
                else:
                    result = mutator(r.doc, rng)
            except AssertionError:
                # A mutator's own internal invariant didn't hold for this
                # particular document (e.g. a malformed field from noisy
                # generated input defeated the "this mutation always
                # produces a finding" assumption). This is a per-document,
                # per-rule skip, not a pipeline failure: the same document
                # is still tried against every other rule normally.
                mutation_precondition_failures += 1
                continue
            if result is not None:
                pools[rid].append((r, result))

    quotas = _resolve_quotas(args, pools)

    examples = []
    for rid in RULE_IDS:
        pool = pools[rid]
        rng.shuffle(pool)
        for r, result in pool[: quotas[rid]]:
            assert_round_trip(result.mutated_doc, result.patch, result.canonical, context=f"{rid}/{r.repo}")
            response = Response(
                findings=result.findings,
                patch=result.patch,
                new_resources=result.new_resources,
                notes=[],
            )
            example = make_example(result.mutated_doc, response, r.repo, rid)
            examples.append((r.repo, example))

    negative_quota = quotas["negative"]
    negative_pool = list(canonical_records)
    rng.shuffle(negative_pool)
    for r in negative_pool[:negative_quota]:
        response = Response()
        example = make_example(r.doc, response, r.repo, "negative")
        examples.append((r.repo, example))

    rng.shuffle(examples)

    splits = {"train": [], "val": [], "test": []}
    for repo, example in examples:
        splits[split_bucket(repo, args.train_ratio, args.val_ratio)].append(example)

    for split_name, split_examples in splits.items():
        out_path = output_dir / f"{split_name}.jsonl"
        with out_path.open("w", encoding="utf-8") as fh:
            for example in split_examples:
                fh.write(json.dumps(example, ensure_ascii=False) + "\n")

    diagnostic["quotas"] = quotas
    diagnostic["mutation_precondition_failures"] = mutation_precondition_failures
    diagnostic["examples_emitted"] = {k: len(v) for k, v in splits.items()}
    diagnostic["examples_emitted"]["total"] = sum(len(v) for v in splits.values())

    diag_path = output_dir / "diagnostic.json"
    diag_path.write_text(json.dumps(diagnostic, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n=== Examples emitted ===", file=sys.stderr)
    print(json.dumps(diagnostic["examples_emitted"], indent=2), file=sys.stderr)
    if mutation_precondition_failures:
        print(
            f"\n[WARNING] {mutation_precondition_failures} mutation attempt(s) hit an internal "
            "invariant and were skipped (see mutation_precondition_failures in diagnostic.json). "
            "A handful is expected from noisy generated input; a large count is worth investigating.",
            file=sys.stderr,
        )
    print(f"\nSplits written to {output_dir}/", file=sys.stderr)

    return diagnostic


def _resolve_quotas(args: argparse.Namespace, pools: dict[str, list]) -> dict[str, int]:
    """Decides how many examples per rule + negatives, respecting --total
    and --negative-ratio, but never exceeding how much each rule actually
    managed to mutate (available pool)."""
    total = args.total
    negative_target = round(total * args.negative_ratio)
    positive_target = total - negative_target
    per_rule_target = positive_target // len(RULE_IDS)

    quotas = {}
    for rid in RULE_IDS:
        available = len(pools[rid])
        quotas[rid] = min(per_rule_target, available)

    # redistributes what's left over (rules with a small pool) to the
    # others, as far as available material allows.
    leftover = positive_target - sum(quotas.values())
    if leftover > 0:
        hungry = sorted(RULE_IDS, key=lambda rid: len(pools[rid]) - quotas[rid], reverse=True)
        for rid in hungry:
            room = len(pools[rid]) - quotas[rid]
            take = min(room, leftover)
            quotas[rid] += take
            leftover -= take
            if leftover <= 0:
                break

    quotas["negative"] = negative_target
    return quotas


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-dir", default="corpus")
    parser.add_argument("--synthetic-dir", default=None, help="folder with *.curated.jsonl from generation/curate.py")
    parser.add_argument("--output-dir", default="dataset/output")
    parser.add_argument("--limit", type=int, default=None, help="cap on parquet rows read (smoke test)")
    parser.add_argument("--total", type=int, default=2000, help="total number of examples to generate")
    parser.add_argument("--negative-ratio", type=float, default=0.35)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    build(args)


if __name__ == "__main__":
    main()
