"""
Synthetic generation of CLEAN manifests via Ollama. The local model only
generates clean input material -- never labels, never the defect. Injecting
the defect and computing findings/patch is always dataset/mutate.py's job,
called later in build.py. This module just needs to produce plausible,
diverse manifests (base/rbac) or deliberate "false positives" for the secret
detector (hard-negative).

Resumable: every item has a deterministic index derived from (--seed, mode,
index); on restart, indices already present in the output file are skipped,
so re-running the command only fills in what's missing, without duplicating.

Usage:
    python -m generation.generate --mode base -n 500
    python -m generation.generate --mode rbac -n 20
    python -m generation.generate --mode hard-negative -n 200
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import re
import sys
import time
from pathlib import Path

import httpx
import yaml
from tqdm import tqdm

from generation.seeds import Seed, sample_seed

OLLAMA_URL = "http://localhost:11434"
MODEL_NAME = "qwen2.5-coder:7b-instruct-q4_K_M"
MAX_RETRIES = 2
FENCE_RE = re.compile(r"^```(?:yaml|yml)?\s*\n?|\n?```\s*$", re.MULTILINE)

BASE_PROMPT_TEMPLATE = """\
You are a senior Kubernetes platform engineer. Generate a REALISTIC and \
SECURE Kubernetes manifest (no security problems whatsoever: no plaintext \
credential, no insecure securityContext, no hostNetwork/hostPID/hostIPC/\
sensitive hostPath, no permissive RBAC, container image always with a \
pinned tag and never 'latest').

Follow exactly these constraints:
{constraints}

Respond with ONLY the manifest YAML. Do not include explanations, do not \
include markdown code fences (```), do not include any text before or after \
the YAML."""

HARD_NEGATIVE_PROMPT_TEMPLATE = """\
You are a senior Kubernetes platform engineer. Generate a REALISTIC \
Kubernetes manifest that is completely SECURE, but that contains values \
which LOOK suspicious at first glance to a naive secret scanner -- without \
being actual secrets. Examples of the pattern you should produce:
- a SHA-256 image digest (high entropy, but it's just a public image hash, \
not a secret)
- an environment variable named something like API_TOKEN_HEADER whose value \
is just the NAME of an HTTP header (e.g. "X-Api-Token"), not an actual token
- a variable like SESSION_TIMEOUT with a numeric value (e.g. "3600")
- explicit placeholder values (e.g. "changeme", "${{DB_PASSWORD}}", \
"<SET_ME>")
- a real credential already correctly resolved via secretKeyRef (never in \
plaintext)

Follow exactly these constraints:
{constraints}

Respond with ONLY the manifest YAML. Do not include explanations, do not \
include markdown code fences (```), do not include any text before or after \
the YAML."""

RETRY_SUFFIX = "\n\nReminder: respond with ONLY plain YAML, no code fences (```) and no extra text."


def build_prompt(seed: Seed) -> str:
    template = HARD_NEGATIVE_PROMPT_TEMPLATE if seed.mode == "hard-negative" else BASE_PROMPT_TEMPLATE
    return template.format(constraints=seed.to_prompt_constraints())


def strip_fences(text: str) -> str:
    return FENCE_RE.sub("", text.strip()).strip()


def looks_like_manifest(text: str) -> bool:
    try:
        docs = list(yaml.safe_load_all(text))
    except yaml.YAMLError:
        return False
    if not docs:
        return False
    return any(isinstance(d, dict) and "kind" in d and "apiVersion" in d for d in docs)


def load_completed_indices(output_path: Path) -> set[int]:
    if not output_path.exists():
        return set()
    completed = set()
    with output_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "index" in record:
                completed.add(record["index"])
    return completed


async def generate_one(client: httpx.AsyncClient, index: int, args: argparse.Namespace) -> dict | None:
    rng = random.Random(args.seed * 1_000_003 + index)
    seed = sample_seed(rng, mode=args.mode)
    prompt = build_prompt(seed)

    for attempt in range(MAX_RETRIES + 1):
        payload = {
            "model": args.model,
            "prompt": prompt if attempt == 0 else prompt + RETRY_SUFFIX,
            "stream": False,
            "options": {"temperature": 0.8, "top_p": 0.9, "num_predict": 1024},
        }
        try:
            resp = await client.post(f"{OLLAMA_URL}/api/generate", json=payload, timeout=120)
            resp.raise_for_status()
        except httpx.HTTPError as e:
            print(f"[http error] index {index}, attempt {attempt}: {e}", file=sys.stderr)
            continue

        raw_text = resp.json().get("response", "")
        cleaned = strip_fences(raw_text)
        if looks_like_manifest(cleaned):
            return {
                "index": index,
                "mode": args.mode,
                "model": args.model,
                "seed": seed.to_dict(),
                "manifest_yaml": cleaned,
            }

    print(f"[discarded] index {index}: model did not produce valid YAML after {MAX_RETRIES + 1} attempts", file=sys.stderr)
    return None


async def run(args: argparse.Namespace) -> None:
    output_path = Path(args.output) if args.output else Path(f"generation/output/{args.mode}.jsonl")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    completed = load_completed_indices(output_path)
    remaining = [i for i in range(args.n) if i not in completed]
    if not remaining:
        print(f"{len(completed)} items are already complete in {output_path}; nothing to do.")
        return
    print(f"Resuming: {len(completed)} already complete, generating the {len(remaining)} remaining.", file=sys.stderr)

    semaphore = asyncio.Semaphore(args.concurrency)
    write_lock = asyncio.Lock()
    start_time = time.monotonic()
    written = 0

    async def worker(client: httpx.AsyncClient, index: int, pbar: tqdm) -> None:
        nonlocal written
        async with semaphore:
            record = await generate_one(client, index, args)
        if record is not None:
            async with write_lock:
                with output_path.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1
        pbar.update(1)

    async with httpx.AsyncClient() as client:
        with tqdm(total=len(remaining), desc=f"generating ({args.mode})") as pbar:
            await asyncio.gather(*(worker(client, i, pbar) for i in remaining))

    elapsed_min = max((time.monotonic() - start_time) / 60, 1e-6)
    rate = written / elapsed_min
    print(f"\nSuccessfully generated: {written}/{len(remaining)} attempted ({rate:.1f} manifests/min).")
    print(f"Output: {output_path}")


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("base", "rbac", "hard-negative"), required=True)
    parser.add_argument("-n", type=int, required=True, help="number of valid manifests to generate")
    parser.add_argument("--output", default=None)
    parser.add_argument("--model", default=MODEL_NAME)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
