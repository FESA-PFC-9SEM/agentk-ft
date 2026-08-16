"""
Extracts the YAML manifests from inside the JSONL files (final dataset or
synthetic generation output) into individual .yaml files, easy to open in an
editor -- no need to dig through the JSON by hand.

Automatically recognizes two line formats:
- final dataset format (dataset/build.py): {"messages": [system, user, assistant]}
  -- the YAML is in messages[1].content, and the response (findings/patch/...)
  is in messages[2].content.
- synthetic generation format (generation/generate.py or *.curated.jsonl):
  {"manifest_yaml": "...", "mode": "...", "seed": {...}}

Usage:
    python -m dataset.view dataset/output/train.jsonl --out /tmp/yamls
    python -m dataset.view generation/output/rbac.curated.jsonl --out /tmp/yamls
    python -m dataset.view dataset/output/train.jsonl --rule-id KSEC-002 --limit 10
    python -m dataset.view dataset/output/test.jsonl --stdout --limit 3
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def _slugify(text: str, max_len: int = 40) -> str:
    text = re.sub(r"[^a-zA-Z0-9._-]+", "-", text).strip("-")
    return text[:max_len] or "item"


def _iter_records(input_path: Path):
    with input_path.open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh):
            line = line.strip()
            if not line:
                continue
            yield line_no, json.loads(line)


def _from_dataset_format(record: dict) -> tuple[str, str, dict | None]:
    """Returns (base_name, yaml_text, response_dict) for a record in the
    {"messages": [...]} format."""
    messages = record["messages"]
    manifest_yaml = messages[1]["content"]
    response = json.loads(messages[2]["content"])
    rule_ids = sorted({f["rule_id"] for f in response.get("findings", [])})
    tag = "-".join(rule_ids) if rule_ids else "clean"
    return tag, manifest_yaml, response


def _from_generation_format(record: dict) -> tuple[str, str, dict | None]:
    """Returns (base_name, yaml_text, None) for a record in the
    {"manifest_yaml": ..., "mode": ..., "index": ...} format."""
    mode = record.get("mode", "synthetic")
    index = record.get("index", "?")
    return f"{mode}-{index}", record.get("manifest_yaml", ""), None


def _matches_rule_filter(response: dict | None, rule_id: str | None) -> bool:
    if rule_id is None:
        return True
    if response is None:
        return False
    return any(f["rule_id"] == rule_id for f in response.get("findings", []))


def extract(args: argparse.Namespace) -> int:
    input_path = Path(args.input)
    out_dir = Path(args.out) if args.out else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)

    written = 0
    shown = 0
    for line_no, record in _iter_records(input_path):
        if args.limit is not None and shown >= args.limit:
            break

        if "messages" in record:
            tag, manifest_yaml, response = _from_dataset_format(record)
        elif "manifest_yaml" in record:
            tag, manifest_yaml, response = _from_generation_format(record)
        else:
            continue

        if not _matches_rule_filter(response, args.rule_id):
            continue
        if args.mode and record.get("mode") != args.mode:
            continue

        shown += 1
        base_name = f"{input_path.stem}_{line_no:04d}_{_slugify(tag)}"

        if args.stdout:
            print(f"\n{'=' * 80}\n# {base_name}\n{'=' * 80}")
            print(manifest_yaml.rstrip())
            if response is not None:
                print(f"\n--- response ({len(response.get('findings', []))} finding(s)) ---")
                print(json.dumps(response, indent=2, ensure_ascii=False))

        if out_dir:
            (out_dir / f"{base_name}.yaml").write_text(manifest_yaml, encoding="utf-8")
            if args.sidecar and response is not None:
                (out_dir / f"{base_name}.response.json").write_text(
                    json.dumps(response, indent=2, ensure_ascii=False), encoding="utf-8"
                )
            written += 1

    if out_dir:
        print(f"\n{written} manifest(s) written to {out_dir}/")
    else:
        print(f"\n{shown} manifest(s) shown (use --out to also write them to disk).")
    return 0


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", help=".jsonl file (final dataset or generation/* output)")
    parser.add_argument("--out", default=None, help="folder to write one .yaml per example")
    parser.add_argument("--stdout", action="store_true", help="also print each manifest to the terminal")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--rule-id", default=None, help="only examples with a finding for this rule, e.g. KSEC-002")
    parser.add_argument("--mode", default=None, help="only examples from this mode (generation format: base/rbac/hard-negative)")
    parser.add_argument("--sidecar", action="store_true", help="also write the response (findings/patch) to .response.json")
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    raise SystemExit(extract(args))


if __name__ == "__main__":
    main()
