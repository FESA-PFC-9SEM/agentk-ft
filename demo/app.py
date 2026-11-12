"""
Minimal demo web UI for the manifest-security auditor (dataset/ + finetune/).

Loads the base model + a LoRA adapter AT RUNTIME (finetune.evaluate.load_model
-- the same call finetune/evaluate.py uses) and serves a single-page chat-style
UI where a manifest can be:
  - fetched live from whatever cluster the current kubectl context points at
    (minikube, typically -- see demo/seed_minikube.sh to put real, flawed
    workloads there first),
  - loaded from one of the hand-curated scenarios/*.yaml files, or
  - pasted directly,
then sent to the model, which replies with the JSON findings/patch contract
dataset/schema.py defines. This is a demo, not a service: one global model
instance, no auth, Flask's dev server.

For an interactive UI that lets you pick which findings to fix and push the
result straight to the cluster, see demo/streamlit_app.py instead -- this
Flask app only previews the full patch, it doesn't apply anything.

Requires the SAME venv finetune/*.py uses (torch/unsloth/transformers) plus
flask -- see demo/requirements.txt.

Usage:
    finetune/.venv/bin/python -m pip install -r demo/requirements.txt
    finetune/.venv/bin/python -m demo.app --adapter runs/<run>/lora_adapter
    # then open http://127.0.0.1:5000
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory

from demo import core

APP_DIR = Path(__file__).resolve().parent

app = Flask(__name__, static_folder=None)

STATE: dict = {
    "model": None,
    "tokenizer": None,
    "adapter": None,
    "base_model": None,
    "max_seq_length": 4096,
    "max_new_tokens": 1280,
}


@app.get("/")
def index():
    return send_from_directory(APP_DIR / "static", "index.html")


@app.get("/api/status")
def status():
    context = core.kubectl_context()
    return jsonify(
        {
            "adapter": STATE["adapter"],
            "base_model": STATE["base_model"],
            "kubectl_context": context,
            "cluster_reachable": bool(context and core.cluster_reachable()),
            "rules": core.RULES,
        }
    )


@app.get("/api/resources")
def resources():
    include_system = request.args.get("all") == "true"
    items, error = core.list_resources(include_system)
    return jsonify({"items": items, "error": error})


@app.get("/api/resource-yaml")
def resource_yaml():
    kind = request.args.get("kind", "")
    namespace = request.args.get("namespace", "")
    name = request.args.get("name", "")
    if not (kind and namespace and name):
        return jsonify({"error": "kind, namespace and name are required"}), 400
    text, error = core.fetch_resource_yaml(kind, namespace, name)
    if error:
        return jsonify({"error": error}), 502
    return jsonify({"yaml": text})


@app.get("/api/scenarios")
def list_scenarios():
    return jsonify({"scenarios": core.list_scenarios()})


@app.get("/api/scenarios/<name>")
def scenario_detail(name: str):
    text, ground_truth = core.load_scenario(name)
    if text is None:
        return jsonify({"error": "not found"}), 404
    return jsonify({"yaml": text, "ground_truth": ground_truth})


@app.post("/api/analyze")
def analyze():
    payload = request.get_json(force=True, silent=True) or {}
    manifest = (payload.get("manifest") or "").strip()
    if not manifest:
        return jsonify({"error": "manifest is empty"}), 400

    started = time.time()
    raw = core.generate_response_text(STATE["model"], STATE["tokenizer"], core.SYSTEM_PROMPT, manifest, STATE["max_new_tokens"])
    elapsed = round(time.time() - started, 2)

    parsed, errors = core.parse_model_output(raw)
    patched_yaml, patch_error = (None, None)
    if parsed and parsed.get("patch"):
        patched_yaml, patch_error = core.apply_patch_to_manifest(manifest, parsed["patch"])

    return jsonify(
        {
            "raw": raw,
            "parsed": parsed,
            "errors": errors,
            "elapsed_seconds": elapsed,
            "patched_yaml": patched_yaml,
            "patch_error": patch_error,
        }
    )


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--adapter", required=True, help="path to a saved LoRA adapter dir, loaded at runtime")
    parser.add_argument("--max-seq-length", type=int, default=4096)
    parser.add_argument("--max-new-tokens", type=int, default=1280)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5000)
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    adapter_path = Path(args.adapter)

    print(f"Loading base model + LoRA adapter from {adapter_path} ...")
    model, tokenizer = core.load_model(str(adapter_path), args.max_seq_length)
    STATE.update(
        model=model,
        tokenizer=tokenizer,
        adapter=str(adapter_path),
        base_model=core.base_model_for_adapter(str(adapter_path)),
        max_seq_length=args.max_seq_length,
        max_new_tokens=args.max_new_tokens,
    )

    print(f"Model ready. Open http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
