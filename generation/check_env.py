"""
Checks whether the environment is ready for Part B (local synthetic
generation): NVIDIA driver, CUDA, free VRAM, the Ollama service is up, and,
if the model has already been pulled, a test response. Downloads nothing --
just diagnoses.

Usage: python generation/check_env.py
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys

import httpx

MODEL_NAME = "qwen2.5-coder:7b-instruct-q4_K_M"
MIN_FREE_VRAM_MIB = 5000
OLLAMA_URL = "http://localhost:11434"


def check_nvidia_smi() -> bool:
    if shutil.which("nvidia-smi") is None:
        print("[FAIL] nvidia-smi not found on PATH.")
        return False
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total,memory.free", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        print(f"[FAIL] nvidia-smi did not respond: {e}")
        return False

    line = out.stdout.strip().splitlines()[0]
    print(f"[OK] GPU detected: {line}")
    try:
        free_mib = int(line.split(",")[-1].strip().split()[0])
    except (ValueError, IndexError):
        print("[WARN] could not parse free VRAM.")
        return True
    if free_mib < MIN_FREE_VRAM_MIB:
        print(f"[WARN] free VRAM ({free_mib} MiB) below the recommended amount ({MIN_FREE_VRAM_MIB} MiB).")
    else:
        print(f"[OK] enough free VRAM: {free_mib} MiB")
    return True


def check_ollama_service() -> bool:
    try:
        resp = httpx.get(f"{OLLAMA_URL}/api/version", timeout=5)
        resp.raise_for_status()
    except httpx.HTTPError as e:
        print(f"[FAIL] Ollama service did not respond at {OLLAMA_URL}: {e}")
        print("       run 'ollama serve' (or check the systemd service) and try again.")
        return False
    print(f"[OK] Ollama responding, version {resp.json().get('version', '?')}")
    return True


def check_model_present() -> bool:
    try:
        resp = httpx.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        resp.raise_for_status()
    except httpx.HTTPError as e:
        print(f"[FAIL] could not list models: {e}")
        return False
    names = {m["name"] for m in resp.json().get("models", [])}
    if MODEL_NAME in names:
        print(f"[OK] model '{MODEL_NAME}' is already pulled.")
        return True
    print(f"[PENDING] model '{MODEL_NAME}' has not been pulled yet.")
    print(f"          run: ollama pull {MODEL_NAME}")
    return False


def check_model_responds() -> bool:
    payload = {
        "model": MODEL_NAME,
        "prompt": "Reply with just the word: ok",
        "stream": False,
        "options": {"num_predict": 5},
    }
    try:
        resp = httpx.post(f"{OLLAMA_URL}/api/generate", json=payload, timeout=60)
        resp.raise_for_status()
    except httpx.HTTPError as e:
        print(f"[FAIL] model did not respond to a test prompt: {e}")
        return False
    text = resp.json().get("response", "").strip()
    print(f"[OK] model responded to the test prompt: {text!r}")
    return True


def main() -> int:
    print("=== Environment check for local synthetic generation ===\n")
    gpu_ok = check_nvidia_smi()
    print()
    service_ok = check_ollama_service()
    print()
    if not service_ok:
        print("\nResult: Ollama service unavailable, check incomplete.")
        return 1

    model_present = check_model_present()
    print()
    model_ok = check_model_responds() if model_present else False

    print("\n=== Summary ===")
    print(json.dumps({
        "gpu_ok": gpu_ok,
        "ollama_service_ok": service_ok,
        "model_present": model_present,
        "model_responds": model_ok,
    }, indent=2, ensure_ascii=False))

    return 0 if (gpu_ok and service_ok) else 1


if __name__ == "__main__":
    sys.exit(main())
