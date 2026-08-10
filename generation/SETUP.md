# Local inference runtime setup (Part B)

## Runtime choice: Ollama

Two viable options on this machine: Ollama or a llama.cpp server. We picked
**Ollama** for the following reasons, in the priority order set by the
project brief ("ease of setup and batch throughput over single-request
latency"):

- Already installed on this machine (`ollama` on PATH) and auto-detects the
  GPU via CUDA.
- Manages GGUF quantization and weight downloads with a single command
  (`ollama pull`), no compilation needed.
- Exposes an OpenAI-compatible API (`/v1/chat/completions`) and a native API
  (`/api/generate`), both trivial to call in parallel with `httpx`/`asyncio`
  to maximize batch throughput. `OLLAMA_NUM_PARALLEL` controls how many
  concurrent requests the server serves.

llama.cpp server's strong point is GBNF grammar-constrained decoding, which
guarantees syntactically valid output -- useful for the requirement to
"generate YAML only, with handling for when the model disobeys." As a
cheaper-to-set-up alternative, `generation/generate.py` handles this with
strict prompting + markdown-fence stripping + a bounded retry when the YAML
doesn't parse. If `curate.py`'s pass rate comes in low AND the reason is
specifically malformed output (not content quality), the documented
fallback is switching to a llama.cpp server with GBNF grammar -- that
wouldn't require rewriting the pipeline, just the model-calling module in
`generate.py`.

## Model: Qwen2.5-Coder-7B-Instruct, Q4_K_M quantization

Kept the brief's suggestion. It's heavily trained on code/config, including
YAML, which is exactly the target format. At Q4_K_M it takes up roughly
4.7 GB of weights, comfortable within the ~7.2 GB of free VRAM on this GPU
(RTX 4060 Ti, 8 GB).

**Model weights are NOT downloaded automatically by this pipeline.**
Run this manually when you're ready for Part B:

```bash
ollama pull qwen2.5-coder:7b-instruct-q4_K_M
```

## Step by step

1. Confirm the Ollama service is running:
   ```bash
   systemctl --user status ollama    # or: ollama serve (in another terminal)
   ```
2. Run the environment check (driver, CUDA, free VRAM, model response):
   ```bash
   .venv/bin/python generation/check_env.py
   ```
3. Tune `OLLAMA_NUM_PARALLEL` (Ollama's default is usually enough for a
   single 8 GB GPU; there's no need to raise it beyond what the VRAM allows
   for more than one loaded model instance).

## kubeconform

Used by `generation/curate.py` to validate synthetic manifests against the
official Kubernetes schema. Installed locally via Go (not a model weight,
just a command-line tool):

```bash
GOBIN=$(go env GOPATH)/bin go install github.com/yannh/kubeconform/cmd/kubeconform@latest
export PATH=$PATH:$(go env GOPATH)/bin
```
