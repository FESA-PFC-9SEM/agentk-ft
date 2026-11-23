# The cluster agent app (Streamlit UI, planner, fine-tuned auditor, MCP
# server) as a container, for running the demo entirely on Kubernetes
# (deploy/agent-app.yaml). Built by deploy/deploy_app_k8s.py from a staged
# context: the Python packages the app imports, one LoRA adapter, and this
# directory -- not the whole repository (25 runs of checkpoints, datasets).
#
# The base model isn't in the image: the pod's init container downloads it
# into a volume once (deploy/agent-app.yaml), so the image stays ~10 GB
# smaller and a new adapter doesn't mean a new 5 GB layer.
FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# A C compiler: Triton (used by Unsloth) compiles small kernels at runtime.
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

# kubectl, matching the cluster's version, checksum-verified.
ARG KUBECTL_VERSION=v1.35.1
RUN curl -fsSLo /usr/local/bin/kubectl "https://dl.k8s.io/release/${KUBECTL_VERSION}/bin/linux/amd64/kubectl" \
    && echo "$(curl -fsSL "https://dl.k8s.io/release/${KUBECTL_VERSION}/bin/linux/amd64/kubectl.sha256")  /usr/local/bin/kubectl" | sha256sum -c - \
    && chmod 0755 /usr/local/bin/kubectl

# Exactly the package set the benchmarks and live tests ran on
# (requirements.lock.txt, frozen from finetune/.venv), installed as-is.
# torch and the NVIDIA libraries come from the CUDA 12.6 index: the wheels
# bundle the CUDA runtime, so the node provides only the driver.
COPY deploy/app/requirements.lock.txt /tmp/requirements.lock.txt
RUN pip install --no-deps --extra-index-url https://download.pytorch.org/whl/cu126 -r /tmp/requirements.lock.txt \
    && python -c "import torch, transformers, peft, bitsandbytes, streamlit, mcp, psycopg, authlib, jsonpatch" \
    && rm /tmp/requirements.lock.txt

RUN useradd --create-home --uid 1000 app
WORKDIR /app
COPY --chown=root:root . /app

# Writable places (the root filesystem is read-only in the pod): the
# base-model cache volume, Unsloth's compiled kernels, HOME (Triton, kubectl
# and Streamlit caches) and /tmp are volumes in deploy/agent-app.yaml.
ENV HF_HOME=/models/hf \
    UNSLOTH_COMPILE_LOCATION=/cache/unsloth_compiled_cache \
    HOME=/home/app

USER 1000
EXPOSE 8501
CMD ["streamlit", "run", "/app/demo/streamlit_app.py", "--server.address", "0.0.0.0", "--server.port", "8501", "--server.headless", "true"]
