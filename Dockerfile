# syntax=docker/dockerfile:1
FROM netbirdio/netbird:0.78.2-rootless@sha256:4a30e98bb0a98848c3b2a3ca4778cebfbe87d210f05b1f40288bb072d627c3fd AS netbird
FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04@sha256:17e2934e1fa96152b14f78078bfbafd0f00f391df995dc6c641a720fce1202bb

ARG COMFY_REV=a2afcdb89910f7ada92e2a6e04c8b5c9126122f1
ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    HF_HOME=/workspace/huggingface TTS_HOME=/workspace/tts \
    PRAXIS_QWEN_TTS_MODEL_DIR=/workspace/qwen3-tts/1.7B-Base VOSK_DEFAULT_LANGUAGE=de \
    NB_USE_NETSTACK_MODE=true NB_ENABLE_NETSTACK_LOCAL_FORWARDING=true \
    NB_NETSTACK_SKIP_PROXY=true NB_DISABLE_DNS=true NB_ENABLE_CAPTURE=false \
    NB_CONFIG=/workspace/netbird/config.json NB_STATE_DIR=/workspace/netbird \
    NB_DAEMON_ADDR=unix:///workspace/netbird/netbird.sock NB_LOG_FILE=console \
    NB_ENTRYPOINT_SERVICE_TIMEOUT=60
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-venv python3-dev build-essential git ca-certificates \
    ffmpeg libsndfile1 espeak-ng sox libsox-fmt-all supervisor tini curl util-linux libssl3 \
    && rm -rf /var/lib/apt/lists/*
COPY --from=netbird /usr/local/bin/netbird /usr/local/bin/netbird
COPY --from=netbird /usr/local/bin/netbird-entrypoint.sh /usr/local/bin/netbird-entrypoint.sh
RUN git init /opt/ComfyUI && cd /opt/ComfyUI \
    && git remote add origin https://github.com/Comfy-Org/ComfyUI.git \
    && git fetch --depth 1 origin "$COMFY_REV" && git checkout --detach FETCH_HEAD \
    && rm -rf .git
COPY constraints.txt /opt/constraints.txt
RUN python3 -m venv /opt/comfy-venv && python3 -m venv /opt/tts-venv \
    && /opt/comfy-venv/bin/pip install --upgrade pip \
    && /opt/tts-venv/bin/pip install --upgrade pip
# Separate layers preserve successful installs when a later dependency fails.
# Cache stays disabled: a retry downloads a fresh wheel; integrity checks stay on.
RUN /opt/comfy-venv/bin/pip install -c /opt/constraints.txt torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
RUN /opt/tts-venv/bin/pip install -c /opt/constraints.txt torch==2.8.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
RUN /opt/comfy-venv/bin/pip install -c /opt/constraints.txt -r /opt/ComfyUI/requirements.txt
RUN /opt/tts-venv/bin/pip install -c /opt/constraints.txt 'coqui-tts[languages]==0.27.5'
RUN /opt/comfy-venv/bin/pip check && /opt/tts-venv/bin/pip check
RUN python3 -m venv /opt/vosk-venv \
    && /opt/vosk-venv/bin/pip install vosk==0.3.45 aiohttp==3.13.3 \
    && /opt/vosk-venv/bin/pip check \
    && /opt/vosk-venv/bin/python -c 'import vosk'
COPY custom_nodes/praxis_xtts /opt/ComfyUI/custom_nodes/praxis_xtts
# Qwen conflicts with XTTS's Transformers constraints: never merge these venvs.
RUN python3 -m venv /opt/qwen-tts-venv && /opt/qwen-tts-venv/bin/pip install --upgrade pip
RUN /opt/qwen-tts-venv/bin/pip install torch==2.8.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
RUN /opt/qwen-tts-venv/bin/pip install -r /opt/ComfyUI/custom_nodes/praxis_xtts/qwen-requirements.txt \
    && /opt/qwen-tts-venv/bin/pip check \
    && /opt/qwen-tts-venv/bin/python -c 'from qwen_tts import Qwen3TTSModel'
# Pinned FlashAttention wheel for the PRAXIS_QWEN_TTS_ATTN=flash_attention_2 opt-in.
# Measured slower than sdpa for this workload on RTX 3090; sdpa stays the default.
RUN curl -fsSL --output "/tmp/flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp310-cp310-linux_x86_64.whl" "https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp310-cp310-linux_x86_64.whl" \
    && echo "75c51f34bb93c5a4438d6b767547ca6c22a9bfc6b29b5646dda98e943cd75d99  /tmp/flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp310-cp310-linux_x86_64.whl" | sha256sum -c - \
    && /opt/qwen-tts-venv/bin/pip install --no-cache-dir "/tmp/flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp310-cp310-linux_x86_64.whl" \
    && rm "/tmp/flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp310-cp310-linux_x86_64.whl" \
    && /opt/qwen-tts-venv/bin/python -c 'import flash_attn; assert flash_attn.__version__ == "2.8.3"'
COPY scripts /opt/praxis/bin
COPY supervisord.conf /etc/supervisor/praxis.conf
COPY workflows /opt/praxis/workflows
RUN chmod +x /opt/praxis/bin/*.sh \
    && cd /opt/ComfyUI && /opt/comfy-venv/bin/python -c 'import torch, folder_paths' \
    && /opt/tts-venv/bin/python -c 'from TTS.api import TTS'
# Do not EXPOSE API ports: Vast automatically publishes EXPOSE declarations.
WORKDIR /opt/ComfyUI
HEALTHCHECK --interval=30s --start-period=300s --timeout=25s --retries=3 \
    CMD python3 /opt/praxis/bin/healthcheck.py
ENTRYPOINT ["/usr/bin/tini", "--", "/opt/praxis/bin/start.sh"]