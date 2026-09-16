# syntax=docker/dockerfile:1
FROM netbirdio/netbird:0.78.2-rootless@sha256:4a30e98bb0a98848c3b2a3ca4778cebfbe87d210f05b1f40288bb072d627c3fd AS netbird
FROM nvidia/cuda:12.8.1-devel-ubuntu22.04@sha256:a99a1860ba8e2916e5c3e73b72ec4c4301653a84586e05bfc9a2aa2d58027e97 AS llama-build
ARG LLAMA_CPP_REV=27c54b4bbcefadedcec6397477cc2e866c1db716
ARG CUDA_DOCKER_ARCH=75;80;86;89;90;120
ARG LLAMA_BUILD_JOBS=4
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake git ca-certificates libssl-dev \
    && rm -rf /var/lib/apt/lists/*
RUN git init /src/llama.cpp && cd /src/llama.cpp \
    && git remote add origin https://github.com/thecodacus/llama.cpp.git \
    && git fetch --depth 1 origin "$LLAMA_CPP_REV" && git checkout --detach FETCH_HEAD \
    && test "$(git rev-parse HEAD)" = "$LLAMA_CPP_REV" \
    && cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
       -DGGML_CUDA=ON -DGGML_NATIVE=OFF -DGGML_BACKEND_DL=ON -DGGML_CPU_ALL_VARIANTS=ON \
       -DCMAKE_CUDA_ARCHITECTURES="$CUDA_DOCKER_ARCH" \
       -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_APP=OFF -DLLAMA_BUILD_UI=OFF \
       -DCMAKE_EXE_LINKER_FLAGS=-Wl,--allow-shlib-undefined \
    && cmake --build build --target llama-server llama-moe-trace --parallel "$LLAMA_BUILD_JOBS" \
    && mkdir -p /opt/llama.cpp/bin \
    && cp build/bin/llama-server build/bin/llama-moe-trace /opt/llama.cpp/bin/ \
    && find build -name '*.so*' -exec cp -P {} /opt/llama.cpp/bin/ \; \
    && cp LICENSE /opt/llama.cpp/LICENSE \
    && git rev-parse HEAD > /opt/llama.cpp/REVISION
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
    ffmpeg libsndfile1 espeak-ng sox libsox-fmt-all nginx supervisor tini curl util-linux libgomp1 libssl3 \
    && rm -rf /var/lib/apt/lists/*
COPY --from=netbird /usr/local/bin/netbird /usr/local/bin/netbird
COPY --from=netbird /usr/local/bin/netbird-entrypoint.sh /usr/local/bin/netbird-entrypoint.sh
# Executables and ALL shared libraries come from the same pinned fork build.
COPY --from=llama-build /opt/llama.cpp /opt/llama.cpp
RUN printf '/opt/llama.cpp/bin\n' > /etc/ld.so.conf.d/llama-cpp.conf && ldconfig
COPY LLAMA_CPP.json /opt/praxis/LLAMA_CPP.json
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
COPY scripts /opt/praxis/bin
COPY supervisord.conf /etc/supervisor/praxis.conf
COPY workflows /opt/praxis/workflows
RUN chmod +x /opt/praxis/bin/*.sh \
    && /opt/llama.cpp/bin/llama-server --help > /tmp/llama-help.txt \
    && grep -q -- '--moe-cache-profile' /tmp/llama-help.txt \
    && grep -q -- '--moe-cache-slots' /tmp/llama-help.txt \
    && grep -q -- 'draft-mtp' /tmp/llama-help.txt \
    && grep -q -- '--spec-draft-model' /tmp/llama-help.txt \
    && grep -q -- '--spec-draft-ngl' /tmp/llama-help.txt \
    && /opt/llama.cpp/bin/llama-moe-trace --help > /dev/null
# Syntax check only: nginx does not start during this build step.
RUN mkdir -p /workspace/logs && nginx -t -c /opt/praxis/bin/llama-proxy.conf
RUN cd /opt/ComfyUI && /opt/comfy-venv/bin/python -c 'import torch, folder_paths' \
    && /opt/tts-venv/bin/python -c 'from TTS.api import TTS'
# Do not EXPOSE API ports: Vast automatically publishes EXPOSE declarations.
WORKDIR /opt/ComfyUI
HEALTHCHECK --interval=30s --start-period=300s --timeout=25s --retries=3 \
    CMD python3 /opt/praxis/bin/healthcheck.py
ENTRYPOINT ["/usr/bin/tini", "--", "/opt/praxis/bin/start.sh"]
