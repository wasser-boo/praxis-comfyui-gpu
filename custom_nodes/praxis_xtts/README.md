# Praxis ComfyUI audio nodes

This package preserves **PraxisXTTS** and adds **PraxisQwen3TTS**, with a shared
German/Japanese segment planner and one combined PCM16 WAV per mixed reply.
Models run in isolated child processes; importing the node does not download a
model or import TTS libraries into ComfyUI's environment.

## Install carefully on the GPU host

Use the real ComfyUI directory/manager on that host; `/opt/ComfyUI` is the template
example. Back up the existing `custom_nodes/praxis_xtts` directory **outside**
`custom_nodes`, then install this complete directory as its replacement at:

```text
/opt/ComfyUI/custom_nodes/praxis_xtts/
    __init__.py
    node.py
    worker.py
    protocol.py
    warm_worker.py
    segments.py
    audio.py
    ...
```

**Do not install next to the old `praxis_xtts` node**: that registers duplicate
`PraxisXTTS` classes. Do not leave the backup in `custom_nodes`, where ComfyUI
could import it. Keep the old worker at `/opt/praxis/bin/xtts_worker.py` untouched
for rollback. The new nodes use the worker inside their own package.

Files should be readable by the service account and writable only by the operator.
After an authorized maintenance window with no running/queued jobs, restart only
ComfyUI using its actual manager. Do not restart the whole GPU stack, clear its
queue, or disable networking/authentication. Verify `/object_info/PraxisXTTS`
contains language `de-ja` and `/object_info/PraxisQwen3TTS` exists. A read-only node
probe does not run inference. The public API ports must remain unpublished and
reachable only through the configured private NetBird ACLs.

XTTS retains `/opt/tts-venv/bin/python` and the existing coqui-tts installation.
`PRAXIS_XTTS_PYTHON` optionally overrides that interpreter. Its existing
`COQUI_TOS_AGREED=1` requirement is unchanged: **only the operator may accept**
the XTTS model terms. No variable is set on their behalf.

## Optional Qwen3 environment and pinned model

Qwen dependencies conflict with some XTTS installations. Create a **new** venv;
do not pip-install these into ComfyUI's or XTTS's environment. The examples assume
Python 3.10+ and a CUDA-12.8-capable driver. Have `libsndfile`, ffmpeg and SoX
available as system libraries/tools. Download/install packages only after the
operator authorizes provisioning on the rental.

```sh
python3 -m venv /opt/qwen-tts-venv
/opt/qwen-tts-venv/bin/python -m pip install --upgrade pip
/opt/qwen-tts-venv/bin/python -m pip install torch==2.8.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
/opt/qwen-tts-venv/bin/python -m pip install -r /opt/ComfyUI/custom_nodes/praxis_xtts/qwen-requirements.txt
/opt/qwen-tts-venv/bin/python -m pip check
```

The Qwen package is pinned to `qwen-tts==0.1.1`, which pins Transformers 4.57.3
and Accelerate 1.12.0. Transitive dependencies are not fully locked; archive
`pip freeze` and validate the resulting environment. No flash-attn compilation
is required; this integration uses the model's supported PyTorch SDPA backend.

Review the model card/license, then **explicitly** download the public model:

```sh
/opt/qwen-tts-venv/bin/python /opt/ComfyUI/custom_nodes/praxis_xtts/download_qwen_model.py --download
```

This provisions `Qwen/Qwen3-TTS-12Hz-1.7B-Base` at revision
`fd4b254389122332181a7c3db7f27e918eec64e3`, including its speech tokenizer, into
`/workspace/qwen3-tts/1.7B-Base`. It is several GB; allow space for both models and
CUDA environments. The downloader refuses non-empty destinations; if interrupted,
inspect the partial download before deciding how to resume. It never accepts a
gated license, runs inference, restarts services, or uses a Hugging Face token.

Optional server environment overrides:

- `PRAXIS_QWEN_TTS_PYTHON=/opt/qwen-tts-venv/bin/python`
- `PRAXIS_QWEN_TTS_MODEL_DIR=/workspace/qwen3-tts/1.7B-Base`
- `PRAXIS_QWEN_TTS_IDLE_SECONDS=120` opts into a warm model/reference cache;
  default `0` retains one-shot behavior (integer 0–3600, POSIX only when enabled).
  First reply stays cold; later replies before idle expiry reuse the model and
  unchanged reference prompt. Cache contents stay in private child RAM.
  No TCP listener or quality/precision changes. Errors are not retried.
  History `outputs["1"].tts_timings[0]` exposes only timing/boolean metadata.
  If an owned child cannot exit, further warm requests are blocked for operator
  inspection rather than spawning duplicate GPU workers.

Inference requires local model files, `local_files_only=True`,
`trust_remote_code=False`, offline Hugging Face/Transformers mode and CUDA. There
is no implicit model download or CPU/provider fallback. To preserve these changes
when recreating a rental, incorporate the node and isolated environment into your
own image/startup provisioning; never bake reference voices or credentials into it.

## Runtime/acceptance

See [COMFYUI_QWEN3.md](../../docs/COMFYUI_QWEN3.md) for the typed Praxis context
setting `settings.comfyui_tts_language_mode=de-ja`, provider selection and limits.
No shared Ollama/ComfyUI GPU lock is added. An isolated worker frees its own VRAM
when it exits normally; warm mode retains those allocations until idle expiry.
It is outside ComfyUI's model cache. Simultaneous workloads still require
capacity planning, and a driver hang can prevent normal process cleanup.

Offline tests from the Praxis repository:

```sh
python3 -m unittest discover -s tests -p 'test_comfyui*.py'
```

GPU synthesis, sound quality, mixed-language pronunciation and native playback
must be tested after deployment. These source files alone do not activate a backend.
