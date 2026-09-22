# praxis-comfyui-gpu

Speech-only GPU companion for Praxis: **ComfyUI + TTS + STT**, without the
llama.cpp LLM stack. Fork of `praxis-llamap-gpu` stripped to the audio side.

- **ComfyUI** (pinned rev `a2afcdb`) with two native audio nodes:
  - **PraxisQwen3TTS** — Qwen3-TTS 12Hz 1.7B Base (~4.3 GB) in an isolated
    venv; measured-fastest sdpa attention with `torch.compile` (~2× per-token
    decode on RTX 3090); warm worker keeps the model resident
    (`PRAXIS_QWEN_TTS_IDLE_SECONDS=600`). Auto-provisions on first synthesis
    when `PRAXIS_QWEN_TTS_DOWNLOAD=1`.
  - **PraxisXTTS** — Coqui XTTS v2 in its own venv (`COQUI_TOS_AGREED=1`
    required).
  - Mixed DE/JA synthesis into one WAV, DE-FR auto workflows included.
- **Vosk STT** — HTTP + WebSocket server on **2700** with the **large**
  de-0.21/fr-0.22/ja-0.22 models (~2.9/2.7/1.6 GB) provisioned at startup when
  `VOSK_DOWNLOAD_MODELS=1`. Wire protocol: `{"config":{"sample_rate":N}}`,
  binary PCM16 frames, `{"eof":1}`; language via URL path (`/de`, `/fr`,
  `/ja`), default `de`.
- **NetBird** — private overlay access (no public ports). No llama.cpp, no
  nginx proxy, no LLM endpoints: port **8188** (ComfyUI) and **2700** (Vosk).

```sh
cp runtime.env.example runtime.env
chmod 600 runtime.env
# Review model licenses and set NB_SETUP_KEY privately in runtime.env.
bash scripts/build.sh
bash scripts/verify-image.sh
docker run -d --name praxis-comfyui-gpu --gpus all \
  --env-file runtime.env -v praxis-comfyui-gpu-data:/workspace praxis-comfyui-gpu:dev
```

Speech model/reference provisioning stays as in the original stack: put an
authorized clean voice at `/workspace/input/reference.wav` (input-relative name
in the workflows: `reference.wav`). First TTS run downloads the pinned
Qwen3-TTS Base model (~4.3 GB, Apache-2.0, revision `fd4b254`); worker
failures log the real child stderr to `/workspace/logs/qwen-worker.log`.

New local changes (not yet published): Qwen provisioning uses a locked,
revision-specific `.tmp-<revision>` staging directory beside the model. An
interrupted transfer can resume there; only a complete pinned file set is
published atomically. Existing incomplete public model directories still require
inspection; no blanket deletion of user data or other revisions is performed.
`UPSTREAM.json`/`BASE_STACK.json` retain historical upstream hashes; the deliberate
local downloader change is pinned and tested in `tests/test_qwen_provision.py`.

Torch's persistent cache defaults to `/workspace/inductor-cache` with
`TORCHINDUCTOR_FX_GRAPH_CACHE=1`. Reuse depends on compatible GPU/runtime/settings;
this is not a promise of zero compilation time. Keep the warm-worker idle timeout
aligned with router policy. **Zero currently selects one-shot mode, not permanent
warm mode.** Vosk removal/migration and other router-plan items remain unfinished.

Workflows land in `/workspace/workflows` (API) and
`/workspace/user/default/workflows` (GUI) at startup, without overwriting
existing files. Read **[DEPLOY.md](DEPLOY.md)** for Praxis client settings and
operations.