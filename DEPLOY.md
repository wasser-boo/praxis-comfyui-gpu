# Deployment: ComfyUI + TTS + STT (fork)

Speech-only separation fork of `praxis-llamap-gpu`: **no llama.cpp, no Ollama,
no nginx proxy**. Two private listeners over NetBird: ComfyUI on **8188**
(TTS nodes) and Vosk on **2700** (STT).

## Provisioning

Weights and personal references are never baked into the image.

1. Put an authorized clean reference voice at `/workspace/input/reference.wav`
   (preferably mode 600, never commit personal audio). The workflow field
   `reference_audio` takes only the input-relative filename (`reference.wav`),
   not a path. The server input directory is `/workspace/input` (ComfyUI
   derives it from `--base-directory /workspace` at the pinned revision).
2. Review XTTS terms and set `COQUI_TOS_AGREED=1` only if you accept them.
   XTTS may download on first use into persistent `/workspace/tts`.
3. `PRAXIS_QWEN_TTS_DOWNLOAD=1` authorizes the worker to download the pinned
   `Qwen/Qwen3-TTS-12Hz-1.7B-Base` (rev `fd4b254`, Apache-2.0, ~4.3 GB) into
   `/workspace/qwen3-tts/1.7B-Base` before the first synthesis (flock-
   serialized; offline flags are stripped for the download subprocess only).
   A first synthesis may take minutes on a cold directory; an interrupted
   download leaves an incomplete tree that must be removed before retrying.
4. `VOSK_DOWNLOAD_MODELS=1` provisions the large publisher models
   (de-0.21 / fr-0.22 / ja-0.22) at startup via `run_vosk.py` before the
   server execs; without the flag a missing model surfaces as HTTP 503.

## Praxis client settings

```text
/context set settings.voice_tts_type=comfyui_qwen3 settings.comfyui_base_url=http://GPU_NETBIRD_IP:8188 settings.comfyui_tts_workflow=workflows/tts-qwen3-api.json settings.comfyui_qwen_reference_audio=reference.wav settings.comfyui_qwen_language=auto settings.comfyui_tts_language_mode=de-ja settings.use_tts=true settings.web_chat_tts=true
```

XTTS: `settings.voice_tts_type=comfyui_xtts` with `workflows/tts-api.json`.
STT: `settings.voice_vosk_url=ws://GPU_NETBIRD_IP:2700` (no path = German;
`/de`, `/fr`, `/ja` select languages). The WebSocket server rejects browser
`Origin` headers by design (403) — use the Praxis Python client, and serve the
dashboard over HTTPS if browser mic capture is needed.

## Operations

- `supervisorctl -c /etc/supervisor/praxis.conf status` — services: `netbird`,
  `vosk`, `comfyui`. Logs below `/workspace/logs/`.
- Container health requires both `PraxisXTTS` and `PraxisQwen3TTS` registered
  with `de-ja`, a Vosk `/health` (also 503-unhealthy when models are missing),
  and a ready NetBird peer.
- Worker debugging: the warm-parent hides child errors on purpose
  (`Qwen warm worker failed ... NOT retried`), but the bounded child-stderr
  tail is persisted to `/workspace/logs/qwen-worker.log` — check it first.
- Smoke test against a live box:
  `python3 /opt/praxis/bin/smoke.py --url http://<netbird-ip>:8188 --workflow /workspace/workflows/tts-qwen3-api.json --output /tmp/smoke-audio`.

## Image build and verification

```sh
bash scripts/build.sh          # ComfyUI + XTTS/Qwen-TTS venvs + Vosk venv
bash scripts/verify-image.sh   # CPU-only checks: imports, supervisor, CPU API checks
```

`verify-image.sh` runs ComfyUI in CPU router mode to prove both speech nodes
register with `de-ja` and that Vosk answers `/health` (503 until models are
provisioned; expected in the image).