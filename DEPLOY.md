# Praxis llama.cpp GPU services

## Scope and validation

This is the **separate `praxis-llamap-gpu` repository**, based on the tracked files
of `praxis-gpu` recorded in `BASE_STACK.json`. The original repository and running
services are not modified. No runtime identities, voices, models or credentials
were copied into this repository.

The text backend is **https://github.com/thecodacus/llama.cpp/tree/perf**, pinned to
`27c54b4bbcefadedcec6397477cc2e866c1db716`. It is **not** FreeToken, upstream stock
llama.cpp, or the GenerelSchwerz fork linked in the video's description.

Validated: offline source tests and CPU/image API checks, followed by GPU acceptance
of the corrected launcher on **RTX A4000 16 GB** (2026-09-16). The 48-layer/56-slot
CUDA expert pack uploaded **5,072.27 MiB**; the shared Q8 MTP head loaded and accepted
draft tokens. Repeated chat, a tool-call round trip, a **2,044-token prompt with
finite per-token probabilities**, and 768-dimensional Nomic embeddings passed.
Observed GPU memory after the long prompt: **13,373 / 16,376 MiB**. This is a
snapshot, not a peak-memory guarantee or a full-context/concurrency/quality benchmark.
The upstream 16.6 → 24.4 tok/s result is not a measurement of this deployment.

The published **`vayayo/praxis-llamap-gpu:0.2` predates these repairs**. Its deployed
container was hotfixed without replacing the fork binary, trunk weights or routing
profiles. A new image build, image verification and release remain separate steps;
pushing this repository does not update the Docker Hub tag.

## Preserved stack

- Pinned CUDA 12.8.1 / Ubuntu 22.04 runtime, NetBird 0.78.2 and ComfyUI revision.
- Torch 2.8.0 and the existing SDXL workflow.
- `PraxisXTTS`, coqui-tts 0.27.5, `/opt/tts-venv`.
- `PraxisQwen3TTS`, qwen-tts 0.1.1, isolated `/opt/qwen-tts-venv`.
- Existing mixed DE/JA speech routing, optional warm Qwen worker, and DE/FR Auto
  workflows; XTTS remains one-shot. No changes to the vendored speech code.
- SoX, libsndfile, ffmpeg, espeak-ng and four separate Python environments.
- Vosk 0.3.45 + aiohttp 3.13.3, HTTP WAV and native WebSocket APIs, DE/FR/JA models.
- NetBird netstack, loopback-only listeners, no TUN, added capabilities or host networking.

Ollama, its CUDA-link repair and its model-tag provisioner are removed. There is
no FreeToken service, dependency or fallback. Nomic embeddings are retained using
a second **CPU-only llama-server**, not an Ollama process. Its GGUF is the pinned
Nomic v1.5 f16 model; existing Ollama embeddings must not be assumed bit-identical.

Top-level dependencies are pinned, not every transitive Python wheel. Archive
`pip freeze` for all four environments after building. Python 3.10 in the base
reaches end of life in October 2026; upgrading it needs separate validation.

## Codacus Flash recipe

**Model:** `unsloth/Qwen3.8-Flash-Next-GGUF`, **177B UD-IQ3_XXS**, architecture
`qwen4exp`, 512 experts/layer and 10 active. This is **not** `qwen3.8:27b`.
The three shard revisions, byte sizes and publisher LFS SHA256s are pinned in
`scripts/llama-models.json` (81,961,823,936 bytes total, approximately 82 GB).
Those files contain the trunk only. The same pinned repository revision supplies
`MTP/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf` (2,786,568,256 bytes), also pinned by
publisher SHA256 in the manifest. This is the matching MTP head, not another LLM;
see [the publisher's MTP guide](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF/blob/38bb39ee97821de2c9009abb7e93950eec396e66/MTP/README.md).

The configured chat command is equivalent to:

```sh
threads="$(python3 /opt/praxis/bin/cpu_threads.py)"
GGML_CUDA_REGISTER_HOST=1 GGML_SCHED_PREFETCH_EXPERTS=0 \
/opt/llama.cpp/bin/llama-server \
  --model /workspace/llama.cpp/models/qwen3.8-flash-next/UD-IQ3_XXS/Qwen3.8-Flash-Next-UD-IQ3_XXS-00001-of-00003.gguf \
  --alias qwen3.8-flash-next --host 127.0.0.1 --port 11435 \
  --n-gpu-layers 99 --n-cpu-moe 99 --flash-attn on \
  --fit off --load-mode mmap --lazy-mode on \
  --threads "$threads" --threads-batch "$threads" --ctx-size 4096 --parallel 1 \
  --batch-size 512 --ubatch-size 512 --jinja --cache-ram 0 \
  --verbosity 4 --moe-cache-profile /workspace/llama.cpp/profiles/qwen3.8-flash-next-merged.csv \
  --moe-cache-slots 56 --spec-type draft-mtp \
  --spec-draft-model /workspace/llama.cpp/models/qwen3.8-flash-next/MTP/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf \
  --spec-draft-ngl 99 --spec-draft-n-max 2 \
  --spec-draft-threads "$threads" --spec-draft-threads-batch "$threads" \
  --no-webui --no-slots --no-warmup
```

- **56 slots**, from the Flash benchmark row, **not** the generic 112-slot example.
- **`--spec-type draft-mtp`** and **`--spec-draft-model`** are always passed.
  The shared Q8 head borrows the main model's embedding/output tensors; its own
  layers are GPU-offloaded, with at most two draft tokens per step. The pinned
  Codacus fork supports this pairing. Omitting the separate head fails with
  `model doesn't contain MTP layers`; disabling MTP is not the repair.
  Missing/invalid head weights fail before profiling or starting the server.
- **One worker per available physical core**, detected automatically instead of
  hard-coding six or counting SMT threads. Codacus's six-core CPU selects six
  workers if container limits allow it. Generation, prompt processing, MTP,
  routing traces and Nomic embedding worker pools all use this thread budget.
- Code and chat routing traces are captured separately, up to **512 generated
  tokens each** at **4096 context**, then concatenated into one model-specific CSV.
  Generation may stop earlier at EOS. The trace command uses `-ngl 99 -ncmoe 99`,
  Flash Attention and the same thread/loading settings, without an expert cache.
- Host registration remains enabled. **Experimental expert prefetch is disabled**:
  with `GGML_SCHED_PREFETCH_EXPERTS=1`, a ~315-token tool prompt produced repeated
  `/` characters and non-finite probabilities, poisoning subsequent requests.
  Disabling CUDA graphs did not fix it; disabling prefetch did. The launcher removes
  this flag even when an old template still supplies `1`, and warns at chat startup.
  Update old environments to `GGML_SCHED_PREFETCH_EXPERTS=0` too. This fences off the
  observed fork issue; it is not a claimed C++ root-cause repair.
  **CUDA graphs, async CPU splits, host pinning, the 56-slot pack and MTP remain on.**
  No TurboQuant, model/weight changes or alternative fork flags are introduced.
- The README's `-fa 1` example is stale for the pinned parser: use **`--flash-attn on`**.
- **4096 context and 512 batch/ubatch are conservative deployment defaults**, not
  a claim to have recovered every benchmark setting from the video. No automatic
  increase to the old 262144 context, quantization change or VRAM-based tuning.

### Persistent instance tuning without restarting the whole stack

Environment variables still define portable defaults. Optional
**`/workspace/llama.cpp/runtime-settings.json`** takes precedence for the bounded
keys `LLAMA_CONTEXT_SIZE`, `LLAMA_BATCH_SIZE`, `LLAMA_UBATCH_SIZE`,
`LLAMA_MOE_CACHE_SLOTS` and `LLAMA_THREADS` only. Example for a measured 32k instance:

```json
{"LLAMA_CONTEXT_SIZE": 32768}
```

This file stays in the persistent volume, not Git. Preserve any other existing
values, write it atomically with mode 600, and deliberately restart **only**
`llama-chat` after its callers are idle. Launcher and readiness checks read the
same tuning, even if Supervisor inherited an older 4096 environment. Its values
are also passed into the child environment. Embeddings retain their own 2048
context, and routing traces stay at 4096; no profile regeneration is necessary.
A `LLAMA_THREADS` override also changes the embedding worker budget, so deliberately
restart `llama-embedding` too when changing threads (not for a chat-context change).
Unknown keys, symlinks, non-regular files, oversized JSON and out-of-range values
fail closed. This file cannot enable downloads/prefetch or change keys, models or
listeners. The image/example default remains 4096 for smaller cards. Measure VRAM
and prompts before adopting 32768; a configured limit is not a full-window benchmark.

### Hardware and shared VRAM

The video's machine had an RTX 3060 12 GB, Ryzen 5600X and 64 GB RAM. That does not
mean the **whole speech/image stack** also fits concurrently on a 12 GB GPU.
Plan for fast SSD/NVMe storage, **at least 64 GB host RAM with workload headroom**,
and preferably a **48 GB+ GPU** for the combined stack. Allow 200 GB+ disk, more
for image checkpoints and runtime outputs. A 12 GB card requires serialized jobs
and measured memory budgets; this repository does not add a shared GPU scheduler.

The cold expert source still occupies host memory; lazy PLE reads use storage.
Pinned host memory is not free RAM. Do not impose a low container memory limit.
Leave at least the README's ~900 MB additional VRAM headroom, and more for actual
context, MTP, ComfyUI and speech. Larger context competes with the expert pack.

Unlike the previous `OLLAMA_KEEP_ALIVE=0`, llama-server **keeps its model/cache
resident**. If an image/TTS workload needs the VRAM, explicitly stop `llama-chat`
only after its callers are idle, then start it again afterwards. Never unload
someone else's job or clear a shared queue to make a request fit.

### Automatic physical-core thread budget

`LLAMA_THREADS=auto` is the default, including when the variable is unset.
`scripts/cpu_threads.py` counts unique Linux CPU sibling groups **within the
process's allowed affinity** (so taskset/Docker cpusets are respected). It supports
SMT, non-SMT, multiple sockets and restricted access to only secondary SMT siblings.
If sysfs topology is unavailable it uses `/proc/cpuinfo` socket/core IDs; it never
assumes two logical CPUs per core or substitutes the host's logical CPU count.

Visible cgroup v1/v2 CPU quotas, including visible parent quotas, further cap the
budget. Fractional CPU quotas round down, with at least one worker and at most
512. Hidden host limits and shared CPU contention cannot be inferred from inside
the container. If topology/quota data cannot be interpreted, startup fails with
a request for an explicit numeric `LLAMA_THREADS` rather than guessing.

The selected count is logged at startup and passed explicitly to generation,
prefill and MTP; traces and embeddings use the same calculation. This sizes the
**worker pools**, not every HTTP/helper thread, and does not pin workers, reserve
exclusive cores or impose a global CPU budget across ComfyUI/Vosk/other services.
CPU/cgroup changes require a deliberate service restart, not hot retuning.

To activate this in an older `runtime.env`, **replace `LLAMA_THREADS=6` with
`LLAMA_THREADS=auto`** (or remove it). A positive numeric value such as
`LLAMA_THREADS=6` remains an intentional override and bypasses auto detection.
Existing routing profiles remain reusable; no new model download is needed.

Inspect the automatic budget without downloads or GPU inference:

```sh
python3 /opt/praxis/bin/cpu_threads.py
python3 /opt/praxis/bin/run_llama.py chat --print-command
```

## Runtime auto-download and one-time profiling

Copy `runtime.env.example` to untracked `runtime.env`, chmod 600, review the model
licenses, and enter the NetBird setup key privately. Relevant selections:

```dotenv
LLAMA_DOWNLOAD_MODELS=1
LLAMA_MODEL_DIR=/workspace/llama.cpp/models/qwen3.8-flash-next/UD-IQ3_XXS
LLAMA_MTP_MODEL_DIR=/workspace/llama.cpp/models/qwen3.8-flash-next/MTP
LLAMA_GENERATE_PROFILE=1
LLAMA_MOE_CACHE_PROFILE=/workspace/llama.cpp/profiles/qwen3.8-flash-next-merged.csv
LLAMA_MOE_CACHE_SLOTS=56
LLAMA_THREADS=auto
LLAMA_CONTEXT_SIZE=4096
LLAMA_BATCH_SIZE=512
LLAMA_UBATCH_SIZE=512
GGML_CUDA_REGISTER_HOST=1
GGML_SCHED_PREFETCH_EXPERTS=0
LLAMA_EMBEDDING_ENABLED=1
LLAMA_EMBEDDING_MODEL_DIR=/workspace/llama.cpp/models/nomic-embed-text
```

At startup, **`llama-chat`**:

1. Validates settings and checks the **fork binary's CUDA device list**, refusing
   a CPU-only chat fallback.
2. Verifies existing selected GGUF files, including the separate MTP head;
   downloads only missing files if `LLAMA_DOWNLOAD_MODELS=1`, using the pinned
   Hugging Face revision over HTTPS.
   Every new download is streamed to `.part`, SHA256-checked, and published without
   overwriting an existing file. Existing weights with bad sizes/hashes fail clearly.
3. Generates the two generic routing traces if the merged profile is missing and
   `LLAMA_GENERATE_PROFILE=1`. **This is real GPU inference at first startup.**
4. Validates that the CSV has Flash's 10 active experts, in-range IDs and decode
   rows, saves its SHA256/model/fork recipe sidecar, and launches the server.
5. Watches the actual startup output for
   `init_moe_expert_cache: expert cache: ... layers x 56 slots, ... uploaded to CUDA0`.
   A disabled-cache/no-CPU-MoE fallback is fatal. Health additionally requires
   `/health` and a cache-activation receipt tied to the current process and settings.
   Log verbosity is explicitly **4**: this revision hides the library's cache
   activation INFO at the default level 3. The main model must activate its cache
   before MTP loading starts. The GPU-only MTP head's later `no CPU-resident MoE
   layers` message does not invalidate that verified main-model pack.
   No main-model activation line means it cannot pass readiness.

The independent `llama-embedding` program verifies/downloads pinned Nomic and
serves `/v1/embeddings` on CPU at **2048 context**. It never receives Flash's cache,
MTP or 256k context settings. Set `LLAMA_EMBEDDING_ENABLED=0` to explicitly disable
it. Downloads/profiling do not block Vosk or ComfyUI startup, but they share host
resources: keep client GPU workloads idle while the initial profile runs.

Set `LLAMA_DOWNLOAD_MODELS=0` for preinstalled/offline weights and
`LLAMA_GENERATE_PROFILE=0` for an existing verified profile. Unset variables have
these safe zero defaults; the **example explicitly opts into both**. No model
weights are included/downloaded during Docker build. Nothing sets
`COQUI_TOS_AGREED` or accepts XTTS terms on your behalf.

`LLAMA_DOWNLOAD_TIMEOUT_SECONDS` defaults to 21600 (60–86400); per-read socket
timeout is 60 seconds. `LLAMA_TRACE_TIMEOUT_SECONDS` defaults to 3600 per trace
(60–21600). Downloads and failed profiling are **not retried by Supervisor**.
Inspect `.part` files or `flash-trace-*` staging directories before deliberately
starting a new attempt. There is no automatic resume or deletion of partials.

Restarts reuse verified weights and profiles. Private verification receipts use
file size/inode/timestamps to avoid repeatedly hashing 82 GB of unchanged data;
recently modified files are always rehashed. These receipts are a local cache,
not protection against a hostile storage administrator. For a full audit:

```sh
python3 /opt/praxis/bin/provision_llama.py --kind chat --rehash
python3 /opt/praxis/bin/provision_llama.py --kind mtp --rehash
python3 /opt/praxis/bin/provision_llama.py --kind embedding --rehash
```

### Reuse the weights/profile you already have

Point `LLAMA_MODEL_DIR` at the directory containing **all three original shard
filenames**, for example your existing `UD-IQ3_XXS` directory. They must match the
pinned hashes; no unnecessary second download or silent replacement occurs.
Do not point llama.cpp at Ollama manifests/blobs or the old 27B model.
Set `LLAMA_MTP_MODEL_DIR` separately if the matching shared Q8 head already exists
elsewhere. Otherwise only that missing ~2.8 GB file needs downloading. Adding this
head does not change the trunk/profile recipe or regenerate existing valid traces.

Point `LLAMA_MOE_CACHE_PROFILE` at an existing merged **Flash Next** routing CSV.
Profiles generated by this stack are reused automatically. An external CSV has
no provenance sidecar: after verifying which model it came from, explicitly adopt
it with the chat service stopped:

```sh
supervisorctl -c /etc/supervisor/praxis.conf stop llama-chat
python3 /opt/praxis/bin/profile_llama.py --adopt-existing
supervisorctl -c /etc/supervisor/praxis.conf start llama-chat
```

Adoption does not infer which model produced an arbitrary CSV; the operator is
responsible for that association. Without adoption, an existing unverified or
changed profile is preserved and startup fails instead of overwriting it. To
recapture, select a new absent profile path and leave `LLAMA_GENERATE_PROFILE=1`.

Do not carry over `OLLAMA_*`, upstream `LLAMA_ARG_*` or legacy
`GGML_MOE_CACHE_*` variables. They are rejected to avoid silent configuration
conflicts. There is no arbitrary shell `EXTRA_ARGS`. Setting
`GGML_CUDA_REGISTER_HOST=0` through this launcher **unsets** it in the child,
because the pinned CUDA implementation enables registration by presence alone.

## Build and start privately

On the NVIDIA workstation with Docker/NVIDIA Container Toolkit:

```sh
IMAGE_TAG=praxis-llamap-gpu:dev bash scripts/build.sh
IMAGE_TAG=praxis-llamap-gpu:dev bash scripts/verify-image.sh
docker run -d --name praxis-llamap-gpu --gpus all \
  --env-file runtime.env -v praxis-llamap-gpu-data:/workspace praxis-llamap-gpu:dev
```

Build targets `linux/amd64`. The separate pinned CUDA **devel** stage compiles
both server and trace tool; all matching fork `.so` libraries travel together into
the unchanged CUDA/cuDNN runtime. Never mix binaries/libraries from another fork.
`LLAMA_BUILD_JOBS=4` bounds compiler parallelism. To shorten a device-specific
build, e.g. RTX 3060/A6000:

```sh
CUDA_DOCKER_ARCH=86 LLAMA_BUILD_JOBS=4 bash scripts/build.sh
```

Default architectures: 75/80/86/89/90/120. Build expects a large image (three Torch
installations). No API `EXPOSE`, `-p`, TUN, privileged flags, host networking or
extra capabilities. `verify-image.sh` runs with `--network none`, no GPU, no keys,
no model downloads; it checks imports, fork flags, real empty-model HTTP routing,
private proxy rejections, speech node registration and Vosk's unprovisioned state.
It **cannot** validate CUDA, expert cache, MTP output or VPN reachability.

Offline source tests:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install aiohttp==3.13.3
.venv/bin/python -m unittest discover -s tests
bash -n scripts/*.sh
```

### Private NetBird routes

```text
Praxis -> GPU_NETBIRD_IP:11434 -> nginx 127.0.0.1:11434
       -> /v1/chat/completions, /v1/models, /health -> llama-chat 127.0.0.1:11435
       -> /v1/embeddings, /health/embedding        -> llama-embedding 127.0.0.1:11436
Praxis -> GPU_NETBIRD_IP:8188  -> ComfyUI 127.0.0.1:8188
Praxis -> GPU_NETBIRD_IP:2700  -> Vosk    127.0.0.1:2700
```

NetBird's overlay address is not an OS interface in netstack mode. Keep listeners
on loopback; its local forwarding handles overlay connections. Restrict NetBird
ACLs to **only the Praxis peer/group**, TCP **11434, 8188, 2700**. Do not permit
internal **11435/11436**, SOCKS 1080 or broad all-to-all access. The APIs have no
independent user authentication. Public mappings are not safe.

The nginx frontend streams without buffering/retries, rejects browser `Origin`
headers, and exposes only the API routes listed above. **`/` returning 404 is
intentional**, not a failed connection. Use `/health` or `/v1/models`. During model
preparation, the chat routes can return 502 while the independent embedding route
is already healthy; do not interrupt a working download based on that alone. No browser UI, MCP proxy,
model-management or slots endpoints are exposed. Backend `/props` remains
available only locally for diagnostics. **Port 11434 is no longer an Ollama API**;
`/api/chat`, `/api/tags` and `/api/embeddings` are deliberately not emulated.

## Praxis client configuration

On the **Praxis machine**, not in the GPU container's env:

```dotenv
USE_PROVIDER=llamacpp
LLAMACPP_API_BASE=http://GPU_NETBIRD_IP:11434
LLAMACPP_MODEL=qwen3.8-flash-next
LLM_FALLBACK_PROVIDERS=
# Optional local Nomic embeddings via the OpenAI-compatible format:
EMBEDDING_PROVIDER=openai
EMBEDDING_MODEL=nomic-embed-text
OPENAI_BASE_URL=http://GPU_NETBIRD_IP:11434/v1
```

Praxis's `llamacpp` provider itself appends `/v1/chat/completions`, so
**do not append `/v1` to `LLAMACPP_API_BASE`**. Its embedding client uses
`OPENAI_BASE_URL` (different from chat's `OPENAI_API_BASE`) and appends
`/embeddings`, so the embedding URL **does** include `/v1`. `openai` here selects
an API format; with this private base it does not call OpenAI. Do not inject an
unrelated cloud API key into this unauthenticated local endpoint.

For a context that explicitly selected Ollama/the old 27B model, update it:

```text
/context set settings.provider=llamacpp settings.model=qwen3.8-flash-next
```

Set history/compaction budgets to fit the actual server context. For the example
4096 window, **start small** (e.g. 2048 history / 1536 compaction threshold) and
measure room for system instructions, tool definitions/results and output. This
is not a guarantee that a large agent prompt fits. For long-context use, explicitly
raise `LLAMA_CONTEXT_SIZE`, measure peak memory, and adjust client budgets. Do not
keep the old 196608-token history budget with a 4096 server. Do not enable
compaction or delete stored history merely as part of this migration.

The embedding model/precision may differ from an existing Ollama tag. Validate
vectors/dimensions and plan a separate reindex before mixing old and new vectors;
this repository does not mutate the Praxis database. Nomic's `search_query:` /
`search_document:` task prefixes remain the caller's responsibility.

## Speech/image provisioning (unchanged)

Weights and personal references are not baked in. Startup seeds missing generic
workflows in `/workspace/workflows`, with GUI examples in
`/workspace/user/default/workflows`, without overwriting existing files.

1. Put an authorized clean reference at `/workspace/input/reference.wav`, preferably
   mode 600. Never commit personal audio.
2. Review XTTS terms and set `COQUI_TOS_AGREED=1` **only if you accept them**.
   XTTS may download on first use into persistent `/workspace/tts`.
3. Flag-gated speech provisioning mirrors `LLAMA_DOWNLOAD_MODELS` — with
   `PRAXIS_QWEN_TTS_DOWNLOAD=1` and `VOSK_DOWNLOAD_MODELS=1` in the environment
   the stack provisions everything with no manual step: Vosk at startup, the
   Qwen model on the first worker load (before the first synthesis):

   ```sh
   /opt/qwen-tts-venv/bin/python \
     /opt/ComfyUI/custom_nodes/praxis_xtts/download_qwen_model.py --download
   ```

   This installs pinned `Qwen/Qwen3-TTS-12Hz-1.7B-Base` at
   `/workspace/qwen3-tts/1.7B-Base`. Speech inference is offline-only and requires
   CUDA, without silent CPU/provider fallback. Transformers environments remain
   isolated from XTTS. See [speech details](docs/COMFYUI_QWEN3.md). The worker
   runs the same pinned download automatically before its first load when
   `PRAXIS_QWEN_TTS_DOWNLOAD=1`; a flock serializes concurrent workers. A first
   synthesis may therefore take several minutes on a cold model directory; do not
   interrupt it mid-download — an incomplete directory must be removed or
   completed before the worker retries.

   The Vosk service starts through `run_vosk.py`: with `VOSK_DOWNLOAD_MODELS=1`
   it provisions the **large** publisher models (Apache-2.0, CRC- and
   layout-validated, path-safety checked) before executing the unchanged
   `vosk_server.py`: German `vosk-model-de-0.21` (~2.9 GB), French
   `vosk-model-fr-0.22`, Japanese `vosk-model-ja-0.22` (~1.6 GB) in
   `/workspace/vosk/<lang>`. Japanese transcription quality improved from
   garbled small-model output to correct full sentences. Without the flag the
   server starts immediately and missing models surface as HTTP 503, as before.

   **Qwen3-TTS speed policy on the GPU host** (measured on RTX 3090):

   - **torch.compile** on the talker and code predictor roughly halves the
     per-token decode time (123 ms/step to 65 ms). It is enabled by default
     (`PRAXIS_QWEN_TTS_COMPILE=0` disables); the first worker start pays a
     one-time ~50-100 s compile, amortized by the warm idle window
     (`PRAXIS_QWEN_TTS_IDLE_SECONDS=600`).
   - **Attention stays on `sdpa`**: `flash_attention_2` measured consistently
     slower (13.9-15.4 s vs 10.3-11.9 s for identical text), so it remains an
     opt-in via `PRAXIS_QWEN_TTS_ATTN=flash_attention_2` (the pinned wheel is
     installed in the image and verified at build time).
   - Mixed DE/JA replies measured: 75 s (previous deployment, fa2) -> 27 s warm
     with compile; single Japanese sentences synthesize in ~4-5 s.

   A loaded ComfyUI node does **not** mean these ~4.5 GB speech weights are installed.
   In warm mode, missing weights surface as `Qwen warm worker failed; owned worker
   stopped, request NOT retried` because worker errors deliberately do not echo
   private text. The real child failure (e.g. `Install the pinned Qwen3-TTS Base
   model locally before synthesis`) is now persisted together with the bounded
   child-stderr tail in `/workspace/logs/qwen-worker.log` — check it first when a
   warm worker fails. Provision the model before retrying; no ComfyUI restart is
   needed after the model itself lands. In the workflow's `reference_audio` field
   use only the input-relative filename, e.g. `reference.wav`, **not**
   `/workspace/input/reference.wav`. Uploading a voice does not change an existing
   workflow's selected filename.
4. Obtain/verify an SDXL checkpoint separately in `/workspace/models/checkpoints`.
5. Provision Vosk models: the flag-gated `run_vosk.py` startup handles the
   **large** publisher models automatically with `VOSK_DOWNLOAD_MODELS=1`; the
   legacy small-model command stays available for thin hosts:

   ```sh
   python3 /opt/praxis/bin/provision_vosk.py
   supervisorctl -c /etc/supervisor/praxis.conf restart vosk
   ```

   Review https://alphacephei.com/vosk/models. Versioned ZIPs are integrity-checked
   but lack an independently pinned publisher SHA256. Vosk and overall health
   remain unready until all three languages load.

Existing generic workflows are retained: `image-sdxl-api.json`, `tts-api.json`,
`tts-qwen3-api.json`, XTTS/Qwen DE-JA API/UI pairs, Qwen DE-FR Auto API/UI pair.
GUI workflow JSON is **not** an API prompt. Praxis must have the API JSON on its
own local filesystem; a GPU-side workflow does not create that local file.

Example Praxis speech settings:

```text
/context set settings.voice_tts_type=comfyui_qwen3 settings.comfyui_base_url=http://GPU_NETBIRD_IP:8188 settings.comfyui_tts_workflow=workflows/tts-qwen3-api.json settings.comfyui_qwen_reference_audio=reference.wav settings.comfyui_qwen_language=auto settings.comfyui_tts_language_mode=de-ja settings.use_tts=true settings.web_chat_tts=true
```

`PRAXIS_QWEN_TTS_IDLE_SECONDS=120` keeps one warm Qwen worker/reference for repeat
replies; `0` restores per-reply exit. Warm workers occupy VRAM until expiry.
German/Latin and Japanese scripts are synthesized in order into one WAV; romaji
can be marked `[[ja]]arigatou[[/ja]]`. XTTS stays available as `comfyui_xtts`.

### Vosk wire compatibility

Native Praxis: `settings.voice_vosk_url=ws://GPU_NETBIRD_IP:2700/`, German default.
Use `/de`, `/fr`, `/ja` (or `/ws/LANGUAGE`) to choose explicitly. Native Praxis
rejects URL query strings. Protocol:

1. Send `{"config":{"sample_rate":16000}}` (actual integer rate, 8–48 kHz).
2. Send binary **mono PCM16**, not WAV headers/WebM/Opus. Read one `{"partial":""}`
   acknowledgment after **every** frame.
3. Send `{"eof":1}` and receive `{"text":"…","language":"de"}`.

Limits: 16 MiB / five minutes, 120-second connection work deadline. CPU recognition
is serialized across HTTP/WS. No audio is saved, no automatic language guessing,
no retries; silence can return empty text. Browser Origins are rejected.

Legacy HTTP: `POST /transcribe?language=fr`, `Content-Type: audio/wav`,
Content-Length, mono 16 kHz PCM16 WAV; up to 4 MiB / 120 seconds. Dashboard/browser
recording conversion and Discord playback remain client-side acceptance checks.

## Maintenance, stop/resume and Vast

```sh
supervisorctl -c /etc/supervisor/praxis.conf status
# Inspect startup/download/profile/cache logs:
tail -n 80 /workspace/logs/llama-chat.log
# Read-only command inspection:
python3 /opt/praxis/bin/run_llama.py chat --print-command
# After investigating a failed download/profile/OOM:
supervisorctl -c /etc/supervisor/praxis.conf start llama-chat
```

A failed LLM startup leaves the other services running, but readiness stays false.
`RUNNING` alone is not readiness: the process may still be downloading/profiling,
and even an HTTP-200 first reply missed the prefetch regression. Use the bounded
multi-request acceptance below before treating the LLM as usable.

For a separate speech/image machine, stopping only `llama-chat` leaves the Nomic
server and nginx alive. Stop **`llama-chat`, `llama-embedding`, `llama-proxy`** and set
`autostart=false` in those three `/etc/supervisor/praxis.conf` program sections if
they must stay off across Supervisor/container restarts. Apply with `reread`, then
`update llama-chat llama-embedding llama-proxy`, **not** a whole-stack restart.
Do not delete cached models. A replacement container uses its image/template
again. The current whole-stack Docker healthcheck is **not role-aware** and will
remain unhealthy with the LLM intentionally disabled; check ComfyUI/Vosk separately.

If SSH/tmux rejects `TERM=xterm-kitty`, install that terminal's actual definition
rather than changing its advertised capabilities. From the Kitty client machine:

```sh
infocmp -x xterm-kitty | ssh -p SSH_PORT root@GPU_IP 'tic -x -o /root/.terminfo /dev/stdin'
```

This changes only root's terminfo; no service or existing tmux session needs a restart.

A setup key exported in an SSH shell does **not** alter Supervisor's inherited
environment. Correct the Vast instance environment for future starts; recover only
the NetBird component if needed. Preserve `/workspace/netbird` and its identity,
and never put a setup key in the image, repository or command history. A manually
started NetBird daemon can be connected even while Supervisor reports it STOPPED.
No automatic crash retry repeatedly downloads/models/profiles or hammers the GPU.
For intentional idle-GPU maintenance, stop/start only the necessary Supervisor
program. The Unix control socket is private (0700); the startup lock prevents a
second whole stack in the same volume. Existing custom-node directories/links
are preserved and require explicit migration, not blind replacement.

For Vast **SSH mode**, which replaces Docker ENTRYPOINT, use the existing on-start
hook exactly once per container lifecycle:

```sh
nohup /opt/praxis/bin/start.sh >>/workspace/startup.log 2>&1 &
```

Normal Docker ENTRYPOINT starts it automatically. Never publish 11434, 11435,
11436, 8188 or 2700. A CUDA-12.8-capable NVIDIA host driver is required.

Retain/privately back up `/workspace`: GGUFs, verification receipts, traces,
profile sidecars, speech/Vosk caches, images, reference voices, personalized
workflows and NetBird identity. Stop/resume reuses those files; an empty volume
must download/profile again. Rental destruction may destroy the volume.

Do not run two peers with the same NetBird identity. For this separate companion,
prefer a separate volume and fresh private enrollment; copy only specifically
selected model/cache files if desired. New peer IPs require new Praxis endpoints
and restricted ACLs. Keep the Praxis database, Discord tokens and master key on
the Praxis machine. Git is not a backup of runtime data or voices.

## GPU acceptance before publishing

After startup is ready and other clients are idle:

```sh
python3 /opt/praxis/bin/acceptance_llama.py --run \
  --output /workspace/logs/llama-acceptance-$(date -u +%Y%m%dT%H%M%SZ).json
```

This explicitly authorizes six bounded generations (max 128 tokens each) and one
CPU embedding. It checks repeated short prompts, a simulated arithmetic tool round
trip, ~2k prompt tokens, non-greedy generation, a fresh request afterwards, finite
token probabilities and accepted MTP drafts. It uses the existing private-IP,
no-proxy/no-redirect bounded HTTP helpers; failures are not retried, and reports
never overwrite existing files. The report contains generic test outputs, not
personal prompts or audio. Also inspect the main-model cache activation receipt:
this API smoke is not a benchmark or a guarantee about all future model outputs.

The remaining release/whole-stack acceptance checklist still applies:

- [ ] CUDA image build, all four `pip check`s and `scripts/verify-image.sh` pass.
- [ ] Archive image digest and package lists; confirm binary revision matches `LLAMA_CPP.json`.
- [ ] CUDA `--list-devices` lists the intended NVIDIA GPU.
- [ ] First env-enabled startup downloads/verifies every pinned Flash shard.
- [ ] Both code/chat traces have decode rows; merged profile activates **56 slots**.
- [ ] `--spec-type draft-mtp` is in the live command; inspect MTP/draft statistics
      after real generation, not just a loaded model or `nvidia-smi`.
- [ ] A coherent `/v1/chat/completions` reply and a Praxis tool-call round trip work.
- [ ] Nomic `/v1/embeddings` returns the intended vector dimensions/values.
- [ ] Second startup reuses weights/profile; interruption/bad profile does not silently fall back.
- [ ] Authorized VPN peers reach all three ports; other peers/public access cannot.
- [ ] Qwen3/XTTS real mixed WAVs are listened to; Vosk HTTP/WS work with real audio.
- [ ] Browser persistence/replay and Discord playback work in Praxis.
- [ ] Peak RAM/VRAM, long prompts and concurrent speech/image loads are measured.

For a minimal local GPU text smoke test (inside the container):

```sh
curl --fail-with-body http://127.0.0.1:11434/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.8-flash-next","messages":[{"role":"user","content":"Reply with one short sentence."}],"max_tokens":128,"temperature":0}'
```

Short generation is **not** proof of a full 256k context or the video's throughput.
Inspect cache initialization, MTP use, coherent content, prompt/decode rates and
actual memory separately. Never reduce quality or switch providers silently.
