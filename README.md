# praxis-llamap-gpu

Separate GPU companion based on `praxis-gpu`. **Codacus's llama.cpp `perf` fork**
replaces Ollama; **FreeToken is not installed or run**.

- **Qwen3.8-Flash-Next 177B UD-IQ3_XXS**, all three pinned GGUF shards (~82 GB).
- Codacus's **56-slot expert cache + `--spec-type draft-mtp`**, with the separately
  pinned matching **shared Q8 MTP head (~2.8 GB)**. The main GGUFs contain no MTP layers.
- **One CPU worker per available physical core**, auto-detected with affinity/cpuset
  and CPU-quota limits (`LLAMA_THREADS=auto`); no SMT double-counting.
- Host pinning stays on. Experimental expert prefetch is fenced off after
  reproducible non-finite multi-request output; CUDA graphs and async CPU splits stay on.
- Runtime env opts into automatic download, SHA256 verification, two one-time
  code/chat routing traces (up to 512 generated tokens each), and merged-profile reuse.
- Existing **Vosk DE/FR/JA, ComfyUI, XTTS, Qwen3-TTS, workflows and NetBird** retained.
- **Speech provisioning is flag-gated** like the llama downloads: `PRAXIS_QWEN_TTS_DOWNLOAD=1`
  fetches the pinned Qwen3-TTS Base model, `VOSK_DOWNLOAD_MODELS=1` fetches the
  **large** Vosk de-0.21/fr-0.22/ja-0.22 models before the server starts.
- Qwen3-TTS synthesis runs on measured-fastest **sdpa with torch.compile enabled**
  (~2× per-token decode on RTX 3090); `flash_attention_2` stays an opt-in.
- Nomic embeddings remain available through a second, CPU-only llama-server.
- Same private VPN ports: **11434** (now OpenAI-compatible), **8188**, **2700**.

```sh
cp runtime.env.example runtime.env
chmod 600 runtime.env
# Review model licenses and set NB_SETUP_KEY privately in runtime.env.
bash scripts/build.sh
bash scripts/verify-image.sh
docker run -d --name praxis-llamap-gpu --gpus all \
  --env-file runtime.env -v praxis-llamap-gpu-data:/workspace praxis-llamap-gpu:dev
```

**No public port mappings.** The example enables `LLAMA_DOWNLOAD_MODELS=1` and
`LLAMA_GENERATE_PROFILE=1`. First startup downloads weights and performs GPU
profiling; subsequent starts reuse verified files. Speech model/reference
provisioning stays as in the original stack.

Read **[DEPLOY.md](DEPLOY.md)** for hardware sizing, reusing existing weights,
Praxis client settings, maintenance, and acceptance tests. Source pins are in
[`LLAMA_CPP.json`](LLAMA_CPP.json) and [`scripts/llama-models.json`](scripts/llama-models.json);
[`BASE_STACK.json`](BASE_STACK.json) records the unchanged source stack.

**Validation:** offline tests plus GPU acceptance of the corrected launcher on an
RTX A4000 16 GB: active 48-layer/56-slot cache, accepted MTP drafts, repeated chat,
tool round trip, a 2,044-token prompt with finite probabilities, and Nomic embeddings.
This is not a full-context/concurrency benchmark or a reproduction of the video's rate.
The published `vayayo/praxis-llamap-gpu:0.2` predates these fixes; the deployed container
was hotfixed. A new image build/verification/release is still required.

Run `python3 /opt/praxis/bin/acceptance_llama.py --run` against an idle, ready container
for bounded multi-request API regression checks. It does not restart services,
download weights, retry failed generations or submit user data.
