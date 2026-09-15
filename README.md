# praxis-llamap-gpu

Separate GPU companion based on `praxis-gpu`. **Codacus's llama.cpp `perf` fork**
replaces Ollama; **FreeToken is not installed or run**.

- **Qwen3.8-Flash-Next 177B UD-IQ3_XXS**, all three pinned GGUF shards (~82 GB).
- Codacus's **56-slot expert cache + `--spec-type draft-mtp`**.
- **One CPU worker per available physical core**, auto-detected with affinity/cpuset
  and CPU-quota limits (`LLAMA_THREADS=auto`); no SMT double-counting.
- Runtime env opts into automatic download, SHA256 verification, two one-time
  code/chat routing traces (up to 512 generated tokens each), and merged-profile reuse.
- Existing **Vosk DE/FR/JA, ComfyUI, XTTS, Qwen3-TTS, workflows and NetBird** retained.
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

**Validation:** offline tests and a CPU-only build/API check of the pinned fork.
The CUDA Docker image and Flash/MTP inference still need testing on an NVIDIA host.
No claim that this container reproduces the video's 24.4 tokens/s.
