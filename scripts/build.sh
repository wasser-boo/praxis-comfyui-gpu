#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs
image="${IMAGE_TAG:-praxis-comfyui-gpu:dev}"
if docker info >/dev/null 2>&1; then
  engine=(docker)
else
  echo 'Docker requires sudo. Enter your password in this terminal, never in chat.'
  engine=(sudo docker)
  sudo -v
fi
"${engine[@]}" build --platform linux/amd64 --progress=plain \
  -t "$image" . 2>&1 | tee logs/build.log
"${engine[@]}" image inspect "$image" --format '{{.Id}}'
echo 'Build complete. GPU, NetBird peer access, TTS/STT smoke tests still required.'