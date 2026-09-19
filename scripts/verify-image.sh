#!/usr/bin/env bash
# CPU-only image checks. No credentials, model downloads, host mounts or published ports.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs
if docker info >/dev/null 2>&1; then engine=(docker); else engine=(sudo docker); sudo -v; fi
image="${IMAGE_TAG:-praxis-comfyui-gpu:dev}"
"${engine[@]}" image inspect "$image" --format 'Image={{.Id}} Architecture={{.Architecture}} Size={{.Size}} ExposedPorts={{json (index .Config "ExposedPorts")}}' | tee logs/verify-image.log
"${engine[@]}" run --rm -i --network none --entrypoint /bin/bash "$image" <<'CHECKS' 2>&1 | tee -a logs/verify-image.log
set -euo pipefail
for env in comfy tts qwen-tts vosk; do /opt/"$env"-venv/bin/pip check; done
/opt/tts-venv/bin/python -c 'from TTS.api import TTS; print("XTTS import OK")'
/opt/qwen-tts-venv/bin/python -c 'from qwen_tts import Qwen3TTSModel; print("Qwen3 import OK")'
/opt/vosk-venv/bin/python -c 'import vosk, aiohttp; print("Vosk HTTP/WebSocket imports OK")'
/opt/qwen-tts-venv/bin/python -c 'import flash_attn; print("flash_attn", flash_attn.__version__, "wheel OK")'
python3 -m py_compile /opt/praxis/bin/run_vosk.py && echo 'run_vosk wrapper OK'
! command -v ollama
! command -v freetoken
! command -v llama-server
netbird version
python3 - <<'PY'
import configparser
from pathlib import Path
c=configparser.ConfigParser(interpolation=None)
c.read('/etc/supervisor/praxis.conf')
assert {s for s in c if s.startswith('program:')} == {'program:'+p for p in ['netbird','comfyui','vosk']}
assert '--base-directory /workspace' in c['program:comfyui']['command']
for flag in ('--input-directory','--output-directory','--temp-directory','--user-directory'):
    assert flag not in c['program:comfyui']['command']
assert 'run_vosk.py' in c['program:vosk']['command'] and 'vosk_server.py' not in c['program:vosk']['command']
print('Supervisor definitions OK')
PY
mkdir -p /workspace/{custom_nodes,input,output,temp,llama.cpp,user,logs}
python3 /opt/praxis/bin/install_workflows.py
ln -s /opt/ComfyUI/custom_nodes/praxis_xtts /workspace/custom_nodes/praxis_xtts
pids=()
trap 'kill "${pids[@]}" 2>/dev/null || true' EXIT
/opt/comfy-venv/bin/python /opt/ComfyUI/main.py --cpu --listen 127.0.0.1 --port 8188 --base-directory /workspace --user-directory /workspace/user --disable-auto-launch >/tmp/comfy.log 2>&1 &
pids+=("$!")
/opt/vosk-venv/bin/python /opt/praxis/bin/vosk_server.py >/tmp/vosk.log 2>&1 &
pids+=("$!")
python3 - <<'PY'
import json,time,urllib.request,urllib.error
from pathlib import Path
opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
for name,url in [('ComfyUI','http://127.0.0.1:8188/object_info'),('Vosk','http://127.0.0.1:2700/health')]:
    deadline=time.monotonic()+300
    while True:
        try:
            try:
                response=opener.open(urllib.request.Request(url),timeout=3)
            except urllib.error.HTTPError as e:
                if name!='Vosk' or e.code!=503: raise
                response=e
            with response: body=json.load(response)
            if name=='ComfyUI':
                for node in ['PraxisXTTS','PraxisQwen3TTS']:
                    assert node in body and 'de-ja' in body[node]['input']['required']['language'][0]
            if name=='Vosk': assert body['ready'] is False and body['loaded_languages']==[], body
            print(name, 'CPU API check OK',flush=True)
            break
        except Exception:
            if time.monotonic()>=deadline:
                for f in ('comfy','vosk'):
                    print(Path('/tmp/'+f+'.log').read_text()[-12000:])
                raise
            time.sleep(2)
print('PASS: CPU image checks only. Real speech models, GPU synthesis and VPN still require acceptance tests.')
PY
CHECKS