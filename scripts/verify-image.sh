#!/usr/bin/env bash
# CPU-only image checks. No credentials, model downloads, host mounts or published ports.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs
if docker info >/dev/null 2>&1; then engine=(docker); else engine=(sudo docker); sudo -v; fi
image="${IMAGE_TAG:-praxis-llamap-gpu:dev}"
"${engine[@]}" image inspect "$image" --format 'Image={{.Id}} Architecture={{.Architecture}} Size={{.Size}} ExposedPorts={{json (index .Config "ExposedPorts")}}' | tee logs/verify-image.log
"${engine[@]}" run --rm -i --network none --entrypoint /bin/bash "$image" <<'CHECKS' 2>&1 | tee -a logs/verify-image.log
set -euo pipefail
for env in comfy tts qwen-tts vosk; do /opt/"$env"-venv/bin/pip check; done
/opt/tts-venv/bin/python -c 'from TTS.api import TTS; print("XTTS import OK")'
/opt/qwen-tts-venv/bin/python -c 'from qwen_tts import Qwen3TTSModel; print("Qwen3 import OK")'
/opt/vosk-venv/bin/python -c 'import vosk, aiohttp; print("Vosk HTTP/WebSocket imports OK")'
! command -v ollama
! command -v freetoken
/opt/llama.cpp/bin/llama-server --help >/tmp/llama-help
/opt/llama.cpp/bin/llama-moe-trace --help >/dev/null
for flag in --moe-cache-profile --moe-cache-slots draft-mtp; do grep -q -- "$flag" /tmp/llama-help; done
nginx -t -c /opt/praxis/bin/llama-proxy.conf
netbird version
python3 /opt/praxis/bin/run_llama.py chat --print-command
python3 - <<'PY'
import configparser, json
from pathlib import Path
assert Path('/opt/llama.cpp/REVISION').read_text().strip() == json.loads(Path('/opt/praxis/LLAMA_CPP.json').read_text())['commit']
c=configparser.ConfigParser(interpolation=None)
c.read('/etc/supervisor/praxis.conf')
assert {s for s in c if s.startswith('program:')} == {'program:'+p for p in ['netbird','comfyui','llama-chat','llama-proxy','llama-embedding','vosk']}
for kind in ['chat','embedding']:
    assert c['program:llama-'+kind]['autorestart']=='false'
    assert c['program:llama-'+kind]['startretries']=='0'
print('Pinned fork and Supervisor definitions OK')
PY
mkdir -p /workspace/{custom_nodes,input,output,temp,llama.cpp,user,logs} /tmp/empty-models
python3 /opt/praxis/bin/install_workflows.py
ln -s /opt/ComfyUI/custom_nodes/praxis_xtts /workspace/custom_nodes/praxis_xtts
pids=()
trap 'kill "${pids[@]}" 2>/dev/null || true' EXIT
/opt/comfy-venv/bin/python /opt/ComfyUI/main.py --cpu --listen 127.0.0.1 --port 8188 --base-directory /workspace --user-directory /workspace/user --disable-auto-launch >/tmp/comfy.log 2>&1 &
pids+=("$!")
# Router mode exercises the real binaries/HTTP stack without model weights or GPU.
for port in 11435 11436; do
  /opt/llama.cpp/bin/llama-server --host 127.0.0.1 --port "$port" --models-dir /tmp/empty-models --no-models-autoload --no-webui >"/tmp/llama-$port.log" 2>&1 &
  pids+=("$!")
done
nginx -c /opt/praxis/bin/llama-proxy.conf -g 'daemon off;' >/tmp/proxy.log 2>&1 &
pids+=("$!")
/opt/vosk-venv/bin/python /opt/praxis/bin/vosk_server.py >/tmp/vosk.log 2>&1 &
pids+=("$!")
python3 - <<'PY'
import json,time,urllib.request,urllib.error
from pathlib import Path
opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
for name,url in [('ComfyUI','http://127.0.0.1:8188/object_info'),('llama.cpp','http://127.0.0.1:11434/v1/models'),('embedding','http://127.0.0.1:11434/health/embedding'),('Vosk','http://127.0.0.1:2700/health')]:
    deadline=time.monotonic()+300
    while True:
        try:
            try:
                response=opener.open(urllib.request.Request(url,headers={'Host':'100.64.0.9:11434'}),timeout=3)
            except urllib.error.HTTPError as e:
                if name!='Vosk' or e.code!=503: raise
                response=e
            with response: body=json.load(response)
            if name=='ComfyUI':
                for node in ['PraxisXTTS','PraxisQwen3TTS']:
                    assert node in body and 'de-ja' in body[node]['input']['required']['language'][0]
            if name=='llama.cpp': assert body['data']==[], body
            if name=='embedding': assert body['status']=='ok', body
            if name=='Vosk': assert body['ready'] is False and body['loaded_languages']==[], body
            print(name, 'CPU API check OK',flush=True)
            break
        except Exception:
            if time.monotonic()>=deadline:
                for f in ('comfy','llama-11435','llama-11436','proxy','vosk'):
                    print(Path('/tmp/'+f+'.log').read_text()[-12000:])
                raise
            time.sleep(2)
for path,headers,expected in [('/v1/models',{'Origin':'https://untrusted.example'},403),('/api/tags',{},404),('/props',{},404)]:
    try: opener.open(urllib.request.Request('http://127.0.0.1:11434'+path,headers=headers),timeout=3)
    except urllib.error.HTTPError as error: assert error.code==expected
    else: raise AssertionError('Expected private proxy rejection')
print('PASS: CPU image checks only. GPU cache/MTP, real models, speech and VPN still require acceptance tests.')
PY
CHECKS
