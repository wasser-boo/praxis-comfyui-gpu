"""Readiness includes actual Codacus CUDA-cache activation, not model quality."""
import json
import os
import subprocess
import urllib.request

from llama_config import settings
from run_llama import runtime_ready

MAX_JSON = 2 * 1024 * 1024


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None


def main():
    config = settings(os.environ)
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())
    paths = ['http://127.0.0.1:11434/health',
             'http://127.0.0.1:8188/object_info/PraxisXTTS',
             'http://127.0.0.1:8188/object_info/PraxisQwen3TTS',
             'http://127.0.0.1:2700/health']
    if config['embedding']:
        paths.append('http://127.0.0.1:11434/health/embedding')
    for url in paths:
        with opener.open(url, timeout=3) as response:
            raw = response.read(MAX_JSON + 1)
            if len(raw) > MAX_JSON:
                raise RuntimeError('Health response exceeds size limit')
            body = json.loads(raw)
        if ':11434/' in url and body.get('status') != 'ok':
            raise RuntimeError('llama.cpp is not ready')
        if 'object_info' in url:
            name = url.rsplit('/', 1)[1]
            if name not in body or 'de-ja' not in body[name]['input']['required']['language'][0]:
                raise RuntimeError('Mixed speech node failed to load')
    for kind in ['chat'] + (['embedding'] if config['embedding'] else []):
        if not runtime_ready(kind, os.environ):
            raise RuntimeError(f'llama.cpp {kind} readiness receipt is missing/stale; inspect service log')
    subprocess.run(['netbird', 'status', '--check', 'ready'], check=True, timeout=4,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


if __name__ == '__main__':
    main()
