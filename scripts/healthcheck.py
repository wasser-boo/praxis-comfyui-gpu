"""Readiness: ComfyUI speech nodes and Vosk are healthy; NetBird peer is ready."""
import json
import subprocess
import urllib.request

MAX_JSON = 2 * 1024 * 1024


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None


def main():
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())
    for url in ['http://127.0.0.1:8188/object_info/PraxisXTTS',
                'http://127.0.0.1:8188/object_info/PraxisQwen3TTS',
                'http://127.0.0.1:2700/health']:
        with opener.open(url, timeout=3) as response:
            raw = response.read(MAX_JSON + 1)
            if len(raw) > MAX_JSON:
                raise RuntimeError('Health response exceeds size limit')
            body = json.loads(raw)
        if 'object_info' in url:
            name = url.rsplit('/', 1)[1]
            if name not in body or 'de-ja' not in body[name]['input']['required']['language'][0]:
                raise RuntimeError('Mixed speech node failed to load')
    subprocess.run(['netbird', 'status', '--check', 'ready'], check=True, timeout=4,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


if __name__ == '__main__':
    main()