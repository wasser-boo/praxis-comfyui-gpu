#!/usr/bin/env python3
"""Submit one API workflow over a private literal-IP origin; download bounded outputs.

No retries, redirects, proxy environment, shared queue cancellation or overwrites.
Run from the Praxis machine to check NetBird access. Inspect failures before retrying.
"""
import argparse
import ipaddress
import json
import os
from pathlib import Path
import tempfile
import time
import urllib.parse
import urllib.request

MAX_JSON = 2 * 1024 * 1024
MAX_FILE = 32 * 1024 * 1024


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None


OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())


def validate_base(base):
    parsed = urllib.parse.urlsplit(base)
    try:
        address = ipaddress.ip_address(parsed.hostname or '')
        port = parsed.port
    except ValueError as error:
        raise ValueError('Use a private literal-IP HTTP(S) origin') from error
    networks = ['10.0.0.0/8', '172.16.0.0/12', '192.168.0.0/16', '100.64.0.0/10', 'fc00::/7']
    private = address.is_loopback or any(address in ipaddress.ip_network(n) for n in networks)
    if (parsed.scheme not in ['http', 'https'] or not private or address.is_unspecified
            or parsed.username is not None or parsed.password is not None
            or parsed.query or parsed.fragment or parsed.path not in ['', '/']):
        raise ValueError('Use a private literal-IP origin without credentials, path, query or fragment')
    return base.rstrip('/')


def download(url, maximum, deadline):
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError('Diagnostic deadline exceeded')
    with OPENER.open(url, timeout=min(30, remaining)) as response:
        chunks, count = [], 0
        while True:
            if time.monotonic() >= deadline:
                raise TimeoutError('Diagnostic deadline exceeded')
            chunk = response.read1(min(65536, maximum + 1 - count))
            if not chunk:
                return b''.join(chunks)
            count += len(chunk)
            if count > maximum:
                raise ValueError('Response exceeds diagnostic size limit')
            chunks.append(chunk)


def request(url, payload=None, deadline=None):
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, headers={'Content-Type': 'application/json'})
    return json.loads(download(req, MAX_JSON, deadline or time.monotonic() + 30))


def descriptor(item):
    if not isinstance(item, dict) or any(not isinstance(item.get(k), str) for k in ['filename', 'subfolder', 'type']):
        raise ValueError('Invalid output descriptor')
    filename, folder = item['filename'], item['subfolder']
    if (not filename or filename.startswith('.') or '/' in filename or '\\' in filename
            or any(ord(c) < 32 for c in filename + folder) or '\\' in folder
            or (folder and any(p in ['', '.', '..'] for p in folder.split('/')))
            or item['type'] not in ['output', 'temp']):
        raise ValueError('Unsafe output descriptor')
    return {k: item[k] for k in ['filename', 'subfolder', 'type']}


def run(base, workflow, directory, timeout):
    base = validate_base(base)
    if not 1 <= timeout <= 3600:
        raise ValueError('Timeout must be 1–3600 seconds')
    path = Path(workflow)
    if path.stat().st_size > 1024 * 1024:
        raise ValueError('Workflow exceeds 1 MiB')
    graph = json.loads(path.read_text())
    if not isinstance(graph, dict) or not graph or any(not isinstance(n, dict) or not isinstance(n.get('class_type'), str) or not isinstance(n.get('inputs'), dict) for n in graph.values()):
        raise ValueError('Expected API nodes with class_type/inputs, not GUI workflow JSON')
    deadline = time.monotonic() + timeout
    submitted = request(base + '/prompt', {'prompt': graph}, deadline)
    if submitted.get('node_errors'):
        raise RuntimeError('Workflow rejected; inspect server validation without resubmitting')
    prompt_id = submitted.get('prompt_id')
    if not isinstance(prompt_id, str) or not prompt_id:
        raise RuntimeError('Missing prompt acknowledgement; inspect server before retrying')
    while time.monotonic() < deadline:
        entry = request(base + '/history/' + urllib.parse.quote(prompt_id, safe=''), deadline=deadline).get(prompt_id)
        if entry:
            status = entry.get('status', {})
            if status.get('status_str') == 'error':
                raise RuntimeError('Workflow failed; inspect private server logs')
            if status.get('completed'):
                directory = Path(directory)
                directory.mkdir(mode=0o700, parents=True, exist_ok=True)
                files = []
                for output in entry.get('outputs', {}).values():
                    for kind in ('images', 'audio'):
                        for item in output.get(kind, []):
                            if len(files) >= 32:
                                raise ValueError('Too many output files')
                            query = descriptor(item)
                            data = download(base + '/view?' + urllib.parse.urlencode(query), MAX_FILE, deadline)
                            if kind == 'audio' and not (data[:4] == b'RIFF' and data[8:12] == b'WAVE'):
                                raise RuntimeError('Expected WAV audio')
                            if kind == 'images' and not data.startswith(b'\x89PNG\r\n\x1a\n'):
                                raise RuntimeError('Expected PNG image')
                            target = directory / query['filename']
                            with tempfile.NamedTemporaryFile(dir=directory, suffix='.part') as stage:
                                stage.write(data); stage.flush(); os.fsync(stage.fileno())
                                os.link(stage.name, target)  # exclusive publication; no overwrite
                            files.append(str(target))
                if not files:
                    raise RuntimeError('Completed workflow has no image/audio files')
                return files
        time.sleep(min(2, max(0, deadline - time.monotonic())))
    raise TimeoutError('Prompt deadline exceeded; inspect server before retrying. Shared queue untouched.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', required=True)
    parser.add_argument('--workflow', required=True)
    parser.add_argument('--output', default='smoke-output')
    parser.add_argument('--timeout', type=int, default=900)
    args = parser.parse_args()
    print('\n'.join(run(args.url, args.workflow, Path(args.output), args.timeout)))
