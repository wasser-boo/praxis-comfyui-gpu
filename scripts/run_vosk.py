#!/usr/bin/env python3
"""Provision pinned large Vosk models when authorized, then exec the pinned server.

VOSK_DOWNLOAD_MODELS=1 authorizes missing model downloads after license review
(mirrors LLAMA_DOWNLOAD_MODELS). Without it the server starts immediately and a
missing model surfaces as HTTP 503, exactly like the previous behavior.
vosk_server.py and provision_vosk.py stay byte-identical to the base stack.
"""
import fcntl
import os
import shutil
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path('/workspace/vosk')
MODELS = {'fr': 'vosk-model-fr-0.22', 'de': 'vosk-model-de-0.21', 'ja': 'vosk-model-ja-0.22'}
MAX_ARCHIVE = 3 * 1024 * 1024 * 1024
MAX_EXTRACTED = 6 * 1024 * 1024 * 1024


def ready(language: str) -> bool:
    return (ROOT / language / 'am' / 'final.mdl').is_file()


def download(language: str, model: str) -> None:
    archive = ROOT / f'{model}.zip'
    if archive.exists():
        raise RuntimeError(f'Inspect/remove the interrupted download before retrying: {archive}')
    lock_path = ROOT / '.provision.lock'
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if ready(language):
            return  # provisioned by a concurrent start while waiting for the lock
        url = f'https://alphacephei.com/vosk/models/{model}.zip'
        print(f'Downloading {url}', flush=True)
        with urllib.request.urlopen(url, timeout=120) as source, archive.open('wb') as dest:
            total = 0
            while chunk := source.read(1024 * 1024):
                total += len(chunk)
                if total > MAX_ARCHIVE:
                    raise RuntimeError('Unexpectedly large model archive')
                dest.write(chunk)
        stage = ROOT / f'.vosk-staging-{language}'
        shutil.rmtree(stage, ignore_errors=True)
        stage.mkdir(parents=True)
        with zipfile.ZipFile(archive) as z:
            if z.testzip() is not None:
                raise RuntimeError('Model archive failed CRC validation')
            if not any('am/final.mdl' in name for name in z.namelist()):
                raise RuntimeError('Unexpected model layout')
            if sum(item.file_size for item in z.infolist()) > MAX_EXTRACTED:
                raise RuntimeError('Unexpectedly large extracted model')
            for item in z.infolist():
                target = (stage / item.filename).resolve()
                if not str(target).startswith(str(stage.resolve()) + os.sep):
                    raise RuntimeError('Unsafe archive path')
                z.extract(item, stage)
        inner = stage / model
        (inner if inner.is_dir() else stage).rename(ROOT / language)
        shutil.rmtree(stage, ignore_errors=True)
        archive.unlink()
        print(f'Ready: {language} ({model})', flush=True)


def main() -> None:
    if os.environ.get('VOSK_DOWNLOAD_MODELS', '') == '1':
        for language, model in MODELS.items():
            if not ready(language):
                download(language, model)
    else:
        missing = [language for language in MODELS if not ready(language)]
        if missing:
            print(f'Vosk models missing for {missing}; set VOSK_DOWNLOAD_MODELS=1 to '
                  'provision them at startup', flush=True)
    server = Path(__file__).with_name('vosk_server.py')
    os.execv('/opt/vosk-venv/bin/python', ['/opt/vosk-venv/bin/python', str(server)])


if __name__ == '__main__':
    main()