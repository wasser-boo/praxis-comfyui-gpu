"""Explicit, locked provisioning. Publish only a complete pinned model tree.

Interrupted transfers stay in a revision-specific staging directory and can be
resumed by huggingface_hub. Never delete/overwrite an existing incomplete public
model directory or another revision's files automatically.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path

MODEL = 'Qwen/Qwen3-TTS-12Hz-1.7B-Base'
REVISION = 'fd4b254389122332181a7c3db7f27e918eec64e3'
DEFAULT_DIRECTORY = '/workspace/qwen3-tts/1.7B-Base'
REQUIRED_FILES = ('config.json', 'model.safetensors', 'speech_tokenizer/config.json',
                  'speech_tokenizer/model.safetensors', 'tokenizer_config.json',
                  'vocab.json', 'merges.txt')
SOURCE = {'repo_id': MODEL, 'revision': REVISION, 'license': 'apache-2.0'}


def complete(directory):
    return (not any(p.is_symlink() for p in directory.rglob('*'))
            and all((directory / name).is_file() and (directory / name).stat().st_size > 0
                    for name in REQUIRED_FILES))


def provision(directory, download):
    directory = Path(directory)
    if not directory.is_absolute() or any(p.is_symlink() for p in (directory, *directory.parents)):
        raise ValueError('Use an absolute model directory without symlink components')
    directory.parent.mkdir(parents=True, exist_ok=True)
    # Separate from the worker's outer lock: manual and automatic downloaders
    # serialize here too, without deadlocking a worker that holds its own lock.
    lock_path = directory.parent / ('.' + directory.name + '.provision.lock')
    fd = os.open(lock_path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, 'a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if directory.exists() and any(directory.iterdir()):
            try:
                receipt = json.loads((directory / 'PRAXIS_MODEL_SOURCE.json').read_text())
            except (OSError, ValueError):
                receipt = None
            if receipt == SOURCE and complete(directory):
                return directory
            raise ValueError('Destination is not empty; inspect the existing model instead of overwriting it')
        stage = directory.parent / ('.' + directory.name + '.tmp-' + REVISION[:12])
        if stage.is_symlink() or (stage.exists() and not stage.is_dir()):
            raise ValueError('Unsafe staging directory')
        stage.mkdir(exist_ok=True)
        if any(p.is_symlink() for p in stage.rglob('*')):
            raise ValueError('Staging tree contains symlinks; inspect it before retrying')
        download(repo_id=MODEL, revision=REVISION, local_dir=str(stage), token=False,
                 allow_patterns=['*.json', '*.txt', '*.safetensors', 'README.md',
                                 'speech_tokenizer/*.json', 'speech_tokenizer/*.safetensors'])
        if not complete(stage):
            raise RuntimeError('Pinned model download incomplete; staging preserved for retry')
        with (stage / 'PRAXIS_MODEL_SOURCE.json').open('w') as receipt:
            receipt.write(json.dumps(SOURCE, indent=2) + '\n')
            receipt.flush()
            os.fsync(receipt.fileno())
        # Same parent/filesystem: atomic directory publication. A concurrently
        # created nonempty destination makes rename fail instead of deleting it.
        os.rename(stage, directory)
        return directory


def main():
    parser = argparse.ArgumentParser(description='Provision the pinned Apache-2.0 Qwen3-TTS Base model; no service changes')
    parser.add_argument('--download', action='store_true', help='Explicitly authorize downloading the public model weights')
    parser.add_argument('--directory', default=DEFAULT_DIRECTORY)
    args = parser.parse_args()
    if not args.download:
        parser.error('Review the model card, then pass --download to provision the model')
    from huggingface_hub import snapshot_download
    try:
        provision(Path(args.directory), snapshot_download)
    except ValueError as error:
        parser.error(str(error))
    print('Pinned Qwen3-TTS Base model provisioned. No inference or service restart performed.')


if __name__ == '__main__':
    main()
