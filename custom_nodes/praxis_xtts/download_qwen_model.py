"""Explicit provisioning only. Never invoked by a TTS request or node import."""
import argparse
import json
from pathlib import Path

MODEL = 'Qwen/Qwen3-TTS-12Hz-1.7B-Base'
REVISION = 'fd4b254389122332181a7c3db7f27e918eec64e3'
DEFAULT_DIRECTORY = '/workspace/qwen3-tts/1.7B-Base'


def main():
    parser = argparse.ArgumentParser(description='Download the pinned Apache-2.0 Qwen3-TTS Base model; no service changes')
    parser.add_argument('--download', action='store_true', help='Explicitly authorize downloading the public model weights')
    parser.add_argument('--directory', default=DEFAULT_DIRECTORY)
    args = parser.parse_args()
    if not args.download:
        parser.error('Review the model card, then pass --download to provision the model')
    directory = Path(args.directory)
    if not directory.is_absolute() or directory.is_symlink():
        parser.error('Use an absolute, non-symlink model directory')
    if directory.exists() and any(directory.iterdir()):
        parser.error('Destination is not empty; inspect existing/downloaded files instead of overwriting or blindly retrying')
    from huggingface_hub import snapshot_download
    snapshot_download(repo_id=MODEL, revision=REVISION, local_dir=str(directory), token=False,
                      allow_patterns=['*.json', '*.txt', '*.safetensors', 'README.md', 'speech_tokenizer/*.json', 'speech_tokenizer/*.safetensors'])
    (directory / 'PRAXIS_MODEL_SOURCE.json').write_text(json.dumps({'repo_id': MODEL, 'revision': REVISION, 'license': 'apache-2.0'}, indent=2) + '\n')
    print('Pinned Qwen3-TTS Base model provisioned. No inference or service restart performed.')


if __name__ == '__main__':
    main()
