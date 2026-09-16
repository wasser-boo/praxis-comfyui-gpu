"""Isolated XTTS/Qwen worker: one-shot by default, optional private Qwen IPC.

Warm mode retains one Qwen model and one reference prompt only until the parent
idle timeout. No HTTP listener, quality reduction or GPU fallback. When
PRAXIS_QWEN_TTS_DOWNLOAD=1, a missing pinned Qwen3-TTS model is provisioned
once before the first load; no other implicit download path exists.
"""
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

if __package__:
    from .audio import write_speech
    from .segments import speech_segments
else:
    from audio import write_speech
    from segments import speech_segments

QWEN_MODEL_DIR = '/workspace/qwen3-tts/1.7B-Base'
QWEN_LANGUAGES = {'auto': 'Auto', 'en': 'English', 'zh': 'Chinese', 'ja': 'Japanese',
                  'ko': 'Korean', 'de': 'German', 'fr': 'French', 'ru': 'Russian',
                  'pt': 'Portuguese', 'es': 'Spanish', 'it': 'Italian'}


def xtts(request):
    if os.environ.get('COQUI_TOS_AGREED') != '1':
        raise RuntimeError('Explicit XTTS license acceptance required')
    # Mixed work is validated before a model load. Fixed-language XTTS retains
    # its previous tts_to_file path, including its internal sentence handling.
    mixed = request['language'] == 'de-ja'
    segments = speech_segments(request['text'], 'de-ja') if mixed else None
    import torch
    from TTS.api import TTS
    if not torch.cuda.is_available():
        raise RuntimeError('XTTS requires CUDA')
    model = TTS('tts_models/multilingual/multi-dataset/xtts_v2').to('cuda')
    if not mixed:
        model.tts_to_file(text=request['text'], language=request['language'],
                          speaker_wav=request['reference'], file_path=request['output'])
        return
    rate = int(model.synthesizer.output_sample_rate)
    def generate(language, text):
        return model.tts(text=text, language=language, speaker_wav=request['reference']), rate
    write_speech(request['output'], segments, generate)


def ensure_qwen_model(directory: Path) -> None:
    """Provision the pinned Qwen3-TTS model once, authorized by PRAXIS_QWEN_TTS_DOWNLOAD.

    The node pins HF_HUB_OFFLINE for inference. The download subprocess strips those
    variables and reuses the same pinned repository/revision as the manual command.
    A lock serializes concurrent workers; completion is signaled by config.json.
    """
    if (directory / 'config.json').is_file():
        return
    if os.environ.get('PRAXIS_QWEN_TTS_DOWNLOAD', '') != '1':
        raise RuntimeError('Qwen3-TTS model missing; set PRAXIS_QWEN_TTS_DOWNLOAD=1 '
                           'or run download_qwen_model.py --download')
    from download_qwen_model import MODEL, REVISION
    print(f'Qwen3-TTS model missing; downloading {MODEL} @ {REVISION}', flush=True)
    lock_path = directory.parent / '.qwen-download.lock'
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if (directory / 'config.json').is_file():
            return  # provisioned by a concurrent worker while waiting for the lock
        environment = os.environ.copy()
        environment.pop('HF_HUB_OFFLINE', None)
        environment.pop('TRANSFORMERS_OFFLINE', None)
        script = Path(__file__).with_name('download_qwen_model.py')
        subprocess.run([sys.executable, str(script), '--download', '--directory', str(directory)],
                       check=True, timeout=3600, env=environment)
    if not (directory / 'config.json').is_file():
        raise RuntimeError('Qwen3-TTS provisioning did not produce the pinned model')


class QwenEngine:
    def __init__(self):
        self.model = self.prompt = self.reference_key = None

    def speak(self, request):
        started = time.perf_counter()
        segments = speech_segments(request['text'], request['language'])
        directory = Path(os.environ.get('PRAXIS_QWEN_TTS_MODEL_DIR', QWEN_MODEL_DIR))
        required = ['config.json', 'model.safetensors', 'speech_tokenizer/config.json',
                    'speech_tokenizer/model.safetensors', 'tokenizer_config.json', 'vocab.json', 'merges.txt']
        if not directory.is_absolute() or not all((directory / name).is_file() for name in required):
            raise RuntimeError('Install the pinned Qwen3-TTS Base model locally before synthesis')
        os.environ.update({'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1', 'HF_HUB_DISABLE_TELEMETRY': '1'})
        cold = self.model is None
        reference = Path(request['reference'])
        with reference.open('rb') as stream:
            raw = stream.read(20 * 1024 * 1024 + 1)
        if not 0 < len(raw) <= 20 * 1024 * 1024:
            raise ValueError('Reference exceeds the audio byte limit')
        reference_text = request.get('reference_text', '').strip()
        key = (str(reference.resolve()), hashlib.sha256(raw).digest(), reference_text)
        cached = self.reference_key == key
        if not cached:
            import soundfile as sf
            info = sf.info(str(reference))
            if info.channels not in (1, 2) or info.samplerate <= 0 or not 0 < info.frames / info.samplerate <= 120:
                raise ValueError('Use a non-empty mono/stereo reference of at most 120 seconds')
        validated = time.perf_counter()
        if cold:
            import torch
            from qwen_tts import Qwen3TTSModel
            if not torch.cuda.is_available():
                raise RuntimeError('Qwen3-TTS requires CUDA')
            ensure_qwen_model(directory)
            attn = os.environ.get('PRAXIS_QWEN_TTS_ATTN', '')
            if not attn:
                # Measured on RTX 3090: sdpa beats flash_attention_2 for this workload
                # (short decode steps dominate, not attention kernels).
                attn = 'sdpa'
            if attn not in ('sdpa', 'flash_attention_2'):
                raise ValueError('PRAXIS_QWEN_TTS_ATTN must be sdpa or flash_attention_2')
            if attn == 'flash_attention_2' and importlib.util.find_spec('flash_attn') is None:
                raise RuntimeError('PRAXIS_QWEN_TTS_ATTN=flash_attention_2 requires the flash-attn wheel')
            self.model = Qwen3TTSModel.from_pretrained(str(directory), device_map='cuda:0',
                dtype=torch.bfloat16, attn_implementation=attn, local_files_only=True, trust_remote_code=False)
            if os.environ.get('PRAXIS_QWEN_TTS_COMPILE', '1') != '0':
                try:
                    # Cuts ~123 ms/step to ~65 ms on RTX 3090; one-time compile cost
                    # per worker process, amortized by the warm idle window.
                    talker = self.model.model.talker
                    talker.model = torch.compile(talker.model, dynamic=True)
                    talker.code_predictor.model = torch.compile(talker.code_predictor.model, dynamic=True)
                    print('qwen tts: torch.compile enabled for talker + code predictor', flush=True)
                except Exception as error:
                    print(f'qwen tts: torch.compile skipped ({type(error).__name__})', flush=True)
        loaded = time.perf_counter()
        if not cached:
            self.prompt = self.model.create_voice_clone_prompt(ref_audio=str(reference),
                ref_text=reference_text or None, x_vector_only_mode=not bool(reference_text))
            with reference.open('rb') as stream:
                after = stream.read(20 * 1024 * 1024 + 1)
            if hashlib.sha256(after).digest() != key[1]:
                self.prompt = self.reference_key = None
                raise ValueError('Reference changed during preparation; request not retried')
            self.reference_key = key  # one entry; vectors/transcript stay in private worker memory
        prepared = time.perf_counter()
        def generate(language, text):
            wavs, rate = self.model.generate_voice_clone(text=text, language=QWEN_LANGUAGES[language],
                voice_clone_prompt=self.prompt, max_new_tokens=2048)
            if len(wavs) != 1:
                raise ValueError('Expected one waveform per segment')
            return wavs[0], int(rate)
        write_speech(request['output'], segments, generate)
        finished = time.perf_counter()
        return {'cold_model': cold, 'reference_cached': cached,
                'model_load_seconds': round(loaded - validated, 3),
                'reference_seconds': round(validated - started + prepared - loaded, 3),
                'synthesis_seconds': round(finished - prepared, 3),
                'total_seconds': round(finished - started, 3)}


def qwen3(request):
    return QwenEngine().speak(request)


def watch_parent(channel):
    # The parent may disappear during CUDA work, before the next framed read.
    # Peek only: request bytes remain exclusively owned by the serving loop.
    import signal
    import socket
    try:
        while channel.recv(1, socket.MSG_PEEK):
            time.sleep(0.05)
    except OSError:
        return  # normal local channel teardown
    os.kill(os.getpid(), signal.SIGTERM)  # this child only, never a process group


def serve(fd):
    import socket
    import threading
    if __package__:
        from .protocol import receive, encode
    else:
        from protocol import receive, encode
    engine = QwenEngine()
    with socket.socket(fileno=fd) as channel:
        threading.Thread(target=watch_parent, args=(channel,), daemon=True).start()
        while True:
            try:
                request = receive(channel)
            except EOFError:
                return  # parent exited; release all model allocations
            try:
                if request.get('backend') != 'qwen3':
                    raise ValueError('Warm mode supports Qwen only')
                result = engine.speak(request)
            except Exception:
                channel.sendall(encode({'ok': False}))
                return  # do not reuse possibly poisoned CUDA state or echo private errors
            channel.sendall(encode({'ok': True, 'timings': result}))
            del request, result


def main():
    raw = sys.stdin.read(65537)
    if len(raw) > 65536:
        raise ValueError('Worker request exceeds limit')
    request = json.loads(raw)
    if request.get('backend') == 'xtts':
        xtts(request)
    elif request.get('backend') == 'qwen3':
        qwen3(request)
    else:
        raise ValueError('Unknown speech backend')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--server-fd', type=int)
    args = parser.parse_args()
    if args.server_fd is None:
        main()
    else:
        serve(args.server_fd)
