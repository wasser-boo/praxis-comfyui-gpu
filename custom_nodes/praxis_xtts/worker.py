"""Isolated XTTS/Qwen worker: one-shot by default, optional private Qwen IPC.

Warm mode retains one Qwen model and one reference prompt only until the parent
idle timeout. No model downloads, HTTP listener, quality reduction or GPU fallback.
"""
import hashlib
import json
import os
from pathlib import Path
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
            self.model = Qwen3TTSModel.from_pretrained(str(directory), device_map='cuda:0',
                dtype=torch.bfloat16, attn_implementation='sdpa', local_files_only=True, trust_remote_code=False)
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
