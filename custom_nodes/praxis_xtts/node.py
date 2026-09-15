"""Native ComfyUI audio outputs, one isolated worker and one WAV per reply.

No TTS model imports in ComfyUI's Python environment. Neither node mutates the
shared queue, accepts terms, or uploads reference audio. XTTS retains its
existing ComfyUI-cache release; Qwen only owns its worker's GPU allocations.
"""
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import uuid

import folder_paths
from .segments import speech_segments

XTTS_LANGUAGES = ['en', 'es', 'fr', 'de', 'it', 'pt', 'pl', 'tr', 'ru', 'nl', 'cs', 'ar', 'zh-cn', 'hu', 'ko', 'ja', 'hi', 'de-ja']
QWEN_LANGUAGES = ['auto', 'en', 'zh', 'ja', 'ko', 'de', 'fr', 'ru', 'pt', 'es', 'it', 'de-ja']


def warm_idle_seconds():
    value = os.environ.get('PRAXIS_QWEN_TTS_IDLE_SECONDS', '0')
    if not value.isascii() or not value.isdigit() or not 0 <= int(value) <= 3600:
        raise ValueError('PRAXIS_QWEN_TTS_IDLE_SECONDS must be an integer from 0 to 3600')
    return int(value)


def public_timings(values):
    result = {}
    for key in ('cold_model', 'reference_cached'):
        if type(values.get(key)) is not bool:
            raise RuntimeError('Invalid worker timing flags')
        result[key] = values[key]
    for key in ('model_load_seconds', 'reference_seconds', 'synthesis_seconds', 'total_seconds'):
        value = values.get(key)
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise RuntimeError('Invalid worker timing duration')
        result[key] = value
    return result  # Explicit allowlist: no arbitrary worker fields enter history.


def cleanup_output(target):
    target.unlink(missing_ok=True)
    # Only this UUID-named reply's staging files; never sweep a shared directory.
    for partial in target.parent.glob('.praxis-audio-' + target.name + '-*.part'):
        partial.unlink(missing_ok=True)


def reference_path(filename):
    if (not isinstance(filename, str) or not filename or len(filename) > 1024
            or Path(filename).is_absolute() or '\\' in filename or ':' in filename
            or any(p in ('', '.', '..') or p.startswith('.') for p in filename.split('/'))):
        raise ValueError('Reference must be a relative server-input filename')
    root = Path(folder_paths.get_input_directory()).resolve()
    candidate = (root / filename).resolve()
    if not candidate.is_relative_to(root) or not candidate.is_file():
        raise ValueError('Reference must exist inside the server input directory')
    if candidate.suffix.lower() not in {'.wav', '.mp3', '.flac', '.ogg'}:
        raise ValueError('Unsupported reference audio format')
    if not 0 < candidate.stat().st_size <= 20 * 1024 * 1024:
        raise ValueError('Reference audio must contain 1 byte to 20 MiB')
    return candidate


class _SpeechNode:
    RETURN_TYPES = ('STRING',)
    RETURN_NAMES = ('filename',)
    FUNCTION = 'generate'
    CATEGORY = 'Praxis/audio'
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        inputs = {'required': {
            'text': ('STRING', {'multiline': True}),
            'language': (cls.LANGUAGES,),
            'reference_audio': ('STRING', {'default': 'reference.wav'}),
        }}
        if cls.BACKEND == 'qwen3':
            inputs['optional'] = {'reference_text': ('STRING', {'multiline': True, 'default': ''})}
        return inputs

    def generate(self, text, language, reference_audio, reference_text=''):
        if self.BACKEND == 'xtts' and os.environ.get('COQUI_TOS_AGREED') != '1':
            raise ValueError('Review XTTS model license and explicitly set COQUI_TOS_AGREED=1')
        if language not in self.LANGUAGES or not isinstance(text, str) or not text.strip() or len(text) > 5000:
            raise ValueError('Provide 1–5000 characters and a supported language')
        if not isinstance(reference_text, str) or len(reference_text) > 5000:
            raise ValueError('Reference transcript exceeds 5000 characters or is not text')
        if language == 'de-ja' or self.BACKEND == 'qwen3':
            speech_segments(text, language)  # Validate bounded work before spawning.
        ref = reference_path(reference_audio)
        if self.BACKEND == 'xtts':
            # Preserve the previous XTTS node's image-cache release. This does
            # not touch Ollama, interrupt jobs, or mutate the shared queue.
            import comfy.model_management as mm
            mm.unload_all_models()
            mm.soft_empty_cache()
        filename = self.BACKEND + '_' + uuid.uuid4().hex + '.wav'
        target = Path(folder_paths.get_output_directory()).resolve() / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() or target.is_symlink():
            raise RuntimeError('Speech output collision')
        request = {'backend': self.BACKEND, 'text': text, 'language': language,
                   'reference': str(ref), 'reference_text': reference_text, 'output': str(target)}
        python = os.environ.get(self.PYTHON_ENV, self.DEFAULT_PYTHON)
        environment = os.environ.copy()
        if self.BACKEND == 'qwen3':
            environment.update({'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1', 'HF_HUB_DISABLE_TELEMETRY': '1'})
        timings = None
        idle = warm_idle_seconds() if self.BACKEND == 'qwen3' else 0
        try:
            if idle:
                from .warm_worker import run
                timings = public_timings(run(request, python, environment, idle))
            else:
                subprocess.run([python, str(Path(__file__).with_name('worker.py'))],
                               input=json.dumps(request, ensure_ascii=False), text=True, check=True,
                               timeout=600, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                               env=environment)
            if not target.is_file() or not 44 < target.stat().st_size <= 32 * 1024 * 1024:
                raise RuntimeError('Worker did not produce a bounded non-empty WAV')
            with target.open('rb') as stream:
                header = stream.read(12)
            if header[:4] != b'RIFF' or header[8:] != b'WAVE':
                raise RuntimeError('Worker produced invalid WAV media')
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError) as error:
            cleanup_output(target)
            # Never include subprocess output: third-party errors may echo text.
            raise RuntimeError(f'{self.BACKEND} worker failed ({type(error).__name__}); check its environment, model installation, CUDA and limits') from None
        except BaseException:
            cleanup_output(target)
            raise
        ui = {'audio': [{'filename': filename, 'subfolder': '', 'type': 'output'}]}
        if timings is not None:
            ui['tts_timings'] = [timings]  # only timings/booleans, never text, paths or vectors
        return {'ui': ui, 'result': (filename,)}


class PraxisXTTS(_SpeechNode):
    BACKEND = 'xtts'
    LANGUAGES = XTTS_LANGUAGES
    PYTHON_ENV = 'PRAXIS_XTTS_PYTHON'
    DEFAULT_PYTHON = '/opt/tts-venv/bin/python'


class PraxisQwen3TTS(_SpeechNode):
    @classmethod
    def IS_CHANGED(cls, text, language, reference_audio, reference_text=''):
        # ComfyUI otherwise caches by filename alone and may never call the
        # worker after an operator replaces the voice at that same filename.
        reference = reference_path(reference_audio)
        with reference.open('rb') as stream:
            raw = stream.read(20 * 1024 * 1024 + 1)
        if not 0 < len(raw) <= 20 * 1024 * 1024:
            raise ValueError('Reference exceeds the audio byte limit')
        key = [str(reference), hashlib.sha256(raw).hexdigest(), reference_text, warm_idle_seconds()]
        return hashlib.sha256(json.dumps(key, ensure_ascii=False).encode()).hexdigest()

    BACKEND = 'qwen3'
    LANGUAGES = QWEN_LANGUAGES
    PYTHON_ENV = 'PRAXIS_QWEN_TTS_PYTHON'
    DEFAULT_PYTHON = '/opt/qwen-tts-venv/bin/python'
