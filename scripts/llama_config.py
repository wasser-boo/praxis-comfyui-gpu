"""Explicit Codacus perf settings; no shell-expanded extra arguments or provider fallback."""
import json
from pathlib import Path

from cpu_threads import physical_thread_count

BIN = Path('/opt/llama.cpp/bin')
REVISION = '27c54b4bbcefadedcec6397477cc2e866c1db716'
MANIFEST = Path(__file__).with_name('llama-models.json')


def models():
    return json.loads(MANIFEST.read_text())


def number(env, key, default, minimum, maximum):
    value = str(env.get(key, default))
    if not value.isascii() or not value.isdecimal() or not minimum <= int(value) <= maximum:
        raise ValueError(f'{key} must be an integer from {minimum} to {maximum}')
    return int(value)


def enabled(env, key, default='0'):
    value = env.get(key, default)
    if value not in ('0', '1'):
        raise ValueError(f'{key} must be 0 or 1')
    return value == '1'


def absolute_path(value):
    path = Path(value)
    if not path.is_absolute() or '..' in path.parts or any(ord(c) < 32 for c in str(path)):
        raise ValueError('Model/profile directories must be absolute paths without traversal')
    return path


def model_dir(kind, env):
    if kind == 'chat':
        return absolute_path(env.get('LLAMA_MODEL_DIR', '/workspace/llama.cpp/models/qwen3.8-flash-next/UD-IQ3_XXS'))
    if kind == 'embedding':
        return absolute_path(env.get('LLAMA_EMBEDDING_MODEL_DIR', '/workspace/llama.cpp/models/nomic-embed-text'))
    raise ValueError('Unknown model kind')


def model_path(kind, env):
    return model_dir(kind, env) / Path(models()[kind]['files'][0]['path']).name


def profile_path(env):
    return absolute_path(env.get('LLAMA_MOE_CACHE_PROFILE', '/workspace/llama.cpp/profiles/qwen3.8-flash-next-merged.csv'))


def thread_count(env):
    if env.get('LLAMA_THREADS', 'auto') == 'auto':
        return physical_thread_count()
    # A numeric override is deliberate and bypasses automatic topology/quota checks.
    return number(env, 'LLAMA_THREADS', None, 1, 512)


def settings(env):
    # Old upstream/fork variables must not silently override the chosen recipe.
    forbidden = [k for k, v in env.items() if v and (
        k.startswith('LLAMA_ARG_') or k.startswith('GGML_MOE_CACHE_') or k.startswith('OLLAMA_'))]
    if forbidden:
        raise ValueError('Remove conflicting legacy variables: ' + ', '.join(sorted(forbidden)))
    return {
        'threads': thread_count(env),
        'context': number(env, 'LLAMA_CONTEXT_SIZE', 4096, 512, 262144),
        'batch': number(env, 'LLAMA_BATCH_SIZE', 512, 1, 8192),
        'ubatch': number(env, 'LLAMA_UBATCH_SIZE', 512, 1, 8192),
        'slots': number(env, 'LLAMA_MOE_CACHE_SLOTS', 56, 1, 512),
        'download': enabled(env, 'LLAMA_DOWNLOAD_MODELS'),
        'generate_profile': enabled(env, 'LLAMA_GENERATE_PROFILE'),
        'embedding': enabled(env, 'LLAMA_EMBEDDING_ENABLED', '1'),
        'download_timeout': number(env, 'LLAMA_DOWNLOAD_TIMEOUT_SECONDS', 21600, 60, 86400),
        'trace_timeout': number(env, 'LLAMA_TRACE_TIMEOUT_SECONDS', 3600, 60, 21600),
    }


def child_environment(env):
    settings(env)
    child = dict(env)
    # Host registration is presence-based in this revision: setting it to "0"
    # would STILL enable it. Translate our boolean to presence/absence correctly.
    for key in ('GGML_CUDA_REGISTER_HOST', 'GGML_SCHED_PREFETCH_EXPERTS'):
        if enabled(env, key, '1'):
            child[key] = '1'
        else:
            child.pop(key, None)
    return child


def thread_args(config):
    return ['--threads', str(config['threads']), '--threads-batch', str(config['threads'])]


def placement_args(config):
    # -fa 1 in the README is stale: this pinned parser accepts on/off/auto.
    return ['--n-gpu-layers', '99', '--n-cpu-moe', '99', '--flash-attn', 'on',
            '--fit', 'off', '--load-mode', 'mmap', '--lazy-mode', 'on'] + thread_args(config)


def server_command(kind, env):
    config = settings(env)
    args = [str(BIN / 'llama-server'), '--model', str(model_path(kind, env)),
            '--alias', models()[kind]['alias'], '--host', '127.0.0.1',
            '--port', '11435' if kind == 'chat' else '11436',
            '--parallel', '1', '--no-webui', '--no-slots', '--no-warmup']
    if kind == 'chat':
        if config['ubatch'] > config['batch']:
            raise ValueError('LLAMA_UBATCH_SIZE must not exceed LLAMA_BATCH_SIZE')
        args += placement_args(config) + [
            '--ctx-size', str(config['context']), '--batch-size', str(config['batch']),
            '--ubatch-size', str(config['ubatch']), '--jinja', '--cache-ram', '0',
            '--moe-cache-profile', str(profile_path(env)),
            '--moe-cache-slots', str(config['slots']), '--spec-type', 'draft-mtp',
            '--spec-draft-threads', str(config['threads']),
            '--spec-draft-threads-batch', str(config['threads'])]
    else:
        # Keep the existing Nomic capability without taking VRAM from Flash/TTS.
        args += ['--embedding', '--pooling', 'mean', '--n-gpu-layers', '0',
                 '--device', 'none', '--fit', 'off', '--ctx-size', '2048',
                 '--batch-size', '2048', '--ubatch-size', '2048'] + thread_args(config)
    return args
