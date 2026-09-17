"""Explicit Codacus perf settings; no shell-expanded extra arguments or provider fallback."""
import json
import os
from pathlib import Path
import stat

from cpu_threads import physical_thread_count

BIN = Path('/opt/llama.cpp/bin')
REVISION = '27c54b4bbcefadedcec6397477cc2e866c1db716'
MANIFEST = Path(__file__).with_name('llama-models.json')
RUNTIME_SETTINGS = Path('/workspace/llama.cpp/runtime-settings.json')
TUNING_KEYS = {'LLAMA_CONTEXT_SIZE', 'LLAMA_BATCH_SIZE', 'LLAMA_UBATCH_SIZE',
                   'LLAMA_REASONING_EFFORT', 'LLAMA_THINKING_BUDGET', 'LLAMA_REASONING_FORMAT', 'LLAMA_THINKING_BUDGET_MESSAGE', 'LLAMA_POLL',
               'LLAMA_MOE_CACHE_SLOTS', 'LLAMA_THREADS'}


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
    if kind == 'mtp':
        return absolute_path(env.get('LLAMA_MTP_MODEL_DIR', '/workspace/llama.cpp/models/qwen3.8-flash-next/MTP'))
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


def runtime_environment(env):
    """Instance tuning shared by launchers and health checks; never source shell code.

    The optional persistent JSON overrides only bounded tuning values, not model
    identities, downloads, credentials or the private listeners. A restart is required.
    """
    result = dict(env)
    try:
        fd = os.open(RUNTIME_SETTINGS, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    except FileNotFoundError:
        return result
    with os.fdopen(fd, 'rb') as stream:
        if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
            raise ValueError('Runtime settings must be a regular JSON file')
        raw = stream.read(4097)
    if len(raw) > 4096:
        raise ValueError('Runtime settings exceed 4096 bytes')
    values = json.loads(raw)
    if not isinstance(values, dict) or not set(values) <= TUNING_KEYS:
        raise ValueError('Runtime settings may contain only the documented tuning keys')
    if any(type(value) not in (str, int) for value in values.values()):
        raise ValueError('Runtime tuning values must be strings or integers')
    result.update(values)
    return result


def settings(env):
    env = runtime_environment(env)
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
        'prefetch_requested': enabled(env, 'GGML_SCHED_PREFETCH_EXPERTS'),
        'download_timeout': number(env, 'LLAMA_DOWNLOAD_TIMEOUT_SECONDS', 21600, 60, 86400),
        'reasoning_effort': (str(env['LLAMA_REASONING_EFFORT'])
                             if env.get('LLAMA_REASONING_EFFORT') in ('minimal', 'low', 'medium', 'high', 'max', 'xhigh')
                             else None),
        'thinking_budget': int(env['LLAMA_THINKING_BUDGET']) if str(env.get('LLAMA_THINKING_BUDGET', '')).lstrip('-').isdigit() else None,
        'reasoning_format': (str(env['LLAMA_REASONING_FORMAT'])
                             if env.get('LLAMA_REASONING_FORMAT') in ('none', 'deepseek', 'deepseek-legacy')
                             else None),
        'thinking_budget_message': (str(env['LLAMA_THINKING_BUDGET_MESSAGE'])
                                    if env.get('LLAMA_THINKING_BUDGET_MESSAGE') else None),
        'poll': int(env['LLAMA_POLL']) if str(env.get('LLAMA_POLL', '')).lstrip('-').isdigit() else None,
        'trace_timeout': number(env, 'LLAMA_TRACE_TIMEOUT_SECONDS', 3600, 60, 21600),
    }


def child_environment(env):
    settings(env)
    child = {key: str(value) for key, value in runtime_environment(env).items()}
    # Host registration is presence-based in this revision: setting it to "0"
    # would STILL enable it. Translate our boolean to presence/absence correctly.
    if enabled(env, 'GGML_CUDA_REGISTER_HOST', '1'):
        child['GGML_CUDA_REGISTER_HOST'] = '1'
    else:
        child.pop('GGML_CUDA_REGISTER_HOST', None)
    # This pinned Flash/expert-cache recipe produces non-finite inference with
    # the experimental prefetch stream enabled (reproduced on RTX A4000).
    # Fence it off even in old templates that still inherit a value of "1".
    # This does not disable CUDA graphs, host pinning, the expert pack or MTP.
    child.pop('GGML_SCHED_PREFETCH_EXPERTS', None)
    return child


def thread_args(config):
    return ['--threads', str(config['threads']), '--threads-batch', str(config['threads'])]


def placement_args(config):
    # -fa 1 in the README is stale: this pinned parser accepts on/off/auto.
    return ['--n-gpu-layers', '99', '--n-cpu-moe', '99', '--flash-attn', 'on',
            '--fit', 'off', '--load-mode', 'mmap', '--lazy-mode', 'on'] + thread_args(config)


def server_command(kind, env):
    if kind not in ('chat', 'embedding'):
        raise ValueError('Only chat and embedding models can be served; MTP is a draft head')
    config = settings(env)
    args = [str(BIN / 'llama-server'), '--model', str(model_path(kind, env)),
            '--alias', models()[kind]['alias'], '--host', '127.0.0.1',
            '--port', '11435' if kind == 'chat' else '11436',
            '--parallel', '1', '--no-slots', '--no-warmup']
    if kind == 'chat':
        if config['ubatch'] > config['batch']:
            raise ValueError('LLAMA_UBATCH_SIZE must not exceed LLAMA_BATCH_SIZE')
        args += placement_args(config) + [
            '--ctx-size', str(config['context']), '--batch-size', str(config['batch']),
            '--ubatch-size', str(config['ubatch']), '--jinja', '--cache-ram', '0',
            # Fork library INFO (including cache activation) is hidden below trace level 4.
            '--verbosity', '4', '--moe-cache-profile', str(profile_path(env)),
            '--moe-cache-slots', str(config['slots']), '--spec-type', 'draft-mtp',
            # Unsloth publishes the matching head separately; the trunk has no MTP layers.
            # The shared Q8 head borrows the target's embedding/output tensors.
            '--spec-draft-model', str(model_path('mtp', env)),
            '--spec-draft-ngl', '99', '--spec-draft-n-max', '2',
            '--spec-draft-threads', str(config['threads']),
            '--spec-draft-threads-batch', str(config['threads'])]
        if config.get('reasoning_effort'):
            args += ['--reasoning-effort', str(config['reasoning_effort'])]
        if config.get('thinking_budget') is not None:
            args += ['--reasoning-budget', str(config['thinking_budget'])]
        if config.get('reasoning_format'):
            args += ['--reasoning-format', str(config['reasoning_format'])]
        if config.get('thinking_budget_message'):
            args += ['--reasoning-budget-message', str(config['thinking_budget_message'])]
        if config.get('poll') is not None:
            args += ['--poll', str(config['poll'])]
    else:
        # Keep the existing Nomic capability without taking VRAM from Flash/TTS.
        args += ['--embedding', '--pooling', 'mean', '--n-gpu-layers', '0',
                 '--device', 'none', '--fit', 'off', '--ctx-size', '2048',
                 '--batch-size', '2048', '--ubatch-size', '2048'] + thread_args(config)
    return args
