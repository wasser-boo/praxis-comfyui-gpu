import configparser
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))

ROOT = Path(__file__).resolve().parents[1]


def module(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'scripts' / (name + '.py'))
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


class DeploymentTests(unittest.TestCase):
    def test_no_llama_stack_anywhere(self):
        for path in ['LLAMA_CPP.json', 'scripts/run_llama.py', 'scripts/llama_config.py',
                     'scripts/provision_llama.py', 'scripts/profile_llama.py',
                     'scripts/llama-models.json', 'scripts/llama-proxy.conf',
                     'scripts/cpu_threads.py', 'scripts/acceptance_llama.py',
                     'tests/test_llama.py', 'tests/test_llama_acceptance.py', 'tests/test_cpu_threads.py']:
            self.assertFalse((ROOT / path).exists(), path)
        docker = (ROOT / 'Dockerfile').read_text()
        for text in ['llama-build', 'llama-server', 'thecodacus', 'GenerelSchwerz', 'OLLAMA_', 'llama-proxy']:
            self.assertNotIn(text, docker)
        self.assertFalse(any(l.strip().startswith('EXPOSE ') for l in docker.splitlines()))

    def test_supervisor_runs_only_netbird_comfyui_and_vosk(self):
        c = configparser.ConfigParser(interpolation=None); c.read(ROOT / 'supervisord.conf')
        self.assertEqual(c['unix_http_server']['chmod'], '0700')
        names = ['netbird', 'comfyui', 'vosk']
        self.assertEqual({s for s in c if s.startswith('program:')}, {'program:' + n for n in names})
        for name in names:
            self.assertEqual(c['program:' + name]['stopasgroup'], 'true')
        self.assertIn('--base-directory /workspace', c['program:comfyui']['command'])
        for flag in ('--input-directory', '--output-directory', '--temp-directory', '--user-directory'):
            self.assertNotIn(flag, c['program:comfyui']['command'])
        self.assertIn('run_vosk.py', c['program:vosk']['command'])
        self.assertNotIn('vosk_server.py', c['program:vosk']['command'])

    def test_build_pins_comfyui_and_speech_dependencies_only(self):
        docker = (ROOT / 'Dockerfile').read_text()
        self.assertIn('COMFY_REV=a2afcdb89910f7ada92e2a6e04c8b5c9126122f1', docker)
        for text in ['/opt/qwen-tts-venv', 'sox', 'aiohttp==3.13.3', 'coqui-tts', 'ffmpeg']:
            self.assertIn(text, docker)
        self.assertNotIn('download_qwen_model.py --download', docker)
        self.assertNotIn('COQUI_TOS_AGREED=1', docker)
        self.assertNotIn('nvidia/cuda:12.8.1-devel', docker)  # no llama.cpp build stage

    def test_workflows_are_generic_api_and_gui_and_install_without_overwrite(self):
        installer = module('install_workflows')
        for engine, kind in [('xtts', 'PraxisXTTS'), ('qwen3', 'PraxisQwen3TTS')]:
            api = json.loads((ROOT / f'workflows/tts-{engine}-de-ja-api.json').read_text())
            self.assertEqual(api['1']['class_type'], kind)
            self.assertEqual(api['1']['inputs']['reference_audio'], 'reference.wav')
            self.assertEqual(api['1']['inputs']['language'], 'de-ja')
            gui = json.loads((ROOT / f'workflows/tts-{engine}-de-ja-ui.json').read_text())
            self.assertEqual(gui['nodes'][0]['type'], kind)
        with tempfile.TemporaryDirectory() as d:
            dest = Path(d)
            installer.install(ROOT / 'workflows', dest)
            custom = dest / 'user/default/workflows/tts-qwen3-de-ja-ui.json'; custom.write_text('customized')
            installer.install(ROOT / 'workflows', dest)
            self.assertEqual(custom.read_text(), 'customized')
            self.assertTrue((dest / 'workflows/tts-qwen3-de-ja-api.json').is_file())

    def test_french_examples_use_qwen_auto_not_de_ja_routing(self):
        api = json.loads((ROOT / 'workflows/tts-qwen3-de-fr-auto-api.json').read_text())
        self.assertEqual(api['1']['class_type'], 'PraxisQwen3TTS')
        self.assertEqual(api['1']['inputs']['language'], 'auto')
        self.assertEqual(api['1']['inputs']['reference_audio'], 'reference.wav')
        gui = json.loads((ROOT / 'workflows/tts-qwen3-de-fr-auto-ui.json').read_text())
        self.assertEqual(gui['nodes'][0]['widgets_values'][1:3], ['auto', 'reference.wav'])

    def test_vendored_runtime_matches_reviewed_upstream_hashes(self):
        base = ROOT / 'custom_nodes/praxis_xtts'
        provenance = json.loads((base / 'UPSTREAM.json').read_text())
        # worker.py is intentionally divergent from the reviewed upstream commit:
        # the speed/provisioning policies are pinned by test_qwen_worker_speed_and_
        # provisioning_policies instead of byte identity. node.py and warm_worker.py
        # are likewise divergent for the child-stderr diagnostics pinned by
        # test_worker_failure_diagnostics_are_pinned.
        for name, expected in provenance['runtime_files'].items():
            if name in ('worker.py', 'node.py', 'warm_worker.py'):
                continue
            self.assertEqual(hashlib.sha256((base / name).read_bytes()).hexdigest(), expected, name)

    def test_runtime_example_is_speech_only(self):
        env = dict(line.split('=', 1) for line in (ROOT / 'runtime.env.example').read_text().splitlines() if '=' in line and not line.startswith('#'))
        self.assertEqual(env['VOSK_DEFAULT_LANGUAGE'], 'de')
        self.assertEqual(env['VOSK_DOWNLOAD_MODELS'], '1')
        self.assertEqual(env['PRAXIS_QWEN_TTS_DOWNLOAD'], '1')
        self.assertEqual(env['PRAXIS_QWEN_TTS_MODEL_DIR'], '/workspace/qwen3-tts/1.7B-Base')
        for key in env:
            self.assertNotIn('LLAMA', key)
            self.assertNotIn('OLLAMA', key)
        self.assertNotIn('COQUI_TOS_AGREED', env)

    def test_no_models_are_baked_in_and_speech_stays_flag_gated(self):
        docker = (ROOT / 'Dockerfile').read_text()
        for ignore in ['.gitignore', '.dockerignore']:
            text = (ROOT / ignore).read_text()
            for pattern in ['*.wav', '*.safetensors', '__pycache__']:
                self.assertIn(pattern, text)
        self.assertNotIn('download_qwen_model.py --download', docker)
        self.assertNotIn('provision_vosk.py', docker)

    def test_worker_failure_diagnostics_are_pinned(self):
        root = ROOT / 'custom_nodes' / 'praxis_xtts'
        worker = (root / 'worker.py').read_text()
        warm = (root / 'warm_worker.py').read_text()
        node = (root / 'node.py').read_text()
        # The missing-model error must appear only AFTER the authorized download path.
        seek = worker.index('if not qwen_model_complete(directory):')
        after = worker[seek:]
        self.assertLess(after.index('ensure_qwen_model(directory)'),
                        after.index('Install the pinned Qwen3-TTS Base model locally before synthesis'))
        # A partial download is reported, never silently reused.
        self.assertIn('interrupted download', worker)
        # Child stderr is bounded, persisted for operators, but never echoed to users.
        for source, text in [(warm, 'stderr=subprocess.PIPE'), (warm, 'persist_stderr_tail'),
                             (warm, '/workspace/logs/qwen-worker.log'),
                             (node, 'stderr=subprocess.PIPE'), (node, 'persist_stderr_tail')]:
            self.assertIn(text, source)
        self.assertIn('request NOT retried', warm)
        self.assertIn('from None', node)

    def test_qwen_worker_speed_and_provisioning_policies(self):
        source = (ROOT / 'custom_nodes' / 'praxis_xtts' / 'worker.py').read_text()
        self.assertIn("attn = 'sdpa'", source)
        self.assertIn("PRAXIS_QWEN_TTS_ATTN must be sdpa or flash_attention_2", source)
        self.assertIn("find_spec('flash_attn')", source)
        self.assertIn("os.environ.get('PRAXIS_QWEN_TTS_COMPILE', '1') != '0'", source)
        self.assertIn('torch.compile(talker.model, dynamic=True)', source)
        self.assertIn('torch.compile(talker.code_predictor.model, dynamic=True)', source)
        self.assertIn("os.environ.get('PRAXIS_QWEN_TTS_DOWNLOAD', '') != '1'", source)
        self.assertIn("environment.pop('HF_HUB_OFFLINE', None)", source)
        self.assertIn('fcntl.flock', source)

    def test_vosk_startup_provisions_large_models_only_when_authorized(self):
        source = (ROOT / 'scripts' / 'run_vosk.py').read_text()
        self.assertIn("os.environ.get('VOSK_DOWNLOAD_MODELS', '') == '1'", source)
        for model in ('vosk-model-de-0.21', 'vosk-model-fr-0.22', 'vosk-model-ja-0.22'):
            self.assertIn(model, source)
        self.assertIn('am/final.mdl', source)
        self.assertIn('os.execv', source)

    def test_speech_and_vosk_files_are_byte_identical_to_base(self):
        base = json.loads((ROOT / 'BASE_STACK.json').read_text())
        # praxis_xtts worker.py, node.py and warm_worker.py are intentionally
        # divergent from the base stack: measured attention default, flag-gated
        # provisioning, torch.compile and child-stderr diagnostics are pinned by
        # the dedicated tests below instead of byte identity against base hashes.
        for path, expected in base['sha256'].items():
            self.assertTrue((ROOT / path).is_file(), path)
            if path in ('custom_nodes/praxis_xtts/worker.py', 'custom_nodes/praxis_xtts/node.py',
                        'custom_nodes/praxis_xtts/warm_worker.py'):
                continue
            if path.startswith(('custom_nodes/', 'workflows/', 'docs/')) or path in [
                'constraints.txt', 'scripts/vosk_server.py', 'scripts/provision_vosk.py',
                'scripts/xtts_worker.py', 'scripts/install_workflows.py', 'scripts/smoke.py']:
                self.assertEqual(hashlib.sha256((ROOT / path).read_bytes()).hexdigest(), expected, path)

    def test_dockerfile_pins_flashattention_wheel_and_image_checks_it(self):
        docker = (ROOT / 'Dockerfile').read_text()
        self.assertIn('flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp310-cp310-linux_x86_64.whl', docker)
        self.assertIn('75c51f34bb93c5a4438d6b767547ca6c22a9bfc6b29b5646dda98e943cd75d99', docker)
        verify = (ROOT / 'scripts' / 'verify-image.sh').read_text()
        self.assertIn('import flash_attn', verify)
        self.assertIn('py_compile /opt/praxis/bin/run_vosk.py', verify)


if __name__ == '__main__':
    unittest.main()