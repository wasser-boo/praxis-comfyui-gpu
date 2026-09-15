import configparser
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))

ROOT = Path(__file__).resolve().parents[1]


def module(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'scripts' / (name + '.py'))
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); return mod


class DeploymentTests(unittest.TestCase):
    def test_llama_proxy_keeps_private_listeners_and_routes_embeddings(self):
        config = (ROOT / 'scripts/llama-proxy.conf').read_text()
        for text in ['listen 127.0.0.1:11434;', 'proxy_pass http://127.0.0.1:11435;',
                     'proxy_pass http://127.0.0.1:11436;', 'location = /v1/embeddings',
                     'proxy_buffering off;', 'proxy_next_upstream off;',
                     'if ($http_origin != "") { return 403; }', 'location / { return 404; }']:
            self.assertIn(text, config)
        self.assertNotIn('0.0.0.0', config)
        docker = (ROOT / 'Dockerfile').read_text()
        self.assertNotIn('OLLAMA_', docker)
        self.assertFalse(any(l.strip().startswith('EXPOSE ') for l in docker.splitlines()))

    def test_supervisor_keeps_services_independent_without_download_retry_loops(self):
        c = configparser.ConfigParser(interpolation=None); c.read(ROOT / 'supervisord.conf')
        self.assertEqual(c['unix_http_server']['chmod'], '0700')
        for kind in ['chat', 'embedding']:
            setup = c['program:llama-' + kind]
            self.assertEqual(setup['autorestart'], 'false')
            self.assertEqual(setup['startsecs'], '0')
            self.assertEqual(setup['startretries'], '0')
            self.assertIn('run_llama.py ' + kind, setup['command'])
        self.assertIn('--user-directory /workspace/user', c['program:comfyui']['command'])
        names = ['netbird', 'llama-chat', 'llama-embedding', 'llama-proxy', 'comfyui', 'vosk']
        self.assertEqual({s for s in c if s.startswith('program:')}, {'program:' + n for n in names})
        for name in names:
            self.assertEqual(c['program:' + name]['stopasgroup'], 'true')

    def test_build_pins_codacus_perf_and_matching_libraries(self):
        source = json.loads((ROOT / 'LLAMA_CPP.json').read_text())
        docker = (ROOT / 'Dockerfile').read_text()
        self.assertEqual(source['repository'], 'https://github.com/thecodacus/llama.cpp')
        self.assertEqual(source['branch'], 'perf')
        self.assertIn(source['commit'], docker)
        self.assertEqual(source['commit'], module('llama_config').REVISION)
        for text in ['-DGGML_CUDA=ON', '-DGGML_NATIVE=OFF', 'llama-server llama-moe-trace',
                     'COPY --from=llama-build /opt/llama.cpp /opt/llama.cpp']:
            self.assertIn(text, docker)
        for old in ['scripts/provision_ollama.py', 'scripts/ensure_ollama_cuda.py', 'scripts/ollama-proxy.conf']:
            self.assertFalse((ROOT / old).exists())

    def test_workflows_are_generic_api_and_gui_and_install_without_overwrite(self):
        installer = module('install_workflows')
        for engine,kind in [('xtts', 'PraxisXTTS'), ('qwen3', 'PraxisQwen3TTS')]:
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
        import hashlib
        base = ROOT / 'custom_nodes/praxis_xtts'
        provenance = json.loads((base / 'UPSTREAM.json').read_text())
        for name, expected in provenance['runtime_files'].items():
            self.assertEqual(hashlib.sha256((base / name).read_bytes()).hexdigest(), expected, name)

    def test_runtime_example_selects_independent_context_sizes(self):
        env = dict(line.split('=', 1) for line in (ROOT / 'runtime.env.example').read_text().splitlines() if '=' in line and not line.startswith('#'))
        config = module('llama_config')
        self.assertEqual(env['LLAMA_THREADS'], 'auto')
        with patch.object(config, 'physical_thread_count', return_value=6) as detect:
            plan = config.settings(env)
            detect.assert_called_once_with()
        self.assertTrue(plan['download'])
        self.assertTrue(plan['generate_profile'])
        self.assertEqual((plan['slots'], plan['threads'], plan['context']), (56, 6, 4096))
        self.assertIn('draft-mtp', config.server_command('chat', env))
        embedding = config.server_command('embedding', env)
        self.assertEqual(embedding[embedding.index('--ctx-size') + 1], '2048')
        self.assertNotIn('COQUI_TOS_AGREED', env)

    def test_new_tts_dependencies_are_isolated_and_no_models_are_baked_in(self):
        docker = (ROOT / 'Dockerfile').read_text()
        for text in ['/opt/qwen-tts-venv', 'sox', 'nginx', 'aiohttp==3.13.3', 'llama-moe-trace']:
            self.assertIn(text, docker)
        self.assertNotIn('download_qwen_model.py --download', docker)
        self.assertNotIn('COQUI_TOS_AGREED=1', docker)
        for ignore in ['.gitignore', '.dockerignore']:
            text = (ROOT / ignore).read_text()
            for pattern in ['*.wav', '*.safetensors', '__pycache__']:
                self.assertIn(pattern, text)

    def test_existing_speech_vosk_and_workflows_are_byte_identical_to_base(self):
        import hashlib
        base = json.loads((ROOT / 'BASE_STACK.json').read_text())
        for path, expected in base['sha256'].items():
            if path.startswith(('custom_nodes/', 'workflows/', 'docs/')) or path in [
                'constraints.txt', 'scripts/vosk_server.py', 'scripts/provision_vosk.py',
                'scripts/xtts_worker.py', 'scripts/install_workflows.py', 'scripts/smoke.py']:
                self.assertEqual(hashlib.sha256((ROOT / path).read_bytes()).hexdigest(), expected, path)


if __name__ == '__main__': unittest.main()
