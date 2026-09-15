import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


class NodeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        (self.root / 'reference.wav').write_bytes(b'RIFF-test-reference')
        folder = types.SimpleNamespace(get_input_directory=lambda: str(self.root),
                                       get_output_directory=lambda: str(self.root / 'output'))
        mm = types.ModuleType('comfy.model_management')
        mm.unload_all_models = lambda: None
        mm.soft_empty_cache = lambda: None
        comfy = types.ModuleType('comfy')
        comfy.model_management = mm
        self.modules = patch.dict(sys.modules, {'folder_paths': folder, 'comfy': comfy,
                                                'comfy.model_management': mm})
        self.modules.start()
        self.addCleanup(self.modules.stop)
        spec = importlib.util.spec_from_file_location('node', ROOT / 'custom_nodes/praxis_xtts/__init__.py')
        package = importlib.util.module_from_spec(spec)
        sys.modules['node'] = package
        self.addCleanup(lambda: [sys.modules.pop(n, None) for n in list(sys.modules) if n == 'node' or n.startswith('node.')])
        spec.loader.exec_module(package)
        self.node = sys.modules['node.node']

    def test_reference_rejects_escape_and_symlink(self):
        outside = self.root.parent / 'outside-test.wav'
        outside.write_bytes(b'test')
        self.addCleanup(lambda: outside.unlink(missing_ok=True))
        (self.root / 'escape.wav').symlink_to(outside)
        for value in ('../outside-test.wav', str(outside), 'escape.wav', 'missing.wav'):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.node.reference_path(value)

    def test_requires_explicit_license_acceptance(self):
        with patch.dict(os.environ, {'COQUI_TOS_AGREED': ''}):
            with self.assertRaisesRegex(ValueError, 'license'):
                self.node.PraxisXTTS().generate('Hello', 'en', 'reference.wav')

    def test_rejects_invalid_input(self):
        with patch.dict(os.environ, {'COQUI_TOS_AGREED': '1'}):
            for text, lang in [('', 'en'), ('x' * 5001, 'en'), ('Hello', 'invalid')]:
                with self.assertRaises(ValueError):
                    self.node.PraxisXTTS().generate(text, lang, 'reference.wav')

    def test_success_returns_comfy_audio_file_metadata(self):
        import json
        def worker(*args, **kwargs):
            data = json.loads(kwargs['input'])
            Path(data['output']).write_bytes(b'RIFF' + b'0' * 4 + b'WAVE' + b'0' * 80)
        with patch.dict(os.environ, {'COQUI_TOS_AGREED': '1'}), patch.object(self.node.subprocess, 'run', side_effect=worker):
            result = self.node.PraxisXTTS().generate('Hello', 'en', 'reference.wav')
        meta = result['ui']['audio'][0]
        self.assertEqual(meta['type'], 'output')
        self.assertEqual(meta['filename'], result['result'][0])
        self.assertTrue((self.root / 'output' / meta['filename']).is_file())

    def test_failed_worker_cleans_partial_output(self):
        import json
        def worker(*args, **kwargs):
            Path(json.loads(kwargs['input'])['output']).write_bytes(b'partial')
            raise subprocess.TimeoutExpired('worker', 600)
        with patch.dict(os.environ, {'COQUI_TOS_AGREED': '1'}), patch.object(self.node.subprocess, 'run', side_effect=worker):
            # Raw worker exceptions are now private, but cleanup is preserved.
            with self.assertRaisesRegex(RuntimeError, 'TimeoutExpired'):
                self.node.PraxisXTTS().generate('Hello', 'en', 'reference.wav')
        self.assertEqual(list((self.root / 'output').iterdir()), [])


if __name__ == '__main__':
    unittest.main()
