import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('smoke_test_module', Path(__file__).resolve().parents[1] / 'scripts/smoke.py')
smoke = importlib.util.module_from_spec(spec); spec.loader.exec_module(smoke)


class SmokeTests(unittest.TestCase):
    def test_private_literal_origin_only(self):
        for url in ['http://100.64.0.9:8188', 'http://127.0.0.1:8188', 'https://192.168.1.9:8188']:
            self.assertEqual(smoke.validate_base(url), url)
        for url in ['http://8.8.8.8:8188', 'http://example.com', 'http://user:pass@127.0.0.1', 'http://127.0.0.1/api', 'http://0.0.0.0', 'http://127.0.0.1/?key=x']:
            with self.subTest(url=url), self.assertRaises(ValueError): smoke.validate_base(url)

    def test_gui_workflow_rejected_before_submission(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'gui.json'; p.write_text(json.dumps({'nodes': []}))
            with patch.object(smoke, 'request') as request:
                with self.assertRaises(ValueError): smoke.run('http://127.0.0.1:8188', p, Path(d), 1)
                request.assert_not_called()

    def test_unsafe_descriptors_are_rejected(self):
        for item in [{'filename': '../secret.wav', 'subfolder': '', 'type': 'output'}, {'filename': 'x.wav', 'subfolder': '../input', 'type': 'output'}, {'filename': 'x.wav', 'subfolder': '', 'type': 'input'}]:
            with self.assertRaises(ValueError): smoke.descriptor(item)


if __name__ == '__main__': unittest.main()
