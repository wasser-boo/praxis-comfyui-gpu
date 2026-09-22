"""No Hugging Face/model/GPU access: downloads are deterministic fake writers."""
import importlib.util
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('qwen_provision_test', ROOT / 'custom_nodes/praxis_xtts/download_qwen_model.py')
provisioner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(provisioner)


class ProvisionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name); self.target = self.root / 'model'
        self.calls = []

    def download(self, **options):
        self.calls.append(options)
        self.assertEqual(options['revision'], provisioner.REVISION)
        self.assertIs(options['token'], False)
        self.assertFalse(self.target.exists())
        stage = Path(options['local_dir'])
        for name in provisioner.REQUIRED_FILES:
            path = stage / name; path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(b'fake fixture')

    def test_interruption_never_publishes_partial_model_and_retry_resumes_stage(self):
        def interrupted(**options):
            (Path(options['local_dir']) / 'config.json').write_bytes(b'partial')
            raise ConnectionError('fake interruption')
        with self.assertRaises(ConnectionError):
            provisioner.provision(self.target, interrupted)
        self.assertFalse(self.target.exists())
        stages = list(self.root.glob('.model.tmp-*')); self.assertEqual(len(stages), 1)
        provisioner.provision(self.target, self.download)
        self.assertTrue(provisioner.complete(self.target)); self.assertFalse(stages[0].exists())
        self.assertEqual(json.loads((self.target / 'PRAXIS_MODEL_SOURCE.json').read_text()), provisioner.SOURCE)
        provisioner.provision(self.target, self.download)
        self.assertEqual(len(self.calls), 1)

    def test_incomplete_legacy_directory_is_not_deleted_or_overwritten(self):
        self.target.mkdir(); (self.target / 'custom.txt').write_text('keep')
        with self.assertRaises(ValueError): provisioner.provision(self.target, self.download)
        self.assertEqual((self.target / 'custom.txt').read_text(), 'keep'); self.assertEqual(self.calls, [])

    def test_incomplete_download_and_symlink_stage_never_publish(self):
        with self.assertRaises(RuntimeError): provisioner.provision(self.target, lambda **kw: None)
        self.assertFalse(self.target.exists())
        stage = next(self.root.glob('.model.tmp-*')); stage.rmdir(); stage.symlink_to(self.root, target_is_directory=True)
        with self.assertRaises(ValueError): provisioner.provision(self.target, self.download)
        self.assertEqual(self.calls, [])

    def test_parallel_provisioners_download_once(self):
        mutex = threading.Lock()
        def slow(**options):
            with mutex: self.download(**options)
            time.sleep(0.05)
        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(lambda _: provisioner.provision(self.target, slow), range(2)))
        self.assertEqual(results, [self.target, self.target]); self.assertEqual(len(self.calls), 1)

    def test_local_downloader_change_is_pinned_without_rewriting_upstream_history(self):
        import hashlib
        source = ROOT / 'custom_nodes/praxis_xtts/download_qwen_model.py'
        self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(),
                         'fad71ee473130eb489858dc83e37ec0a5555d61c17924771c5021b58c10c2c6d')
        self.assertEqual(provisioner.REVISION, 'fd4b254389122332181a7c3db7f27e918eec64e3')
        self.assertEqual(provisioner.MODEL, 'Qwen/Qwen3-TTS-12Hz-1.7B-Base')

    def test_compile_cache_is_persistent_and_does_not_redefine_zero_idle(self):
        for path in ('Dockerfile', 'runtime.env.example'):
            source = (ROOT / path).read_text()
            self.assertIn('TORCHINDUCTOR_CACHE_DIR=/workspace/inductor-cache', source)
            self.assertIn('TORCHINDUCTOR_FX_GRAPH_CACHE=1', source)
        self.assertIn('PRAXIS_QWEN_TTS_IDLE_SECONDS=600', (ROOT / 'runtime.env.example').read_text())


if __name__ == '__main__': unittest.main()
