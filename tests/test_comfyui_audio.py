"""Offline tests: no Torch, model downloads, CUDA or speech generation."""
import importlib.util
import json
import os
from pathlib import Path
import struct
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
import wave

ROOT = Path(__file__).resolve().parents[1] / 'custom_nodes/praxis_xtts'


def module(name):
    package = 'praxis_test_audio_pkg'
    if package not in sys.modules:
        stub = types.ModuleType(package)
        stub.__path__ = [str(ROOT)]
        sys.modules[package] = stub
    spec = importlib.util.spec_from_file_location(package + '.' + name, ROOT / (name + '.py'))
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


class SegmentsTests(unittest.TestCase):
    def setUp(self):
        self.split = module('segments').speech_segments

    def test_german_japanese_german_including_pure_kanji(self):
        text = 'Auf Japanisch heißt Schule 学校. Das liest man がっこう. Bitte wiederholen.'
        parts = self.split(text, 'de-ja')
        self.assertEqual(''.join(text for _, text in parts), text)
        self.assertEqual([lang for lang, _ in parts], ['de', 'ja', 'de', 'ja', 'de'])
        self.assertIn('学校', parts[1][1])
        self.assertIn('がっこう', parts[3][1])

    def test_japanese_digits_and_halfwidth_katakana(self):
        parts = self.split('日本で3杯。ﾊﾛｰ！ Dann Deutsch.', 'de-ja')
        self.assertEqual([lang for lang, _ in parts], ['ja', 'de'])
        self.assertIn('3杯', parts[0][1])
        self.assertIn('ﾊﾛｰ', parts[0][1])

    def test_romaji_has_explicit_override_not_guessed(self):
        parts = self.split('Sprich [[ja]]arigatou gozaimasu[[/ja]] und dann danke.', 'de-ja')
        self.assertEqual([lang for lang, _ in parts], ['de', 'ja', 'de'])
        self.assertEqual(parts[1][1], 'arigatou gozaimasu')
        self.assertEqual(self.split('arigatou', 'de-ja'), [('de', 'arigatou')])

    def test_malformed_or_nested_markers_fail(self):
        for text in ['[[ja]]Hallo', 'Hallo[[/ja]]', '[[ja]]a[[de]]b[[/de]][[/ja]]', '[[ja]]a[[/de]]']:
            with self.subTest(text=text), self.assertRaises(ValueError):
                self.split(text, 'de-ja')

    def test_single_language_modes_do_not_autodetect(self):
        self.assertEqual(self.split('Hallo 日本語', 'de'), [('de', 'Hallo 日本語')])
        self.assertEqual(self.split('Hallo 日本語', 'ja'), [('ja', 'Hallo 日本語')])
        self.assertEqual(self.split('Hallo 日本語', 'auto'), [('auto', 'Hallo 日本語')])

    def test_chunking_preserves_text_and_bounds_work(self):
        text = 'Hallo Welt. ' * 100
        parts = self.split(text, 'de-ja')
        self.assertEqual(''.join(p[1] for p in parts), text)
        self.assertTrue(all(len(p[1]) <= 200 for p in parts))
        for text in [' ', 'x' * 5001, 'a日' * 40]:
            with self.subTest(length=len(text)), self.assertRaises(ValueError):
                self.split(text, 'de-ja')


class AudioTests(unittest.TestCase):
    def test_concatenated_pcm_wav_and_order(self):
        audio = module('audio')
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'out.wav'
            calls = []
            def generate(language, text):
                calls.append((language, text))
                return ([0.25] * 20 if language == 'de' else [-0.25] * 20), 24000
            audio.write_speech(path, [('de', 'Hallo'), ('ja', 'こんにちは'), ('de', 'Ende')], generate)
            with wave.open(str(path), 'rb') as wav:
                self.assertEqual((wav.getnchannels(), wav.getsampwidth(), wav.getframerate()), (1, 2, 24000))
                frames = wav.readframes(wav.getnframes())
                self.assertEqual(wav.getnframes(), 60 + 2 * 2880)
            self.assertEqual([lang for lang, _ in calls], ['de', 'ja', 'de'])
            self.assertGreater(struct.unpack_from('<h', frames, 10)[0], 0)
            self.assertLess(struct.unpack_from('<h', frames, (20 + 2880 + 5) * 2)[0], 0)

    def test_invalid_audio_never_publishes_partial_output(self):
        audio = module('audio')
        for samples in ([], [float('nan')], [float('inf')], [[1.0]]):
            with self.subTest(samples=str(samples)), tempfile.TemporaryDirectory() as d:
                path = Path(d) / 'out.wav'
                with self.assertRaises((ValueError, TypeError)):
                    audio.write_speech(path, [('de', 'Hallo')], lambda *_: (samples, 24000))
                self.assertFalse(path.exists())
                self.assertEqual(list(Path(d).iterdir()), [])

    def test_failure_and_sample_rate_change_clean_staging(self):
        audio = module('audio')
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'out.wav'
            with self.assertRaises(ValueError):
                audio.write_speech(path, [('de', 'Hallo'), ('ja', '日本語')], lambda lang, _: ([0.1], 24000 if lang == 'de' else 22050))
            self.assertEqual(list(Path(d).iterdir()), [])


class NodeTests(unittest.TestCase):
    def setUp(self):
        idle = patch.dict(os.environ, {'PRAXIS_QWEN_TTS_IDLE_SECONDS': '0'})
        idle.start(); self.addCleanup(idle.stop)
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        (self.root / 'reference.wav').write_bytes(b'RIFF-test')
        folder = types.SimpleNamespace(get_input_directory=lambda: str(self.root), get_output_directory=lambda: str(self.root / 'output'))
        mm = types.ModuleType('comfy.model_management')
        mm.unload_all_models = lambda: None
        mm.soft_empty_cache = lambda: None
        comfy = types.ModuleType('comfy'); comfy.model_management = mm
        self.patcher = patch.dict(sys.modules, {'folder_paths': folder, 'comfy': comfy, 'comfy.model_management': mm})
        self.patcher.start(); self.addCleanup(self.patcher.stop)
        self.node = module('node')

    def test_both_nodes_advertise_mixed_mode(self):
        for node in (self.node.PraxisXTTS, self.node.PraxisQwen3TTS):
            self.assertIn('de-ja', node.INPUT_TYPES()['required']['language'][0])
        self.assertIn('auto', self.node.PraxisQwen3TTS.INPUT_TYPES()['required']['language'][0])

    def test_qwen_graph_cache_detects_replaced_reference_and_transcript(self):
        changed = self.node.PraxisQwen3TTS.IS_CHANGED
        first = changed('Hallo', 'de', 'reference.wav')
        self.assertEqual(first, changed('Hallo', 'de', 'reference.wav'))
        (self.root / 'reference.wav').write_bytes(b'changed-synthetic-voice')
        second = changed('Hallo', 'de', 'reference.wav')
        self.assertNotEqual(first, second)
        self.assertNotEqual(second, changed('Hallo', 'de', 'reference.wav', 'Synthetic caption'))

    def test_reference_escape_absolute_and_symlink_rejected(self):
        outside = self.root.parent / (self.root.name + '.wav')
        outside.write_bytes(b'not-a-reference'); self.addCleanup(lambda: outside.unlink(missing_ok=True))
        (self.root / 'escape.wav').symlink_to(outside)
        for path in ('../outside.wav', str(outside), str(self.root / 'reference.wav'), 'escape.wav', 'missing.wav'):
            with self.subTest(path=path), self.assertRaises(ValueError):
                self.node.reference_path(path)

    def test_xtts_does_not_accept_license_for_user(self):
        with patch.dict(os.environ, {'COQUI_TOS_AGREED': ''}), self.assertRaisesRegex(ValueError, 'license'):
            self.node.PraxisXTTS().generate('Hallo 日本語', 'de-ja', 'reference.wav')

    def test_one_worker_per_reply_and_native_output_descriptor(self):
        def worker(*args, **kwargs):
            request = json.loads(kwargs['input'])
            self.assertEqual(request['language'], 'de-ja')
            self.assertEqual(request['reference'], str(self.root / 'reference.wav'))
            with wave.open(request['output'], 'wb') as f:
                f.setparams((1, 2, 24000, 0, 'NONE', 'not compressed')); f.writeframes(b'\0\0' * 20)
        for cls in (self.node.PraxisXTTS, self.node.PraxisQwen3TTS):
            with patch.dict(os.environ, {'COQUI_TOS_AGREED': '1'}), patch.object(self.node.subprocess, 'run', side_effect=worker) as run:
                value = cls().generate('Hallo 日本語', 'de-ja', 'reference.wav')
            self.assertEqual(run.call_count, 1)
            meta = value['ui']['audio'][0]
            self.assertEqual(meta['type'], 'output'); self.assertEqual(meta['subfolder'], '')
            self.assertEqual(meta['filename'], value['result'][0])

    def test_worker_failure_cleans_output_and_hides_raw_error(self):
        def worker(*args, **kwargs):
            Path(json.loads(kwargs['input'])['output']).write_bytes(b'partial')
            raise self.node.subprocess.CalledProcessError(1, args[0], stderr='private reference transcript')
        with patch.object(self.node.subprocess, 'run', side_effect=worker):
            with self.assertRaises(RuntimeError) as caught:
                self.node.PraxisQwen3TTS().generate('Hallo', 'de', 'reference.wav')
        self.assertNotIn('private reference', str(caught.exception))
        self.assertEqual(list((self.root / 'output').iterdir()), [])


class BundleTests(unittest.TestCase):
    def test_runtime_companions_match_the_recorded_upstream_commit(self):
        import hashlib
        manifest = json.loads((ROOT / 'UPSTREAM.json').read_text())
        self.assertIn('protocol.py', manifest['runtime_files'])
        self.assertIn('warm_worker.py', manifest['runtime_files'])
        # worker.py is intentionally divergent (attention default, flag-gated
        # provisioning, torch.compile); its policy is pinned by dedicated tests.
        for name, digest in manifest['runtime_files'].items():
            if name == 'worker.py':
                continue
            self.assertEqual(hashlib.sha256((ROOT / name).read_bytes()).hexdigest(), digest, name)


class WorkerTests(unittest.TestCase):
    def test_xtts_loads_once_and_keeps_monolingual_path(self):
        worker = module('worker')
        class FakeTTS:
            def __init__(self):
                self.synthesizer = types.SimpleNamespace(output_sample_rate=24000)
                self.calls = []
                self.single = []
            def to(self, device):
                self.device = device
                return self
            def tts(self, **kwargs):
                self.calls.append(kwargs)
                return [0.1] * 20
            def tts_to_file(self, **kwargs):
                self.single.append(kwargs)
        model = FakeTTS()
        from unittest.mock import Mock
        factory = Mock(return_value=model)
        torch = types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: True))
        with tempfile.TemporaryDirectory() as d, patch.dict(os.environ, {'COQUI_TOS_AGREED': '1'}), patch.dict(sys.modules, {'torch': torch, 'TTS': types.ModuleType('TTS'), 'TTS.api': types.SimpleNamespace(TTS=factory)}):
            request = {'text': 'Hallo 日本語 Ende', 'language': 'de-ja', 'reference': '/reference.wav', 'output': str(Path(d) / 'mixed.wav')}
            worker.xtts(request)
            self.assertEqual(factory.call_count, 1)
            self.assertEqual([c['language'] for c in model.calls], ['de', 'ja', 'de'])
            self.assertTrue(Path(request['output']).is_file())
            request['language'] = 'de'
            worker.xtts(request)
            self.assertEqual(len(model.single), 1)
            self.assertEqual(model.single[0]['text'], request['text'])
            self.assertEqual(model.single[0]['language'], 'de')

    def test_qwen_reuses_one_reference_prompt_and_loads_offline(self):
        worker = module('worker')
        from unittest.mock import Mock
        model = Mock()
        model.create_voice_clone_prompt.return_value = ['private-speaker-vector']
        model.generate_voice_clone.return_value = ([[0.1] * 20], 24000)
        factory = Mock(return_value=model)
        torch = types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: True), bfloat16='bf16')
        soundfile = types.SimpleNamespace(info=lambda _: types.SimpleNamespace(channels=1, samplerate=24000, frames=72000))
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            for name in ['config.json', 'model.safetensors', 'speech_tokenizer/config.json', 'speech_tokenizer/model.safetensors', 'tokenizer_config.json', 'vocab.json', 'merges.txt']:
                p = root / name; p.parent.mkdir(parents=True, exist_ok=True); p.touch()
            with patch.dict(os.environ, {'PRAXIS_QWEN_TTS_MODEL_DIR': d}), patch.dict(sys.modules, {'torch': torch, 'soundfile': soundfile, 'qwen_tts': types.SimpleNamespace(Qwen3TTSModel=types.SimpleNamespace(from_pretrained=factory))}):
                (root / 'reference.wav').write_bytes(b'synthetic-reference')
                request = {'text': 'Hallo 学校 Ende', 'language': 'de-ja', 'reference': str(root / 'reference.wav'), 'output': str(root / 'speech.wav')}
                worker.qwen3(request)
                self.assertEqual(factory.call_count, 1)
                self.assertTrue(factory.call_args.kwargs['local_files_only'])
                self.assertFalse(factory.call_args.kwargs['trust_remote_code'])
                self.assertEqual(model.create_voice_clone_prompt.call_count, 1)
                self.assertTrue(model.create_voice_clone_prompt.call_args.kwargs['x_vector_only_mode'])
                self.assertEqual([c.kwargs['language'] for c in model.generate_voice_clone.call_args_list], ['German', 'Japanese', 'German'])
                self.assertTrue((root / 'speech.wav').is_file())


if __name__ == '__main__':
    unittest.main()
