import importlib.util
import io
from pathlib import Path
import unittest
import wave
from unittest.mock import patch
import types

spec = importlib.util.spec_from_file_location('vosk_server', Path(__file__).resolve().parents[1] / 'scripts/vosk_server.py')
server = importlib.util.module_from_spec(spec)
spec.loader.exec_module(server)


def wav(channels=1, rate=16000, frames=160):
    stream = io.BytesIO()
    with wave.open(stream, 'wb') as audio:
        audio.setnchannels(channels)
        audio.setsampwidth(2)
        audio.setframerate(rate)
        audio.writeframes(b'\0\0' * frames * channels)
    return stream.getvalue()


class AudioTests(unittest.TestCase):
    def test_pcm(self):
        self.assertEqual(len(server.pcm_from_wav(wav())), 320)

    def test_rejects_formats_and_truncation(self):
        for data in [b'invalid', wav(channels=2), wav(rate=48000), wav(frames=0), wav()[:-2]]:
            with self.subTest(size=len(data)), self.assertRaises(ValueError):
                server.pcm_from_wav(data)

    def test_rejects_unknown_language(self):
        with self.assertRaises(ValueError):
            server.transcribe(wav(), 'en')

    def test_joins_final_segments(self):
        class Recognizer:
            def __init__(self, *args): pass
            def AcceptWaveform(self, data): return True
            def Result(self): return '{"text": "bonjour"}'
            def FinalResult(self): return '{"text": "monde"}'
        fake = types.SimpleNamespace(Model=lambda p: None, KaldiRecognizer=Recognizer)
        with patch.dict('sys.modules', {'vosk': fake}), patch.dict(server.MODELS, {'fr': object()}):
            self.assertEqual(server.transcribe(wav(), 'fr'), {'text': 'bonjour monde', 'language': 'fr'})


if __name__ == '__main__':
    unittest.main()
