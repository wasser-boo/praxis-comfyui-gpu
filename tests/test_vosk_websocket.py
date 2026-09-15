"""CPU-only protocol tests; no models, network peers, or recordings."""
import importlib.util
import json
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch
from aiohttp.test_utils import TestClient, TestServer

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('vosk_ws_test_server', ROOT / 'scripts/vosk_server.py')
server = importlib.util.module_from_spec(spec)
spec.loader.exec_module(server)


class ProtocolTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.rates = []
        rates = self.rates
        class Recognizer:
            def __init__(self, model, rate): rates.append(rate)
            def AcceptWaveform(self, data): return True
            def Result(self): return '{"text":"eins"}'
            def FinalResult(self): return '{"text":"zwei"}'
        fake = types.SimpleNamespace(KaldiRecognizer=Recognizer, Model=lambda path: object())
        self.patch = patch.dict(sys.modules, {'vosk': fake}); self.patch.start()
        self.models = patch.dict(server.MODELS, {'de': object(), 'ja': object(), 'fr': object()}); self.models.start()
        self.client = TestClient(TestServer(server.create_app()))
        await self.client.start_server()

    async def asyncTearDown(self):
        await self.client.close(); self.models.stop(); self.patch.stop()

    async def test_native_config_pcm_eof_returns_one_final_transcript(self):
        ws = await self.client.ws_connect('/ja')
        await ws.send_json({'config': {'sample_rate': 48000}})
        await ws.send_bytes(b'\0\0' * 480)
        self.assertEqual(await ws.receive_json(), {'partial': ''})
        await ws.send_json({'eof': 1})
        result = await ws.receive_json()
        self.assertEqual(result['text'], 'eins zwei')
        self.assertEqual(result['language'], 'ja')
        self.assertEqual(self.rates, [48000])
        await ws.close()

    async def test_default_german_and_silence_are_valid(self):
        ws = await self.client.ws_connect('/')
        await ws.send_json({'config': {'sample_rate': 16000}})
        await ws.send_bytes(b'\0\0' * 160)
        self.assertEqual(await ws.receive_json(), {'partial': ''})
        await ws.send_json({'eof': 1})
        self.assertEqual((await ws.receive_json())['language'], 'de')
        await ws.close()

    async def test_invalid_configuration_fails_without_constructing_recognizer(self):
        for value in [True, 0, 96000, '16000']:
            ws = await self.client.ws_connect('/')
            await ws.send_json({'config': {'sample_rate': value}})
            self.assertIn('error', await ws.receive_json())
            await ws.close()
        self.assertEqual(self.rates, [])

    async def test_pcm_before_configuration_and_empty_eof_are_rejected(self):
        for value in [b'\0\0', {'eof': 1}]:
            ws = await self.client.ws_connect('/')
            if isinstance(value, bytes): await ws.send_bytes(value)
            else: await ws.send_json(value)
            self.assertIn('error', await ws.receive_json())
            await ws.close()

    async def test_repeated_configuration_and_odd_pcm_are_rejected(self):
        for payload in [{'config': {'sample_rate': 16000}}, b'x']:
            ws = await self.client.ws_connect('/')
            await ws.send_json({'config': {'sample_rate': 16000}})
            if isinstance(payload, bytes): await ws.send_bytes(payload)
            else: await ws.send_json(payload)
            self.assertIn('error', await ws.receive_json()); await ws.close()

    async def test_bounded_work_and_timeout_release_session(self):
        with patch.object(server, 'MAX_WS_BYTES', 100):
            ws = await self.client.ws_connect('/')
            await ws.send_json({'config': {'sample_rate': 16000}})
            await ws.send_bytes(b'\0\0' * 30)
            self.assertEqual(await ws.receive_json(), {'partial': ''})
            await ws.send_bytes(b'\0\0' * 30)
            self.assertIn('error', await ws.receive_json()); await ws.close()
        with patch.object(server, 'WS_TIMEOUT', 0.05):
            ws = await self.client.ws_connect('/')
            self.assertIn('error', await ws.receive_json()); await ws.close()

    async def test_http_health_and_existing_post_contract_remain(self):
        from test_vosk import wav
        response = await self.client.get('/health')
        self.assertEqual(response.status, 200)
        response = await self.client.post('/transcribe?language=fr', data=wav(), headers={'Content-Type': 'audio/wav'})
        self.assertEqual(response.status, 200)
        self.assertEqual(await response.json(), {'text': 'eins zwei', 'language': 'fr'})
        response = await self.client.post('/transcribe?language=xx', data=wav(), headers={'Content-Type': 'audio/wav'})
        self.assertEqual(response.status, 400)

    async def test_model_error_never_echoes_private_details(self):
        with patch.object(server, 'Utterance', side_effect=RuntimeError('PRIVATE_TRANSCRIPT')):
            ws = await self.client.ws_connect('/')
            await ws.send_json({'config': {'sample_rate': 16000}})
            response = await ws.receive_json()
            self.assertIn('error', response)
            self.assertNotIn('PRIVATE_TRANSCRIPT', json.dumps(response))
            await ws.close()

    async def test_browser_origins_are_not_implicitly_trusted(self):
        response = await self.client.get('/', headers={'Origin': 'https://untrusted.example'})
        self.assertEqual(response.status, 403)


if __name__ == '__main__': unittest.main()
