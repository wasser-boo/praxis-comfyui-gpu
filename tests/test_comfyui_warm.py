"""No models/GPU/network: mock Qwen engine plus a real local IPC child fixture."""
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import types
import unittest
from unittest.mock import Mock, patch
from test_comfyui_audio import module


class WarmEngineTests(unittest.TestCase):
    def test_model_and_reference_reused_but_changed_voice_invalidates_cache(self):
        worker = module('worker')
        model = Mock()
        model.create_voice_clone_prompt.return_value = ['synthetic-vector']
        model.generate_voice_clone.return_value = ([[0.1] * 20], 24000)
        factory = Mock(return_value=model)
        torch = types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: True), bfloat16='bf16')
        sf = types.SimpleNamespace(info=lambda _: types.SimpleNamespace(channels=1, samplerate=24000, frames=72000))
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            for name in ['config.json', 'model.safetensors', 'speech_tokenizer/config.json', 'speech_tokenizer/model.safetensors', 'tokenizer_config.json', 'vocab.json', 'merges.txt']:
                p=root/name;p.parent.mkdir(parents=True,exist_ok=True);p.touch()
            reference=root/'reference.wav';reference.write_bytes(b'synthetic-reference')
            with patch.dict(os.environ, {'PRAXIS_QWEN_TTS_MODEL_DIR':d}), patch.dict(sys.modules,{'torch':torch,'soundfile':sf,'qwen_tts':types.SimpleNamespace(Qwen3TTSModel=types.SimpleNamespace(from_pretrained=factory))}):
                engine=worker.QwenEngine()
                request={'text':'Hallo 学校 Ende','language':'de-ja','reference':str(reference),'output':str(root/'a.wav')}
                first=engine.speak(request)
                request['output']=str(root/'b.wav');second=engine.speak(request)
                self.assertEqual(factory.call_count,1)
                self.assertEqual(model.create_voice_clone_prompt.call_count,1)
                self.assertTrue(first['cold_model']);self.assertFalse(second['cold_model'])
                self.assertTrue(second['reference_cached'])
                reference.write_bytes(b'changed-synthetic-reference')
                request['output']=str(root/'c.wav');engine.speak(request)
                self.assertEqual(model.create_voice_clone_prompt.call_count,2)
                request['reference_text']='Changed reference caption';request['output']=str(root/'d.wav');engine.speak(request)
                self.assertEqual(model.create_voice_clone_prompt.call_count,3)
                self.assertEqual(factory.call_count,1)


FIXTURE = '''import json,os,socket,struct,sys,time
s=socket.socket(fileno=int(sys.argv[sys.argv.index('--server-fd')+1]))
def read(n):
 data=b''
 while len(data)<n:
  chunk=s.recv(n-len(data))
  if not chunk:raise EOFError
  data+=chunk
 return data
while True:
 try:r=json.loads(read(struct.unpack('!I',read(4))[0]))
 except EOFError:break
 if r.get('sleep'):time.sleep(r['sleep'])
 data=json.dumps({'ok':not r.get('fail'),'timings':{'pid':os.getpid()}}).encode()
 s.sendall(struct.pack('!I',len(data))+data)
 if r.get('fail'):break
'''


class WarmIPCTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.script=Path(self.temp.name)/'worker.py';self.script.write_text(FIXTURE)
        self.warm=module('warm_worker')

    def worker(self, idle=5, timeout=3):
        result=self.warm.WarmWorker(sys.executable,self.script,os.environ.copy(),idle,timeout)
        self.addCleanup(result.close);return result

    def test_one_child_for_two_replies_and_explicit_close(self):
        w=self.worker();first=w.run({});second=w.run({})
        self.assertEqual(first['pid'],second['pid'])
        process=w.process;w.close();self.assertIsNotNone(process.poll())

    def test_idle_expiry_releases_child_without_touching_other_processes(self):
        w=self.worker(idle=0.05);w.run({});process=w.process
        until=time.monotonic()+3
        while process.poll() is None and time.monotonic()<until:time.sleep(0.02)
        self.assertIsNotNone(process.poll())
        self.assertIsNone(w.process)

    def test_failure_and_timeout_never_retry(self):
        for request,timeout in [({'fail':True},3),({'sleep':0.5},0.1)]:
            with self.subTest(request=request):
                w=self.worker(timeout=timeout)
                with patch.object(self.warm.subprocess,'Popen',wraps=self.warm.subprocess.Popen) as spawn:
                    with self.assertRaises(RuntimeError):w.run(request)
                    self.assertEqual(spawn.call_count,1)
                self.assertIsNone(w.process)

    def test_stale_idle_timer_cannot_kill_a_newer_request(self):
        w=self.worker();w.run({});old=w.generation;w.run({})
        w.expire(old)
        self.assertIsNone(w.process.poll())

    def test_real_worker_exits_when_parent_channel_closes_during_work(self):
        from test_comfyui_audio import ROOT
        from importlib import import_module
        protocol=import_module(self.warm.__package__+'.protocol')
        marker=Path(self.temp.name)/'started'
        self.script.write_text('import sys,time\nfrom pathlib import Path\nsys.path.insert(0,'+repr(str(ROOT))+')\nimport worker\nclass Engine:\n def speak(self,r):Path('+repr(str(marker))+').touch();time.sleep(30);return {}\nworker.QwenEngine=Engine\nworker.serve(int(sys.argv[sys.argv.index("--server-fd")+1]))\n')
        w=self.worker();w._start();process=w.process
        w.socket.sendall(protocol.encode({'backend':'qwen3'}))
        until=time.monotonic()+3
        while not marker.exists() and time.monotonic()<until:time.sleep(0.01)
        self.assertTrue(marker.exists(),'Synthetic generation started before parent disconnect')
        w.socket.close();w.socket=None
        self.assertIsNotNone(process.wait(timeout=4))

    def test_unreaped_child_is_retained_and_new_inference_blocked(self):
        w=self.worker();process=Mock();process.poll.return_value=None
        process.wait.side_effect=self.warm.subprocess.TimeoutExpired('owned-worker',5)
        w.process=process
        with self.assertRaises(RuntimeError):w.close()
        self.assertIs(w.process,process)
        with patch.object(self.warm.subprocess,'Popen') as spawn:
            with self.assertRaises(RuntimeError):w.run({})
            spawn.assert_not_called()
        process.poll.return_value=0  # only test fixture cleanup; no real process

    def test_oversized_request_does_not_spawn_worker(self):
        w=self.worker()
        with self.assertRaises(ValueError):w.run({'text':'x'*65536})
        self.assertIsNone(w.process)


class WarmNodeTests(unittest.TestCase):
    def test_warm_node_publishes_only_allowlisted_timings_and_one_wav(self):
        import importlib
        import wave
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'reference.wav').write_bytes(b'synthetic')
            folders=types.SimpleNamespace(get_input_directory=lambda:d,get_output_directory=lambda:d)
            with patch.dict(sys.modules,{'folder_paths':folders}):node=module('node')
            warm=importlib.import_module(node.__package__+'.warm_worker')
            def run(request,*_):
                with wave.open(request['output'],'wb') as out:
                    out.setparams((1,2,24000,0,'NONE','not compressed'));out.writeframes(b'\x00\x01'*20)
                return {'cold_model':False,'reference_cached':True,'model_load_seconds':0,
                        'reference_seconds':0.1,'synthesis_seconds':1,'total_seconds':1.1,'private':'DO_NOT_PUBLISH'}
            with patch.dict(os.environ,{'PRAXIS_QWEN_TTS_IDLE_SECONDS':'120'}), patch.object(warm,'run',side_effect=run),patch.object(node.subprocess,'run') as cold:
                result=node.PraxisQwen3TTS().generate('Hallo','de','reference.wav')
                cold.assert_not_called()
            self.assertEqual(len(result['ui']['audio']),1)
            self.assertNotIn('DO_NOT_PUBLISH',json.dumps(result))
            self.assertTrue(result['ui']['tts_timings'][0]['reference_cached'])
            with self.assertRaises(RuntimeError):node.public_timings({'cold_model':True,'reference_cached':False,'model_load_seconds':float('nan')})
            for value in ['-1','3601','forever','1.2']:
                with patch.dict(os.environ,{'PRAXIS_QWEN_TTS_IDLE_SECONDS':value}):
                    with self.assertRaises(ValueError):node.warm_idle_seconds()

    def test_cleanup_removes_only_this_reply_staging_files(self):
        with patch.dict(sys.modules,{'folder_paths':types.SimpleNamespace()}):node=module('node')
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);target=root/'qwen3_owned.wav'
            own=root/('.praxis-audio-'+target.name+'-random.part')
            unrelated=root/'.praxis-audio-another.wav-random.part'
            for p in (target,own,unrelated):p.write_bytes(b'synthetic')
            node.cleanup_output(target)
            self.assertFalse(target.exists());self.assertFalse(own.exists())
            self.assertTrue(unrelated.exists())


if __name__=='__main__':unittest.main()
