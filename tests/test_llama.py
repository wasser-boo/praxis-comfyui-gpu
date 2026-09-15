import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import llama_config as config
import provision_llama as provision
import profile_llama as profile
import run_llama as runner


def csv_data(position=0):
    return f'{position},0,' + ','.join(str(i) for i in range(10)) + '\n'


class ConfigTests(unittest.TestCase):
    @patch.object(config, 'physical_thread_count', return_value=6)
    def test_flash_recipe_uses_codacus_not_other_fork_flags(self, detect):
        args = config.server_command('chat', {})
        detect.assert_called_once_with()
        for key, value in [('--threads', '6'), ('--threads-batch', '6'),
                           ('--spec-draft-threads', '6'), ('--spec-draft-threads-batch', '6'),
                           ('--moe-cache-slots', '56'),
                           ('--spec-type', 'draft-mtp'), ('--n-cpu-moe', '99'), ('--n-gpu-layers', '99'),
                           ('--flash-attn', 'on'), ('--load-mode', 'mmap'), ('--lazy-mode', 'on'),
                           ('--host', '127.0.0.1'), ('--port', '11435'), ('--parallel', '1')]:
            self.assertEqual(args[args.index(key) + 1], value)
        self.assertIn('--jinja', args)
        self.assertNotIn('--moe-expert-cache-size', args)
        self.assertNotIn('--spec-draft-model', args)
        self.assertIn('UD-IQ3_XXS-00001-of-00003.gguf', ' '.join(args))

    def test_auto_and_unset_apply_detected_count_to_all_cpu_worker_pools(self):
        with patch.object(config, 'physical_thread_count', return_value=12):
            for env in [{}, {'LLAMA_THREADS': 'auto'}]:
                for kind in ['chat', 'embedding']:
                    args = config.server_command(kind, env)
                    keys = ['--threads', '--threads-batch']
                    if kind == 'chat':
                        keys += ['--spec-draft-threads', '--spec-draft-threads-batch']
                    for key in keys:
                        self.assertEqual(args[args.index(key) + 1], '12')

    def test_numeric_thread_override_does_not_require_topology(self):
        with patch.object(config, 'physical_thread_count', side_effect=AssertionError('must not detect')) as detect:
            for kind in ['chat', 'embedding']:
                args = config.server_command(kind, {'LLAMA_THREADS': '7'})
                self.assertEqual(args[args.index('--threads') + 1], '7')
                self.assertEqual(args[args.index('--threads-batch') + 1], '7')
            detect.assert_not_called()

    def test_boolean_zero_removes_presence_based_host_pinning(self):
        self.assertEqual(config.child_environment({})['GGML_CUDA_REGISTER_HOST'], '1')
        child = config.child_environment({'GGML_CUDA_REGISTER_HOST': '0', 'GGML_SCHED_PREFETCH_EXPERTS': '0'})
        self.assertNotIn('GGML_CUDA_REGISTER_HOST', child)
        self.assertNotIn('GGML_SCHED_PREFETCH_EXPERTS', child)

    def test_invalid_and_legacy_settings_rejected(self):
        for env in [{'LLAMA_THREADS': '0'}, {'LLAMA_THREADS': '1; echo nope'},
                    {'LLAMA_THREADS': ''}, {'LLAMA_THREADS': '513'}, {'LLAMA_THREADS': 'AUTO'},
                    {'LLAMA_MOE_CACHE_SLOTS': '513'}, {'LLAMA_DOWNLOAD_MODELS': 'yes'},
                    {'OLLAMA_MODEL': 'qwen3.8:27b'}, {'LLAMA_ARG_HOST': '0.0.0.0'},
                    {'GGML_MOE_CACHE_SLOTS': '112'}, {'LLAMA_UBATCH_SIZE': '1024'},
                    {'LLAMA_MODEL_DIR': '/tmp/../secret'}, {'LLAMA_MOE_CACHE_PROFILE': 'relative.csv'}]:
            with self.subTest(env=env), self.assertRaises(ValueError):
                config.server_command('chat', env)

    def test_embedding_keeps_own_context_without_mtp_or_gpu_cache(self):
        args = config.server_command('embedding', {'LLAMA_CONTEXT_SIZE': '262144'})
        for key, value in [('--ctx-size', '2048'), ('--port', '11436'), ('--device', 'none'), ('--pooling', 'mean')]:
            self.assertEqual(args[args.index(key) + 1], value)
        self.assertIn('--embedding', args)
        self.assertNotIn('--spec-type', args)
        self.assertNotIn('--moe-cache-slots', args)

    def test_pins_have_all_shards_and_independent_nomic_weights(self):
        models = config.models()
        self.assertEqual(len(models['chat']['files']), 3)
        self.assertEqual(sum(f['size'] for f in models['chat']['files']), 81961823936)
        for model in models.values():
            self.assertRegex(model['revision'], r'^[0-9a-f]{40}$')
            for entry in model['files']:
                self.assertRegex(entry['sha256'], r'^[0-9a-f]{64}$')


class DownloadTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.env = {'LLAMA_MODEL_DIR': str(self.directory)}
        self.data = b'GGUFtest-fixture'
        self.entry = {'path': 'UD-IQ3_XXS/test.gguf', 'size': len(self.data), 'sha256': hashlib.sha256(self.data).hexdigest()}
        self.model = {'repository': 'owner/model', 'revision': 'a'*40, 'files': [self.entry]}
        self.target = self.directory / 'test.gguf'
        self.patcher = patch.object(provision, 'models', return_value={'chat': self.model})
        self.patcher.start(); self.addCleanup(self.patcher.stop)

    def response(self, data=None):
        result = io.BytesIO(self.data if data is None else data)
        result.status = 200
        result.headers = {'Content-Length': str(len(self.data))}
        return result

    def test_download_is_opt_in(self):
        with patch.object(provision.OPENER, 'open') as request:
            with self.assertRaisesRegex(RuntimeError, 'Missing pinned'):
                provision.provision('chat', self.env)
            request.assert_not_called()

    def test_first_download_pinned_and_second_start_reuses_verified_files(self):
        with patch.object(provision.OPENER, 'open', return_value=self.response()) as request:
            provision.provision('chat', self.env, download=True)
            self.assertEqual(self.target.read_bytes(), self.data)
            self.assertIn('/resolve/' + 'a'*40 + '/UD-IQ3_XXS/test.gguf', request.call_args.args[0])
        with patch.object(provision.OPENER, 'open') as request:
            provision.provision('chat', self.env, download=True)
            request.assert_not_called()
        self.assertFalse(self.target.with_suffix('.gguf.part').exists())

    def test_preexisting_weights_are_verified_without_network(self):
        self.target.write_bytes(self.data)
        with patch.object(provision.OPENER, 'open') as request:
            provision.provision('chat', self.env, download=True)
            request.assert_not_called()

    def test_modified_cached_file_and_symlink_are_not_replaced(self):
        self.target.write_bytes(self.data)
        provision.provision('chat', self.env)
        self.target.write_bytes(b'X' * len(self.data))
        with self.assertRaisesRegex(RuntimeError, 'SHA256 mismatch'):
            provision.provision('chat', self.env, download=True)
        self.assertEqual(self.target.read_bytes(), b'X' * len(self.data))
        self.target.unlink(); self.target.symlink_to('/etc/passwd')
        with self.assertRaisesRegex(RuntimeError, 'regular file'):
            provision.provision('chat', self.env, download=True)

    def test_failed_download_retained_but_not_published_or_retried(self):
        with patch.object(provision.OPENER, 'open', return_value=self.response(b'X' * len(self.data))) as request:
            with self.assertRaisesRegex(RuntimeError, 'integrity'):
                provision.provision('chat', self.env, download=True)
            self.assertEqual(request.call_count, 1)
        self.assertFalse(self.target.exists())
        self.assertTrue(self.target.with_suffix('.gguf.part').exists())
        with patch.object(provision.OPENER, 'open') as request:
            with self.assertRaisesRegex(RuntimeError, 'interrupted download'):
                provision.provision('chat', self.env, download=True)
            request.assert_not_called()

    def test_short_oversized_and_timed_out_downloads_fail_closed(self):
        for data, timeout in [(b'short', 60), (self.data + b'extra', 60), (self.data, -1)]:
            with self.subTest(data=data, timeout=timeout):
                self.target.with_suffix('.gguf.part').unlink(missing_ok=True)
                with patch.object(provision.OPENER, 'open', return_value=self.response(data)):
                    with self.assertRaises((RuntimeError, TimeoutError)):
                        provision.download_file(self.model, self.entry, self.target, time.monotonic() + timeout)
                self.assertFalse(self.target.exists())

    def test_full_disk_and_lock_collision_do_not_download(self):
        with patch.object(provision.shutil, 'disk_usage', return_value=type('Disk', (), {'free': 0})()):
            with self.assertRaisesRegex(RuntimeError, 'Insufficient'):
                provision.provision('chat', self.env, download=True)
        with provision.lock(self.directory / '.provision.lock'):
            with self.assertRaises(BlockingIOError):
                provision.provision('chat', self.env, download=True)


class ProfileTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.target = Path(self.temp.name) / 'merged.csv'
        self.env = {'LLAMA_MOE_CACHE_PROFILE': str(self.target)}

    def fake_trace(self, command, **kwargs):
        self.assertEqual(command[0], '/opt/llama.cpp/bin/llama-moe-trace')
        self.assertEqual(command[command.index('--predict')+1], '512')
        self.assertEqual(command[command.index('--ctx-size')+1], '4096')
        self.assertEqual(command[command.index('--threads')+1], '4')
        self.assertEqual(command[command.index('--threads-batch')+1], '4')
        self.assertNotIn('--moe-cache-profile', command)
        Path(kwargs['env']['MOE_TRACE_OUT']).write_text(csv_data())

    def test_two_workloads_merge_once_and_restart_reuses_profile(self):
        with patch.object(config, 'physical_thread_count', return_value=4), \
                patch.object(profile.subprocess, 'run', side_effect=self.fake_trace) as trace:
            profile.ensure_profile(self.env, generate=True)
            self.assertEqual(trace.call_count, 2)
            prompts = [call.args[0][-1] for call in trace.call_args_list]
            self.assertNotEqual(prompts[0], prompts[1])
        self.assertEqual(self.target.read_text(), csv_data() * 2)
        with patch.object(profile.subprocess, 'run') as trace:
            profile.ensure_profile(self.env, generate=True)
            trace.assert_not_called()

    def test_missing_profile_never_silently_serves_baseline(self):
        with self.assertRaisesRegex(RuntimeError, 'Missing routing profile'):
            profile.ensure_profile(self.env)

    def test_existing_profile_needs_explicit_adoption(self):
        self.target.write_text(csv_data())
        with self.assertRaisesRegex(RuntimeError, 'unverified'):
            profile.ensure_profile(self.env, generate=True)
        profile.ensure_profile(self.env, adopt=True)
        profile.ensure_profile(self.env)
        self.target.write_text(csv_data(1))
        with self.assertRaisesRegex(RuntimeError, 'unverified'):
            profile.ensure_profile(self.env)

    def test_bad_or_prefill_only_profiles_fail(self):
        for data in ['', csv_data(-1), '0,0,1\n', csv_data().replace(',9', ',512'), csv_data().replace(',9', ',8')]:
            self.target.write_text(data)
            with self.subTest(data=data), self.assertRaises(RuntimeError):
                profile.validate_profile(self.target)

    def test_failed_trace_is_not_published_or_retried(self):
        with patch.object(profile.subprocess, 'run', side_effect=subprocess.CalledProcessError(1, 'trace')) as trace:
            with self.assertRaises(subprocess.CalledProcessError):
                profile.ensure_profile(self.env, generate=True)
            self.assertEqual(trace.call_count, 1)
        self.assertFalse(self.target.exists())


class RuntimeTests(unittest.TestCase):
    def test_cache_activation_and_fallback(self):
        self.assertTrue(runner.cache_active('init_moe_expert_cache: expert cache: 40 layers x 56 slots, 8164.00 MiB uploaded to CUDA0', 56))
        self.assertFalse(runner.cache_active('model loaded; no cache message', 56))
        for line in ['pack allocation failed - expert cache disabled', 'no CPU-resident MoE layers',
                     'expert cache: 40 layers x 112 slots, 8164.00 MiB uploaded to CUDA0']:
            with self.assertRaises(RuntimeError): runner.cache_active(line, 56)

    def test_no_cuda_refuses_cpu_chat_fallback(self):
        with patch.object(runner.subprocess, 'run', return_value=subprocess.CompletedProcess([], 0, 'Available devices:\n', '')):
            with self.assertRaisesRegex(RuntimeError, 'No CUDA device'): runner.require_cuda({})
        with patch.object(runner.subprocess, 'run', return_value=subprocess.CompletedProcess([], 0, 'Available devices:\n  CUDA0: NVIDIA RTX 3060\n', '')):
            runner.require_cuda({})

    def test_startup_env_drives_download_profile_and_mtp_launch(self):
        from unittest.mock import MagicMock
        child = MagicMock()
        child.pid = os.getpid()
        child.stdout = io.StringIO('expert cache: 40 layers x 56 slots, 8164.00 MiB uploaded to CUDA0\n')
        child.wait.return_value = 0
        child.poll.return_value = 0
        env = {'LLAMA_DOWNLOAD_MODELS': '1', 'LLAMA_GENERATE_PROFILE': '1'}
        with tempfile.TemporaryDirectory() as d, patch.object(runner, 'STATE_DIR', Path(d)), \
                patch.object(runner, 'require_cuda') as cuda, \
                patch.object(runner, 'provision') as download, \
                patch.object(runner, 'ensure_profile') as trace, \
                patch.object(runner.subprocess, 'Popen', return_value=child) as launch:
            self.assertEqual(runner.serve('chat', env), 0)
            cuda.assert_called_once_with(env)
            download.assert_called_once_with('chat', env, download=True)
            trace.assert_called_once_with(env, generate=True)
            self.assertIn('draft-mtp', launch.call_args.args[0])
            self.assertFalse((Path(d) / 'chat.json').exists())

    def test_disabled_embedding_does_not_download_or_launch(self):
        with tempfile.TemporaryDirectory() as d, patch.object(runner, 'STATE_DIR', Path(d)), \
                patch.object(runner, 'provision') as download, patch.object(runner.subprocess, 'Popen') as launch:
            self.assertEqual(runner.serve('embedding', {'LLAMA_EMBEDDING_ENABLED': '0'}), 0)
            download.assert_not_called(); launch.assert_not_called()

    def test_preparation_failure_does_not_launch_a_baseline_server(self):
        with tempfile.TemporaryDirectory() as d, patch.object(runner, 'STATE_DIR', Path(d)), \
                patch.object(runner, 'require_cuda'), patch.object(runner, 'provision'), \
                patch.object(runner, 'ensure_profile', side_effect=RuntimeError('bad profile')), \
                patch.object(runner.subprocess, 'Popen') as launch:
            with self.assertRaisesRegex(RuntimeError, 'bad profile'):
                runner.serve('chat', {'LLAMA_GENERATE_PROFILE': '1'})
            launch.assert_not_called()

    def test_readiness_is_bound_to_running_process_and_recipe(self):
        with tempfile.TemporaryDirectory() as d, patch.object(runner, 'STATE_DIR', Path(d)):
            self.assertFalse(runner.runtime_ready('chat', {}))
            receipt = {'pid': os.getpid(), 'process': runner.process_identity(os.getpid()),
                       'command': runner.command_digest(config.server_command('chat', {})), 'cache_active': True}
            path = Path(d) / 'chat.json'; path.write_text(json.dumps(receipt))
            self.assertTrue(runner.runtime_ready('chat', {}))
            other_count = config.thread_count({}) % 512 + 1
            with patch.object(config, 'physical_thread_count', return_value=other_count):
                self.assertFalse(runner.runtime_ready('chat', {}))
            self.assertFalse(runner.runtime_ready('chat', {'LLAMA_MOE_CACHE_SLOTS': '48'}))
            receipt['cache_active'] = False; path.write_text(json.dumps(receipt))
            self.assertFalse(runner.runtime_ready('chat', {}))
            receipt['cache_active'] = True; receipt['process'] = ['stale', 'pid']; path.write_text(json.dumps(receipt))
            self.assertFalse(runner.runtime_ready('chat', {}))


if __name__ == '__main__':
    unittest.main()
