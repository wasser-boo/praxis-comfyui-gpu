"""Optional Qwen-only process reuse. Idle expiry kills only our own child.

No TCP listener, retries, queue mutation, model imports in ComfyUI or inference
fallback. The parent serializes requests and owns idle expiry, avoiding a child
exiting on an idle deadline while a new request is being submitted.
"""
import atexit
import logging
import os
from pathlib import Path
import socket
import subprocess
import threading
import time
from .protocol import encode, receive


class WarmWorker:
    def __init__(self, python, script, environment, idle_seconds, timeout=600):
        if os.name != 'posix':
            raise ValueError('Warm mode requires POSIX private sockets; use idle=0 for one-shot mode')
        self.python, self.script, self.environment = python, str(script), environment
        self.idle_seconds, self.timeout = idle_seconds, timeout
        self.process = self.socket = self.timer = None
        self.lock = threading.Lock()
        self.generation = 0
        self.blocked = False

    def _close(self):
        self.generation += 1
        if self.timer:
            self.timer.cancel(); self.timer = None
        sock, self.socket = self.socket, None
        process, self.process = self.process, None
        if sock:
            sock.close()
        if process and process.poll() is None:
            try:
                process.terminate()
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.process, self.blocked = process, True
                    raise RuntimeError('Owned Qwen worker has not exited; inspect before further work') from None
        self.blocked = False

    def close(self):
        with self.lock:
            self._close()

    def expire(self, generation):
        with self.lock:
            if generation == self.generation:
                try:
                    self._close()
                except RuntimeError:
                    logging.getLogger(__name__).error('Qwen idle worker did not exit; warm requests blocked until operator cleanup')

    def _start(self):
        if self.process and self.process.poll() is None:
            return
        self._close()
        parent, child = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            self.process = subprocess.Popen(
                [self.python, self.script, '--server-fd', str(child.fileno())],
                pass_fds=(child.fileno(),), start_new_session=True, env=self.environment,
                stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            self.socket = parent
        except BaseException:
            parent.close()
            raise
        finally:
            child.close()

    def run(self, request):
        frame = encode(request)  # reject oversized work before spawning
        with self.lock:
            if self.blocked:
                raise RuntimeError('Prior Qwen worker did not exit; operator cleanup required')
            self.generation += 1
            if self.timer:
                self.timer.cancel(); self.timer = None
            deadline = time.monotonic() + self.timeout
            try:
                self._start()
                self.socket.settimeout(max(0.001, deadline - time.monotonic()))
                self.socket.sendall(frame)
                result = receive(self.socket, deadline)
                if result.get('ok') is not True or not isinstance(result.get('timings'), dict):
                    raise RuntimeError('Invalid worker acknowledgement')
            except Exception:
                self._close()
                raise RuntimeError('Qwen warm worker failed; owned worker stopped, request NOT retried') from None
            except BaseException:
                self._close()
                raise
            self.timer = threading.Timer(self.idle_seconds, self.expire, (self.generation,))
            self.timer.daemon = True
            self.timer.start()
            return result['timings']


_lock = threading.Lock()
_worker = None
_key = None


def run(request, python, environment, idle_seconds):
    global _worker, _key
    key = (python, environment.get('PRAXIS_QWEN_TTS_MODEL_DIR'), environment.get('CUDA_VISIBLE_DEVICES'), idle_seconds)
    with _lock:
        if key != _key or _worker is None:
            if _worker:
                _worker.close()
            _worker = WarmWorker(python, Path(__file__).with_name('worker.py'), environment, idle_seconds)
            _key = key
        return _worker.run(request)


def close():
    global _worker
    with _lock:
        if _worker:
            _worker.close(); _worker = None


atexit.register(close)
