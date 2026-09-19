"""Optional Qwen-only process reuse. Idle expiry kills only our own child.

No TCP listener, retries, queue mutation, model imports in ComfyUI or inference
fallback. The parent serializes requests and owns idle expiry, avoiding a child
exiting on an idle deadline while a new request is being submitted.
"""
import atexit
import collections
import logging
import os
from pathlib import Path
import socket
import subprocess
import threading
import time
from .protocol import encode, receive


STDERR_LOG = '/workspace/logs/qwen-worker.log'
STDERR_BUDGET = 16 * 1024
STDERR_PERSIST = 8 * 1024
STDERR_FILE_CAP = 256 * 1024


def _drain_stderr(process, budget=STDERR_BUDGET):
    """Drain child stderr into a bounded byte buffer; return the buffer list."""
    buffer = []
    total = 0
    def drain():
        nonlocal total
        try:
            while True:
                chunk = process.stderr.readline()
                if not chunk:
                    return
                buffer.append(chunk)
                total += len(chunk)
                while total > budget and buffer:
                    total -= len(buffer[0])
                    buffer.pop(0)
        except Exception:
            pass
        finally:
            try:
                process.stderr.close()
            except OSError:
                pass
    threading.Thread(target=drain, daemon=True, name='qwen-worker-stderr').start()
    return buffer


def persist_stderr_tail(raw, label):
    """Append a bounded worker-stderr tail to a size-capped debugging log.

    This is the only persisted child output: exception text still never reaches
    ComfyUI users, but an operator can read the real failure locally.
    """
    if not raw:
        return
    try:
        path = Path(STDERR_LOG)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('ab') as stream:
            stream.write(f'\n--- {label} ---\n'.encode('utf-8', 'replace'))
            stream.write(raw[-STDERR_PERSIST:])
        if path.stat().st_size > STDERR_FILE_CAP:
            with path.open('rb') as stream:
                stream.seek(-STDERR_FILE_CAP // 2, 2)
                keep = stream.read()
            path.write_bytes(b'[qwen-worker.log truncated]\n' + keep)
    except OSError:
        pass


class WarmWorker:
    def __init__(self, python, script, environment, idle_seconds, timeout=600):
        if os.name != 'posix':
            raise ValueError('Warm mode requires POSIX private sockets; use idle=0 for one-shot mode')
        self.python, self.script, self.environment = python, str(script), environment
        self.idle_seconds, self.timeout = idle_seconds, timeout
        self.process = self.socket = self.timer = None
        self.stderr_tail = None
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
                stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            self.stderr_tail = _drain_stderr(self.process)
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
                if self.stderr_tail:
                    time.sleep(0.25)  # let the drainer finish reading the failure
                    persist_stderr_tail(b''.join(self.stderr_tail), 'Qwen warm worker failure')
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
