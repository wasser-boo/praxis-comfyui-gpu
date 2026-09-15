"""Bounded JSON framing on an inherited private socket, never a network listener."""
import json
import struct
import time

MAX_FRAME = 65536


def encode(value):
    data = json.dumps(value, ensure_ascii=False, allow_nan=False).encode('utf-8')
    if len(data) > MAX_FRAME:
        raise ValueError('Speech worker message exceeds 64 KiB')
    return struct.pack('!I', len(data)) + data


def _read(sock, size, deadline):
    result = bytearray()
    while len(result) < size:
        if deadline is not None:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError('Speech worker deadline exceeded')
            sock.settimeout(remaining)
        chunk = sock.recv(size - len(result))
        if not chunk:
            raise EOFError('Speech worker disconnected')
        result.extend(chunk)
    return bytes(result)


def receive(sock, deadline=None):
    size = struct.unpack('!I', _read(sock, 4, deadline))[0]
    if not 0 < size <= MAX_FRAME:
        raise ValueError('Invalid speech worker frame size')
    value = json.loads(_read(sock, size, deadline))
    if not isinstance(value, dict):
        raise ValueError('Speech worker message must be an object')
    return value
