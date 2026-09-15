"""Combine ordered model outputs into ONE atomic, bounded PCM16 WAV.

Never concatenate RIFF files byte-for-byte. Each segment must have the same
sample rate. A failure discards the entire incomplete reply, not just a segment.
"""
from array import array
import math
import os
from pathlib import Path
import sys
import tempfile
import wave

MAX_SECONDS = 600
MAX_BYTES = 32 * 1024 * 1024
GAP_MS = 120


def write_speech(output, segments, generate):
    output = Path(output)
    if output.exists() or output.is_symlink():
        raise ValueError('Refusing to replace an existing speech output')
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix='.praxis-audio-' + output.name + '-', suffix='.part', dir=output.parent)
    temporary = Path(temporary)
    total = 0
    rate = None
    try:
        with os.fdopen(fd, 'wb') as stream, wave.open(stream, 'wb') as writer:
            for index, (language, text) in enumerate(segments):
                samples, sample_rate = generate(language, text)
                if not isinstance(sample_rate, int) or not 8000 <= sample_rate <= 48000:
                    raise ValueError('Unsupported model sample rate')
                if rate is None:
                    rate = sample_rate
                    writer.setparams((1, 2, rate, 0, 'NONE', 'not compressed'))
                elif sample_rate != rate:
                    raise ValueError('Speech segment sample rates differ')
                if getattr(samples, 'ndim', 1) != 1 or len(samples) == 0:
                    raise ValueError('Model returned empty or non-mono speech')
                gap = rate * GAP_MS // 1000 if index else 0
                total += gap + len(samples)
                if total > rate * MAX_SECONDS or total * 2 + 44 > MAX_BYTES:
                    raise ValueError('Combined speech exceeds the duration/byte budget')
                if gap:
                    writer.writeframesraw(b'\0\0' * gap)
                pcm = array('h')
                for sample in samples:
                    sample = float(sample)
                    if not math.isfinite(sample):
                        raise ValueError('Model returned non-finite audio')
                    pcm.append(round(max(-1.0, min(1.0, sample)) * 32767))
                    if len(pcm) == 8192:
                        if sys.byteorder != 'little':
                            pcm.byteswap()
                        writer.writeframesraw(pcm.tobytes())
                        pcm = array('h')
                if sys.byteorder != 'little':
                    pcm.byteswap()
                writer.writeframesraw(pcm.tobytes())
            if rate is None or total == 0:
                raise ValueError('No speech segments were generated')
        with temporary.open('rb') as stream:
            os.fsync(stream.fileno())
        # Hard-link publishing is exclusive: a concurrent output is never replaced.
        os.link(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
