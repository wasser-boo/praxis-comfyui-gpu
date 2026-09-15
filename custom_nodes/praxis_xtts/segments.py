"""Bounded DE/JA speech planning; no model, network, translation or language guessing.

In de-ja mode Japanese scripts (including bare kanji) select Japanese. Other
letters select German. Romaji is ambiguous: explicitly mark [[ja]]...[[/ja]].
Returned text retains its order and punctuation; only explicit markers disappear.
"""
import re

MAX_TEXT = 5000
MAX_SEGMENTS = 64
MARKER = re.compile(r'\[\[(/?)(de|ja)\]\]')


def _script(char):
    code = ord(char)
    if (0x3040 <= code <= 0x30FF or 0x31F0 <= code <= 0x31FF
            or 0x3400 <= code <= 0x4DBF or 0x4E00 <= code <= 0x9FFF
            or 0xF900 <= code <= 0xFAFF or 0xFF66 <= code <= 0xFF9F
            or 0x20000 <= code <= 0x323AF or char in '々〆〇'):
        return 'ja'
    return 'de' if char.isalpha() else None


def _automatic(text):
    first = next((_script(c) for c in text if _script(c)), 'de')
    language, start = first, 0
    for index, char in enumerate(text):
        detected = _script(char)
        if detected and detected != language:
            yield language, text[start:index]
            start, language = index, detected
    if start < len(text):
        yield language, text[start:]


def _marked(text):
    language = None
    position = 0
    for match in MARKER.finditer(text):
        body = text[position:match.start()]
        if language:
            if body:
                yield language, body
        else:
            yield from _automatic(body)
        closing, code = match.groups()
        if closing:
            if language != code:
                raise ValueError('Mismatched or unexpected speech-language closing marker')
            language = None
        else:
            if language is not None:
                raise ValueError('Nested speech-language markers are not supported')
            language = code
        position = match.end()
    if language is not None:
        raise ValueError('Unclosed speech-language marker')
    yield from _automatic(text[position:])


def _chunks(language, text):
    limit = 70 if language == 'ja' else 200
    while len(text) > limit:
        # Prefer a sentence/word boundary, but never drop text or exceed bounds.
        split = max((i + 1 for i, c in enumerate(text[:limit])
                     if i >= limit // 2 and (c.isspace() or c in '.!?。！？、,')), default=limit)
        yield language, text[:split]
        text = text[split:]
    if text:
        yield language, text


def speech_segments(text, language):
    if not isinstance(text, str) or not text.strip() or len(text) > MAX_TEXT:
        raise ValueError('Speech requires 1–5000 characters')
    # Fixed-language modes preserve their language; automatic detection is opt-in.
    raw = list(_marked(text)) if language == 'de-ja' else [(language, text)]
    merged = []
    for lang, body in raw:
        if not body:
            continue
        # Attach whitespace/punctuation-only runs instead of synthesizing them.
        if merged and (merged[-1][0] == lang or not any(c.isalnum() for c in body)):
            merged[-1] = (merged[-1][0], merged[-1][1] + body)
        else:
            merged.append((lang, body))
    result = [chunk for lang, body in merged for chunk in _chunks(lang, body)]
    if not result or not any(c.isalnum() for _, body in result for c in body):
        raise ValueError('Speech contains no pronounceable text')
    if len(result) > MAX_SEGMENTS:
        raise ValueError('Speech exceeds 64 segments; shorten the reply')
    return result
