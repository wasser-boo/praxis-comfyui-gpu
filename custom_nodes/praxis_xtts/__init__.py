"""Install as the replacement for the old praxis_xtts directory, not beside it."""
from .node import PraxisXTTS, PraxisQwen3TTS

NODE_CLASS_MAPPINGS = {'PraxisXTTS': PraxisXTTS, 'PraxisQwen3TTS': PraxisQwen3TTS}
NODE_DISPLAY_NAME_MAPPINGS = {
    'PraxisXTTS': 'Praxis XTTS-v2 → WAV (DE/JA mix)',
    'PraxisQwen3TTS': 'Praxis Qwen3-TTS → WAV (DE/JA mix)',
}
