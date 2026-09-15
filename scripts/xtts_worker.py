"""One synthesis per process so XTTS GPU allocations are released on exit."""
import json
import os
import sys


def main():
    if os.environ.get("COQUI_TOS_AGREED") != "1":
        raise RuntimeError("Explicit XTTS model license acceptance required")
    import torch
    from TTS.api import TTS
    if not torch.cuda.is_available():
        raise RuntimeError("XTTS requires a working CUDA GPU in this deployment")
    request = json.load(sys.stdin)
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to("cuda")
    tts.tts_to_file(text=request["text"], language=request["language"],
                    speaker_wav=request["reference"], file_path=request["output"])


if __name__ == "__main__":
    main()
