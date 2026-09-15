"""Download the selected small Vosk models from the official publisher."""
from pathlib import Path
import tempfile
import urllib.request
import zipfile

MODELS = {"fr": "vosk-model-small-fr-0.22", "de": "vosk-model-small-de-0.15", "ja": "vosk-model-small-ja-0.22"}


def main():
    root = Path("/workspace/vosk")
    root.mkdir(parents=True, exist_ok=True)
    for language, model in MODELS.items():
        target = root / language
        if target.exists():
            print(f"Keeping existing {language} model at {target}")
            continue
        with tempfile.TemporaryDirectory(dir=root) as temporary:
            stage = Path(temporary)
            archive = stage / "model.zip"
            url = f"https://alphacephei.com/vosk/models/{model}.zip"
            print(f"Downloading {url}", flush=True)
            with urllib.request.urlopen(url, timeout=120) as source, archive.open("wb") as dest:
                total = 0
                while chunk := source.read(1024 * 1024):
                    total += len(chunk)
                    if total > 200 * 1024 * 1024:
                        raise RuntimeError("Unexpectedly large model archive")
                    dest.write(chunk)
            with zipfile.ZipFile(archive) as z:
                for item in z.infolist():
                    if not (stage / item.filename).resolve().is_relative_to(stage.resolve()):
                        raise RuntimeError("Unsafe archive path")
                if sum(i.file_size for i in z.infolist()) > 500 * 1024 * 1024:
                    raise RuntimeError("Unexpectedly large extracted model")
                if z.testzip():
                    raise RuntimeError("Model archive failed CRC validation")
                z.extractall(stage)
            extracted = stage / model
            if not (extracted / "am/final.mdl").is_file():
                raise RuntimeError("Unexpected model layout")
            extracted.rename(target)
        print(f"Ready: {language}")
    print("Restart the container to load all models, then check /health on port 2700.")


if __name__ == "__main__":
    main()
