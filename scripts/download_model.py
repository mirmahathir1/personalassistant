import os
import sys
import time
import urllib.request
from pathlib import Path

MODEL_URL = os.environ["MODEL_URL"]
MODEL_FILENAME = os.environ["MODEL_FILENAME"]
MODELS_DIR = Path(os.getenv("MODELS_DIR", "/workspace/models"))
CHUNK_SIZE = 1024 * 1024
RETRIES = 3


def download_file(url: str, destination: Path) -> None:
    temp_path = destination.with_suffix(destination.suffix + ".part")

    for attempt in range(1, RETRIES + 1):
        try:
            with urllib.request.urlopen(url, timeout=60) as response:
                total_bytes = int(response.headers.get("Content-Length", "0"))
                downloaded = 0

                with temp_path.open("wb") as handle:
                    while True:
                        chunk = response.read(CHUNK_SIZE)
                        if not chunk:
                            break

                        handle.write(chunk)
                        downloaded += len(chunk)

                        if total_bytes:
                            percent = downloaded / total_bytes * 100
                            print(
                                f"Downloading {destination.name}: "
                                f"{downloaded / 1024 / 1024:.1f}MB / "
                                f"{total_bytes / 1024 / 1024:.1f}MB ({percent:.1f}%)",
                                flush=True,
                            )
                        else:
                            print(
                                f"Downloading {destination.name}: "
                                f"{downloaded / 1024 / 1024:.1f}MB",
                                flush=True,
                            )

            temp_path.replace(destination)
            return
        except Exception as exc:  # noqa: BLE001
            if temp_path.exists():
                temp_path.unlink()

            if attempt == RETRIES:
                raise RuntimeError(f"Failed to download {url}") from exc

            print(
                f"Download attempt {attempt} failed: {exc}. "
                f"Retrying in {attempt * 2} seconds...",
                flush=True,
            )
            time.sleep(attempt * 2)


def main() -> int:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / MODEL_FILENAME

    if model_path.exists() and model_path.stat().st_size > 0:
        print(f"Model already present at {model_path}", flush=True)
        return 0

    print(f"Downloading GGUF model to {model_path}", flush=True)
    download_file(MODEL_URL, model_path)
    print(f"Model download complete: {model_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
