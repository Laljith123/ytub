import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "output"
FINAL_VIDEO_NAME = os.getenv("FINAL_VIDEO_NAME", "final.mp4")
FINAL_VIDEO = OUTPUT_DIR / FINAL_VIDEO_NAME
THUMBNAIL_PATH = Path(os.getenv("THUMBNAIL_PATH", str(OUTPUT_DIR / "thumbnail.jpg")))
CHUNKS_DIR = OUTPUT_DIR / "chunks"
FINAL_AUDIO = OUTPUT_DIR / "final.wav"
VIDEO_DIR = OUTPUT_DIR / "video"
FINAL_VIDEO_SILENT = VIDEO_DIR / "final_silent.mp4"


def _run(script: str, env: dict[str, str] | None = None) -> None:
    cmd = [sys.executable, str(ROOT / script)]
    subprocess.run(cmd, check=True, env=env)


def _has_audio_chunks() -> bool:
    return CHUNKS_DIR.exists() and any(CHUNKS_DIR.glob("*.wav"))


def _clear_audio_chunks() -> None:
    if CHUNKS_DIR.exists():
        for path in CHUNKS_DIR.glob("*.wav"):
            try:
                path.unlink()
            except OSError as exc:
                print(f"Unable to delete {path}: {exc}")


def _run_voice_with_retries(env: dict[str, str]) -> None:
    retries = int(os.getenv("VOICE_RETRIES", "3"))
    backoff = float(os.getenv("VOICE_RETRY_BACKOFF", "3"))
    last_exc: Exception | None = None

    for attempt in range(1, retries + 1):
        _clear_audio_chunks()
        if FINAL_AUDIO.exists():
            try:
                FINAL_AUDIO.unlink()
            except OSError:
                pass
        try:
            _run("generating/voice.py", env=env)
        except subprocess.CalledProcessError as exc:
            last_exc = exc
        if _has_audio_chunks() and FINAL_AUDIO.exists():
            return

        if attempt < retries:
            sleep_for = backoff * attempt
            print(
                f"Voice chunks missing or voice failed. Retrying in {sleep_for:.1f}s "
                f"(attempt {attempt}/{retries})..."
            )
            time.sleep(sleep_for)

    if last_exc:
        raise last_exc
    raise RuntimeError("Voice chunks missing after retries.")


def _video_ready() -> bool:
    if not FINAL_VIDEO.exists():
        return False
    try:
        return FINAL_VIDEO.stat().st_size > 1024
    except OSError:
        return False


def _run_video_with_retries(env: dict[str, str]) -> None:
    retries = int(os.getenv("VIDEO_RETRIES", "2"))
    backoff = float(os.getenv("VIDEO_RETRY_BACKOFF", "3"))
    last_exc: Exception | None = None

    for attempt in range(1, retries + 1):
        if FINAL_VIDEO.exists():
            try:
                FINAL_VIDEO.unlink()
            except OSError:
                pass
        if FINAL_VIDEO_SILENT.exists():
            try:
                FINAL_VIDEO_SILENT.unlink()
            except OSError:
                pass
        try:
            _run("generating/video.py", env=env)
        except subprocess.CalledProcessError as exc:
            last_exc = exc
        if _video_ready():
            return

        if attempt < retries:
            sleep_for = backoff * attempt
            print(
                f"Final video missing after render. Retrying in {sleep_for:.1f}s "
                f"(attempt {attempt}/{retries})..."
            )
            time.sleep(sleep_for)

    if last_exc:
        raise last_exc
    raise RuntimeError("Final video missing after retries.")


def _cleanup_output_folder() -> None:
    if not FINAL_VIDEO.exists():
        raise RuntimeError(f"Expected final video at {FINAL_VIDEO} but it was not found.")

    for child in OUTPUT_DIR.iterdir():
        if child == FINAL_VIDEO or child == THUMBNAIL_PATH:
            continue
        if child.name in {"youtube_token.json"}:
            continue
        if child.is_dir() and child.name in {"queue", "uploads"}:
            continue
        try:
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()
        except OSError as exc:
            print(f"Unable to delete {child}: {exc}")


def _cleanup_after_upload() -> None:
    if OUTPUT_DIR.exists():
        try:
            shutil.rmtree(OUTPUT_DIR)
            print("Cleared output folder after upload.")
        except OSError as exc:
            print(f"Unable to delete output folder: {exc}")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    _run("generating/trend.py")
    _run("generating/images.py")

    voice_env = os.environ.copy()
    voice_env["VOICE_CLEANUP"] = "0"
    _run_voice_with_retries(voice_env)

    video_env = os.environ.copy()
    video_env["FINAL_VIDEO_NAME"] = FINAL_VIDEO_NAME
    video_env["VIDEO_CLEANUP"] = "0"
    _run_video_with_retries(video_env)

    subtitle_env = os.environ.copy()
    subtitle_env["FINAL_VIDEO_NAME"] = FINAL_VIDEO_NAME
    _run("generating/subtitle.py", env=subtitle_env)

    thumb_env = os.environ.copy()
    thumb_env["FINAL_VIDEO_NAME"] = FINAL_VIDEO_NAME
    _run("generating/thumbnail.py", env=thumb_env)

    _cleanup_output_folder()
    print(f"Saved final video: {FINAL_VIDEO}")
    print(f"Saved thumbnail: {THUMBNAIL_PATH}")

    if os.getenv("RUN_UPLOAD", "0") == "1":
        _run("upload.py")
        if os.getenv("CLEAN_OUTPUT_AFTER_UPLOAD", "1") == "1":
            _cleanup_after_upload()


if __name__ == "__main__":
    main()
