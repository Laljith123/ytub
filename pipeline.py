import os
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "output"
FINAL_VIDEO_NAME = os.getenv("FINAL_VIDEO_NAME", "final.mp4")
FINAL_VIDEO = OUTPUT_DIR / FINAL_VIDEO_NAME
THUMBNAIL_PATH = Path(os.getenv("THUMBNAIL_PATH", str(OUTPUT_DIR / "thumbnail.jpg")))


def _run(script: str, env: dict[str, str] | None = None) -> None:
    cmd = [sys.executable, str(ROOT / script)]
    subprocess.run(cmd, check=True, env=env)


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
    _run("generating/voice.py", env=voice_env)

    video_env = os.environ.copy()
    video_env["FINAL_VIDEO_NAME"] = FINAL_VIDEO_NAME
    video_env["VIDEO_CLEANUP"] = "0"
    _run("generating/video.py", env=video_env)

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
