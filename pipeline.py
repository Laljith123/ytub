import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "output"
OUTPUT_JSON = Path(os.getenv("OUTPUT_JSON_PATH", str(ROOT / "output.json")))
LEGACY_OUTPUT_JSON = ROOT / "ouput.json"
FINAL_VIDEO_NAME = os.getenv("FINAL_VIDEO_NAME", "final.mp4")
FINAL_VIDEO = OUTPUT_DIR / FINAL_VIDEO_NAME
THUMBNAIL_PATH = Path(os.getenv("THUMBNAIL_PATH", str(OUTPUT_DIR / "thumbnail.jpg")))
CHUNKS_DIR = OUTPUT_DIR / "chunks"
FINAL_AUDIO = OUTPUT_DIR / "final.wav"
VIDEO_DIR = OUTPUT_DIR / "video"
FINAL_VIDEO_SILENT = VIDEO_DIR / "final_silent.mp4"
QUEUE_DIR = Path(os.getenv("UPLOAD_QUEUE_DIR", str(OUTPUT_DIR / "queue")))
GENERATE_COUNT = int(os.getenv("GENERATE_COUNT", "1"))
RUN_UPLOAD = os.getenv("RUN_UPLOAD", "0") == "1"
UPLOAD_EACH = os.getenv("UPLOAD_EACH", "1") == "1"
BACKGROUND_MUSIC_ENABLED = os.getenv("BACKGROUND_MUSIC_ENABLED", "1") == "1"
CLEAN_OUTPUT_JSON_AFTER_UPLOAD = os.getenv("CLEAN_OUTPUT_JSON_AFTER_UPLOAD", "1") == "1"
PERSIST_HISTORY_FILES = os.getenv(
    "PERSIST_HISTORY_FILES",
    str(ROOT / "generating" / "history.json"),
)


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
            if exc.returncode == 42:
                raise RuntimeError(
                    "The configured NVIDIA Riva TTS function is not available for this NVIDIA_API_KEY/account. "
                    "Update the GitHub secret NVIDIA_API_KEY with a key from an account that can access "
                    "the selected Riva TTS model, then rerun the workflow."
                ) from exc
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

    if CLEAN_OUTPUT_JSON_AFTER_UPLOAD:
        for path in (OUTPUT_JSON, LEGACY_OUTPUT_JSON):
            _delete_temporary_file(path)


def _history_paths() -> set[Path]:
    paths: set[Path] = set()
    for raw in PERSIST_HISTORY_FILES.split(os.pathsep):
        value = raw.strip()
        if not value:
            continue
        try:
            paths.add(Path(value).resolve())
        except OSError:
            continue
    return paths


def _delete_temporary_file(path: Path) -> None:
    try:
        resolved = path.resolve()
    except OSError:
        resolved = path
    if resolved in _history_paths():
        return
    if not path.exists() or path.is_dir():
        return
    try:
        path.unlink()
        print(f"Deleted temporary file after upload: {path}")
    except OSError as exc:
        print(f"Unable to delete temporary file {path}: {exc}")


def _queue_outputs(index: int) -> None:
    QUEUE_DIR.mkdir(parents=True, exist_ok=True)
    suffix = f"{index:02d}"
    queued_video = QUEUE_DIR / f"short_{suffix}.mp4"
    queued_thumb = QUEUE_DIR / f"short_{suffix}.jpg"

    if FINAL_VIDEO.exists():
        shutil.copy2(FINAL_VIDEO, queued_video)
    if THUMBNAIL_PATH.exists():
        shutil.copy2(THUMBNAIL_PATH, queued_thumb)


def _reset_iteration_outputs() -> None:
    if not OUTPUT_DIR.exists():
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for child in OUTPUT_DIR.iterdir():
        if child == QUEUE_DIR:
            continue
        if child.name in {"youtube_token.json", "music"}:
            continue
        try:
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()
        except OSError as exc:
            print(f"Unable to delete {child}: {exc}")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for index in range(1, max(GENERATE_COUNT, 1) + 1):
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        _run("generating/trend.py")

        voice_env = os.environ.copy()
        voice_env["VOICE_CLEANUP"] = "0"
        _run_voice_with_retries(voice_env)

        _run("generating/images.py")

        if BACKGROUND_MUSIC_ENABLED:
            _run("generating/music.py")

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

        if RUN_UPLOAD and UPLOAD_EACH:
            upload_env = os.environ.copy()
            upload_env["UPLOAD_SINGLE_VIDEO"] = str(FINAL_VIDEO)
            upload_env["THUMBNAIL_PATH"] = str(THUMBNAIL_PATH)
            upload_env["YOUTUBE_UPLOADS_PER_DAY"] = "1"
            upload_env["YOUTUBE_MAX_SUCCESS"] = "1"
            upload_env["YOUTUBE_LOOP"] = "0"
            upload_env["YOUTUBE_WAIT_BETWEEN_UPLOADS"] = "0"
            _run("upload.py", env=upload_env)
            if os.getenv("CLEAN_OUTPUT_AFTER_UPLOAD", "1") == "1":
                _cleanup_after_upload()
            elif GENERATE_COUNT > 1:
                _reset_iteration_outputs()
            continue

        if GENERATE_COUNT > 1:
            _queue_outputs(index)
            _reset_iteration_outputs()

    if GENERATE_COUNT <= 1:
        _cleanup_output_folder()
        print(f"Saved final video: {FINAL_VIDEO}")
        print(f"Saved thumbnail: {THUMBNAIL_PATH}")

    if RUN_UPLOAD and not UPLOAD_EACH:
        _run("upload.py")
        if os.getenv("CLEAN_OUTPUT_AFTER_UPLOAD", "1") == "1":
            _cleanup_after_upload()


if __name__ == "__main__":
    main()
