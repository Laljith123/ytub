import json
import os
import shutil
import subprocess
from pathlib import Path

from yt_dlp import YoutubeDL

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "output"
OUTPUT_JSON = PROJECT_ROOT / "output.json"

MUSIC_DIR = OUTPUT_DIR / "music"
BACKGROUND_WAV = MUSIC_DIR / "background.wav"


def _require_tool(name: str) -> None:
    if shutil.which(name) is None:
        raise RuntimeError(f"{name} not found in PATH. Please install ffmpeg and ffprobe.")


def _run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


def _file_ready(path: Path, min_bytes: int = 1024) -> bool:
    try:
        return path.exists() and path.stat().st_size >= min_bytes
    except OSError:
        return False


def _load_background_query(path: Path) -> str | None:
    if not path.exists():
        raise RuntimeError(f"output.json not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list) or not data:
        raise RuntimeError("output.json does not contain a non-empty list")
    last = data[-1]
    if not isinstance(last, dict):
        raise RuntimeError("Last item in output.json is not an object")
    value = str(last.get("background_music") or "").strip()
    if not value:
        return None
    return value


def _is_url(value: str) -> bool:
    return value.startswith("http://") or value.startswith("https://")


def _download_audio(query: str, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    target = query if _is_url(query) else f"ytsearch1:{query}"
    cookies_file = os.getenv("YTDLP_COOKIES")
    js_runtime = os.getenv("YTDLP_JS_RUNTIME")
    player_client = os.getenv("YTDLP_PLAYER_CLIENT", "android")
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": str(out_dir / "%(id)s.%(ext)s"),
        "noplaylist": True,
        "quiet": False,
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",
                "preferredquality": "192",
            }
        ],
        "extractor_args": {
            "youtube": {
                "player_client": [player_client],
            }
        },
    }
    if cookies_file:
        ydl_opts["cookiefile"] = cookies_file
    if js_runtime:
        ydl_opts["js_runtimes"] = {js_runtime: {}}

    with YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(target, download=True)
        if isinstance(info, dict) and "entries" in info:
            entries = [e for e in info.get("entries") or [] if e]
            if not entries:
                raise RuntimeError(f"No YouTube results found for query: {query}")
            info = entries[0]
        filename = Path(ydl.prepare_filename(info)).with_suffix(".wav")

    if not _file_ready(filename):
        raise RuntimeError(f"Downloaded audio is missing or too small: {filename}")
    return filename


def create_background_music_wav() -> Path | None:
    _require_tool("ffmpeg")

    query = _load_background_query(OUTPUT_JSON)
    if not query:
        print("No background_music found in output.json; skipping music download.")
        if BACKGROUND_WAV.exists():
            try:
                BACKGROUND_WAV.unlink()
            except OSError as exc:
                print(f"Unable to delete old background wav: {exc}")
        return None
    print(f"Background music query: {query}")

    downloaded = _download_audio(query, MUSIC_DIR)
    print(f"Downloaded audio: {downloaded}")

    if BACKGROUND_WAV.exists():
        try:
            BACKGROUND_WAV.unlink()
        except OSError as exc:
            print(f"Unable to delete old background wav: {exc}")

    if downloaded.resolve() != BACKGROUND_WAV.resolve():
        downloaded.replace(BACKGROUND_WAV)

    if not _file_ready(BACKGROUND_WAV):
        raise RuntimeError("Background wav is missing or too small.")

    print(f"Saved background wav: {BACKGROUND_WAV}")
    return BACKGROUND_WAV
def main() -> None:
    try:
        create_background_music_wav()
    except Exception as exc:
        print(f"Error creating background music wav: {exc}")
if __name__ == "__main__":
    main()
