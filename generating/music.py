import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from yt_dlp import YoutubeDL

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "output"
OUTPUT_JSON = PROJECT_ROOT / "output.json"

MUSIC_DIR = OUTPUT_DIR / "music"
BACKGROUND_WAV = MUSIC_DIR / "background.wav"
BACKGROUND_GAIN = float(os.getenv("BACKGROUND_MUSIC_GAIN", "1.0"))

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


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


def _load_background_query(path: Path) -> tuple[str | None, int]:
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
        return None, len(data)
    return value, len(data)


def _safe_text(value: str) -> str:
    if value is None:
        return ""
    text = str(value)
    text = text.replace("\u2011", "-")
    enc = sys.stdout.encoding or "utf-8"
    try:
        return text.encode(enc, "replace").decode(enc, "replace")
    except Exception:
        return text.encode("utf-8", "replace").decode("utf-8", "replace")


def _is_url(value: str) -> bool:
    return value.startswith("http://") or value.startswith("https://")


def _parse_cookies_from_browser(value: str) -> tuple[str, str] | tuple[str]:
    parts = [p.strip() for p in value.split(":", 1)]
    if len(parts) == 1:
        return (parts[0],)
    return (parts[0], parts[1])


def _browser_installed(browser: str) -> bool:
    exe_by_browser = {
        "chrome": "chrome.exe",
        "edge": "msedge.exe",
        "brave": "brave.exe",
        "firefox": "firefox.exe",
    }
    exe = exe_by_browser.get(browser, "")
    if exe and shutil.which(exe):
        return True
    known_paths = {
        "chrome": [
            r"C:\Program Files\Google\Chrome\Application\chrome.exe",
            r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
        ],
        "edge": [
            r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
            r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
        ],
        "brave": [
            r"C:\Program Files\BraveSoftware\Brave-Browser\Application\brave.exe",
            r"C:\Program Files (x86)\BraveSoftware\Brave-Browser\Application\brave.exe",
        ],
        "firefox": [
            r"C:\Program Files\Mozilla Firefox\firefox.exe",
            r"C:\Program Files (x86)\Mozilla Firefox\firefox.exe",
        ],
    }
    for path in known_paths.get(browser, []):
        if Path(path).exists():
            return True
    return False


def _detect_browser_for_cookies() -> str | None:
    order = os.getenv("YTDLP_COOKIES_BROWSER_ORDER", "chrome,edge,brave,firefox")
    candidates = [b.strip().lower() for b in order.split(",") if b.strip()]
    for browser in candidates:
        if _browser_installed(browser):
            return browser
    return None


def _resolve_cookies_source() -> tuple[str | None, tuple | None]:
    cookies_from_browser = os.getenv("YTDLP_COOKIES_FROM_BROWSER")
    if cookies_from_browser:
        return None, _parse_cookies_from_browser(cookies_from_browser.strip())

    cookies_file = os.getenv("YTDLP_COOKIES")
    if cookies_file:
        return cookies_file, None

    prefer_browser = os.getenv("YTDLP_COOKIES_AUTO_PREFER_BROWSER", "1") == "1"
    auto_enabled = os.getenv("YTDLP_COOKIES_AUTO", "1") == "1"
    default_cookie = PROJECT_ROOT / "cookies.txt"

    detected = None
    if auto_enabled:
        detected = _detect_browser_for_cookies()
        if detected:
            return None, (detected,)

    if default_cookie.exists() and (not prefer_browser or detected is None):
        return str(default_cookie), None

    return None, None


def _download_audio(query: str, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    target = query if _is_url(query) else f"ytsearch1:{query}"
    cookies_file, cookies_from_browser = _resolve_cookies_source()
    js_runtime = os.getenv("YTDLP_JS_RUNTIME")
    remote_components = os.getenv("YTDLP_REMOTE_COMPONENTS")
    default_client = "web" if (cookies_file or cookies_from_browser) else "android"
    player_client = os.getenv("YTDLP_PLAYER_CLIENT", default_client)
    if cookies_file and player_client == "android":
        player_client = "web"
    if cookies_from_browser and player_client == "android":
        player_client = "web"
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
    if cookies_from_browser:
        ydl_opts["cookiesfrombrowser"] = cookies_from_browser
    if js_runtime:
        ydl_opts["js_runtimes"] = {js_runtime: {}}
    if remote_components:
        parts = [p.strip() for p in remote_components.split(",") if p.strip()]
        if parts:
            ydl_opts["remote_components"] = parts

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


def _apply_gain(path: Path, gain: float) -> None:
    if abs(gain - 1.0) < 1e-3:
        return
    temp_path = path.with_suffix(".gain.wav")
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(path),
        "-filter:a",
        f"volume={gain:.3f}",
        "-c:a",
        "pcm_s16le",
        str(temp_path),
    ]
    _run(cmd)
    temp_path.replace(path)
    print(f"Applied background gain: {gain:.2f}x")


def create_background_music_wav() -> Path | None:
    _require_tool("ffmpeg")

    query, index = _load_background_query(OUTPUT_JSON)
    if not query:
        print("No background_music found in output.json; skipping music download.")
        if BACKGROUND_WAV.exists():
            try:
                BACKGROUND_WAV.unlink()
            except OSError as exc:
                print(f"Unable to delete old background wav: {exc}")
        return None
    print(f"Background music query: {_safe_text(query)}")

    downloaded = _download_audio(query, MUSIC_DIR)
    print(f"Downloaded audio: {_safe_text(downloaded)}")

    if BACKGROUND_WAV.exists():
        try:
            BACKGROUND_WAV.unlink()
        except OSError as exc:
            print(f"Unable to delete old background wav: {exc}")

    if downloaded.resolve() != BACKGROUND_WAV.resolve():
        downloaded.replace(BACKGROUND_WAV)

    _apply_gain(BACKGROUND_WAV, BACKGROUND_GAIN)

    if not _file_ready(BACKGROUND_WAV):
        raise RuntimeError("Background wav is missing or too small.")

    if index <= 0:
        index = 1
    versioned = MUSIC_DIR / f"background_{index:03d}.wav"
    if versioned.resolve() != BACKGROUND_WAV.resolve():
        try:
            shutil.copy2(BACKGROUND_WAV, versioned)
        except OSError as exc:
            print(f"Unable to save versioned background wav: {exc}")

    print(f"Saved background wav: {_safe_text(BACKGROUND_WAV)}")
    if versioned.exists():
        print(f"Saved versioned background wav: {_safe_text(versioned)}")
    return BACKGROUND_WAV
def main() -> None:
    try:
        create_background_music_wav()
    except Exception as exc:
        print(f"Error creating background music wav: {exc}")
if __name__ == "__main__":
    main()
