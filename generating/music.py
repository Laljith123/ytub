import json
import os
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import requests


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "output"
OUTPUT_JSON = Path(os.getenv("OUTPUT_JSON_PATH", str(PROJECT_ROOT / "output.json")))

MUSIC_DIR = OUTPUT_DIR / "music"
BACKGROUND_WAV = MUSIC_DIR / "background.wav"
BACKGROUND_SOURCE_JSON = MUSIC_DIR / "background_source.json"
BACKGROUND_ATTRIBUTION_TXT = MUSIC_DIR / "background_attribution.txt"

OPENVERSE_API_URL = os.getenv("OPENVERSE_AUDIO_API_URL", "https://api.openverse.org/v1/audio/")
OPENVERSE_LICENSES = os.getenv("OPENVERSE_MUSIC_LICENSES", "cc0,by")
OPENVERSE_SOURCES = os.getenv("OPENVERSE_MUSIC_SOURCES", "").strip()
OPENVERSE_PAGE_SIZE = int(os.getenv("OPENVERSE_MUSIC_PAGE_SIZE", "20"))
OPENVERSE_PAGES = int(os.getenv("OPENVERSE_MUSIC_PAGES", "3"))
OPENVERSE_TIMEOUT = float(os.getenv("OPENVERSE_MUSIC_TIMEOUT", "30"))
OPENVERSE_RETRIES = int(os.getenv("OPENVERSE_MUSIC_RETRIES", "3"))
OPENVERSE_BACKOFF = float(os.getenv("OPENVERSE_MUSIC_RETRY_BACKOFF", "1.5"))
OPENVERSE_MIN_SECONDS = float(os.getenv("OPENVERSE_MUSIC_MIN_SECONDS", "30"))
OPENVERSE_MAX_SECONDS = float(os.getenv("OPENVERSE_MUSIC_MAX_SECONDS", "600"))
OPENVERSE_FALLBACK_QUERIES = [
    item.strip()
    for item in os.getenv(
        "OPENVERSE_MUSIC_FALLBACK_QUERIES",
        "dark ambient|ambient suspense|cinematic tension|mystery ambient|eerie drone|quiet thriller",
    ).split("|")
    if item.strip()
]

BACKGROUND_GAIN = float(os.getenv("BACKGROUND_MUSIC_GAIN", "1.0"))
TARGET_SAMPLE_RATE = int(os.getenv("BACKGROUND_MUSIC_SAMPLE_RATE", "44100"))
TARGET_CHANNELS = int(os.getenv("BACKGROUND_MUSIC_CHANNELS", "2"))

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


def _safe_text(value: object) -> str:
    text = "" if value is None else str(value)
    text = text.replace("\u2011", "-")
    enc = sys.stdout.encoding or "utf-8"
    try:
        return text.encode(enc, "replace").decode(enc, "replace")
    except Exception:
        return text.encode("utf-8", "replace").decode("utf-8", "replace")


def _headers() -> dict[str, str]:
    headers = {"User-Agent": os.getenv("OPENVERSE_USER_AGENT", "ytub-music/1.0")}
    token = os.getenv("OPENVERSE_ACCESS_TOKEN", "").strip()
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _get_json_with_retries(url: str, params: dict[str, Any]) -> dict[str, Any]:
    last_exc: Exception | None = None
    for attempt in range(1, max(1, OPENVERSE_RETRIES) + 1):
        try:
            response = requests.get(
                url,
                params=params,
                headers=_headers(),
                timeout=OPENVERSE_TIMEOUT,
            )
            if response.status_code in {429, 500, 502, 503, 504} and attempt < OPENVERSE_RETRIES:
                sleep_for = OPENVERSE_BACKOFF * attempt
                print(
                    f"Openverse returned {response.status_code}. Retrying in {sleep_for:.1f}s "
                    f"(attempt {attempt}/{OPENVERSE_RETRIES})..."
                )
                time.sleep(sleep_for)
                continue
            response.raise_for_status()
            payload = response.json()
            return payload if isinstance(payload, dict) else {}
        except Exception as exc:
            last_exc = exc
            if attempt >= OPENVERSE_RETRIES:
                break
            sleep_for = OPENVERSE_BACKOFF * attempt
            print(f"Openverse request failed. Retrying in {sleep_for:.1f}s ({attempt}/{OPENVERSE_RETRIES})...")
            time.sleep(sleep_for)
    raise RuntimeError(f"Openverse request failed: {last_exc}") from last_exc


def _duration_seconds(item: dict[str, Any]) -> float:
    value = item.get("duration")
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    return number / 1000.0 if number > 10000 else number


def _usable_result(item: dict[str, Any]) -> bool:
    if not isinstance(item, dict):
        return False
    if not str(item.get("url") or "").startswith(("http://", "https://")):
        return False
    if str(item.get("category") or "").lower() != "music":
        return False
    if item.get("mature") is True:
        return False
    duration = _duration_seconds(item)
    if duration and duration < OPENVERSE_MIN_SECONDS:
        return False
    if duration and duration > OPENVERSE_MAX_SECONDS:
        return False
    return True


def _search_openverse(query: str) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    seen: set[str] = set()
    for page in range(1, max(1, OPENVERSE_PAGES) + 1):
        params: dict[str, Any] = {
            "q": query,
            "category": "music",
            "license": OPENVERSE_LICENSES,
            "page_size": max(1, min(50, OPENVERSE_PAGE_SIZE)),
            "page": page,
            "mature": "false",
        }
        if OPENVERSE_SOURCES:
            params["source"] = OPENVERSE_SOURCES
        payload = _get_json_with_retries(OPENVERSE_API_URL, params)
        for item in payload.get("results") or []:
            if not _usable_result(item):
                continue
            key = str(item.get("id") or item.get("url") or "")
            if not key or key in seen:
                continue
            seen.add(key)
            results.append(item)
    return results


def _pick_track(query: str) -> dict[str, Any]:
    queries = [query, *OPENVERSE_FALLBACK_QUERIES]
    for candidate_query in queries:
        print(f"Searching Openverse music: {_safe_text(candidate_query)}")
        results = _search_openverse(candidate_query)
        if not results:
            continue
        random.shuffle(results)
        return results[0]
    raise RuntimeError("No usable Openverse background music results found.")


def _download_file(url: str, out_dir: Path, filetype: str | None = None) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = f".{filetype.strip('.').lower()}" if filetype else Path(urlparse(url).path).suffix
    if suffix.lower() not in {".mp3", ".wav", ".ogg", ".flac", ".m4a", ".aac"}:
        suffix = ".mp3"
    out_path = out_dir / f"background_source{suffix}"

    with requests.get(url, headers=_headers(), timeout=OPENVERSE_TIMEOUT, stream=True) as response:
        response.raise_for_status()
        with out_path.open("wb") as f:
            for chunk in response.iter_content(chunk_size=1024 * 256):
                if chunk:
                    f.write(chunk)

    if not _file_ready(out_path):
        raise RuntimeError(f"Downloaded music is missing or too small: {out_path}")
    return out_path


def _convert_to_background_wav(source_path: Path, out_path: Path, gain: float) -> None:
    filters = []
    if abs(gain - 1.0) > 1e-3:
        filters.append(f"volume={gain:.3f}")

    cmd = ["ffmpeg", "-y", "-i", str(source_path)]
    if filters:
        cmd.extend(["-filter:a", ",".join(filters)])
    cmd.extend(
        [
            "-ar",
            str(TARGET_SAMPLE_RATE),
            "-ac",
            str(TARGET_CHANNELS),
            "-c:a",
            "pcm_s16le",
            str(out_path),
        ]
    )
    _run(cmd)
    if abs(gain - 1.0) > 1e-3:
        print(f"Applied background gain: {gain:.2f}x")


def _attribution_text(item: dict[str, Any]) -> str:
    title = str(item.get("title") or "Untitled background music").strip()
    creator = str(item.get("creator") or "Unknown creator").strip()
    license_name = str(item.get("license") or "").upper()
    license_version = str(item.get("license_version") or "").strip()
    license_label = f"{license_name} {license_version}".strip()
    landing = str(item.get("foreign_landing_url") or "").strip()
    license_url = str(item.get("license_url") or "").strip()

    parts = [f'Music: "{title}" by {creator}']
    if license_label:
        parts.append(f"License: {license_label}")
    if landing:
        parts.append(f"Source: {landing}")
    if license_url:
        parts.append(f"License URL: {license_url}")
    parts.append("Discovered via Openverse.")
    return "\n".join(parts)


def _write_metadata(item: dict[str, Any], index: int) -> None:
    MUSIC_DIR.mkdir(parents=True, exist_ok=True)
    metadata = {
        "id": item.get("id"),
        "title": item.get("title"),
        "creator": item.get("creator"),
        "license": item.get("license"),
        "license_version": item.get("license_version"),
        "license_url": item.get("license_url"),
        "source": item.get("source"),
        "provider": item.get("provider"),
        "foreign_landing_url": item.get("foreign_landing_url"),
        "url": item.get("url"),
        "duration_seconds": _duration_seconds(item),
        "output_index": index,
        "attribution": _attribution_text(item),
    }
    BACKGROUND_SOURCE_JSON.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    BACKGROUND_ATTRIBUTION_TXT.write_text(metadata["attribution"], encoding="utf-8")


def _cleanup_old_music() -> None:
    MUSIC_DIR.mkdir(parents=True, exist_ok=True)
    for pattern in ("background_source.*", "background.wav", "background_*.wav", "background_source.json", "background_attribution.txt"):
        for path in MUSIC_DIR.glob(pattern):
            try:
                path.unlink()
            except OSError as exc:
                print(f"Unable to delete old music artifact {path}: {exc}")


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

    _cleanup_old_music()
    print(f"Background music query: {_safe_text(query)}")
    track = _pick_track(query)
    print(
        "Selected background track: "
        f"{_safe_text(track.get('title'))} by {_safe_text(track.get('creator'))} "
        f"({_safe_text(track.get('license'))})"
    )

    downloaded = _download_file(str(track.get("url")), MUSIC_DIR, str(track.get("filetype") or "mp3"))
    _convert_to_background_wav(downloaded, BACKGROUND_WAV, BACKGROUND_GAIN)

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

    _write_metadata(track, index)
    print(f"Saved background wav: {_safe_text(BACKGROUND_WAV)}")
    print(f"Saved attribution: {_safe_text(BACKGROUND_ATTRIBUTION_TXT)}")
    return BACKGROUND_WAV


def main() -> None:
    try:
        create_background_music_wav()
    except Exception as exc:
        print(f"Error creating background music wav: {exc}")


if __name__ == "__main__":
    main()
