import json
import os
import sys
import time
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parent
MUSIC_DIR = ROOT / "music"
URLS_FILE = Path(os.getenv("MUSIC_URLS_FILE", str(MUSIC_DIR / "urls.txt")))
COUNT = int(os.getenv("MUSIC_COUNT", "5"))
USE_AI = os.getenv("MUSIC_USE_AI", "1") == "1"
MAX_CANDIDATES = int(os.getenv("MUSIC_MAX_CANDIDATES", "50"))
HISTORY_FILE = Path(os.getenv("MUSIC_HISTORY_FILE", str(MUSIC_DIR / "history.json")))
ALLOW_REUSE = os.getenv("MUSIC_ALLOW_REUSE", "0") == "1"

SCRIPT_FILE = Path(os.getenv("MUSIC_SCRIPT_FILE", str(ROOT / "output.json")))
SCRIPT_TEXT_ENV = os.getenv("MUSIC_SCRIPT_TEXT", "").strip()
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY", "").strip()

# --- TEST INJECTION (leave empty in production) ---
TEST_SCRIPT_TEXT = ""
TEST_URLS: list[str] = []
TEST_NVIDIA_KEY = ""
# --------------------------------------------------


def _ensure_urls_file() -> None:
    MUSIC_DIR.mkdir(parents=True, exist_ok=True)
    if not URLS_FILE.exists():
        URLS_FILE.write_text(
            "\n".join(
                [
                    "# Add one direct audio URL per line.",
                    "# Optional format: url | title | tags (comma separated)",
                    "# Example:",
                    "# https://example.com/audio1.mp3 | Dark Ambient | mystery, cinematic",
                ]
            )
            + "\n",
            encoding="utf-8",
        )


def _load_lines() -> list[str]:
    if TEST_URLS:
        return [line.strip() for line in TEST_URLS if str(line).strip()]
    env_urls = os.getenv("MUSIC_URLS", "").strip()
    if env_urls:
        return [line.strip() for line in env_urls.splitlines() if line.strip()]
    if URLS_FILE.exists():
        return [line.strip() for line in URLS_FILE.read_text(encoding="utf-8").splitlines()]
    return []


def _parse_sources(lines: list[str]) -> list[dict]:
    sources = []
    for line in lines:
        if not line or line.startswith("#"):
            continue
        parts = [p.strip() for p in line.split("|")]
        url = parts[0]
        title = parts[1] if len(parts) > 1 else ""
        tags = []
        if len(parts) > 2:
            tags = [t.strip().lower() for t in parts[2].split(",") if t.strip()]
        sources.append({"url": url, "title": title, "tags": tags})
    return sources


def _load_script_text() -> str:
    if TEST_SCRIPT_TEXT:
        return TEST_SCRIPT_TEXT
    if SCRIPT_TEXT_ENV:
        return SCRIPT_TEXT_ENV
    if SCRIPT_FILE.exists():
        try:
            data = json.loads(SCRIPT_FILE.read_text(encoding="utf-8"))
            if isinstance(data, list) and data:
                last = data[-1]
                if isinstance(last, dict):
                    script = last.get("script")
                    if isinstance(script, list):
                        return " ".join(str(x) for x in script if str(x).strip())
                    if script:
                        return str(script)
        except json.JSONDecodeError:
            pass
    return ""


def _resolve_nvidia_key() -> str:
    if NVIDIA_API_KEY:
        return NVIDIA_API_KEY
    if TEST_NVIDIA_KEY:
        return TEST_NVIDIA_KEY
    try:
        value = input("Enter NVIDIA_API_KEY (leave blank to skip AI selection): ").strip()
    except EOFError:
        return ""
    return value


def _extract_json_object(text: str) -> str | None:
    start = text.find("[")
    if start == -1:
        start = text.find("{")
    if start == -1:
        return None
    depth = 0
    in_str = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == "\"":
                in_str = False
        else:
            if ch == "\"":
                in_str = True
            elif ch in "[{":
                depth += 1
            elif ch in "]}":
                depth -= 1
                if depth == 0:
                    return text[start : i + 1]
    return None


def _pick_sources_ai(sources: list[dict], script_text: str) -> tuple[list[dict], dict]:
    api_key = _resolve_nvidia_key()
    if not sources or not api_key or not USE_AI:
        return [], {}
    try:
        from openai import OpenAI
    except Exception:
        return [], {}

    options = []
    for i, src in enumerate(sources[:MAX_CANDIDATES], 1):
        title = src.get("title") or f"Track {i}"
        tags = ", ".join(src.get("tags") or [])
        options.append(f"{i}. {title} | tags: {tags} | url: {src.get('url')}")

    prompt = (
        "Pick background music that fits this narration. "
        "Prefer dark, cinematic, suspense, ambient, minimal, no vocals. "
        f"Choose exactly {min(COUNT, len(options))} unique items. "
        "Return ONLY a JSON object: "
        "{\"picks\":[...],\"mood\":\"...\",\"slang_level\":\"low|med|high\"}.\n\n"
        f"Script:\n{script_text}\n\nOptions:\n" + "\n".join(options)
    )

    client = OpenAI(base_url="https://integrate.api.nvidia.com/v1", api_key=api_key)
    try:
        completion = client.chat.completions.create(
            model="nvidia/nemotron-3-super-120b-a12b",
            messages=[
                {"role": "system", "content": "Return only valid JSON. No extra text."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            top_p=0.9,
            max_tokens=256,
        )
        raw = (completion.choices[0].message.content or "").strip()
        candidate = _extract_json_object(raw) or raw
        data = json.loads(candidate)
        picks = data.get("picks")
        if isinstance(picks, list):
            selected = []
            for idx in picks:
                try:
                    n = int(idx)
                except (TypeError, ValueError):
                    continue
                if 1 <= n <= len(options):
                    selected.append(sources[n - 1])
            return selected[:COUNT], data
    except Exception:
        return [], {}
    return [], {}


def _load_history() -> set[str]:
    if not HISTORY_FILE.exists():
        return set()
    try:
        data = json.loads(HISTORY_FILE.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return {str(item.get("url")) for item in data if isinstance(item, dict) and item.get("url")}
    except json.JSONDecodeError:
        return set()
    return set()


def _append_history(entries: list[dict]) -> None:
    existing = []
    if HISTORY_FILE.exists():
        try:
            existing = json.loads(HISTORY_FILE.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            existing = []
    if not isinstance(existing, list):
        existing = []
    existing.extend(entries)
    HISTORY_FILE.write_text(json.dumps(existing, indent=2, ensure_ascii=False), encoding="utf-8")


def _ext_from_url(url: str) -> str:
    path = urlparse(url).path
    ext = Path(path).suffix.lower()
    if ext in {".mp3", ".wav", ".ogg", ".m4a"}:
        return ext
    return ".mp3"


def _download(url: str, dest: Path) -> None:
    req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(req) as resp, dest.open("wb") as f:
        while True:
            chunk = resp.read(1024 * 256)
            if not chunk:
                break
            f.write(chunk)


def main() -> None:
    lines = _load_lines()
    if not lines:
        _ensure_urls_file()
        print(f"No URLs provided. Add links to {URLS_FILE} and re-run.")
        sys.exit(1)

    sources = _parse_sources(lines)
    if not sources:
        print("No valid sources found.")
        sys.exit(1)

    used_urls = _load_history()
    unused = [s for s in sources if s.get("url") and s.get("url") not in used_urls]
    pool = unused if unused else (sources if ALLOW_REUSE else unused)
    if not pool:
        print("All tracks are already used. Set MUSIC_ALLOW_REUSE=1 to reuse.")
        sys.exit(1)

    script_text = _load_script_text()
    ai_selected, ai_meta = ([], {})
    if script_text:
        ai_selected, ai_meta = _pick_sources_ai(pool, script_text)
    picked = ai_selected if ai_selected else pool[:COUNT]

    MUSIC_DIR.mkdir(parents=True, exist_ok=True)
    manifest = []
    history_entries = []
    for idx, src in enumerate(picked, 1):
        ext = _ext_from_url(src["url"])
        filename = f"bg_{idx:02d}{ext}"
        target = MUSIC_DIR / filename
        if not target.exists():
            _download(src["url"], target)
        entry = {
            "filename": filename,
            "url": src["url"],
            "title": src.get("title", ""),
            "tags": src.get("tags", []),
            "downloaded_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        manifest.append(entry)
        history_entries.append(
            {
                "url": src["url"],
                "title": src.get("title", ""),
                "tags": src.get("tags", []),
                "used_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
        )

    _append_history(history_entries)
    (MUSIC_DIR / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    selected_payload = {"tracks": manifest, "ai": ai_meta}
    (MUSIC_DIR / "selected.json").write_text(
        json.dumps(selected_payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Saved {len(manifest)} tracks to {MUSIC_DIR}")


if __name__ == "__main__":
    main()
