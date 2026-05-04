import json
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "output"

FINAL_VIDEO_NAME = os.getenv("FINAL_VIDEO_NAME", "final.mp4")
FINAL_VIDEO = OUTPUT_DIR / FINAL_VIDEO_NAME

THUMBNAIL_PATH = Path(os.getenv("THUMBNAIL_PATH", str(OUTPUT_DIR / "thumbnail.jpg")))
WIDTH = int(os.getenv("VIDEO_WIDTH", "1080"))
HEIGHT = int(os.getenv("VIDEO_HEIGHT", "1920"))
THUMBNAIL_TIME_SEC = float(os.getenv("THUMBNAIL_TIME_SEC", "0"))
THUMBNAIL_SOURCE = os.getenv("THUMBNAIL_SOURCE", "image").lower()
THUMBNAIL_IMAGE_PATH = Path(
    os.getenv("THUMBNAIL_IMAGE_PATH", str(OUTPUT_DIR / "image" / "image_01.png"))
)
THUMBNAIL_IMAGE_PATTERN = os.getenv(
    "THUMBNAIL_IMAGE_PATTERN",
    str(OUTPUT_DIR / "image" / "image_{index:02d}.png"),
)
THUMBNAIL_FONT = os.getenv("THUMBNAIL_FONT", "Arial")
THUMBNAIL_FONT_FILE = os.getenv("THUMBNAIL_FONT_FILE", "")
THUMBNAIL_FONT_SIZE = int(os.getenv("THUMBNAIL_FONT_SIZE", "86"))
THUMBNAIL_FONT_COLOR = os.getenv("THUMBNAIL_FONT_COLOR", "white")
THUMBNAIL_BOX_COLOR = os.getenv("THUMBNAIL_BOX_COLOR", "black@0.6")
THUMBNAIL_BOX_BORDER = int(os.getenv("THUMBNAIL_BOX_BORDER", "22"))
THUMBNAIL_LINE_SPACING = int(os.getenv("THUMBNAIL_LINE_SPACING", "8"))
THUMBNAIL_MAX_CHARS = int(os.getenv("THUMBNAIL_MAX_CHARS", "18"))
THUMBNAIL_MAX_LINES = int(os.getenv("THUMBNAIL_MAX_LINES", "3"))
THUMBNAIL_TEXT_POSITION = os.getenv("THUMBNAIL_TEXT_POSITION", "top").lower()
THUMBNAIL_Y_TOP = os.getenv("THUMBNAIL_Y_TOP", "(h*0.08)")
THUMBNAIL_Y_CENTER = os.getenv("THUMBNAIL_Y_CENTER", "(h-text_h)/2")
THUMBNAIL_Y_BOTTOM = os.getenv("THUMBNAIL_Y_BOTTOM", "h-(text_h+h*0.12)")
THUMBNAIL_UPPERCASE = os.getenv("THUMBNAIL_UPPERCASE", "1") == "1"
TITLE_FILE = OUTPUT_DIR / "thumbnail_title.txt"

NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
THUMBNAIL_PLAN_ENABLED = os.getenv("THUMBNAIL_PLAN_ENABLED", "1") == "1"
THUMBNAIL_PLAN_MODEL = os.getenv("THUMBNAIL_PLAN_MODEL", os.getenv("CONTENT_MODEL", "openai/gpt-oss-120b"))
THUMBNAIL_PLAN_BASE_URL = os.getenv(
    "THUMBNAIL_PLAN_BASE_URL",
    os.getenv("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1"),
)
THUMBNAIL_PLAN_MAX_ATTEMPTS = int(os.getenv("THUMBNAIL_PLAN_MAX_ATTEMPTS", "3"))
THUMBNAIL_PLAN_MAX_TOKENS = int(os.getenv("THUMBNAIL_PLAN_MAX_TOKENS", "2048"))
THUMBNAIL_PLAN_REASONING_BUDGET = int(os.getenv("THUMBNAIL_PLAN_REASONING_BUDGET", "2048"))
THUMBNAIL_PLAN_TEMPERATURE = float(os.getenv("THUMBNAIL_PLAN_TEMPERATURE", "0.35"))
THUMBNAIL_PLAN_TOP_P = float(os.getenv("THUMBNAIL_PLAN_TOP_P", "0.9"))
THUMBNAIL_PLAN_ENABLE_THINKING = os.getenv("THUMBNAIL_PLAN_ENABLE_THINKING", "1") == "1"
THUMBNAIL_PLAN_PICK_IMAGE = os.getenv("THUMBNAIL_PLAN_PICK_IMAGE", "1") == "1"
THUMBNAIL_PLAN_USE_POSITION = os.getenv("THUMBNAIL_PLAN_USE_POSITION", "1") == "1"
THUMBNAIL_PLAN_TEXT_MAX_WORDS = int(os.getenv("THUMBNAIL_PLAN_TEXT_MAX_WORDS", "4"))
THUMBNAIL_PLAN_SUBTEXT_MAX_WORDS = int(os.getenv("THUMBNAIL_PLAN_SUBTEXT_MAX_WORDS", "4"))
THUMBNAIL_PLAN_FIELD = os.getenv("THUMBNAIL_PLAN_FIELD", "thumbnail_text")

if not THUMBNAIL_FONT_FILE and os.name == "nt":
    win_font = Path(os.getenv("WINDIR", "C:\\Windows")) / "Fonts" / "arialbd.ttf"
    if win_font.exists():
        THUMBNAIL_FONT_FILE = str(win_font)


def _run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


def _ffprobe_duration(path: Path) -> float:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=nw=1:nk=1",
        str(path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    value = result.stdout.strip()
    if not value:
        raise RuntimeError(f"Could not read duration for {path}")
    return float(value)


def _pick_time(duration: float) -> float:
    if THUMBNAIL_TIME_SEC > 0:
        return min(THUMBNAIL_TIME_SEC, max(0.0, duration - 0.1))
    # Auto: 20% into the video, but not earlier than 1s.
    return min(max(1.0, duration * 0.2), max(0.0, duration - 0.1))


def _load_latest_content() -> dict[str, Any]:
    output_json = PROJECT_ROOT / "output.json"
    if not output_json.exists():
        return {}
    try:
        data = json.loads(output_json.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    if isinstance(data, list) and data and isinstance(data[-1], dict):
        return data[-1]
    if isinstance(data, dict):
        return data
    return {}


def _fallback_title(item: dict[str, Any]) -> str:
    for key in (THUMBNAIL_PLAN_FIELD, "thumbnail_text", "hook", "title"):
        value = str(item.get(key) or "").strip()
        if value:
            return value
    return "Breaking Mystery"


def _extract_json_object(text: str) -> str | None:
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
            elif ch == '"':
                in_str = False
        else:
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[start : i + 1]
    return None


def _strip_reasoning_lines(text: str) -> str:
    if not text:
        return text
    filtered: list[str] = []
    for line in text.splitlines():
        stripped = line.lstrip()
        if stripped.startswith("[Reasoning]") or stripped.startswith("Reasoning:"):
            continue
        if stripped.startswith("[Analysis]") or stripped.startswith("Analysis:"):
            continue
        filtered.append(line)
    return "\n".join(filtered).strip()


def _contains_reasoning(text: str) -> bool:
    t = text.lower()
    return "[reasoning]" in t or "reasoning:" in t or "[analysis]" in t or "analysis:" in t


def _parse_json_object(text: str) -> dict[str, Any]:
    if not text or not text.strip():
        return {}
    try:
        data = json.loads(text)
        return data if isinstance(data, dict) else {}
    except json.JSONDecodeError:
        candidate = _extract_json_object(text)
        if not candidate:
            return {}
        try:
            data = json.loads(candidate)
            return data if isinstance(data, dict) else {}
        except json.JSONDecodeError:
            return {}


def _clean_overlay_part(text: str, max_words: int) -> str:
    value = str(text or "").replace("\n", " ").strip()
    value = re.sub(r"[^A-Za-z0-9\s'\-]", " ", value)
    value = re.sub(r"\s+", " ", value).strip()
    words = value.split()
    if max_words > 0:
        value = " ".join(words[:max_words])
    if THUMBNAIL_UPPERCASE:
        value = value.upper()
    return value.strip()


def _is_valid_position(value: str) -> bool:
    return value in {"top", "center", "middle", "bottom"}


def _default_thumbnail_plan(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "text": _fallback_title(item),
        "subtext": "",
        "source_scene": 1,
        "text_position": THUMBNAIL_TEXT_POSITION,
    }


def _validate_thumbnail_plan(plan: dict[str, Any], item: dict[str, Any]) -> dict[str, Any]:
    text = _clean_overlay_part(plan.get("text") or _fallback_title(item), THUMBNAIL_PLAN_TEXT_MAX_WORDS)
    subtext = _clean_overlay_part(plan.get("subtext") or "", THUMBNAIL_PLAN_SUBTEXT_MAX_WORDS)
    if not text:
        text = _clean_overlay_part(_fallback_title(item), THUMBNAIL_PLAN_TEXT_MAX_WORDS)

    script = item.get("script")
    scene_count = len(script) if isinstance(script, list) else 1
    try:
        source_scene = int(plan.get("source_scene") or 1)
    except (TypeError, ValueError):
        source_scene = 1
    source_scene = min(max(source_scene, 1), max(scene_count, 1))

    text_position = str(plan.get("text_position") or THUMBNAIL_TEXT_POSITION).lower().strip()
    if text_position == "middle":
        text_position = "center"
    if not _is_valid_position(text_position):
        text_position = THUMBNAIL_TEXT_POSITION

    return {
        "text": text,
        "subtext": subtext,
        "source_scene": source_scene,
        "text_position": text_position,
    }


def _build_thumbnail_prompt(item: dict[str, Any]) -> str:
    title = str(item.get("title") or "").strip()
    hook = str(item.get("hook") or "").strip()
    thumbnail_text = str(item.get(THUMBNAIL_PLAN_FIELD) or item.get("thumbnail_text") or "").strip()
    background_music = str(item.get("background_music") or "").strip()
    script = item.get("script") if isinstance(item.get("script"), list) else [str(item.get("script") or "")]
    image = item.get("image") if isinstance(item.get("image"), list) else []
    retention = item.get("retention_triggers") if isinstance(item.get("retention_triggers"), list) else []

    return (
        "You are a YouTube Shorts thumbnail director for a safe true-crime mystery channel. "
        "Return ONE valid JSON object ONLY. No markdown, no commentary, no code block. "
        "Do NOT rewrite the story. Do NOT invent facts. Use only the provided video context. "
        "Create short thumbnail overlay text that is readable on mobile and creates curiosity. "
        "Avoid graphic words, gore, explicit violence, emojis, hashtags, and clickbait lies. "
        f"Main text must be {THUMBNAIL_PLAN_TEXT_MAX_WORDS} words or less. "
        f"Subtext must be {THUMBNAIL_PLAN_SUBTEXT_MAX_WORDS} words or less, or empty. "
        "Choose source_scene as the best visual scene number from the script/image lists. "
        "Choose text_position from: top, center, bottom. "
        "Return exactly these keys: "
        '{"text":"...","subtext":"...","source_scene":1,"text_position":"top"}. '
        "Video context: "
        + json.dumps(
            {
                "title": title,
                "hook": hook,
                "thumbnail_text": thumbnail_text,
                "background_music": background_music,
                "script": script,
                "image": image,
                "retention_triggers": retention,
            },
            ensure_ascii=False,
        )
    )


def _run_thumbnail_completion(prompt: str) -> str:
    from openai import OpenAI

    client = OpenAI(base_url=THUMBNAIL_PLAN_BASE_URL, api_key=NVIDIA_API_KEY)
    completion = client.chat.completions.create(
        model=THUMBNAIL_PLAN_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You must output ONLY valid JSON. "
                    "No markdown, no code fences, no extra commentary. "
                    "Do not output reasoning or analysis."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        temperature=THUMBNAIL_PLAN_TEMPERATURE,
        top_p=THUMBNAIL_PLAN_TOP_P,
        max_tokens=THUMBNAIL_PLAN_MAX_TOKENS,
        extra_body={
            "chat_template_kwargs": {"enable_thinking": THUMBNAIL_PLAN_ENABLE_THINKING},
            "reasoning_budget": THUMBNAIL_PLAN_REASONING_BUDGET if THUMBNAIL_PLAN_ENABLE_THINKING else 0,
        },
        stop=["\n[Reasoning]", "\nReasoning:", "\n[Analysis]", "\nAnalysis:"],
    )
    return (completion.choices[0].message.content or "").strip()


def _build_thumbnail_plan(item: dict[str, Any]) -> dict[str, Any]:
    default_plan = _default_thumbnail_plan(item)
    if not THUMBNAIL_PLAN_ENABLED:
        return _validate_thumbnail_plan(default_plan, item)
    if not NVIDIA_API_KEY:
        print("Thumbnail planner skipped: NVIDIA_API_KEY is missing. Using title fallback.")
        return _validate_thumbnail_plan(default_plan, item)

    prompt = _build_thumbnail_prompt(item)
    last_error = ""
    for attempt in range(1, max(1, THUMBNAIL_PLAN_MAX_ATTEMPTS) + 1):
        try:
            print(f"[Thumbnail] Planning attempt {attempt}/{THUMBNAIL_PLAN_MAX_ATTEMPTS}...")
            response = _run_thumbnail_completion(prompt)
            response = _strip_reasoning_lines(response)
            if _contains_reasoning(response):
                last_error = "reasoning leaked"
                continue
            plan = _parse_json_object(response)
            if not plan:
                last_error = "invalid JSON"
                continue
            return _validate_thumbnail_plan(plan, item)
        except Exception as exc:
            last_error = str(exc)
            print(f"[Thumbnail] Planner error on attempt {attempt}: {exc}")
    print(f"[Thumbnail] Planner failed ({last_error}). Using title fallback.")
    return _validate_thumbnail_plan(default_plan, item)


def _plan_overlay_text(plan: dict[str, Any], item: dict[str, Any]) -> str:
    text = str(plan.get("text") or "").strip()
    subtext = str(plan.get("subtext") or "").strip()
    if not text:
        text = _fallback_title(item)
    if subtext:
        return f"{text}\n{subtext}"
    return text


def _wrap_title(text: str, max_chars: int, max_lines: int) -> str:
    raw_lines = [line.strip() for line in str(text or "").splitlines() if line.strip()]
    if not raw_lines:
        raw_lines = [str(text or "").strip()]

    lines: list[str] = []
    for raw_line in raw_lines:
        words = raw_line.split()
        if not words:
            continue
        current = ""
        for word in words:
            candidate = f"{current} {word}".strip()
            if len(candidate) <= max_chars or not current:
                current = candidate
            else:
                lines.append(current)
                current = word
                if len(lines) >= max_lines:
                    return "\n".join(lines)
        if current:
            lines.append(current)
            if len(lines) >= max_lines:
                return "\n".join(lines)
    return "\n".join(lines[:max_lines])


def _text_y_expression(text_position: str) -> str:
    if not THUMBNAIL_PLAN_USE_POSITION:
        return THUMBNAIL_Y_TOP
    value = str(text_position or THUMBNAIL_TEXT_POSITION).lower().strip()
    if value in {"center", "middle"}:
        return THUMBNAIL_Y_CENTER
    if value == "bottom":
        return THUMBNAIL_Y_BOTTOM
    return THUMBNAIL_Y_TOP


def _drawtext_filter(text_position: str = "top") -> str:
    textfile = TITLE_FILE.as_posix().replace(":", "\\:")
    if THUMBNAIL_FONT_FILE:
        font_path = Path(THUMBNAIL_FONT_FILE).as_posix().replace(":", "\\:")
        font_spec = f"fontfile='{font_path}'"
    else:
        font_spec = f"font='{THUMBNAIL_FONT}'"
    y_expr = _text_y_expression(text_position)
    return (
        f"drawtext={font_spec}:textfile='{textfile}':"
        f"x=(w-text_w)/2:y={y_expr}:"
        f"fontsize={THUMBNAIL_FONT_SIZE}:"
        f"fontcolor={THUMBNAIL_FONT_COLOR}:"
        "borderw=2:bordercolor=black@0.7:"
        f"box=1:boxcolor={THUMBNAIL_BOX_COLOR}:boxborderw={THUMBNAIL_BOX_BORDER}:"
        f"line_spacing={THUMBNAIL_LINE_SPACING}"
    )


def _resolve_thumbnail_image(plan: dict[str, Any]) -> Path:
    if not THUMBNAIL_PLAN_PICK_IMAGE:
        return THUMBNAIL_IMAGE_PATH
    try:
        index = int(plan.get("source_scene") or 1)
    except (TypeError, ValueError):
        index = 1
    index = max(index, 1)
    try:
        candidate = Path(THUMBNAIL_IMAGE_PATTERN.format(index=index, index0=index - 1))
    except Exception:
        candidate = THUMBNAIL_IMAGE_PATH
    if candidate.exists():
        return candidate
    return THUMBNAIL_IMAGE_PATH


def main() -> None:
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except Exception:
        pass

    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        raise RuntimeError("ffmpeg/ffprobe not found; cannot generate thumbnail.")
    if not FINAL_VIDEO.exists():
        raise RuntimeError(f"Final video not found: {FINAL_VIDEO}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    THUMBNAIL_PATH.parent.mkdir(parents=True, exist_ok=True)

    latest_content = _load_latest_content()
    thumbnail_plan = _build_thumbnail_plan(latest_content)

    title = _wrap_title(_plan_overlay_text(thumbnail_plan, latest_content), THUMBNAIL_MAX_CHARS, THUMBNAIL_MAX_LINES)
    TITLE_FILE.write_text(title, encoding="utf-8")

    drawtext = _drawtext_filter(str(thumbnail_plan.get("text_position") or THUMBNAIL_TEXT_POSITION))
    vf = (
        f"scale={WIDTH}:{HEIGHT}:force_original_aspect_ratio=decrease,"
        f"pad={WIDTH}:{HEIGHT}:(ow-iw)/2:(oh-ih)/2,"
        f"{drawtext}"
    )

    thumbnail_image_path = _resolve_thumbnail_image(thumbnail_plan)
    use_image = THUMBNAIL_SOURCE == "image" and thumbnail_image_path.exists()
    if use_image:
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(thumbnail_image_path),
            "-frames:v",
            "1",
            "-update",
            "1",
            "-vf",
            vf,
            "-q:v",
            "2",
            str(THUMBNAIL_PATH),
        ]
    else:
        duration = _ffprobe_duration(FINAL_VIDEO)
        timestamp = _pick_time(duration)
        cmd = [
            "ffmpeg",
            "-y",
            "-ss",
            f"{timestamp:.2f}",
            "-i",
            str(FINAL_VIDEO),
            "-frames:v",
            "1",
            "-update",
            "1",
            "-vf",
            vf,
            "-q:v",
            "2",
            str(THUMBNAIL_PATH),
        ]
    _run(cmd)
    print(f"Thumbnail saved: {THUMBNAIL_PATH}")


if __name__ == "__main__":
    main()
