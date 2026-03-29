import json
import os
import shutil
import subprocess
from pathlib import Path


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
THUMBNAIL_FONT = os.getenv("THUMBNAIL_FONT", "Arial")
THUMBNAIL_FONT_FILE = os.getenv("THUMBNAIL_FONT_FILE", "")
THUMBNAIL_FONT_SIZE = int(os.getenv("THUMBNAIL_FONT_SIZE", "86"))
THUMBNAIL_FONT_COLOR = os.getenv("THUMBNAIL_FONT_COLOR", "white")
THUMBNAIL_BOX_COLOR = os.getenv("THUMBNAIL_BOX_COLOR", "black@0.6")
THUMBNAIL_BOX_BORDER = int(os.getenv("THUMBNAIL_BOX_BORDER", "22"))
THUMBNAIL_LINE_SPACING = int(os.getenv("THUMBNAIL_LINE_SPACING", "8"))
THUMBNAIL_MAX_CHARS = int(os.getenv("THUMBNAIL_MAX_CHARS", "18"))
THUMBNAIL_MAX_LINES = int(os.getenv("THUMBNAIL_MAX_LINES", "3"))
TITLE_FILE = OUTPUT_DIR / "thumbnail_title.txt"

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


def _load_title() -> str:
    output_json = PROJECT_ROOT / "output.json"
    if output_json.exists():
        try:
            data = json.loads(output_json.read_text(encoding="utf-8"))
            if isinstance(data, list) and data:
                last = data[-1]
                if isinstance(last, dict):
                    title = str(last.get("title") or "").strip()
                    if title:
                        return title
        except json.JSONDecodeError:
            pass
    return "Breaking Mystery"


def _wrap_title(text: str, max_chars: int, max_lines: int) -> str:
    words = text.split()
    if not words:
        return text
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if len(candidate) <= max_chars or not current:
            current = candidate
        else:
            lines.append(current)
            current = word
            if len(lines) >= max_lines:
                break
    if current and len(lines) < max_lines:
        lines.append(current)
    return "\n".join(lines)


def _drawtext_filter() -> str:
    textfile = TITLE_FILE.as_posix().replace(":", "\\:")
    if THUMBNAIL_FONT_FILE:
        font_path = Path(THUMBNAIL_FONT_FILE).as_posix().replace(":", "\\:")
        font_spec = f"fontfile='{font_path}'"
    else:
        font_spec = f"font='{THUMBNAIL_FONT}'"
    return (
        f"drawtext={font_spec}:textfile='{textfile}':"
        "x=(w-text_w)/2:y=(h*0.08):"
        f"fontsize={THUMBNAIL_FONT_SIZE}:"
        f"fontcolor={THUMBNAIL_FONT_COLOR}:"
        "borderw=2:bordercolor=black@0.7:"
        f"box=1:boxcolor={THUMBNAIL_BOX_COLOR}:boxborderw={THUMBNAIL_BOX_BORDER}:"
        f"line_spacing={THUMBNAIL_LINE_SPACING}"
    )


def main() -> None:
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        raise RuntimeError("ffmpeg/ffprobe not found; cannot generate thumbnail.")
    if not FINAL_VIDEO.exists():
        raise RuntimeError(f"Final video not found: {FINAL_VIDEO}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    THUMBNAIL_PATH.parent.mkdir(parents=True, exist_ok=True)

    title = _wrap_title(_load_title(), THUMBNAIL_MAX_CHARS, THUMBNAIL_MAX_LINES)
    TITLE_FILE.write_text(title, encoding="utf-8")

    drawtext = _drawtext_filter()
    vf = (
        f"scale={WIDTH}:{HEIGHT}:force_original_aspect_ratio=decrease,"
        f"pad={WIDTH}:{HEIGHT}:(ow-iw)/2:(oh-ih)/2,"
        f"{drawtext}"
    )

    use_image = THUMBNAIL_SOURCE == "image" and THUMBNAIL_IMAGE_PATH.exists()
    if use_image:
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(THUMBNAIL_IMAGE_PATH),
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
