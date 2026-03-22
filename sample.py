import os
import re
import subprocess
import unicodedata
from pathlib import Path
from typing import List


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "output"
OUTPUT_JSON = ROOT / "output.json"

ASS_PATH = OUTPUT_DIR / "sample_subtitles.ass"
SAMPLE_VIDEO = OUTPUT_DIR / "sample_demo.mp4"

PLAY_RES_X = int(os.getenv("SUBTITLE_RES_X", "1080"))
PLAY_RES_Y = int(os.getenv("SUBTITLE_RES_Y", "1920"))
SUB_POS_X = int(os.getenv("SUBTITLE_POS_X", str(PLAY_RES_X // 2)))
SUB_POS_Y = int(os.getenv("SUBTITLE_POS_Y", str(PLAY_RES_Y // 2)))
SUB_FONT = os.getenv("SUBTITLE_FONT", "Arial")
SUB_FONT_SIZE = int(os.getenv("SUBTITLE_FONT_SIZE", "54"))
HIGHLIGHT_SIZE = int(os.getenv("SUBTITLE_HIGHLIGHT_SIZE", str(SUB_FONT_SIZE + 12)))
WORDS_PER_SECOND = float(os.getenv("SUBTITLE_WORDS_PER_SECOND", "5"))
HIGHLIGHT_COLOR = os.getenv("SUBTITLE_HIGHLIGHT_COLOR", "&H0000FFFF")
MIN_WORDS_PER_LINE = int(os.getenv("SUBTITLE_MIN_WORDS_PER_LINE", "5"))
MAX_WORDS_PER_LINE = int(os.getenv("SUBTITLE_MAX_WORDS_PER_LINE", "6"))


def _run(cmd: List[str]) -> None:
    subprocess.run(cmd, check=True)


def _clean_text(text: str) -> str:
    clean = unicodedata.normalize("NFKC", text)
    clean = re.sub(r"\[[^\]]*\]", "", clean)
    clean = re.sub(r"\b[A-Za-z ]+:\s*", "", clean)
    clean = clean.replace("\\n", "\n")
    clean = re.sub(r"[^A-Za-z0-9\s]", " ", clean)
    clean = re.sub(r"\s+", " ", clean).strip()
    return clean


def _split_sentences(text: str) -> List[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def _timecode(seconds: float) -> str:
    total_cs = int(round(seconds * 100))
    cs = total_cs % 100
    total_s = total_cs // 100
    s = total_s % 60
    total_m = total_s // 60
    m = total_m % 60
    h = total_m // 60
    return f"{h:d}:{m:02d}:{s:02d}.{cs:02d}"


def _word_durations_ms(words: int, total_seconds: float, words_per_second: float) -> List[int]:
    if words <= 0:
        return []
    total_ms = max(1, int(round(total_seconds * 1000)))
    target_word_ms = max(1, int(round(1000 / max(words_per_second, 0.1))))
    target_total = target_word_ms * words
    scale = total_ms / target_total if target_total else 1.0
    base = max(1, int(round(target_word_ms * scale)))
    durations = [base] * words
    diff = total_ms - sum(durations)
    if diff > 0:
        for i in range(diff):
            durations[i % words] += 1
    elif diff < 0:
        for i in range(-diff):
            idx = i % words
            if durations[idx] > 1:
                durations[idx] -= 1
    return durations


def _group_words(words: List[str], min_words: int, max_words: int) -> List[List[str]]:
    if not words:
        return []
    if len(words) <= max_words:
        return [words]
    groups = []
    i = 0
    while i < len(words):
        remaining = len(words) - i
        if remaining <= max_words:
            groups.append(words[i:])
            break
        groups.append(words[i:i + max_words])
        i += max_words
    if len(groups) > 1 and len(groups[-1]) < min_words:
        deficit = min_words - len(groups[-1])
        idx = len(groups) - 2
        while deficit > 0 and idx >= 0:
            while deficit > 0 and len(groups[idx]) > min_words:
                groups[-1].insert(0, groups[idx].pop())
                deficit -= 1
            idx -= 1
    return groups


def _split_chunk_lines(text: str, duration: float) -> List[tuple[str, float]]:
    words = text.split()
    groups = _group_words(words, MIN_WORDS_PER_LINE, MAX_WORDS_PER_LINE)
    if not groups:
        return []
    if len(groups) == 1:
        return [(" ".join(groups[0]), duration)]
    counts = [len(g) for g in groups]
    total_words = sum(counts)
    total_ms = max(1, int(round(duration * 1000)))
    line_ms = [int(round(total_ms * (c / total_words))) for c in counts]
    diff = total_ms - sum(line_ms)
    if diff > 0:
        for i in range(diff):
            line_ms[i % len(line_ms)] += 1
    elif diff < 0:
        for i in range(-diff):
            idx = i % len(line_ms)
            if line_ms[idx] > 1:
                line_ms[idx] -= 1
    return [(" ".join(g), ms / 1000.0) for g, ms in zip(groups, line_ms)]


def _timed_word_text(text: str, duration: float, words_per_second: float) -> str:
    words = re.findall(r"\S+", text)
    if not words:
        return ""
    durations = _word_durations_ms(len(words), duration, words_per_second)
    parts = []
    current_ms = 0
    for word, dur_ms in zip(words, durations):
        start = current_ms
        end = current_ms + dur_ms
        parts.append(
            "{"
            f"\\fs{SUB_FONT_SIZE}"
            "\\c&H00FFFFFF&"
            f"\\t({start},{start},\\fs{HIGHLIGHT_SIZE}\\c{HIGHLIGHT_COLOR}&)"
            f"\\t({end},{end},\\fs{SUB_FONT_SIZE}\\c&H00FFFFFF&)"
            "}"
            f"{word}"
        )
        current_ms = end
    return " ".join(parts)


def _write_ass(sentences: List[str], durations: List[float]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    header = f"""[Script Info]
Title: Subtitle Demo
ScriptType: v4.00+
PlayResX: {PLAY_RES_X}
PlayResY: {PLAY_RES_Y}
WrapStyle: 2
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Base,{SUB_FONT},{SUB_FONT_SIZE},&H00FFFFFF,&H00FFFFFF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,4,0,5,0,0,0,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""

    lines = [header]
    current = 0.0
    for sentence, duration in zip(sentences, durations):
        for line_text, line_duration in _split_chunk_lines(sentence, duration):
            start = _timecode(current)
            end = _timecode(current + line_duration)
            timed = _timed_word_text(line_text, line_duration, WORDS_PER_SECOND)
            text = f"{{\\pos({SUB_POS_X},{SUB_POS_Y})}}{timed}"
            lines.append(f"Dialogue: 0,{start},{end},Base,,0,0,0,,{text}\n")
            current += line_duration

    ASS_PATH.write_text("".join(lines), encoding="utf-8")


def _find_sample_image() -> Path | None:
    image_dir = OUTPUT_DIR / "image"
    if image_dir.exists():
        for pattern in ("*.png", "*.jpg", "*.jpeg"):
            images = sorted(image_dir.glob(pattern))
            if images:
                return images[0]
    return None


def _make_demo_video(total_duration: float) -> None:
    ass_path = ASS_PATH.as_posix().replace("'", "\\'").replace(":", "\\:")
    image = _find_sample_image()
    if image:
        vf = (
            f"scale={PLAY_RES_X}:{PLAY_RES_Y}:force_original_aspect_ratio=increase,"
            f"crop={PLAY_RES_X}:{PLAY_RES_Y},"
            f"ass=filename='{ass_path}'"
        )
        cmd = [
            "ffmpeg",
            "-y",
            "-loop",
            "1",
            "-i",
            str(image),
            "-t",
            f"{total_duration:.2f}",
            "-vf",
            vf,
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(SAMPLE_VIDEO),
        ]
    else:
        cmd = [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"color=c=black:s={PLAY_RES_X}x{PLAY_RES_Y}:d={total_duration:.2f}",
            "-vf",
            f"ass=filename='{ass_path}'",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(SAMPLE_VIDEO),
        ]
    _run(cmd)


def main() -> None:
    if not OUTPUT_JSON.exists():
        raise RuntimeError("output.json not found. Run the pipeline once first.")

    import json

    with OUTPUT_JSON.open("r", encoding="utf-8") as f:
        items = json.load(f)
    if not isinstance(items, list) or not items:
        raise RuntimeError("output.json does not contain any items.")

    script_data = items[-1].get("script")
    if isinstance(script_data, list):
        script_text = " ".join(str(x) for x in script_data if str(x).strip())
    else:
        script_text = str(script_data or "")

    clean = _clean_text(script_text)
    sentences = _split_sentences(clean)[:2]
    if len(sentences) < 2:
        sentences = (sentences + [clean])[:2]

    durations = []
    for sentence in sentences:
        words = max(1, len(sentence.split()))
        seconds = max(2.0, words / max(WORDS_PER_SECOND, 0.1))
        durations.append(seconds)

    _write_ass(sentences, durations)

    try:
        _make_demo_video(sum(durations))
        print(f"Demo video saved: {SAMPLE_VIDEO}")
    except FileNotFoundError:
        print("ffmpeg not found. Subtitle file only was generated.")

    print(f"Subtitle file saved: {ASS_PATH}")


if __name__ == "__main__":
    main()
