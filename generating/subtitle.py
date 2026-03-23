import os
import re
import subprocess
import unicodedata
from pathlib import Path
from typing import List

from pydub import AudioSegment
from pydub.silence import detect_nonsilent


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "output"
OUTPUT_JSON = PROJECT_ROOT / "output.json"
CHUNKS_DIR = OUTPUT_DIR / "chunks"
FINAL_AUDIO = OUTPUT_DIR / "final.wav"

FINAL_VIDEO_NAME = os.getenv("FINAL_VIDEO_NAME", "final.mp4")
FINAL_VIDEO = OUTPUT_DIR / FINAL_VIDEO_NAME
ASS_PATH = OUTPUT_DIR / "subtitles.ass"
TEMP_VIDEO = OUTPUT_DIR / "final_subtitled.mp4"

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
SILENCE_MIN_MS = int(os.getenv("SUBTITLE_SILENCE_MIN_MS", "120"))
SILENCE_THRESH = int(os.getenv("SUBTITLE_SILENCE_THRESH", "-42"))


def _run(cmd: List[str]) -> None:
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


def _split_for_xtts(text: str, max_words: int = 40) -> List[str]:
    paragraphs = re.split(r"\n\s*\n", text)
    chunks = []
    for para in paragraphs:
        sentences = re.split(r"(?<=[.!?]) +", para)
        current = ""
        for s in sentences:
            if len((current + " " + s).split()) <= max_words:
                current += " " + s
            else:
                if current.strip():
                    chunks.append(current.strip())
                current = s
        if current.strip():
            chunks.append(current.strip())
    return chunks


def _clean_text(text: str) -> str:
    clean = unicodedata.normalize("NFKC", text)
    clean = re.sub(r"\[[^\]]*\]", "", clean)
    clean = re.sub(r"\b[A-Za-z ]+:\s*", "", clean)
    clean = clean.replace("\\n", "\n")
    clean = re.sub(r"[^A-Za-z0-9\s]", " ", clean)
    clean = re.sub(r"\s+", " ", clean).strip()
    return clean


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


def _allocate_words_to_segments(words: List[str], segments: List[tuple[int, int]]) -> List[List[str]]:
    if not words or not segments:
        return []
    total_words = len(words)
    seg_durations = [max(1, end - start) for start, end in segments]
    total_ns = sum(seg_durations)
    counts = [int(round(total_words * (d / total_ns))) for d in seg_durations]
    diff = total_words - sum(counts)
    if diff > 0:
        for i in range(diff):
            counts[i % len(counts)] += 1
    elif diff < 0:
        for i in range(-diff):
            idx = i % len(counts)
            if counts[idx] > 0:
                counts[idx] -= 1
    groups: List[List[str]] = []
    pos = 0
    for count in counts:
        groups.append(words[pos:pos + count])
        pos += count
    if pos < total_words:
        groups[-1].extend(words[pos:])
    return groups


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


def _write_ass(chunks: List[str], durations: List[float]) -> None:
    ASS_PATH.parent.mkdir(parents=True, exist_ok=True)
    header = f"""[Script Info]
Title: Generated Subtitles
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
    for chunk, duration in zip(chunks, durations):
        for line_text, line_duration in _split_chunk_lines(chunk, duration):
            start = _timecode(current)
            end = _timecode(current + line_duration)
            text = _timed_word_text(line_text, line_duration, WORDS_PER_SECOND)
            if text:
                text = f"{{\\pos({SUB_POS_X},{SUB_POS_Y})}}{text}"
                lines.append(f"Dialogue: 0,{start},{end},Base,,0,0,0,,{text}\n")
            current += line_duration

    ASS_PATH.write_text("".join(lines), encoding="utf-8")


def _burn_subtitles() -> None:
    if not FINAL_VIDEO.exists():
        raise RuntimeError(f"Final video not found: {FINAL_VIDEO}")
    if not ASS_PATH.exists():
        raise RuntimeError(f"Subtitle file not found: {ASS_PATH}")
    ass_path = ASS_PATH.as_posix().replace("'", "\\'")
    ass_path = ass_path.replace(":", "\\:")
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(FINAL_VIDEO),
        "-vf",
        f"ass=filename='{ass_path}'",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "copy",
        str(TEMP_VIDEO),
    ]
    _run(cmd)
    TEMP_VIDEO.replace(FINAL_VIDEO)


def main() -> None:
    if not CHUNKS_DIR.exists():
        raise RuntimeError(f"No audio chunks folder found: {CHUNKS_DIR}")

    chunk_files = sorted(CHUNKS_DIR.glob("*.wav"))
    use_fallback_timing = False
    if not chunk_files:
        if FINAL_AUDIO.exists():
            use_fallback_timing = True
        else:
            raise RuntimeError("No audio chunks found for subtitle timing.")

    import json

    with OUTPUT_JSON.open("r", encoding="utf-8") as f:
        items = json.load(f)
    if not isinstance(items, list) or not items:
        raise RuntimeError("output.json does not contain any items.")

    script_data = items[-1].get("script")
    if isinstance(script_data, list):
        raw_chunks = [str(x) for x in script_data if str(x).strip()]
        chunks = [_clean_text(chunk) for chunk in raw_chunks if _clean_text(chunk)]
    else:
        script = str(script_data or "")
        clean = _clean_text(script)
        chunks = _split_for_xtts(clean, 40)

    durations = []
    nonsilent_segments: List[List[tuple[int, int]]] = []
    if use_fallback_timing:
        total_duration = _ffprobe_duration(FINAL_AUDIO)
        word_counts = [max(1, len(chunk.split())) for chunk in chunks]
        total_words = sum(word_counts)
        if total_words <= 0:
            durations = [total_duration / max(len(chunks), 1)] * max(len(chunks), 1)
        else:
            durations = [total_duration * (count / total_words) for count in word_counts]
        nonsilent_segments = [[(0, int(duration * 1000))] for duration in durations]
    else:
        for path in chunk_files:
            audio = AudioSegment.from_wav(str(path))
            durations.append(len(audio) / 1000.0)
            segments = detect_nonsilent(audio, min_silence_len=SILENCE_MIN_MS, silence_thresh=SILENCE_THRESH)
            nonsilent_segments.append([(start, end) for start, end in segments])

    if len(chunks) != len(durations):
        if len(chunks) > len(durations):
            chunks = chunks[: len(durations)]
        else:
            last = chunks[-1] if chunks else ""
            chunks = chunks + [last] * (len(durations) - len(chunks))

    ASS_PATH.parent.mkdir(parents=True, exist_ok=True)
    header = f"""[Script Info]
Title: Generated Subtitles
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
    for chunk, duration, segments in zip(chunks, durations, nonsilent_segments):
        if not chunk.strip():
            current += duration
            continue

        words = chunk.split()
        if not words:
            current += duration
            continue

        if not segments:
            current += duration
            continue

        seg_words = _allocate_words_to_segments(words, segments)
        for (seg_start, seg_end), words_in_seg in zip(segments, seg_words):
            if not words_in_seg:
                continue
            seg_duration = max(0.01, (seg_end - seg_start) / 1000.0)
            line_groups = _group_words(words_in_seg, MIN_WORDS_PER_LINE, MAX_WORDS_PER_LINE)
            if not line_groups:
                continue
            counts = [len(g) for g in line_groups]
            total_words = sum(counts)
            seg_ms = max(1, int(round(seg_duration * 1000)))
            line_ms = [int(round(seg_ms * (c / total_words))) for c in counts]
            diff = seg_ms - sum(line_ms)
            if diff > 0:
                for i in range(diff):
                    line_ms[i % len(line_ms)] += 1
            elif diff < 0:
                for i in range(-diff):
                    idx = i % len(line_ms)
                    if line_ms[idx] > 1:
                        line_ms[idx] -= 1

            line_offset_ms = 0
            for group, ms in zip(line_groups, line_ms):
                line_text = " ".join(group)
                start = _timecode(current + (seg_start / 1000.0) + (line_offset_ms / 1000.0))
                end = _timecode(current + (seg_start / 1000.0) + ((line_offset_ms + ms) / 1000.0))
                timed = _timed_word_text(line_text, ms / 1000.0, WORDS_PER_SECOND)
                if timed:
                    text = f"{{\\pos({SUB_POS_X},{SUB_POS_Y})}}{timed}"
                    lines.append(f"Dialogue: 0,{start},{end},Base,,0,0,0,,{text}\n")
                line_offset_ms += ms

        current += duration

    ASS_PATH.write_text("".join(lines), encoding="utf-8")
    _burn_subtitles()


if __name__ == "__main__":
    main()
