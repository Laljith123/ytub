import math
import os
import random
import re
import shutil
import subprocess
from pathlib import Path
from typing import Iterable, List


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "output"
IMAGES_DIR = OUTPUT_DIR / "image"
AUDIO_DIR = OUTPUT_DIR / "chunks"
VIDEO_DIR = OUTPUT_DIR / "video"

AUDIO_FFCONCAT = OUTPUT_DIR / "_audio.ffconcat"
VIDEO_FFCONCAT = VIDEO_DIR / "_video.ffconcat"

FINAL_AUDIO = OUTPUT_DIR / "final_audio.wav"
FINAL_VIDEO_SILENT = VIDEO_DIR / "final_silent.mp4"
FINAL_VIDEO_NAME = os.getenv("FINAL_VIDEO_NAME", "final.mp4")
FINAL_VIDEO = OUTPUT_DIR / FINAL_VIDEO_NAME

DEFAULT_WIDTH = int(os.getenv("VIDEO_WIDTH", "1080"))
DEFAULT_HEIGHT = int(os.getenv("VIDEO_HEIGHT", "1920"))
DEFAULT_FPS = int(os.getenv("VIDEO_FPS", "60"))
DEFAULT_CLIP_SECONDS = float(os.getenv("VIDEO_CLIP_SECONDS", "5"))
DEFAULT_TRANSITION_SECONDS = float(os.getenv("VIDEO_TRANSITION_SECONDS", "0.5"))
DEFAULT_TRANSITION_MODE = os.getenv("VIDEO_TRANSITION_MODE", "random").lower()
DEFAULT_TRANSITIONS = os.getenv(
    "VIDEO_TRANSITIONS",
    "fade,slideleft,slideright,slideup,slidedown,wipeleft,wiperight,wipeup,wipedown,rectcrop,circlecrop,radial,fadeblack,fadewhite"
)
DEFAULT_TRANSITION_SEED = int(os.getenv("VIDEO_TRANSITION_SEED", "42"))
DEFAULT_MIN_SECONDS = float(os.getenv("VIDEO_MIN_SECONDS", "40"))
DEFAULT_MAX_SECONDS = float(os.getenv("VIDEO_MAX_SECONDS", "50"))


def _require_tool(name: str) -> None:
    if shutil.which(name) is None:
        raise RuntimeError(f"{name} not found in PATH. Please install ffmpeg and ffprobe.")


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


def _natural_sort_key(path: Path) -> tuple:
    name = path.name
    match = re.search(r"(\d+)", name)
    return (int(match.group(1)) if match else 0, name)


def _list_files(path: Path, pattern: str) -> List[Path]:
    return sorted(path.glob(pattern), key=_natural_sort_key)


def _ffconcat_path(path: Path) -> str:
    return str(path.resolve()).replace("\\", "/")


def _write_ffconcat(paths: Iterable[Path], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        f.write("ffconcat version 1.0\n")
        for p in paths:
            f.write(f"file '{_ffconcat_path(p)}'\n")


def _build_segments(
    images: List[Path],
    durations: List[float],
    *,
    width: int,
    height: int,
    fps: int,
) -> List[Path]:
    VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    segments: List[Path] = []

    for idx, (image, duration) in enumerate(zip(images, durations), start=1):
        frames = max(1, int(math.ceil(duration * fps)))
        zoom_in = idx % 2 == 1
        z_start = 1.00 if zoom_in else 1.04
        z_end = 1.04 if zoom_in else 1.00
        denom = max(frames - 1, 1)
        z_expr = (
            f"{z_start}+({z_end}-{z_start})*(1-cos(PI*on/{denom}))/2"
        )

        x_expr = "iw/2-(iw/zoom/2)"
        y_expr = "ih/2-(ih/zoom/2)"

        vf = (
            f"scale={width}:{height}:force_original_aspect_ratio=increase,"
            f"crop={width}:{height},"
            "setsar=1,"
            f"zoompan=z='{z_expr}':x='{x_expr}':y='{y_expr}':d={frames}:s={width}x{height}:fps={fps},"
            "format=yuv420p"
        )

        seg_path = VIDEO_DIR / f"segment_{idx:03d}.mp4"
        cmd = [
            "ffmpeg",
            "-y",
            "-loop",
            "1",
            "-framerate",
            str(fps),
            "-i",
            str(image),
            "-t",
            f"{duration:.3f}",
            "-vf",
            vf,
            "-r",
            str(fps),
            "-pix_fmt",
            "yuv420p",
            "-c:v",
            "libx264",
            "-profile:v",
            "high",
            "-level",
            "4.1",
            str(seg_path),
        ]
        _run(cmd)
        segments.append(seg_path)

    return segments


def _concat_video(segments: List[Path], out_path: Path) -> None:
    _write_ffconcat(segments, VIDEO_FFCONCAT)
    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(VIDEO_FFCONCAT),
        "-c",
        "copy",
        str(out_path),
    ]
    try:
        _run(cmd)
    except subprocess.CalledProcessError:
        cmd = [
            "ffmpeg",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(VIDEO_FFCONCAT),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(out_path),
        ]
        _run(cmd)


def _concat_audio(audio_files: List[Path], out_path: Path) -> None:
    _write_ffconcat(audio_files, AUDIO_FFCONCAT)
    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(AUDIO_FFCONCAT),
        "-c",
        "copy",
        str(out_path),
    ]
    _run(cmd)


def _trim_audio(audio_path: Path, target_seconds: float) -> None:
    temp_path = audio_path.with_suffix(".trim.wav")
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(audio_path),
        "-t",
        f"{target_seconds:.3f}",
        "-c",
        "copy",
        str(temp_path),
    ]
    try:
        _run(cmd)
    except subprocess.CalledProcessError:
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(audio_path),
            "-t",
            f"{target_seconds:.3f}",
            "-c:a",
            "pcm_s16le",
            str(temp_path),
        ]
        _run(cmd)
    temp_path.replace(audio_path)


def _repeat_audio_to_min(
    audio_files: List[Path],
    durations: List[float],
    min_seconds: float,
) -> tuple[List[Path], List[float]]:
    total = sum(durations)
    if total >= min_seconds or not audio_files:
        return audio_files, durations
    base_files = list(audio_files)
    base_durations = list(durations)
    idx = 0
    while total < min_seconds:
        audio_files.append(base_files[idx % len(base_files)])
        durations.append(base_durations[idx % len(base_durations)])
        total += base_durations[idx % len(base_durations)]
        idx += 1
    return audio_files, durations


def _truncate_durations(durations: List[float], max_seconds: float) -> List[float]:
    trimmed: List[float] = []
    total = 0.0
    for dur in durations:
        if total >= max_seconds:
            break
        remaining = max_seconds - total
        if dur > remaining:
            trimmed.append(remaining)
            total = max_seconds
            break
        trimmed.append(dur)
        total += dur
    return trimmed


def _pick_transitions(count: int) -> List[str]:
    options = [t.strip() for t in DEFAULT_TRANSITIONS.split(",") if t.strip()]
    if not options:
        options = ["fade"]
    rng = random.Random(DEFAULT_TRANSITION_SEED)
    return [rng.choice(options) for _ in range(count)]


def _xfade_video(
    segments: List[Path],
    durations: List[float],
    out_path: Path,
    *,
    fps: int,
    transition_seconds: float,
) -> None:
    if len(segments) <= 1:
        _concat_video(segments, out_path)
        return

    transitions = _pick_transitions(len(segments) - 1)
    inputs: List[str] = []
    for seg in segments:
        inputs.extend(["-i", str(seg)])

    filter_lines: List[str] = []
    for i in range(len(segments)):
        filter_lines.append(
            f"[{i}:v]settb=1/{fps},setpts=PTS-STARTPTS,fps={fps},format=yuv420p[v{i}]"
        )

    current_label = "v0"
    current_duration = durations[0]
    for i in range(1, len(segments)):
        transition = transitions[i - 1]
        offset = max(0.0, current_duration - transition_seconds)
        next_label = f"v{i}x"
        filter_lines.append(
            f"[{current_label}][v{i}]xfade=transition={transition}:"
            f"duration={transition_seconds:.3f}:offset={offset:.3f}[{next_label}]"
        )
        current_duration = current_duration + durations[i] - transition_seconds
        current_label = next_label

    filter_complex = ";".join(filter_lines)
    cmd = [
        "ffmpeg",
        "-y",
        *inputs,
        "-filter_complex",
        filter_complex,
        "-map",
        f"[{current_label}]",
        "-r",
        str(fps),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        str(out_path),
    ]
    _run(cmd)


def _merge_audio_video(video_path: Path, audio_path: Path, out_path: Path) -> None:
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-i",
        str(audio_path),
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-shortest",
        str(out_path),
    ]
    _run(cmd)


def _cleanup_video_artifacts(segments: List[Path]) -> None:
    if os.getenv("VIDEO_CLEANUP", "1") != "1":
        return
    to_delete = [VIDEO_FFCONCAT, FINAL_VIDEO_SILENT, FINAL_AUDIO, AUDIO_FFCONCAT]
    to_delete.extend(segments)
    for path in to_delete:
        try:
            if path.exists():
                path.unlink()
        except OSError as exc:
            print(f"Unable to delete {path}: {exc}")


def main() -> None:
    _require_tool("ffmpeg")
    _require_tool("ffprobe")

    images = _list_files(IMAGES_DIR, "*.png")
    if not images:
        raise RuntimeError(f"No images found in {IMAGES_DIR}")

    audio_files = _list_files(AUDIO_DIR, "*.wav")
    if not audio_files:
        raise RuntimeError(f"No audio chunks found in {AUDIO_DIR}")

    durations = [_ffprobe_duration(p) for p in audio_files]
    total_audio = sum(durations)

    min_seconds = DEFAULT_MIN_SECONDS
    max_seconds = DEFAULT_MAX_SECONDS
    if max_seconds < min_seconds:
        max_seconds = min_seconds

    target_seconds = total_audio
    if total_audio < min_seconds:
        target_seconds = min_seconds
    elif total_audio > max_seconds:
        target_seconds = max_seconds

    if total_audio < min_seconds:
        audio_files, durations = _repeat_audio_to_min(audio_files, durations, min_seconds)
        total_audio = sum(durations)

    if total_audio < target_seconds:
        target_seconds = total_audio
    if total_audio > target_seconds:
        trim_to_seconds = target_seconds
    else:
        trim_to_seconds = None

    if DEFAULT_CLIP_SECONDS > 0:
        clip_seconds = DEFAULT_CLIP_SECONDS
        clips = max(1, int(math.ceil(target_seconds / clip_seconds)))
        durations = [clip_seconds] * clips
        durations[-1] = max(0.1, target_seconds - clip_seconds * (clips - 1))
        if len(images) < clips:
            images = [images[i % len(images)] for i in range(clips)]
        elif len(images) > clips:
            images = images[:clips]
        print(f"Using fixed clip length: {clip_seconds:.1f}s ({clips} clips).")
    else:
        if trim_to_seconds is not None:
            durations = _truncate_durations(durations, trim_to_seconds)
        if len(images) < len(durations):
            print(
                f"Only {len(images)} images for {len(durations)} audio chunks. "
                "Reusing the last image for remaining chunks."
            )
            last = images[-1]
            images = images + [last] * (len(durations) - len(images))
        elif len(images) > len(durations):
            print(
                f"Found {len(images)} images but only {len(durations)} audio chunks. "
                "Extra images will be ignored."
            )
            images = images[: len(durations)]

    transition_enabled = (
        DEFAULT_TRANSITION_MODE != "none"
        and DEFAULT_TRANSITION_SECONDS > 0
        and len(durations) > 1
    )

    segment_durations = list(durations)
    if transition_enabled:
        for i in range(len(segment_durations) - 1):
            segment_durations[i] += DEFAULT_TRANSITION_SECONDS

    segments = _build_segments(
        images,
        segment_durations,
        width=DEFAULT_WIDTH,
        height=DEFAULT_HEIGHT,
        fps=DEFAULT_FPS,
    )

    if transition_enabled:
        _xfade_video(
            segments,
            segment_durations,
            FINAL_VIDEO_SILENT,
            fps=DEFAULT_FPS,
            transition_seconds=DEFAULT_TRANSITION_SECONDS,
        )
    else:
        _concat_video(segments, FINAL_VIDEO_SILENT)
    _concat_audio(audio_files, FINAL_AUDIO)
    if trim_to_seconds is not None:
        _trim_audio(FINAL_AUDIO, trim_to_seconds)
    _merge_audio_video(FINAL_VIDEO_SILENT, FINAL_AUDIO, FINAL_VIDEO)
    _cleanup_video_artifacts(segments)

    print(f"Saved final video: {FINAL_VIDEO}")


if __name__ == "__main__":
    main()
