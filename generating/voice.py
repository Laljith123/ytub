import json
import os
import re
import shutil
import subprocess
import unicodedata
from pathlib import Path

import torch
from TTS.api import TTS
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import XttsAudioConfig
from pydub import AudioSegment
from pydub.silence import detect_nonsilent

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "output"
CHUNKS_DIR = OUTPUT_DIR / "chunks"
OUTPUT_JSON = PROJECT_ROOT / "output.json"
VOICE_SAMPLE = PROJECT_ROOT / "voices" / "master.wav"
FINAL_WAV = OUTPUT_DIR / "final.wav"
VOICE_SPEED = float(os.getenv("VOICE_SPEED", "1.25"))

def split_for_xtts(text, max_words=40):
    paragraphs = re.split(r'\n\s*\n', text)
    chunks = []
    for para in paragraphs:
        sentences = re.split(r'(?<=[.!?]) +', para)
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


def _run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


def _atempo_chain(speed: float) -> str:
    parts = []
    rate = speed
    while rate > 2.0:
        parts.append("atempo=2.0")
        rate /= 2.0
    while rate < 0.5:
        parts.append("atempo=0.5")
        rate /= 0.5
    parts.append(f"atempo={rate:.5f}")
    return ",".join(parts)


def _speed_audio(path: Path, speed: float) -> None:
    if abs(speed - 1.0) < 1e-3:
        return
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg not found; cannot apply VOICE_SPEED.")
    temp_path = path.with_suffix(".speed.wav")
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(path),
        "-filter:a",
        _atempo_chain(speed),
        "-c:a",
        "pcm_s16le",
        str(temp_path),
    ]
    _run(cmd)
    temp_path.replace(path)


if hasattr(torch.serialization, "add_safe_globals"):
    torch.serialization.add_safe_globals([XttsConfig, XttsAudioConfig])

tts = TTS(
    model_name="tts_models/multilingual/multi-dataset/xtts_v2",
    gpu=False
)

content = json.load(open(OUTPUT_JSON, "r", encoding="utf-8"))

script_data = content[-1]["script"]


def _clean_text(text: str) -> str:
    clean = unicodedata.normalize("NFKC", text)
    clean = re.sub(r"\[[^\]]*\]", "", clean)
    clean = re.sub(r"\b[A-Za-z ]+:\s*", "", clean)
    clean = clean.replace("\\n", "\n")
    clean = re.sub(r"[^\w\s.,!?;:'\"()\-\n]", "", clean)
    clean = re.sub(r"\s+", " ", clean).strip()
    return clean


if isinstance(script_data, list):
    raw_chunks = [str(x) for x in script_data if str(x).strip()]
else:
    raw_chunks = [str(script_data)]

clean_chunks = [_clean_text(chunk) for chunk in raw_chunks]
clean_chunks = [chunk for chunk in clean_chunks if chunk]

if len(clean_chunks) == 1 and not isinstance(script_data, list):
    chunks = split_for_xtts(clean_chunks[0], 40)
else:
    chunks = clean_chunks

CHUNKS_DIR.mkdir(parents=True, exist_ok=True)

paths = []

for i, chunk in enumerate(chunks):
    path = CHUNKS_DIR / f"output_{i:03d}.wav"
    tts.tts_to_file(
        text=chunk,
        speaker_wav=str(VOICE_SAMPLE),
        language="en",
        file_path=str(path)
    )
    _speed_audio(path, VOICE_SPEED)
    paths.append(path)

combined = AudioSegment.empty()

for i, p in enumerate(paths):
    audio = AudioSegment.from_wav(str(p))
    nonsilent = detect_nonsilent(audio, min_silence_len=120, silence_thresh=-42)
    if nonsilent:
        start = nonsilent[0][0]
        end = nonsilent[-1][1]
        audio = audio[start:end]
    if i == 0:
        combined = audio
    else:
        combined = combined.append(audio, crossfade=180)

combined = combined.normalize()
combined.export(str(FINAL_WAV), format="wav")

if os.getenv("VOICE_CLEANUP", "1") == "1":
    for p in paths:
        try:
            p.unlink()
        except OSError as exc:
            print(f"Unable to delete {p}: {exc}")
