import json
import os
import re
import shutil
import subprocess
import unicodedata
from pathlib import Path

from pydub import AudioSegment
from pydub.silence import detect_nonsilent

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "output"
CHUNKS_DIR = OUTPUT_DIR / "chunks"
OUTPUT_JSON = PROJECT_ROOT / "output.json"
FINAL_WAV = OUTPUT_DIR / "final.wav"

VOICE_SPEED = float(os.getenv("VOICE_SPEED", "1.25"))
VOICE_GAIN_DB = float(os.getenv("VOICE_GAIN_DB", "0"))

RIVA_VOICE = os.getenv("RIVA_VOICE", "Magpie-Multilingual.EN-US.Aria")
RIVA_LANGUAGE = os.getenv("RIVA_LANGUAGE", "en-US")
RIVA_URI = os.getenv("RIVA_URI", "grpc.nvcf.nvidia.com:443")
RIVA_FUNCTION_ID = os.getenv("RIVA_FUNCTION_ID", "877104f7-e885-42b9-8de8-f6e4c6303969")
RIVA_SAMPLE_RATE_HZ = int(os.getenv("RIVA_SAMPLE_RATE_HZ", "44100"))
RIVA_USE_SSL = os.getenv("RIVA_USE_SSL", "1") == "1"
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")

VOICE_PLAN_ENABLED = os.getenv("VOICE_PLAN_ENABLED", "1") == "1"
VOICE_PLAN_MODEL = os.getenv("VOICE_PLAN_MODEL", os.getenv("CONTENT_MODEL", "openai/gpt-oss-120b"))
VOICE_PLAN_BASE_URL = os.getenv("VOICE_PLAN_BASE_URL", os.getenv("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1"))
VOICE_PLAN_MAX_ATTEMPTS = int(os.getenv("VOICE_PLAN_MAX_ATTEMPTS", "3"))
VOICE_PLAN_MAX_TOKENS = int(os.getenv("VOICE_PLAN_MAX_TOKENS", "4096"))
VOICE_PLAN_TEMPERATURE = float(os.getenv("VOICE_PLAN_TEMPERATURE", "0.35"))
VOICE_PLAN_TOP_P = float(os.getenv("VOICE_PLAN_TOP_P", "0.9"))
VOICE_PLAN_ENABLE_THINKING = os.getenv("VOICE_PLAN_ENABLE_THINKING", "1") == "1"
VOICE_PLAN_REASONING_BUDGET = int(os.getenv("VOICE_PLAN_REASONING_BUDGET", "4096"))
VOICE_PLAN_MIN_SPEED_MULTIPLIER = float(os.getenv("VOICE_PLAN_MIN_SPEED_MULTIPLIER", "0.82"))
VOICE_PLAN_MAX_SPEED_MULTIPLIER = float(os.getenv("VOICE_PLAN_MAX_SPEED_MULTIPLIER", "1.12"))
VOICE_PLAN_MIN_GAIN_DB = float(os.getenv("VOICE_PLAN_MIN_GAIN_DB", "-2.5"))
VOICE_PLAN_MAX_GAIN_DB = float(os.getenv("VOICE_PLAN_MAX_GAIN_DB", "2.5"))
VOICE_PLAN_MIN_PAUSE_MS = int(os.getenv("VOICE_PLAN_MIN_PAUSE_MS", "0"))
VOICE_PLAN_MAX_PAUSE_MS = int(os.getenv("VOICE_PLAN_MAX_PAUSE_MS", "700"))
VOICE_PLAN_ALLOWED_SFX = {
    item.strip().lower()
    for item in os.getenv("VOICE_PLAN_ALLOWED_SFX", "none,bass_hit,subtle_riser,tape_glitch,camera_click,silence_drop").split(",")
    if item.strip()
}

VOICE_SFX_ENABLED = os.getenv("VOICE_SFX_ENABLED", "0") == "1"
VOICE_SFX_DIR = Path(os.getenv("VOICE_SFX_DIR", str(PROJECT_ROOT / "sfx")))
VOICE_SFX_EXT = os.getenv("VOICE_SFX_EXT", ".wav")
VOICE_SFX_GAIN_DB = float(os.getenv("VOICE_SFX_GAIN_DB", "-8"))
VOICE_SFX_CROSSFADE_MS = int(os.getenv("VOICE_SFX_CROSSFADE_MS", "80"))


def split_text(text, max_words=40):
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


def _clean_text(text: str) -> str:
    clean = unicodedata.normalize("NFKC", text)
    clean = re.sub(r"\[[^\]]*\]", "", clean)
    clean = re.sub(r"\b[A-Za-z ]+:\s*", "", clean)
    clean = clean.replace("\\n", "\n")
    clean = re.sub(r"[^\w\s.,!?;:'\"()\-\n]", "", clean)
    clean = re.sub(r"\s+", " ", clean).strip()
    return clean


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
    lines = text.splitlines()
    filtered: list[str] = []
    for line in lines:
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


def _parse_json_object(text: str) -> dict:
    if not text or not text.strip():
        return {}
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        candidate = _extract_json_object(text)
        if not candidate:
            return {}
        try:
            parsed = json.loads(candidate)
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            return {}


def _clamp_float(value, default: float, low: float, high: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = default
    return max(low, min(high, number))


def _clamp_int(value, default: int, low: int, high: int) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError):
        number = default
    return max(low, min(high, number))


def _safe_ascii_tag(value: str, default: str = "none") -> str:
    tag = str(value or default).strip().lower()
    tag = re.sub(r"[^a-z0-9_\-]+", "_", tag).strip("_")
    if not tag:
        tag = default
    if VOICE_PLAN_ALLOWED_SFX and tag not in VOICE_PLAN_ALLOWED_SFX:
        return default
    return tag


def _clean_short_text(value: str, max_chars: int = 80) -> str:
    text = str(value or "").strip()
    text = re.sub(r"\s+", " ", text)
    return text[:max_chars]


def _default_voice_plan(chunks: list[str]) -> list[dict]:
    return [
        {
            "index": i + 1,
            "text": chunk,
            "delivery": "clean serious narration",
            "emotion": "controlled suspense",
            "speed_multiplier": 1.0,
            "gain_db": 0.0,
            "pause_after_ms": 0,
            "sfx": "none",
            "sfx_timing": "none",
        }
        for i, chunk in enumerate(chunks)
    ]


def _video_context(video_data: dict, chunks: list[str]) -> dict:
    keys = [
        "title",
        "hook",
        "caption",
        "thumbnail_text",
        "background_music",
        "retention_triggers",
        "trend",
        "image",
    ]
    context = {key: video_data.get(key) for key in keys if key in video_data}
    context["script"] = chunks
    return context


def _build_voice_plan_prompt(video_data: dict, chunks: list[str]) -> str:
    context = _video_context(video_data, chunks)
    return (
        "You are a professional voice director for short-form true-crime narration. "
        "Create ONLY an in-memory voice direction plan for RIVA text-to-speech. "
        "Do NOT rewrite, paraphrase, summarize, translate, censor, or add new narration. "
        "Do NOT add slang to the script. Use respectful suspense, not comedy. "
        "Return ONE valid JSON object ONLY with this exact shape: "
        "{\"scenes\":[{\"index\":1,\"delivery\":\"...\",\"emotion\":\"...\","
        "\"speed_multiplier\":1.0,\"gain_db\":0.0,\"pause_after_ms\":0,"
        "\"sfx\":\"none\",\"sfx_timing\":\"none\"}]}. "
        f"There must be exactly {len(chunks)} scene objects, one for each script line. "
        "index must start at 1 and increase by 1. "
        f"speed_multiplier must be between {VOICE_PLAN_MIN_SPEED_MULTIPLIER} and {VOICE_PLAN_MAX_SPEED_MULTIPLIER}. "
        f"gain_db must be between {VOICE_PLAN_MIN_GAIN_DB} and {VOICE_PLAN_MAX_GAIN_DB}. "
        f"pause_after_ms must be between {VOICE_PLAN_MIN_PAUSE_MS} and {VOICE_PLAN_MAX_PAUSE_MS}. "
        f"sfx must be one of: {sorted(VOICE_PLAN_ALLOWED_SFX)}. "
        "sfx_timing must be one of: none, start, end. "
        "Use faster energy for hooks, calmer pacing for facts, slower pacing for reveals, "
        "and a longer pause after the final question. "
        "Output JSON only. No markdown. No analysis. No commentary. "
        "Video content to direct: "
        f"{json.dumps(context, ensure_ascii=False)}"
    )


def _run_voice_plan_completion(prompt: str) -> str:
    if not NVIDIA_API_KEY:
        raise RuntimeError("NVIDIA_API_KEY is missing.")

    from openai import OpenAI

    client = OpenAI(
        base_url=VOICE_PLAN_BASE_URL,
        api_key=NVIDIA_API_KEY,
    )

    completion = client.chat.completions.create(
        model=VOICE_PLAN_MODEL,
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
        temperature=VOICE_PLAN_TEMPERATURE,
        top_p=VOICE_PLAN_TOP_P,
        max_tokens=VOICE_PLAN_MAX_TOKENS,
        extra_body={
            "chat_template_kwargs": {"enable_thinking": VOICE_PLAN_ENABLE_THINKING},
            "reasoning_budget": VOICE_PLAN_REASONING_BUDGET if VOICE_PLAN_ENABLE_THINKING else 0,
        },
        stop=["\n[Reasoning]", "\nReasoning:", "\n[Analysis]", "\nAnalysis:"],
    )

    return (completion.choices[0].message.content or "").strip()


def _coerce_voice_plan(data: dict, chunks: list[str]) -> list[dict] | None:
    scenes = data.get("scenes") if isinstance(data, dict) else None
    if not isinstance(scenes, list):
        return None

    by_index: dict[int, dict] = {}
    for scene in scenes:
        if not isinstance(scene, dict):
            continue
        idx = _clamp_int(scene.get("index"), 0, 1, len(chunks))
        by_index[idx] = scene

    if len(by_index) != len(chunks):
        return None

    plan: list[dict] = []
    for i, chunk in enumerate(chunks, start=1):
        scene = by_index.get(i, {})
        timing = str(scene.get("sfx_timing") or "none").strip().lower()
        if timing not in {"none", "start", "end"}:
            timing = "none"

        plan.append(
            {
                "index": i,
                "text": chunk,
                "delivery": _clean_short_text(scene.get("delivery"), 90) or "clean serious narration",
                "emotion": _clean_short_text(scene.get("emotion"), 60) or "controlled suspense",
                "speed_multiplier": _clamp_float(
                    scene.get("speed_multiplier"),
                    1.0,
                    VOICE_PLAN_MIN_SPEED_MULTIPLIER,
                    VOICE_PLAN_MAX_SPEED_MULTIPLIER,
                ),
                "gain_db": _clamp_float(
                    scene.get("gain_db"),
                    0.0,
                    VOICE_PLAN_MIN_GAIN_DB,
                    VOICE_PLAN_MAX_GAIN_DB,
                ),
                "pause_after_ms": _clamp_int(
                    scene.get("pause_after_ms"),
                    0,
                    VOICE_PLAN_MIN_PAUSE_MS,
                    VOICE_PLAN_MAX_PAUSE_MS,
                ),
                "sfx": _safe_ascii_tag(scene.get("sfx"), "none"),
                "sfx_timing": timing,
            }
        )

    return plan


def build_voice_plan(video_data: dict, chunks: list[str]) -> list[dict]:
    fallback = _default_voice_plan(chunks)
    if not VOICE_PLAN_ENABLED:
        return fallback

    prompt = _build_voice_plan_prompt(video_data, chunks)

    for attempt in range(1, VOICE_PLAN_MAX_ATTEMPTS + 1):
        try:
            response = _run_voice_plan_completion(prompt)
        except Exception as exc:
            print(f"Voice plan request failed ({attempt}/{VOICE_PLAN_MAX_ATTEMPTS}): {exc}")
            continue

        response = _strip_reasoning_lines(response)
        if _contains_reasoning(response):
            print(f"Voice plan reasoning leak detected ({attempt}/{VOICE_PLAN_MAX_ATTEMPTS}).")
            continue

        data = _parse_json_object(response)
        plan = _coerce_voice_plan(data, chunks)
        if plan:
            return plan

        print(f"Voice plan validation failed ({attempt}/{VOICE_PLAN_MAX_ATTEMPTS}).")

    print("Using default voice plan.")
    return fallback


def generate_riva(chunk: str, path: Path):
    if not NVIDIA_API_KEY:
        raise RuntimeError("NVIDIA_API_KEY is missing.")

    import riva.client

    auth = riva.client.Auth(
        use_ssl=RIVA_USE_SSL,
        uri=RIVA_URI,
        metadata_args=[
            ["function-id", RIVA_FUNCTION_ID],
            ["authorization", f"Bearer {NVIDIA_API_KEY}"],
        ],
    )

    service = riva.client.SpeechSynthesisService(auth)

    response = service.synthesize(
        text=chunk,
        voice_name=RIVA_VOICE,
        language_code=RIVA_LANGUAGE,
        sample_rate_hz=RIVA_SAMPLE_RATE_HZ,
        encoding=riva.client.AudioEncoding.LINEAR_PCM,
    )

    raw_audio = response.audio

    temp_raw = path.with_suffix(".raw")
    temp_raw.write_bytes(raw_audio)

    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "s16le",
        "-ar",
        str(RIVA_SAMPLE_RATE_HZ),
        "-ac",
        "1",
        "-i",
        str(temp_raw),
        "-c:a",
        "pcm_s16le",
        str(path),
    ]

    _run(cmd)
    temp_raw.unlink(missing_ok=True)


def generate_voice_chunk(chunk: str, path: Path):
    generate_riva(chunk, path)


def _trim_silence(audio: AudioSegment) -> AudioSegment:
    nonsilent = detect_nonsilent(
        audio,
        min_silence_len=120,
        silence_thresh=-42,
    )

    if nonsilent:
        start = nonsilent[0][0]
        end = nonsilent[-1][1]
        audio = audio[start:end]

    return audio


def _load_sfx(tag: str) -> AudioSegment | None:
    if not VOICE_SFX_ENABLED or tag == "none":
        return None

    sfx_path = VOICE_SFX_DIR / f"{tag}{VOICE_SFX_EXT}"
    if not sfx_path.exists():
        print(f"SFX skipped, file not found: {sfx_path}")
        return None

    try:
        return AudioSegment.from_file(str(sfx_path)).apply_gain(VOICE_SFX_GAIN_DB)
    except Exception as exc:
        print(f"SFX skipped, unable to load {sfx_path}: {exc}")
        return None


def _apply_sfx(audio: AudioSegment, tag: str, timing: str) -> AudioSegment:
    sfx = _load_sfx(tag)
    if sfx is None:
        return audio

    if timing == "start":
        return sfx.append(audio, crossfade=min(VOICE_SFX_CROSSFADE_MS, len(sfx), len(audio)))

    if timing == "end":
        return audio.append(sfx, crossfade=min(VOICE_SFX_CROSSFADE_MS, len(sfx), len(audio)))

    return audio


OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CHUNKS_DIR.mkdir(parents=True, exist_ok=True)

content = json.load(open(OUTPUT_JSON, "r", encoding="utf-8"))
video_data = content[-1]
script_data = video_data["script"]

if isinstance(script_data, list):
    raw_chunks = [str(x) for x in script_data if str(x).strip()]
else:
    raw_chunks = [str(script_data)]

clean_chunks = [_clean_text(chunk) for chunk in raw_chunks]
clean_chunks = [chunk for chunk in clean_chunks if chunk]

if len(clean_chunks) == 1 and not isinstance(script_data, list):
    chunks = split_text(clean_chunks[0], 40)
else:
    chunks = clean_chunks

voice_plan = build_voice_plan(video_data, chunks)
paths = []

for i, scene in enumerate(voice_plan):
    chunk = scene["text"]
    path = CHUNKS_DIR / f"output_{i:03d}.wav"
    speed = VOICE_SPEED * float(scene.get("speed_multiplier", 1.0))

    print(
        f"Generating RIVA voice chunk {i + 1}/{len(voice_plan)}: "
        f"{str(scene.get('emotion', ''))} | {chunk[:80]}"
    )

    generate_voice_chunk(chunk, path)
    _speed_audio(path, speed)

    if not path.exists() or path.stat().st_size < 1024:
        raise RuntimeError(f"Generated chunk is missing or too small: {path}")

    paths.append(path)

combined = AudioSegment.empty()

for i, p in enumerate(paths):
    audio = AudioSegment.from_wav(str(p))
    audio = _trim_silence(audio)

    scene = voice_plan[i]
    scene_gain = float(scene.get("gain_db", 0.0))
    if abs(scene_gain) > 1e-3:
        audio = audio.apply_gain(scene_gain)

    audio = _apply_sfx(audio, str(scene.get("sfx", "none")), str(scene.get("sfx_timing", "none")))

    pause_after_ms = int(scene.get("pause_after_ms", 0))
    if pause_after_ms > 0:
        audio += AudioSegment.silent(duration=pause_after_ms)

    if i == 0:
        combined = audio
    else:
        combined = combined.append(audio, crossfade=180)

combined = combined.normalize()

if abs(VOICE_GAIN_DB) > 1e-3:
    combined = combined.apply_gain(VOICE_GAIN_DB)

combined.export(str(FINAL_WAV), format="wav")

if not FINAL_WAV.exists() or FINAL_WAV.stat().st_size < 1024:
    raise RuntimeError("Final voice output is missing or too small.")

if os.getenv("VOICE_CLEANUP", "1") == "1":
    for p in paths:
        try:
            p.unlink()
        except OSError as exc:
            print(f"Unable to delete {p}: {exc}")

print(f"Final voice saved: {FINAL_WAV}")
