import json
import math
import os
import re
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_JSON = PROJECT_ROOT / "output.json"
CLIP_SECONDS = float(os.getenv("VIDEO_CLIP_SECONDS", "5"))
VIDEO_MIN_SECONDS = int(os.getenv("VIDEO_MIN_SECONDS", "40"))
VIDEO_MAX_SECONDS = int(os.getenv("VIDEO_MAX_SECONDS", "50"))
VIDEO_MIN_MINUTES = float(os.getenv("VIDEO_MIN_MINUTES", "1"))
VIDEO_MAX_MINUTES = float(os.getenv("VIDEO_MAX_MINUTES", "1"))
DEFAULT_CLIP_SECONDS = 5.0
SCENE_SECONDS = CLIP_SECONDS if CLIP_SECONDS > 0 else DEFAULT_CLIP_SECONDS
MIN_TOTAL_SECONDS = VIDEO_MIN_SECONDS if VIDEO_MIN_SECONDS > 0 else VIDEO_MIN_MINUTES * 60
MAX_TOTAL_SECONDS = VIDEO_MAX_SECONDS if VIDEO_MAX_SECONDS > 0 else VIDEO_MAX_MINUTES * 60
if MAX_TOTAL_SECONDS < MIN_TOTAL_SECONDS:
    MAX_TOTAL_SECONDS = MIN_TOTAL_SECONDS
MIN_SCENES = max(1, int(math.ceil(MIN_TOTAL_SECONDS / SCENE_SECONDS)))
MAX_SCENES = max(MIN_SCENES, int(math.ceil(MAX_TOTAL_SECONDS / SCENE_SECONDS)))

MAX_TOKENS = int(os.getenv("CONTENT_MAX_TOKENS", "16384"))
REASONING_BUDGET = int(os.getenv("CONTENT_REASONING_BUDGET", "16384"))
TEMPERATURE = float(os.getenv("CONTENT_TEMPERATURE", "0.7"))
TOP_P = float(os.getenv("CONTENT_TOP_P", "0.9"))
ENABLE_THINKING = os.getenv("CONTENT_ENABLE_THINKING", "1") == "1"
STREAM_OUTPUT = os.getenv("CONTENT_STREAM", "0") == "1"
SIMILARITY_THRESHOLD = float(os.getenv("CONTENT_DUP_SIM", "0.82"))


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
            elif ch == "\"":
                in_str = False
        else:
            if ch == "\"":
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[start : i + 1]
    return None


def extract_json(s, trends):
    if not s or not s.strip():
        print("Empty response from model.")
        return {}
    try:
        parsed = json.loads(s)
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        candidate = _extract_json_object(s)
        if not candidate:
            print("No JSON object found in response.")
            return {}
        try:
            parsed = json.loads(candidate)
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            print("Failed to parse JSON object from response.")
            return {}


def is_valid(data):
    if not isinstance(data, dict):
        return False
    required = {"title", "script", "image", "trend"}
    if set(data.keys()) != required:
        return False
    return True


def _normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _token_set(text: str) -> set[str]:
    text = _normalize(text)
    return set(text.split())


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def _too_similar(items: list[str], threshold: float) -> bool:
    tokens = [_token_set(item) for item in items]
    for i in range(len(tokens)):
        for j in range(i + 1, len(tokens)):
            if _jaccard(tokens[i], tokens[j]) >= threshold:
                return True
    return False


def _has_duplicates(items: list[str]) -> bool:
    seen: set[str] = set()
    for item in items:
        norm = _normalize(item)
        if not norm:
            continue
        if norm in seen:
            return True
        seen.add(norm)
    return False


def _previous_sets(file_data: list[dict]) -> tuple[set[str], set[str], set[str]]:
    titles: set[str] = set()
    scripts: set[str] = set()
    images: set[str] = set()
    for item in file_data:
        if not isinstance(item, dict):
            continue
        title = str(item.get("title") or "")
        if title:
            titles.add(_normalize(title))
        script = item.get("script")
        if isinstance(script, list):
            for s in script:
                if s:
                    scripts.add(_normalize(str(s)))
        elif script:
            scripts.add(_normalize(str(script)))
        image = item.get("image")
        if isinstance(image, list):
            for img in image:
                if img:
                    images.add(_normalize(str(img)))
        elif image:
            images.add(_normalize(str(image)))
    return titles, scripts, images


def _previous_tokens(file_data: list[dict]) -> tuple[list[set[str]], list[set[str]], list[set[str]]]:
    titles: list[set[str]] = []
    scripts: list[set[str]] = []
    images: list[set[str]] = []
    for item in file_data:
        if not isinstance(item, dict):
            continue
        title = str(item.get("title") or "")
        if title:
            titles.append(_token_set(title))
        script = item.get("script")
        if isinstance(script, list):
            for s in script:
                if s:
                    scripts.append(_token_set(str(s)))
        elif script:
            scripts.append(_token_set(str(script)))
        image = item.get("image")
        if isinstance(image, list):
            for img in image:
                if img:
                    images.append(_token_set(str(img)))
        elif image:
            images.append(_token_set(str(image)))
    return titles, scripts, images


def _validate_no_repeats(data: dict, file_data: list[dict]) -> tuple[bool, str]:
    title = str(data.get("title") or "")
    script = data.get("script")
    images = data.get("image")

    if not isinstance(script, list) or not isinstance(images, list):
        return False, "script/image must be lists."
    if len(script) != len(images):
        return False, "script/image length mismatch."
    if _has_duplicates([str(s) for s in script]):
        return False, "Duplicate lines inside script."
    if _has_duplicates([str(i) for i in images]):
        return False, "Duplicate prompts inside image list."
    if _too_similar([str(s) for s in script], SIMILARITY_THRESHOLD):
        return False, "Script lines too similar."
    if _too_similar([str(i) for i in images], SIMILARITY_THRESHOLD):
        return False, "Image prompts too similar."

    titles_prev, scripts_prev, images_prev = _previous_sets(file_data)
    if title and _normalize(title) in titles_prev:
        return False, "Title already used in output.json."

    for s in script:
        if _normalize(str(s)) in scripts_prev:
            return False, "Script line repeats a previous output."

    for img in images:
        if _normalize(str(img)) in images_prev:
            return False, "Image prompt repeats a previous output."

    prev_title_tokens, prev_script_tokens, prev_image_tokens = _previous_tokens(file_data)
    if title:
        title_tokens = _token_set(title)
        for t in prev_title_tokens:
            if _jaccard(title_tokens, t) >= SIMILARITY_THRESHOLD:
                return False, "Title too similar to a previous output."

    for s in script:
        tokens = _token_set(str(s))
        for prev in prev_script_tokens:
            if _jaccard(tokens, prev) >= SIMILARITY_THRESHOLD:
                return False, "Script line too similar to a previous output."

    for img in images:
        tokens = _token_set(str(img))
        for prev in prev_image_tokens:
            if _jaccard(tokens, prev) >= SIMILARITY_THRESHOLD:
                return False, "Image prompt too similar to a previous output."

    return True, ""


def _build_prompt(trends: list[str], repeated: list[str]) -> str:
    return (
        "You are a professional YouTuber specializing in TRUE CRIME. "
        f"This is a YouTube Short: vertical 9:16, {int(MIN_TOTAL_SECONDS)}-{int(MAX_TOTAL_SECONDS)} seconds total. "
        f"Select the most popular REAL true crime case from this list of trends: {trends}. "
        "It MUST be a real case (NOT movies, NOT TV shows, NOT fiction). "
        "It must be interesting but NOT overly violent or disturbing. "
        "Use safe, censored wording suitable for YouTube. "
        f"Ignore these trends completely: {repeated}. "
        f"Create a {int(MIN_TOTAL_SECONDS)}-{int(MAX_TOTAL_SECONDS)} second YouTube documentary package. "
        "Return ONE JSON object only (no extra text, no markdown) with these fields: "
        "{\"title\": \"...\", \"script\": [\"scene 1 narration\", \"scene 2 narration\", \"...\"], "
        "\"image\": [\"scene 1 image prompt\", \"scene 2 image prompt\", \"...\"], "
        "\"trend\": \"selected trend number only\"}. "
        "Rules: "
        "0. The first 5 seconds MUST be a strong hook with a sudden-stop beat. "
        "0b. Focus on the main story beats and important facts only (no filler). "
        "Include the core facts: who, what, where, when, how - but keep it concise. "
        "1. script MUST be a list of narration scenes. "
        "2. image MUST be a list of image prompts. "
        "3. script and image lists MUST be the SAME LENGTH. "
        f"4. Each script item = 1-2 short sentences (tight pacing ~{int(SCENE_SECONDS)}s per clip). "
        f"5. Total scenes: {MIN_SCENES}-{MAX_SCENES} "
        f"(about one scene every {int(SCENE_SECONDS)} seconds). "
        "6. Images should describe cinematic, photorealistic, documentary b-roll visuals "
        "   matched to the narration. Each prompt must include a clear subject, setting, "
        "   time-of-day, and action. Add subtle camera cues like "
        "   'slow dolly', 'tracking shot', 'handheld', 'wide establishing', 'close-up', "
        "   but still describe a single frame. Use vertical 9:16 framing cues. "
        "7. Keep people/locations consistent across scenes where applicable "
        "   (same clothing, same location details). "
        "8. Do NOT include graphic or disturbing imagery. "
        "9. Do NOT mention movies, actors, or TV shows. "
        "10. Do NOT include scene labels like 'Scene 1'. "
        "11. Do NOT include timestamps. "
        "12. Do NOT include quotes outside JSON. "
        "13. Write in engaging YouTube storytelling style. "
        "14. Make it interactive: ask short questions to the viewer, add sudden-stop beats "
        "(short fragments for tension), and sprinkle light Gen Z-leaning slang "
        "(lowkey, no cap, vibes) sparingly. Keep it professional and safe. "
        "15. The final scene MUST end with a mystery-style question to the viewer. "
        "16. If you cannot comply, return an empty JSON object {}. "
        "17. Do NOT repeat any lines or image prompts within the output. "
        "18. Avoid repeating titles, lines, or image prompts from earlier videos. "
        "19. Avoid paraphrasing or reusing the same phrasing from earlier videos. "
        "20. If none of the provided trends are usable, pick ANY real murder case "
        "that does NOT appear in output.json and proceed. "
        "Return ONLY the JSON object."
    )


def _run_completion(client: OpenAI, prompt: str, enable_thinking: bool, stream: bool) -> str:
    completion = client.chat.completions.create(
        model="nvidia/nemotron-3-super-120b-a12b",
        messages=[
            {
                "role": "system",
                "content": (
                    "You must output ONLY valid JSON. "
                    "No markdown, no code fences, no extra commentary."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        temperature=TEMPERATURE,
        top_p=TOP_P,
        max_tokens=MAX_TOKENS,
        extra_body={
            "chat_template_kwargs": {"enable_thinking": enable_thinking},
            "reasoning_budget": REASONING_BUDGET if enable_thinking else 0,
        },
        stream=stream,
    )

    if not stream:
        return (completion.choices[0].message.content or "").strip()

    s = []
    for chunk in completion:
        if not chunk.choices:
            continue
        if chunk.choices[0].delta.content is not None:
            s.append(chunk.choices[0].delta.content)
    return "".join(s).strip()


def contents(trends):
    load_dotenv()
    client = OpenAI(
        base_url="https://integrate.api.nvidia.com/v1",
        api_key=os.getenv("NVIDIA_API_KEY")
    )
    repeated = []
    if OUTPUT_JSON.exists():
        with OUTPUT_JSON.open("r", encoding="utf-8") as f:
            try:
                file_data = json.load(f)
            except json.JSONDecodeError:
                file_data = []
    else:
        file_data = []
    if len(file_data) > 0:
        repeated = [item["trend"] for item in file_data]

    max_attempts = int(os.getenv("CONTENT_MAX_ATTEMPTS", "3"))
    prompt = _build_prompt(trends, repeated)
    for attempt in range(1, max_attempts + 1):
        try:
            s = _run_completion(client, prompt, enable_thinking=ENABLE_THINKING, stream=STREAM_OUTPUT)
        except Exception as exc:
            print(f"\ncontent request failed: {exc}\n")
            s = ""

        if not s:
            # Fallback: disable thinking and streaming for a strict JSON-only response.
            try:
                s = _run_completion(client, prompt, enable_thinking=False, stream=False)
            except Exception as exc:
                print(f"\ncontent fallback failed: {exc}\n")
                s = ""

        if not s:
            print(f"\nempty response, retrying ({attempt}/{max_attempts})...\n")
            continue
        data = extract_json(s, trends)
        if data and is_valid(data):
            ok, reason = _validate_no_repeats(data, file_data)
            if not ok:
                print(f"\nretrying ({attempt}/{max_attempts})... {reason}\n")
                continue
            if data.get("trend") in repeated:
                print(f"\nretrying ({attempt}/{max_attempts})... trend already used\n")
                continue
            break
        print(f"\nretrying ({attempt}/{max_attempts})...\n")
    else:
        raise RuntimeError("Failed to get valid JSON after 3 attempts.")
    file_data.append(data)
    with OUTPUT_JSON.open("w", encoding="utf-8") as f:
        json.dump(file_data, f, indent=2, ensure_ascii=False)
