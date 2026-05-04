import json
import math
import os
import re
from pathlib import Path

from dotenv import load_dotenv
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
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
MAX_OUTPUT_CHARS = int(os.getenv("CONTENT_MAX_OUTPUT_CHARS", "8000"))
MAX_NGRAM_REPEAT = int(os.getenv("CONTENT_MAX_NGRAM_REPEAT", "20"))
FALLBACK_MAX_TOKENS = int(os.getenv("CONTENT_FALLBACK_MAX_TOKENS", "4096"))
CONTENT_MODEL = os.getenv("CONTENT_MODEL", "openai/gpt-oss-120b")
NVIDIA_BASE_URL = os.getenv("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1")
TITLE_MAX_CHARS = int(os.getenv("CONTENT_TITLE_MAX_CHARS", "80"))
HOOK_MAX_WORDS = int(os.getenv("CONTENT_HOOK_MAX_WORDS", "18"))
CAPTION_MAX_CHARS = int(os.getenv("CONTENT_CAPTION_MAX_CHARS", "220"))
THUMBNAIL_TEXT_MAX_WORDS = int(os.getenv("CONTENT_THUMBNAIL_TEXT_MAX_WORDS", "5"))
HASHTAG_MIN_COUNT = int(os.getenv("CONTENT_HASHTAG_MIN_COUNT", "8"))
HASHTAG_MAX_COUNT = int(os.getenv("CONTENT_HASHTAG_MAX_COUNT", "15"))
RETENTION_TRIGGER_MIN_COUNT = int(os.getenv("CONTENT_RETENTION_TRIGGER_MIN_COUNT", "3"))
RETENTION_TRIGGER_MAX_COUNT = int(os.getenv("CONTENT_RETENTION_TRIGGER_MAX_COUNT", "6"))
BACKGROUND_MUSIC_MIN_WORDS = int(os.getenv("CONTENT_BACKGROUND_MUSIC_MIN_WORDS", "4"))
BACKGROUND_MUSIC_MAX_WORDS = int(os.getenv("CONTENT_BACKGROUND_MUSIC_MAX_WORDS", "8"))
PROMPT_CAMERA_CUES = os.getenv(
    "CONTENT_CAMERA_CUES",
    "slow dolly,tracking shot,handheld,wide establishing,close-up",
)
PROMPT_STORY_STRUCTURE = os.getenv(
    "CONTENT_STORY_STRUCTURE",
    "Hook, discovery, escalation, twist or unanswered question",
)
PROMPT_STYLE_NOTE = os.getenv(
    "CONTENT_STYLE_NOTE",
    "respectful suspense, simple spoken English, high retention pacing",
)
REQUIRED_FIELDS = {
    "title",
    "hook",
    "script",
    "image",
    "caption",
    "thumbnail_text",
    "hashtags",
    "retention_triggers",
    "trend",
    "background_music",
}
YOUTUBE_UPLOAD_SCOPE = "https://www.googleapis.com/auth/youtube.upload"
YOUTUBE_HISTORY_SCOPE = "https://www.googleapis.com/auth/youtube.readonly"
YOUTUBE_HISTORY_REQUIRED_SCOPES = [YOUTUBE_UPLOAD_SCOPE, YOUTUBE_HISTORY_SCOPE]
CONCEPT_STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "from",
    "into",
    "this",
    "that",
    "still",
    "after",
    "before",
    "about",
    "crime",
    "true",
    "case",
    "story",
    "mystery",
    "murder",
    "killer",
    "victim",
    "victims",
    "woman",
    "man",
    "girl",
    "boy",
    "teen",
    "documentary",
    "short",
    "shorts",
    "viral",
    "unsolved",
    "cold",
    "police",
    "missing",
    "files",
    "file",
    "explained",
    "haunting",
    "dark",
    "disturbing",
    "shocking",
}
_history_skip_logged = False


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


def _looks_glitched(text: str) -> bool:
    if not text:
        return False
    if "{" not in text and len(text) > MAX_OUTPUT_CHARS:
        return True
    tokens = _normalize(text).split()
    if len(tokens) < 60:
        return False
    counts: dict[tuple[str, str, str], int] = {}
    for i in range(len(tokens) - 2):
        gram = (tokens[i], tokens[i + 1], tokens[i + 2])
        counts[gram] = counts.get(gram, 0) + 1
    return bool(counts) and max(counts.values()) >= MAX_NGRAM_REPEAT


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


def _word_count(text: str) -> int:
    return len(str(text or "").strip().split())


def _is_plain_ascii_words(text: str) -> bool:
    value = str(text or "").strip()
    if not value:
        return False
    return bool(re.fullmatch(r"[A-Za-z0-9 ]+", value))


def _valid_hashtags(items) -> bool:
    if not isinstance(items, list):
        return False
    if not (HASHTAG_MIN_COUNT <= len(items) <= HASHTAG_MAX_COUNT):
        return False
    seen: set[str] = set()
    for item in items:
        tag = str(item or "").strip()
        if not tag or not tag.startswith("#") or " " in tag:
            return False
        norm = tag.lower()
        if norm in seen:
            return False
        seen.add(norm)
    return True


def _valid_string_list(items, min_count: int, max_count: int) -> bool:
    if not isinstance(items, list):
        return False
    if not (min_count <= len(items) <= max_count):
        return False
    return all(bool(str(item or "").strip()) for item in items)


def is_valid(data):
    if not isinstance(data, dict):
        return False
    if set(data.keys()) != REQUIRED_FIELDS:
        return False

    title = str(data.get("title") or "").strip()
    if not title or len(title) > TITLE_MAX_CHARS:
        return False

    hook = str(data.get("hook") or "").strip()
    if not hook or _word_count(hook) > HOOK_MAX_WORDS:
        return False

    script = data.get("script")
    image = data.get("image")
    if not isinstance(script, list) or not isinstance(image, list):
        return False
    if not (MIN_SCENES <= len(script) <= MAX_SCENES):
        return False
    if len(script) != len(image):
        return False
    if not all(bool(str(item or "").strip()) for item in script):
        return False
    if not all(bool(str(item or "").strip()) for item in image):
        return False

    caption = str(data.get("caption") or "").strip()
    if not caption or len(caption) > CAPTION_MAX_CHARS:
        return False

    thumbnail_text = str(data.get("thumbnail_text") or "").strip()
    if not thumbnail_text or _word_count(thumbnail_text) > THUMBNAIL_TEXT_MAX_WORDS:
        return False

    if not _valid_hashtags(data.get("hashtags")):
        return False

    if not _valid_string_list(
        data.get("retention_triggers"),
        RETENTION_TRIGGER_MIN_COUNT,
        RETENTION_TRIGGER_MAX_COUNT,
    ):
        return False

    bg = str(data.get("background_music") or "").strip()
    bg_words = _word_count(bg)
    if not bg or not _is_plain_ascii_words(bg):
        return False
    if not (BACKGROUND_MUSIC_MIN_WORDS <= bg_words <= BACKGROUND_MUSIC_MAX_WORDS):
        return False

    trend = str(data.get("trend") or "").strip()
    if not trend or trend.isdigit():
        return False
    return True


def _normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _log_history_skip(message: str) -> None:
    global _history_skip_logged
    if _history_skip_logged:
        return
    print(message)
    _history_skip_logged = True


def _youtube_history_enabled() -> bool:
    return os.getenv("YOUTUBE_HISTORY_ENABLED", "1") == "1"


def _youtube_history_limit() -> int:
    return max(1, int(os.getenv("YOUTUBE_HISTORY_LIMIT", "200")))


def _youtube_history_prompt_limit() -> int:
    return max(0, int(os.getenv("YOUTUBE_HISTORY_PROMPT_LIMIT", "40")))


def _youtube_history_title_similarity() -> float:
    return float(os.getenv("YOUTUBE_HISTORY_TITLE_SIM", "0.78"))


def _youtube_history_concept_similarity() -> float:
    return float(os.getenv("YOUTUBE_HISTORY_CONCEPT_SIM", "0.55"))


def _normalize_scopes(raw_scopes) -> list[str]:
    if raw_scopes is None:
        return []
    if isinstance(raw_scopes, str):
        raw_scopes = raw_scopes.split()

    scopes: list[str] = []
    seen: set[str] = set()
    for scope in raw_scopes:
        value = str(scope).strip()
        if not value or value in seen:
            continue
        scopes.append(value)
        seen.add(value)
    return scopes


def _history_token_paths() -> list[Path]:
    candidates: list[Path] = []
    env_token_path = os.getenv("YOUTUBE_TOKEN_FILE", "").strip()
    if env_token_path:
        candidates.append(Path(env_token_path))
    candidates.extend(
        [
            PROJECT_ROOT / "output" / "youtube_token.json",
            PROJECT_ROOT / "youtube_token.json",
        ]
    )
    deduped: list[Path] = []
    seen: set[str] = set()
    for path in candidates:
        key = str(path.resolve()) if path.is_absolute() else str(path)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(path)
    return deduped


def _load_history_token_info(data: dict) -> tuple[Credentials, list[str]]:
    scopes = _normalize_scopes(data.get("scopes"))
    creds = Credentials.from_authorized_user_info(data, scopes or None)
    return creds, scopes


def _load_history_credentials() -> tuple[Credentials | None, list[str], str | None]:
    issues: list[str] = []
    token_json = os.getenv("YOUTUBE_TOKEN_JSON", "").strip()
    if token_json:
        try:
            data = json.loads(token_json)
        except json.JSONDecodeError:
            issues.append("YOUTUBE_TOKEN_JSON is not valid JSON.")
        else:
            creds, scopes = _load_history_token_info(data)
            return creds, scopes, None

    for path in _history_token_paths():
        if not path.exists():
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            creds, scopes = _load_history_token_info(data)
            return creds, scopes, None
        except json.JSONDecodeError as exc:
            issues.append(f"token file {path} is not valid JSON: {exc}")
        except Exception as exc:
            issues.append(f"unable to read token file {path}: {exc}")
    return None, [], issues[-1] if issues else None


def _build_history_service():
    if not _youtube_history_enabled():
        return None

    creds, granted_scopes, error_message = _load_history_credentials()
    if creds is None:
        if error_message:
            _log_history_skip(f"Channel history skipped: {error_message}")
        else:
            _log_history_skip("Channel history skipped: no YouTube OAuth token found.")
        return None

    scope_set = set(granted_scopes)
    if not all(scope in scope_set for scope in YOUTUBE_HISTORY_REQUIRED_SCOPES):
        _log_history_skip(
            "Channel history skipped: token only has upload scope. Run "
            "`python generate_youtube_token.py --history` and update YOUTUBE_TOKEN_JSON "
            "to enable channel history checks."
        )
        return None

    if not creds.valid:
        if creds.expired and creds.refresh_token:
            try:
                creds.refresh(Request())
            except Exception as exc:
                _log_history_skip(f"Channel history skipped: token refresh failed: {exc}")
                return None
        else:
            _log_history_skip("Channel history skipped: OAuth token is invalid and cannot be refreshed.")
            return None

    return build("youtube", "v3", credentials=creds, cache_discovery=False)


def _resolve_uploads_playlist_id(youtube) -> str:
    channel_id = os.getenv("YOUTUBE_CHANNEL_ID", "").strip()
    try:
        if channel_id:
            response = youtube.channels().list(part="contentDetails", id=channel_id).execute()
        else:
            response = youtube.channels().list(part="contentDetails", mine=True).execute()
    except HttpError as exc:
        status = exc.resp.status if exc.resp is not None else None
        if status == 403 and not channel_id:
            _log_history_skip(
                "Channel history skipped: token cannot read your channel metadata. "
                "Regenerate the token with generate_youtube_token.py after adding "
                "youtube.readonly scope."
            )
            return ""
        _log_history_skip(f"Channel history skipped: failed to load channel metadata ({exc}).")
        return ""

    items = response.get("items") or []
    if not items:
        _log_history_skip("Channel history skipped: channel metadata returned no items.")
        return ""

    content_details = items[0].get("contentDetails") or {}
    related = content_details.get("relatedPlaylists") or {}
    uploads = str(related.get("uploads") or "").strip()
    if not uploads:
        _log_history_skip("Channel history skipped: uploads playlist was not found.")
    return uploads


def _fetch_channel_titles() -> list[str]:
    youtube = _build_history_service()
    if youtube is None:
        return []

    uploads_playlist_id = _resolve_uploads_playlist_id(youtube)
    if not uploads_playlist_id:
        return []

    titles: list[str] = []
    seen: set[str] = set()
    next_page_token = None
    remaining = _youtube_history_limit()

    while remaining > 0:
        response = youtube.playlistItems().list(
            part="snippet",
            playlistId=uploads_playlist_id,
            maxResults=min(50, remaining),
            pageToken=next_page_token,
        ).execute()

        for item in response.get("items", []):
            snippet = item.get("snippet") or {}
            title = str(snippet.get("title") or "").strip()
            if not title or title in {"Deleted video", "Private video"}:
                continue
            norm = _normalize(title)
            if not norm or norm in seen:
                continue
            titles.append(title)
            seen.add(norm)
            remaining -= 1
            if remaining <= 0:
                break

        next_page_token = response.get("nextPageToken")
        if not next_page_token:
            break

    return titles


def _strip_trend_text(text: str) -> str:
    if not text:
        return ""
    value = str(text).strip()
    value = re.sub(r"^\s*\d+\s*[\.\)\-]\s*", "", value)
    value = re.sub(r"^\s*\d+\s+", "", value)
    value = re.sub(r"\s*\([^)]*\)\s*$", "", value)
    return value.strip()


def _clean_trend_list(items: list[str], *, allow_numeric: bool = True) -> list[str]:
    cleaned: list[str] = []
    seen: set[str] = set()
    for item in items:
        if item is None:
            continue
        value = _strip_trend_text(str(item))
        if not value:
            continue
        if not allow_numeric and value.isdigit():
            continue
        norm = _normalize(value)
        if not norm or norm in seen:
            continue
        cleaned.append(value)
        seen.add(norm)
    return cleaned


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


def _previous_trends(file_data: list[dict]) -> list[str]:
    trends: list[str] = []
    for item in file_data:
        if not isinstance(item, dict):
            continue
        trend = item.get("trend")
        if trend:
            trends.append(str(trend))
    return _clean_trend_list(trends, allow_numeric=False)


def _previous_trend_tokens(file_data: list[dict]) -> list[set[str]]:
    return [_token_set(t) for t in _previous_trends(file_data)]


def _concept_tokens(text: str) -> set[str]:
    return {
        token
        for token in _token_set(text)
        if len(token) > 2 and not token.isdigit() and token not in CONCEPT_STOPWORDS
    }


def _title_matches_channel_history(candidate: str, channel_title: str) -> bool:
    candidate_norm = _normalize(candidate)
    channel_norm = _normalize(channel_title)
    if not candidate_norm or not channel_norm:
        return False
    if candidate_norm == channel_norm:
        return True
    if len(candidate_norm.split()) >= 3 and candidate_norm in channel_norm:
        return True
    return _jaccard(_token_set(candidate), _token_set(channel_title)) >= _youtube_history_title_similarity()


def _concept_matches_channel_history(candidate: str, channel_title: str) -> bool:
    candidate_norm = _normalize(candidate)
    channel_norm = _normalize(channel_title)
    if not candidate_norm or not channel_norm:
        return False

    candidate_tokens = _concept_tokens(candidate)
    channel_tokens = _concept_tokens(channel_title)
    if not candidate_tokens or not channel_tokens:
        return False

    overlap = candidate_tokens & channel_tokens
    if len(candidate_tokens) >= 2 and candidate_tokens.issubset(channel_tokens):
        return True
    if len(channel_tokens) >= 2 and channel_tokens.issubset(candidate_tokens):
        return True
    if len(overlap) >= 2 and _jaccard(candidate_tokens, channel_tokens) >= _youtube_history_concept_similarity():
        return True
    if len(candidate_norm.split()) >= 2 and candidate_norm in channel_norm:
        return True
    return False


def _validate_channel_history(data: dict, channel_titles: list[str]) -> tuple[bool, str]:
    if not channel_titles:
        return True, ""

    title = str(data.get("title") or "")
    trend_clean = _strip_trend_text(str(data.get("trend") or ""))

    for channel_title in channel_titles:
        if title and _title_matches_channel_history(title, channel_title):
            return False, f'Generated title overlaps existing channel title "{channel_title}".'
        if title and _concept_matches_channel_history(title, channel_title):
            return False, f'Generated concept overlaps existing channel title "{channel_title}".'
        if trend_clean and _concept_matches_channel_history(trend_clean, channel_title):
            return False, f'Generated trend overlaps existing channel title "{channel_title}".'

    return True, ""


def _validate_no_repeats(
    data: dict,
    file_data: list[dict],
    channel_titles: list[str] | None = None,
) -> tuple[bool, str]:
    title = str(data.get("title") or "")
    script = data.get("script")
    images = data.get("image")
    trend_value = str(data.get("trend") or "")

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

    trend_clean = _strip_trend_text(trend_value)
    if trend_clean:
        prev_trends = _previous_trends(file_data)
        prev_trends_norm = {_normalize(t) for t in prev_trends}
        if _normalize(trend_clean) in prev_trends_norm:
            return False, "Trend topic already used in output.json."

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

    if trend_clean:
        trend_tokens = _token_set(trend_clean)
        for prev in _previous_trend_tokens(file_data):
            if _jaccard(trend_tokens, prev) >= SIMILARITY_THRESHOLD:
                return False, "Trend topic too similar to a previous output."

    ok, reason = _validate_channel_history(data, channel_titles or [])
    if not ok:
        return False, reason

    return True, ""


def _all_trends_used(trends: list[str], repeated: list[str]) -> bool:
    trend_topics = _clean_trend_list(trends, allow_numeric=False)
    if not trend_topics:
        return False
    used = {_normalize(t) for t in _clean_trend_list(repeated, allow_numeric=False)}
    return all(_normalize(t) in used for t in trend_topics)


def _build_prompt(trends: list[str], repeated: list[str], channel_titles: list[str]) -> str:
    trend_topics = _clean_trend_list(trends, allow_numeric=False)
    used_topics = _clean_trend_list(repeated, allow_numeric=False)
    used_norm = {_normalize(t) for t in used_topics}
    unused_topics = [t for t in trend_topics if _normalize(t) not in used_norm]
    all_used = bool(trend_topics) and not unused_topics
    prompt_channel_titles = channel_titles[: _youtube_history_prompt_limit()]
    camera_cues = [cue.strip() for cue in PROMPT_CAMERA_CUES.split(",") if cue.strip()]
    camera_cue_line = ", ".join(camera_cues) if camera_cues else "natural documentary camera movement"
    required_fields_line = ", ".join(sorted(REQUIRED_FIELDS))
    ignore_line = (
        f"Do NOT choose these trend topics from the list: {used_topics}. " if used_topics else ""
    )
    history_line = (
        f"These YouTube channel titles already exist and are off-limits for concept reuse: {prompt_channel_titles}. "
        if prompt_channel_titles
        else ""
    )
    history_rule = (
        "If a case overlaps with any existing channel title, reject it and choose a different case. "
        "Never cover the same case again with a new headline. "
        if prompt_channel_titles
        else ""
    )
    history_suffix = " or already covered on the YouTube channel" if prompt_channel_titles else ""
    if not all_used and unused_topics:
        if len(unused_topics) == 1:
            pick_line = f"You MUST choose this unused trend topic: \"{unused_topics[0]}\". "
        else:
            pick_line = f"Choose ONLY from these unused trend topics: {unused_topics}. "
    else:
        pick_line = ""
    trend_rule = (
        f"Select the strongest REAL true crime case from this list of trends: {trend_topics}. "
        if not all_used
        else (
            f"All listed trend topics are already used in output.json. "
            f"Ignore the list {trend_topics} and pick ANY real true crime case NOT in output.json{history_suffix}. "
            "Set \"trend\" to the exact case name you chose."
        )
    )
    return (
        "You are a professional YouTube Shorts true crime writer and retention editor. "
        f"This is a vertical 9:16 YouTube Short, {int(MIN_TOTAL_SECONDS)}-{int(MAX_TOTAL_SECONDS)} seconds total. "
        f"{trend_rule}"
        "You MUST select a real, well-documented true crime case. "
        "Do NOT choose movies, TV shows, fictional stories, creepypasta, urban legends, or invented cases. "
        "Use only public, documentable facts. Do NOT invent updates, evidence, quotes, dates, or police actions. "
        "If a provided trend has a current public-interest angle, connect it naturally without forcing it. "
        "Keep it suitable for YouTube: no gore, no graphic injury details, no cruelty, no victim-blaming, and no jokes about victims. "
        "Focus on mystery, timeline, investigation, decisions, clues, and unanswered questions. "
        f"{ignore_line}"
        f"{history_line}"
        f"{history_rule}"
        f"{pick_line}"
        f"Create a high-retention {int(MIN_TOTAL_SECONDS)}-{int(MAX_TOTAL_SECONDS)} second YouTube Shorts content package. "
        f"Story structure: {PROMPT_STORY_STRUCTURE}. "
        f"Style: {PROMPT_STYLE_NOTE}. "
        "Return ONE valid JSON object ONLY. No extra text, no markdown, no code blocks, no commentary. "
        f"The JSON object MUST contain exactly these fields: {required_fields_line}. "
        "Use this exact JSON shape: "
        "{"
        "\"title\": \"...\", "
        "\"hook\": \"...\", "
        "\"script\": [\"scene narration\"], "
        "\"image\": [\"scene image prompt\"], "
        "\"caption\": \"...\", "
        "\"thumbnail_text\": \"...\", "
        "\"hashtags\": [\"#tag\"], "
        "\"retention_triggers\": [\"...\"], "
        "\"trend\": \"selected trend topic text only\", "
        "\"background_music\": \"short music search query\""
        "}. "
        "STRICT RULES: "
        f"0. title MUST be under {TITLE_MAX_CHARS} characters and written for curiosity, not clickbait lies. "
        f"1. hook MUST be under {HOOK_MAX_WORDS} words and MUST create curiosity, danger, contradiction, or an unanswered question. "
        "2. The first script item MUST deliver the hook immediately with a sudden-stop beat or sharp curiosity gap. "
        "3. script MUST be a list of narration scenes. "
        "4. image MUST be a list of image prompts. "
        "5. script and image lists MUST be the SAME LENGTH. "
        f"6. Total scenes: {MIN_SCENES}-{MAX_SCENES} scenes, about one scene every {int(SCENE_SECONDS)} seconds. "
        f"7. Each script item = 1-2 short spoken sentences, tight pacing around {int(SCENE_SECONDS)} seconds per clip. "
        "8. Every script item MUST add one new fact, clue, decision, location, time shift, or turning point. "
        "9. No filler, no recap, no repeated phrasing, no generic true-crime lines. "
        "10. Include only core facts: who, what, where, when, how, and why it still matters if known. "
        "11. Make the narration human and suspenseful: short questions, natural pauses, and curiosity gaps are allowed. "
        "12. Do NOT use slang. Do NOT make memes. Do NOT sound like a robot. "
        "13. Empathy first, curiosity second. Respect victims and families. "
        "14. The final script item MUST end with a mystery-style question to the viewer. "
        "15. Images MUST be cinematic, photorealistic, documentary b-roll visuals matched to each narration scene. "
        "16. Each image prompt MUST include a clear subject, setting, time-of-day, action, vertical 9:16 framing, and one camera cue. "
        f"17. Camera cues can follow this style: {camera_cue_line}. "
        "18. People, clothing, props, weather, and locations MUST stay consistent across scenes where applicable. "
        "19. Do NOT show graphic, disturbing, exploitative, or inappropriate imagery. "
        "20. Do NOT mention movies, actors, TV shows, fictional elements, or unrelated pop culture. "
        "21. Do NOT include scene labels, timestamps, camera metadata labels, or quotes outside JSON. "
        f"22. caption MUST be under {CAPTION_MAX_CHARS} characters and should invite comments without begging. "
        f"23. thumbnail_text MUST be {THUMBNAIL_TEXT_MAX_WORDS} words or fewer, high curiosity, no false claim. "
        f"24. hashtags MUST contain {HASHTAG_MIN_COUNT}-{HASHTAG_MAX_COUNT} short hashtags, each starting with # and containing no spaces. "
        f"25. retention_triggers MUST contain {RETENTION_TRIGGER_MIN_COUNT}-{RETENTION_TRIGGER_MAX_COUNT} short reasons this Short keeps attention. "
        "26. trend MUST be the exact case/topic text you chose, with no numbering and no source labels. "
        f"27. background_music MUST be {BACKGROUND_MUSIC_MIN_WORDS}-{BACKGROUND_MUSIC_MAX_WORDS} plain ASCII words only, no punctuation, no emojis, no quotes, no special characters. "
        "28. Do NOT repeat or paraphrase titles, lines, cases, or image prompts from earlier videos. "
        f"29. If none of the provided trends are usable, pick ANY real true crime case that does NOT appear in output.json{history_suffix}. "
        "30. If you cannot comply with ALL rules, return an empty JSON object {} ONLY. "
        "31. If you start repeating or looping, STOP and return {} ONLY. "
        "32. Output MUST be valid JSON and nothing else."
    )

def _run_completion(
    client: OpenAI,
    prompt: str,
    enable_thinking: bool,
    stream: bool,
    max_tokens: int | None = None,
    stop: list[str] | None = None,
) -> str:
    completion = client.chat.completions.create(
        model=CONTENT_MODEL,
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
        temperature=TEMPERATURE,
        top_p=TOP_P,
        max_tokens=max_tokens or MAX_TOKENS,
        extra_body={
            "chat_template_kwargs": {"enable_thinking": enable_thinking},
            "reasoning_budget": REASONING_BUDGET if enable_thinking else 0,
        },
        stream=stream,
        stop=stop,
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
        base_url=NVIDIA_BASE_URL,
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
        repeated = [str(item.get("trend") or "") for item in file_data]

    channel_titles = _fetch_channel_titles()
    if channel_titles:
        print(f"Loaded {len(channel_titles)} existing YouTube titles for history checks.")

    max_attempts = int(os.getenv("CONTENT_MAX_ATTEMPTS", "3"))
    prompt = _build_prompt(trends, repeated, channel_titles)
    retry_reason = ""
    for attempt in range(1, max_attempts + 1):
        attempt_prompt = prompt
        if retry_reason:
            attempt_prompt = (
                f"{prompt} Previous attempt failed validation: {retry_reason} "
                "Generate a completely different case, title, and script."
            )
        try:
            s = _run_completion(
                client,
                attempt_prompt,
                enable_thinking=ENABLE_THINKING,
                stream=STREAM_OUTPUT,
                stop=["\n[Reasoning]", "\nReasoning:", "\n[Analysis]", "\nAnalysis:"],
            )
        except Exception as exc:
            print(f"\ncontent request failed: {exc}\n")
            s = ""

        if s:
            s = _strip_reasoning_lines(s)
            if _contains_reasoning(s) or _looks_glitched(s):
                print("\nReasoning leak or glitch detected. Retrying with strict JSON mode.\n")
                s = ""

        if not s:
            print(f"\nempty response, retrying ({attempt}/{max_attempts})...\n")
            continue
        data = extract_json(s, trends)
        if data and is_valid(data):
            ok, reason = _validate_no_repeats(data, file_data, channel_titles)
            if not ok:
                retry_reason = reason
                print(f"\nretrying ({attempt}/{max_attempts})... {reason}\n")
                continue
            break
        print(f"\nretrying ({attempt}/{max_attempts})...\n")
    else:
        raise RuntimeError("Failed to get valid JSON after 3 attempts.")
    file_data.append(data)
    with OUTPUT_JSON.open("w", encoding="utf-8") as f:
        json.dump(file_data, f, indent=2, ensure_ascii=False)
