import json
import os
import re
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Tuple

from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaFileUpload
from google.auth.exceptions import RefreshError
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from openai import OpenAI

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "output"
OUTPUT_JSON = Path(os.getenv("OUTPUT_JSON_PATH", str(ROOT / "output.json")))

UPLOAD_SCOPE = "https://www.googleapis.com/auth/youtube.upload"
HISTORY_SCOPE = "https://www.googleapis.com/auth/youtube.readonly"
UPLOAD_SCOPES = [UPLOAD_SCOPE]
HISTORY_SCOPES = [UPLOAD_SCOPE, HISTORY_SCOPE]

CLIENT_SECRETS = Path(os.getenv("YOUTUBE_CLIENT_SECRETS", "client_secrets.json"))
TOKEN_FILE = Path(os.getenv("YOUTUBE_TOKEN_FILE", str(OUTPUT_DIR / "youtube_token.json")))
PRIVACY_STATUS = os.getenv("YOUTUBE_PRIVACY_STATUS", "public")
CATEGORY_ID = os.getenv("YOUTUBE_CATEGORY_ID", "24")
DEFAULT_TAGS = os.getenv(
    "YOUTUBE_TAGS",
    "shorts,ytshorts,youtubeshorts,shortsvideo,viral,trending,true crime,crime,documentary,mystery,unsolved,cold case,investigation,case file,missing person",
).split(",")
POPULAR_HASHTAGS = os.getenv(
    "YOUTUBE_HASHTAGS",
    "#shorts #ytshorts #youtubeshorts #shortsvideo #viral #trending "
    "#truecrime #mystery #crime #unsolved #coldcase #documentary",
).strip()
DESCRIPTION_SUFFIX = os.getenv("YOUTUBE_DESCRIPTION_SUFFIX", POPULAR_HASHTAGS)
KEYWORDS_LINE = os.getenv("YOUTUBE_KEYWORDS_LINE", "").strip()
REFERENCE_TEXT = os.getenv("YOUTUBE_REFERENCE_TEXT", "").strip()

UPLOADS_PER_DAY = int(os.getenv("YOUTUBE_UPLOADS_PER_DAY", "10"))
LOOP_UPLOADS = os.getenv("YOUTUBE_LOOP", "1") == "1"
WAIT_SECONDS = float(os.getenv("YOUTUBE_WAIT_BETWEEN_UPLOADS", "0"))
MAX_SUCCESS_UPLOADS = int(os.getenv("YOUTUBE_MAX_SUCCESS", str(UPLOADS_PER_DAY)))

VIDEO_QUEUE_DIR = Path(os.getenv("UPLOAD_QUEUE_DIR", str(OUTPUT_DIR / "queue")))
DEFAULT_VIDEO = Path(os.getenv("UPLOAD_SINGLE_VIDEO", str(OUTPUT_DIR / "final.mp4")))
THUMBNAIL_PATH = Path(os.getenv("THUMBNAIL_PATH", str(OUTPUT_DIR / "thumbnail.jpg")))
NON_INTERACTIVE = os.getenv("NON_INTERACTIVE", "").lower() in {"1", "true", "yes"} or os.getenv(
    "GITHUB_ACTIONS", ""
) == "true" or os.getenv("CI", "").lower() in {"1", "true", "yes"}

CLIENT_SECRETS_JSON = os.getenv("YOUTUBE_CLIENT_SECRETS_JSON", "")
TOKEN_JSON = os.getenv("YOUTUBE_TOKEN_JSON", "")

METADATA_AI_ENABLED = os.getenv("YOUTUBE_METADATA_AI_ENABLED", "1") == "1"
METADATA_MODEL = os.getenv("YOUTUBE_METADATA_MODEL", os.getenv("CONTENT_MODEL", "openai/gpt-oss-120b"))
METADATA_BASE_URL = os.getenv(
    "YOUTUBE_METADATA_BASE_URL",
    os.getenv("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1"),
)
METADATA_MAX_ATTEMPTS = int(os.getenv("YOUTUBE_METADATA_MAX_ATTEMPTS", "3"))
METADATA_MAX_TOKENS = int(os.getenv("YOUTUBE_METADATA_MAX_TOKENS", "4096"))
METADATA_TEMPERATURE = float(os.getenv("YOUTUBE_METADATA_TEMPERATURE", "0.35"))
METADATA_TOP_P = float(os.getenv("YOUTUBE_METADATA_TOP_P", "0.9"))
METADATA_ENABLE_THINKING = os.getenv("YOUTUBE_METADATA_ENABLE_THINKING", "1") == "1"
METADATA_REASONING_BUDGET = int(os.getenv("YOUTUBE_METADATA_REASONING_BUDGET", "4096"))
METADATA_TITLE_MAX_CHARS = int(os.getenv("YOUTUBE_METADATA_TITLE_MAX_CHARS", "75"))
METADATA_DESCRIPTION_MAX_CHARS = int(os.getenv("YOUTUBE_METADATA_DESCRIPTION_MAX_CHARS", "1200"))
METADATA_TAG_MIN_COUNT = int(os.getenv("YOUTUBE_METADATA_TAG_MIN_COUNT", "8"))
METADATA_TAG_MAX_COUNT = int(os.getenv("YOUTUBE_METADATA_TAG_MAX_COUNT", "15"))
METADATA_TAG_CHAR_LIMIT = int(os.getenv("YOUTUBE_METADATA_TAG_CHAR_LIMIT", "450"))
METADATA_HASHTAG_MIN_COUNT = int(os.getenv("YOUTUBE_METADATA_HASHTAG_MIN_COUNT", "8"))
METADATA_HASHTAG_MAX_COUNT = int(os.getenv("YOUTUBE_METADATA_HASHTAG_MAX_COUNT", "15"))
METADATA_USE_CONTENT_HASHTAGS = os.getenv("YOUTUBE_METADATA_USE_CONTENT_HASHTAGS", "1") == "1"

_nvidia_client = None
if os.getenv("NVIDIA_API_KEY"):
    _nvidia_client = OpenAI(
        base_url=METADATA_BASE_URL,
        api_key=os.getenv("NVIDIA_API_KEY"),
    )


def _safe_log_text(value: object) -> str:
    text = str(value)
    encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
    return text.encode(encoding, errors="replace").decode(encoding, errors="replace")


def _clean_control_chars(text: str) -> str:
    return re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", str(text or "")).strip()


def _clean_title(text: str, max_chars: int | None = None) -> str:
    value = _clean_control_chars(text)
    value = re.sub(r"\s+", " ", value)
    value = value.strip(" -|:.\n\t")
    limit = max_chars or METADATA_TITLE_MAX_CHARS
    if len(value) > limit:
        value = value[:limit].rstrip(" -|:.,")
    return value or "Untitled Short"


def _clean_description(text: str, max_chars: int | None = None) -> str:
    value = _clean_control_chars(text)
    value = re.sub(r"\n{3,}", "\n\n", value)
    limit = max_chars or METADATA_DESCRIPTION_MAX_CHARS
    if len(value) > limit:
        value = value[:limit].rstrip()
    return value


def _normalize_tag(tag: object) -> str:
    value = _clean_control_chars(tag)
    value = value.replace("#", "")
    value = re.sub(r"\s+", " ", value).strip(" ,.;:-|")
    if len(value) > 45:
        value = value[:45].rstrip()
    return value


def _clean_tags(tags: list | tuple | str | None) -> list[str]:
    if tags is None:
        values: list[object] = []
    elif isinstance(tags, str):
        values = re.split(r"[,;\n]", tags)
    else:
        values = list(tags)

    cleaned: list[str] = []
    seen: set[str] = set()
    for tag in values:
        value = _normalize_tag(tag)
        if not value:
            continue
        key = value.lower()
        if key in seen:
            continue
        cleaned.append(value)
        seen.add(key)

    limited: list[str] = []
    used_chars = 0
    for tag in cleaned:
        extra = len(tag) + (1 if limited else 0)
        if used_chars + extra > METADATA_TAG_CHAR_LIMIT:
            break
        limited.append(tag)
        used_chars += extra

    return limited


def _normalize_hashtag(tag: object) -> str:
    value = _clean_control_chars(tag)
    if not value:
        return ""
    if not value.startswith("#"):
        value = f"#{value}"
    body = value[1:]
    body = re.sub(r"[^A-Za-z0-9_]", "", body)
    if not body:
        return ""
    return f"#{body[:40]}"


def _clean_hashtags(hashtags: list | tuple | str | None) -> list[str]:
    if hashtags is None:
        values: list[object] = []
    elif isinstance(hashtags, str):
        values = re.findall(r"#[A-Za-z0-9_]+", hashtags)
        if not values:
            values = re.split(r"[,;\s\n]+", hashtags)
    else:
        values = list(hashtags)

    cleaned: list[str] = []
    seen: set[str] = set()
    for tag in values:
        value = _normalize_hashtag(tag)
        if not value:
            continue
        key = value.lower()
        if key in seen:
            continue
        cleaned.append(value)
        seen.add(key)
    return cleaned[:METADATA_HASHTAG_MAX_COUNT]


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


def _parse_json_object(text: str) -> dict:
    if not text:
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


def _load_latest_content() -> dict:
    if not OUTPUT_JSON.exists():
        return {}
    try:
        data = json.loads(OUTPUT_JSON.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    if isinstance(data, list) and data and isinstance(data[-1], dict):
        return data[-1]
    if isinstance(data, dict):
        return data
    return {}


def _list_to_text(value: object, max_items: int = 8) -> str:
    if isinstance(value, list):
        parts = [str(x).strip() for x in value[:max_items] if str(x).strip()]
        return " ".join(parts)
    return str(value or "").strip()


def _content_hashtags(latest: dict) -> list[str]:
    if not METADATA_USE_CONTENT_HASHTAGS:
        return []
    return _clean_hashtags(latest.get("hashtags"))


def _ensure_hashtags(description: str, hashtags: list[str] | str) -> str:
    hashtag_list = _clean_hashtags(hashtags)
    if not hashtag_list:
        return description

    hashtag_string = " ".join(hashtag_list)
    description = description.replace(POPULAR_HASHTAGS, "").strip()
    description = re.sub(r"(?:^|\n)(?:#[A-Za-z0-9_]+\s*){3,}\s*$", "", description).strip()

    if hashtag_string not in description:
        description = f"{description}\n\n{hashtag_string}".strip()
    return description


def _metadata_plan_is_valid(plan: dict) -> tuple[bool, str]:
    if not isinstance(plan, dict):
        return False, "metadata plan is not an object."

    title = _clean_title(plan.get("title", ""))
    description = _clean_description(plan.get("description", ""))
    tags = _clean_tags(plan.get("tags"))
    hashtags = _clean_hashtags(plan.get("hashtags"))

    if not title:
        return False, "missing title."
    if not description:
        return False, "missing description."
    if len(tags) < METADATA_TAG_MIN_COUNT:
        return False, f"too few tags ({len(tags)})."
    if len(hashtags) < METADATA_HASHTAG_MIN_COUNT:
        return False, f"too few hashtags ({len(hashtags)})."

    return True, ""


def _sanitize_metadata_plan(plan: dict, fallback_title: str, fallback_description: str) -> dict:
    title = _clean_title(plan.get("title") or fallback_title)
    description = _clean_description(plan.get("description") or fallback_description)
    tags = _clean_tags(plan.get("tags"))
    hashtags = _clean_hashtags(plan.get("hashtags"))

    if len(tags) > METADATA_TAG_MAX_COUNT:
        tags = tags[:METADATA_TAG_MAX_COUNT]
    if len(hashtags) > METADATA_HASHTAG_MAX_COUNT:
        hashtags = hashtags[:METADATA_HASHTAG_MAX_COUNT]

    if not tags:
        tags = _clean_tags(DEFAULT_TAGS)
    if not hashtags:
        hashtags = _clean_hashtags(POPULAR_HASHTAGS)

    description = _ensure_hashtags(description, hashtags)

    return {
        "title": title,
        "description": description,
        "tags": tags,
        "hashtags": hashtags,
    }


def _build_metadata_prompt(latest: dict, fallback_title: str, fallback_description: str) -> str:
    script = latest.get("script")
    script_text = _list_to_text(script, max_items=6)
    image_text = _list_to_text(latest.get("image"), max_items=4)
    retention_text = _list_to_text(latest.get("retention_triggers"), max_items=8)
    existing_hashtags = _content_hashtags(latest)
    payload = {
        "current_title": fallback_title,
        "current_description": fallback_description[:700],
        "content_title": latest.get("title", ""),
        "hook": latest.get("hook", ""),
        "caption": latest.get("caption", ""),
        "thumbnail_text": latest.get("thumbnail_text", ""),
        "trend": latest.get("trend", ""),
        "background_music": latest.get("background_music", ""),
        "script_preview": script_text,
        "image_mood_preview": image_text,
        "retention_triggers": retention_text,
        "existing_hashtags": existing_hashtags,
    }

    return (
        "You are a YouTube Shorts SEO metadata editor for a safe true-crime mystery channel. "
        "Create upload metadata using ONLY the provided video facts. Do not invent names, dates, "
        "locations, victims, suspects, or case details. Package the content for curiosity and retention. "
        "Avoid graphic wording, gore, insults, clickbait lies, and disrespectful slang. "
        "Prefer short curiosity titles over boring documentary titles. "
        "Return ONE valid JSON object ONLY with exactly these fields: "
        "{\"title\":\"...\",\"description\":\"...\",\"tags\":[\"...\"],\"hashtags\":[\"#...\"]}. "
        f"Rules: title must be <= {METADATA_TITLE_MAX_CHARS} characters. "
        f"description must be <= {METADATA_DESCRIPTION_MAX_CHARS} characters and include a short viewer question. "
        f"tags must be {METADATA_TAG_MIN_COUNT}-{METADATA_TAG_MAX_COUNT} YouTube upload tags without # symbols. "
        f"hashtags must be {METADATA_HASHTAG_MIN_COUNT}-{METADATA_HASHTAG_MAX_COUNT} hashtags, each starting with #. "
        "Include broad Shorts tags and topic-specific tags when supported by the provided facts. "
        "Output JSON only, no markdown, no code block, no commentary. "
        f"VIDEO DATA:\n{json.dumps(payload, ensure_ascii=False)}"
    )


def generate_metadata_plan(latest: dict, fallback_title: str, fallback_description: str) -> dict:
    if not METADATA_AI_ENABLED:
        return {}
    if _nvidia_client is None:
        print("[Metadata] NVIDIA_API_KEY not set - using local defaults.")
        return {}

    prompt = _build_metadata_prompt(latest, fallback_title, fallback_description)
    last_error = ""

    for attempt in range(1, METADATA_MAX_ATTEMPTS + 1):
        print(f"\n[Metadata] Attempt {attempt}/{METADATA_MAX_ATTEMPTS} using {METADATA_MODEL} ...")
        try:
            completion = _nvidia_client.chat.completions.create(
                model=METADATA_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You must output ONLY valid JSON. No markdown, no code fences, "
                            "no reasoning, no analysis, no extra commentary."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=METADATA_TEMPERATURE,
                top_p=METADATA_TOP_P,
                max_tokens=METADATA_MAX_TOKENS,
                extra_body={
                    "chat_template_kwargs": {"enable_thinking": METADATA_ENABLE_THINKING},
                    "reasoning_budget": METADATA_REASONING_BUDGET if METADATA_ENABLE_THINKING else 0,
                },
                stream=False,
                stop=["\n[Reasoning]", "\nReasoning:", "\n[Analysis]", "\nAnalysis:"],
            )
            raw_output = (completion.choices[0].message.content or "").strip()
            raw_output = _strip_reasoning_lines(raw_output)
            plan = _parse_json_object(raw_output)
            ok, reason = _metadata_plan_is_valid(plan)
            if not ok:
                last_error = reason
                print(_safe_log_text(f"[Metadata] Validation failed: {reason}"))
                continue

            clean_plan = _sanitize_metadata_plan(plan, fallback_title, fallback_description)
            print(_safe_log_text(f"[Metadata] Title: {clean_plan['title']}"))
            print(_safe_log_text(f"[Metadata] Hashtags: {' '.join(clean_plan['hashtags'])}"))
            return clean_plan

        except Exception as exc:
            last_error = str(exc)
            print(_safe_log_text(f"[Metadata] API error on attempt {attempt}: {exc}"))
            if attempt < METADATA_MAX_ATTEMPTS:
                print("[Metadata] Retrying...")

    print(_safe_log_text(f"[Metadata] All attempts failed ({last_error}). Using local defaults."))
    return {}


def generate_hashtags(title: str, description: str, max_retries: int = 3) -> tuple[str, str]:
    if _nvidia_client is None:
        print("[Hashtags] NVIDIA_API_KEY not set - using default hashtags.")
        return POPULAR_HASHTAGS, ""

    max_retries = max(1, int(os.getenv("YOUTUBE_HASHTAG_MAX_ATTEMPTS", str(max_retries))))
    model = os.getenv("YOUTUBE_HASHTAG_MODEL", METADATA_MODEL)
    max_tokens = int(os.getenv("YOUTUBE_HASHTAG_MAX_TOKENS", "1024"))
    temperature = float(os.getenv("YOUTUBE_HASHTAG_TEMPERATURE", "0.35"))
    top_p = float(os.getenv("YOUTUBE_HASHTAG_TOP_P", "0.9"))
    enable_thinking = os.getenv("YOUTUBE_HASHTAG_ENABLE_THINKING", "1") == "1"
    reasoning_budget = int(os.getenv("YOUTUBE_HASHTAG_REASONING_BUDGET", "1024"))

    prompt = (
        "You are a YouTube Shorts SEO expert. Generate content-matched hashtags only.\n\n"
        f"TITLE: {title}\n\n"
        f"DESCRIPTION: {description[:700]}\n\n"
        f"Generate {METADATA_HASHTAG_MIN_COUNT}-{METADATA_HASHTAG_MAX_COUNT} hashtags for this video.\n"
        "STRICT FORMAT RULES:\n"
        "- Output ONLY one valid JSON object: {\"hashtags\":[\"#tagOne\",\"#tagTwo\"]}\n"
        "- Every hashtag MUST start with #\n"
        "- No explanations, no markdown, no extra fields\n"
        "- No spaces inside hashtags; use camelCase or simple words\n"
        "- Do not invent case facts"
    )

    last_error = None

    for attempt in range(1, max_retries + 1):
        print(f"\n[Hashtags] Attempt {attempt}/{max_retries} using {model} ...")

        try:
            completion = _nvidia_client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You must output ONLY valid JSON. No markdown, no code fences, "
                            "no reasoning, no analysis."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                extra_body={
                    "chat_template_kwargs": {"enable_thinking": enable_thinking},
                    "reasoning_budget": reasoning_budget if enable_thinking else 0,
                },
                stream=False,
                stop=["\n[Reasoning]", "\nReasoning:", "\n[Analysis]", "\nAnalysis:"],
            )

            raw_output = (completion.choices[0].message.content or "").strip()
            raw_output = _strip_reasoning_lines(raw_output)
            data = _parse_json_object(raw_output)
            hashtags = _clean_hashtags(data.get("hashtags") if data else raw_output)

            if len(hashtags) < 5:
                raise ValueError(f"Too few hashtags ({len(hashtags)}).")

            hashtag_string = " ".join(hashtags)
            print(_safe_log_text(f"[Hashtags] OK {hashtag_string}"))
            return hashtag_string, ""

        except ValueError as exc:
            last_error = exc
            print(_safe_log_text(f"\n[Hashtags] Format error on attempt {attempt}: {exc}"))
            if attempt < max_retries:
                print("[Hashtags] Retrying...")

        except Exception as exc:
            last_error = exc
            print(_safe_log_text(f"\n[Hashtags] API error on attempt {attempt}: {exc}"))
            if attempt < max_retries:
                print("[Hashtags] Retrying...")

    print(f"[Hashtags] All {max_retries} attempts failed ({last_error}). Using defaults.")
    return POPULAR_HASHTAGS, ""


def _write_json_env(payload: str, path: Path, label: str) -> None:
    if not payload:
        return
    try:
        data = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"{label} is not valid JSON.") from exc
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


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


def _load_token_data() -> dict | None:
    if TOKEN_JSON:
        try:
            return json.loads(TOKEN_JSON)
        except json.JSONDecodeError as exc:
            raise RuntimeError("YOUTUBE_TOKEN_JSON is not valid JSON.") from exc
    if TOKEN_FILE.exists():
        try:
            return json.loads(TOKEN_FILE.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Stored OAuth token at {TOKEN_FILE} is not valid JSON.") from exc
    return None


def _load_credentials():
    data = _load_token_data()
    if data is None:
        return None
    scopes = _normalize_scopes(data.get("scopes"))
    return Credentials.from_authorized_user_info(data, scopes or None)


def _credential_scopes(creds: Credentials) -> list[str]:
    return _normalize_scopes(creds.scopes)


def _missing_scopes(creds: Credentials, required_scopes: list[str]) -> list[str]:
    granted = set(_credential_scopes(creds))
    return [scope for scope in required_scopes if scope not in granted]


def _save_credentials(creds: Credentials) -> None:
    TOKEN_FILE.parent.mkdir(parents=True, exist_ok=True)
    TOKEN_FILE.write_text(creds.to_json(), encoding="utf-8")


def _run_oauth_flow(scopes: list[str] | None = None) -> Credentials:
    requested_scopes = _normalize_scopes(scopes) or list(UPLOAD_SCOPES)
    if not CLIENT_SECRETS.exists():
        raise RuntimeError(
            f"Missing client secrets at {CLIENT_SECRETS}. "
            "Create OAuth client credentials in Google Cloud Console."
        )
    flow = InstalledAppFlow.from_client_secrets_file(str(CLIENT_SECRETS), requested_scopes)
    # Force an offline consent so Google returns a refresh token we can reuse.
    return flow.run_local_server(port=0, access_type="offline", prompt="consent")


def _refresh_error_details(exc: RefreshError) -> tuple[str, str]:
    error = ""
    description = ""
    for arg in exc.args:
        if isinstance(arg, dict):
            error = str(arg.get("error") or error)
            description = str(arg.get("error_description") or description)
        elif isinstance(arg, str) and not description:
            description = arg
    return error, description or str(exc)


def _refresh_credentials(creds: Credentials) -> Credentials:
    try:
        creds.refresh(Request())
        return creds
    except RefreshError as exc:
        error, description = _refresh_error_details(exc)
        if NON_INTERACTIVE:
            if error == "invalid_scope":
                raise RuntimeError(
                    "Google rejected the stored YouTube refresh token in CI because it does not "
                    "include the requested OAuth scope. Existing upload-only tokens can still be "
                    "used for uploads, but channel history requires a new token. Run "
                    "`python generate_youtube_token.py --history` locally and update the GitHub "
                    "secret YOUTUBE_TOKEN_JSON to enable history checks."
                ) from exc
            if error == "invalid_grant":
                raise RuntimeError(
                    "Google rejected the stored YouTube refresh token in CI "
                    "(invalid_grant: expired or revoked). Set the OAuth consent screen "
                    "to In production, generate a new refresh token locally, and update "
                    "the GitHub secret YOUTUBE_TOKEN_JSON."
                ) from exc
            raise RuntimeError(
                f"Google rejected the stored YouTube refresh token in CI ({error or 'refresh_error'}: "
                f"{description})."
            ) from exc
        print("Stored YouTube refresh token was rejected. Opening the browser to re-authorize...")
        return _run_oauth_flow(_credential_scopes(creds) or list(UPLOAD_SCOPES))


def _require_scopes(creds: Credentials, required_scopes: list[str]) -> None:
    missing = _missing_scopes(creds, required_scopes)
    if not missing:
        return

    if UPLOAD_SCOPE in missing:
        raise RuntimeError(
            "The stored YouTube OAuth token does not include youtube.upload. "
            "Generate a fresh upload token and update YOUTUBE_TOKEN_JSON."
        )

    if HISTORY_SCOPE in missing:
        raise RuntimeError(
            "The stored YouTube OAuth token does not include youtube.readonly. "
            "Run `python generate_youtube_token.py --history` and update YOUTUBE_TOKEN_JSON "
            "to enable channel history checks."
        )

    raise RuntimeError(f"The stored YouTube OAuth token is missing required scopes: {missing}")


def _get_service(
    requested_scopes: list[str] | None = None,
    *,
    force_reauth: bool = False,
):
    required_scopes = _normalize_scopes(requested_scopes) or list(UPLOAD_SCOPES)
    if CLIENT_SECRETS_JSON and not CLIENT_SECRETS.exists():
        _write_json_env(CLIENT_SECRETS_JSON, CLIENT_SECRETS, "YOUTUBE_CLIENT_SECRETS_JSON")
    creds = None if force_reauth else _load_credentials()
    if not creds:
        if NON_INTERACTIVE:
            raise RuntimeError(
                "Missing OAuth credentials. Provide YOUTUBE_TOKEN_JSON (preferred) or "
                "mount YOUTUBE_TOKEN_FILE in GitHub Actions."
            )
        creds = _run_oauth_flow(required_scopes)
    else:
        missing = _missing_scopes(creds, required_scopes)
        if missing:
            if NON_INTERACTIVE:
                _require_scopes(creds, required_scopes)
            creds = _run_oauth_flow(required_scopes)
    if not creds.valid:
        if creds.expired and creds.refresh_token:
            creds = _refresh_credentials(creds)
        else:
            if NON_INTERACTIVE:
                raise RuntimeError(
                    "OAuth credentials are invalid/expired and cannot be refreshed in CI. "
                    "Recreate a refresh token and set YOUTUBE_TOKEN_JSON."
                )
            creds = _run_oauth_flow(required_scopes)
    _require_scopes(creds, required_scopes)
    _save_credentials(creds)
    return build("youtube", "v3", credentials=creds)


def _load_defaults() -> Tuple[str, str]:
    title = "Untitled Short"
    description = "Uploaded via ownytub."
    latest = _load_latest_content()

    if latest:
        title = str(latest.get("title") or title).strip() or title
        caption = str(latest.get("caption") or "").strip()
        hook = str(latest.get("hook") or "").strip()
        script = latest.get("script")

        if caption:
            description = caption
        elif hook:
            description = hook
        elif isinstance(script, list):
            description = " ".join(str(x) for x in script[:3] if str(x).strip())
        elif script:
            description = str(script)

    content_hashtags = _content_hashtags(latest)
    if content_hashtags:
        description = _ensure_hashtags(description, content_hashtags)
    elif DESCRIPTION_SUFFIX and DESCRIPTION_SUFFIX.strip() not in description:
        description = f"{description}\n\n{DESCRIPTION_SUFFIX.strip()}"

    if KEYWORDS_LINE:
        description = f"{description}\n\n{KEYWORDS_LINE}"
    else:
        keywords = ", ".join(t.strip() for t in DEFAULT_TAGS if t.strip())
        if keywords:
            description = f"{description}\n\n{keywords}"
    if REFERENCE_TEXT:
        description = f"{description}\n\n{REFERENCE_TEXT}"
    return _clean_title(title), _clean_description(description, max_chars=4900)


def _prompt_text(label: str, default: str) -> str:
    if NON_INTERACTIVE:
        return default
    value = input(f"{label} [{default}]: ").strip()
    return value or default


def _list_videos() -> List[Path]:
    videos: List[Path] = []
    if VIDEO_QUEUE_DIR.exists():
        videos = sorted(VIDEO_QUEUE_DIR.glob("*.mp4"))
    if not videos and DEFAULT_VIDEO.exists():
        videos = [DEFAULT_VIDEO]
    if not videos:
        raise RuntimeError("No videos found to upload.")
    return videos


def _make_title(base: str, index: int, total: int) -> str:
    if "{n}" in base:
        return base.format(n=index + 1)
    if total > 1:
        return f"{base} #{index + 1}"
    return base


def _upload_video(
    youtube,
    video_path: Path,
    title: str,
    description: str,
    tags: list[str] | None = None,
) -> str:
    upload_tags = _clean_tags(tags if tags else DEFAULT_TAGS)
    if not upload_tags:
        upload_tags = _clean_tags(DEFAULT_TAGS)

    body = {
        "snippet": {
            "title": _clean_title(title, max_chars=100),
            "description": _clean_description(description, max_chars=5000),
            "tags": upload_tags,
            "categoryId": CATEGORY_ID,
        },
        "status": {"privacyStatus": PRIVACY_STATUS},
    }
    media = MediaFileUpload(str(video_path), chunksize=-1, resumable=True)
    request = youtube.videos().insert(part="snippet,status", body=body, media_body=media)
    response = None
    while response is None:
        status, response = request.next_chunk()
        if status:
            print(f"Upload progress: {int(status.progress() * 100)}%")
    video_id = response.get("id")
    print(f"Uploaded: {video_id}")
    return video_id


def _http_error_reason(exc: HttpError) -> str:
    try:
        data = json.loads(exc.content.decode("utf-8", errors="replace"))
    except Exception:
        return ""
    errors = data.get("error", {}).get("errors", [])
    if errors and isinstance(errors[0], dict):
        return str(errors[0].get("reason") or "")
    return ""


def _upload_thumbnail(youtube, video_id: str, thumbnail: Path) -> None:
    if not thumbnail.exists():
        print(f"Thumbnail not found, skipping: {thumbnail}")
        return
    media = MediaFileUpload(str(thumbnail), mimetype="image/jpeg")
    request = youtube.thumbnails().set(videoId=video_id, media_body=media)
    for attempt in range(1, 4):
        try:
            request.execute()
            print("Thumbnail uploaded.")
            return
        except HttpError as exc:
            status = exc.resp.status if exc.resp is not None else None
            reason = _http_error_reason(exc)
            if status == 403:
                print("Thumbnail upload forbidden (403). Skipping and continuing.")
                return
            if status == 429 or reason == "uploadRateLimitExceeded":
                print("Thumbnail upload rate-limited. Skipping thumbnail and continuing.")
                return
            if status == 404 and attempt < 3:
                sleep_for = 10 * attempt
                print(f"Thumbnail not ready (404). Retrying in {sleep_for}s (attempt {attempt}/3)...")
                time.sleep(sleep_for)
                continue
            if status == 404:
                print("Thumbnail upload failed (404). Skipping and continuing.")
                return
            raise


def _pick_thumbnail_for_video(video_path: Path) -> Path:
    candidate = video_path.with_suffix(".jpg")
    if candidate.exists():
        return candidate
    candidate = video_path.with_suffix(".jpeg")
    if candidate.exists():
        return candidate
    candidate = video_path.with_suffix(".png")
    if candidate.exists():
        return candidate
    return THUMBNAIL_PATH


def main() -> None:
    youtube = _get_service()
    videos = _list_videos()

    latest_content = _load_latest_content()
    default_title, default_description = _load_defaults()

    base_title = os.getenv("YOUTUBE_TITLE")
    base_description = os.getenv("YOUTUBE_DESCRIPTION")

    if not base_title:
        base_title = _prompt_text("Title", default_title)
    if not base_description:
        base_description = _prompt_text("Description", default_description)

    metadata_plan = generate_metadata_plan(latest_content, base_title, base_description)

    dynamic_tags = _clean_tags(DEFAULT_TAGS)

    if metadata_plan:
        if not os.getenv("YOUTUBE_TITLE"):
            base_title = metadata_plan["title"]
        if not os.getenv("YOUTUBE_DESCRIPTION"):
            base_description = metadata_plan["description"]
        dynamic_tags = _clean_tags(metadata_plan.get("tags")) or dynamic_tags
    else:
        generated_hashtags, _reasoning = generate_hashtags(base_title, base_description)
        if generated_hashtags != POPULAR_HASHTAGS:
            base_description = _ensure_hashtags(base_description, generated_hashtags)

    total_uploads = UPLOADS_PER_DAY if LOOP_UPLOADS else min(UPLOADS_PER_DAY, len(videos))
    if MAX_SUCCESS_UPLOADS > 0:
        total_uploads = min(total_uploads, MAX_SUCCESS_UPLOADS)
    if total_uploads <= 0:
        raise RuntimeError("YOUTUBE_UPLOADS_PER_DAY must be > 0")

    if WAIT_SECONDS <= 0:
        WAIT_SECONDS_LOCAL = 0 if NON_INTERACTIVE else (24 * 60 * 60 / max(total_uploads, 1))
    else:
        WAIT_SECONDS_LOCAL = WAIT_SECONDS

    start = datetime.now()
    for i in range(total_uploads):
        video_path = videos[i % len(videos)] if LOOP_UPLOADS else videos[i]
        title = _make_title(base_title, i, total_uploads)
        description = _make_title(base_description, i, total_uploads)

        print(f"\nUploading {video_path.name} ({i + 1}/{total_uploads})")
        video_id = _upload_video(youtube, video_path, title, description, dynamic_tags)
        _upload_thumbnail(youtube, video_id, _pick_thumbnail_for_video(video_path))

        if i < total_uploads - 1:
            next_time = datetime.now() + timedelta(seconds=WAIT_SECONDS_LOCAL)
            print(f"Next upload at ~{next_time.strftime('%Y-%m-%d %H:%M:%S')}")
            time.sleep(WAIT_SECONDS_LOCAL)

    elapsed = datetime.now() - start
    print(f"Done. Uploaded {total_uploads} videos in {elapsed}.")


if __name__ == "__main__":
    main()
