import json
import os
import re
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

ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "output"
OUTPUT_JSON = ROOT / "output.json"

SCOPES = ["https://www.googleapis.com/auth/youtube.upload"]

CLIENT_SECRETS = Path(os.getenv("YOUTUBE_CLIENT_SECRETS", "client_secrets.json"))
TOKEN_FILE = Path(os.getenv("YOUTUBE_TOKEN_FILE", str(OUTPUT_DIR / "youtube_token.json")))
PRIVACY_STATUS = "public"
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

_nvidia_client = None
if os.getenv("NVIDIA_API_KEY"):
    _nvidia_client = OpenAI(
        base_url="https://integrate.api.nvidia.com/v1",
        api_key=os.getenv("NVIDIA_API_KEY"),
    )

def generate_hashtags(title: str, description: str, max_retries: int = 3) -> tuple[str, str]:
    if _nvidia_client is None:
        print("[Hashtags] NVIDIA_API_KEY not set — using default hashtags.")
        return POPULAR_HASHTAGS, ""

    prompt = (
        f"You are a YouTube SEO expert. A video has this title and description:\n\n"
        f"TITLE: {title}\n\n"
        f"DESCRIPTION: {description[:500]}\n\n"
        f"Generate exactly 15 trending YouTube hashtags that best fit this content.\n\n"
        f"STRICT FORMAT RULES:\n"
        f"- Output ONLY a single line of space-separated hashtags\n"
        f"- Every hashtag MUST start with #\n"
        f"- No numbers, bullets, explanations, or extra lines\n"
        f"- No spaces inside hashtags (use camelCase)\n\n"
        f"Example output:\n"
        f"#shorts #viral #truecrime #mystery #coldcase #crime #documentary "
        f"#unsolved #investigation #trending #ytshorts #youtubeshorts #casefile "
        f"#missingperson #shortsvideo"
    )

    reasoning_captured = ""
    last_error = None

    for attempt in range(1, max_retries + 1):
        print(f"\n[Hashtags] Attempt {attempt}/{max_retries} ...")
        reasoning_parts: list[str] = []
        content_parts: list[str] = []

        try:
            completion = _nvidia_client.chat.completions.create(
                model="nvidia/nemotron-super-49b-v1",
                messages=[{"role": "user", "content": prompt}],
                temperature=1,
                top_p=0.95,
                max_tokens=512,
                extra_body={
                    "chat_template_kwargs": {"enable_thinking": True},
                    "reasoning_budget": 2048,
                },
                stream=True,
            )

            for chunk in completion:
                if not chunk.choices:
                    continue
                reasoning = getattr(chunk.choices[0].delta, "reasoning_content", None)
                if reasoning:
                    reasoning_parts.append(reasoning)
                    print(reasoning, end="", flush=True)
                if chunk.choices[0].delta.content is not None:
                    content_parts.append(chunk.choices[0].delta.content)

            reasoning_captured = "".join(reasoning_parts)
            raw_output = "".join(content_parts).strip()
            print(f"\n[Hashtags] Raw output: {raw_output}")

            hashtags = re.findall(r"#\w+", raw_output)

            if not hashtags:
                raise ValueError(f"No hashtags found in output: '{raw_output}'")
            if len(hashtags) < 5:
                raise ValueError(
                    f"Too few hashtags ({len(hashtags)}), expected at least 5. Got: {hashtags}"
                )

            hashtag_string = " ".join(hashtags)
            print(f"[Hashtags] ✅ {hashtag_string}")
            return hashtag_string, reasoning_captured

        except ValueError as exc:
            last_error = exc
            print(f"\n[Hashtags] ⚠️  Format error on attempt {attempt}: {exc}")
            if attempt < max_retries:
                print("[Hashtags] Retrying...")

        except Exception as exc:
            last_error = exc
            print(f"\n[Hashtags] ❌ API error on attempt {attempt}: {exc}")
            if attempt < max_retries:
                print("[Hashtags] Retrying...")

    print(f"[Hashtags] All {max_retries} attempts failed ({last_error}). Using defaults.")
    return POPULAR_HASHTAGS, reasoning_captured

def _write_json_env(payload: str, path: Path, label: str) -> None:
    if not payload:
        return
    try:
        data = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"{label} is not valid JSON.") from exc
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")

def _load_credentials():
    if TOKEN_JSON:
        try:
            data = json.loads(TOKEN_JSON)
        except json.JSONDecodeError as exc:
            raise RuntimeError("YOUTUBE_TOKEN_JSON is not valid JSON.") from exc
        return Credentials.from_authorized_user_info(data, SCOPES)
    if TOKEN_FILE.exists():
        return Credentials.from_authorized_user_file(str(TOKEN_FILE), SCOPES)
    return None

def _run_oauth_flow() -> Credentials:
    if not CLIENT_SECRETS.exists():
        raise RuntimeError(
            f"Missing client secrets at {CLIENT_SECRETS}. "
            "Create OAuth client credentials in Google Cloud Console."
        )
    flow = InstalledAppFlow.from_client_secrets_file(str(CLIENT_SECRETS), SCOPES)
    # Force an offline consent so Google returns a refresh token we can reuse.
    return flow.run_local_server(port=0, access_type="offline", prompt="consent")

def _refresh_credentials(creds: Credentials) -> Credentials:
    try:
        creds.refresh(Request())
        return creds
    except RefreshError as exc:
        if NON_INTERACTIVE:
            raise RuntimeError(
                "Google rejected the stored YouTube refresh token in CI "
                "(invalid_grant: expired or revoked). Set the OAuth consent screen "
                "to In production, generate a new refresh token locally, and update "
                "the GitHub secret YOUTUBE_TOKEN_JSON."
            ) from exc
        print("Stored YouTube refresh token was rejected. Opening the browser to re-authorize...")
        return _run_oauth_flow()

def _get_service():
    if CLIENT_SECRETS_JSON and not CLIENT_SECRETS.exists():
        _write_json_env(CLIENT_SECRETS_JSON, CLIENT_SECRETS, "YOUTUBE_CLIENT_SECRETS_JSON")
    creds = _load_credentials()
    if not creds:
        if NON_INTERACTIVE:
            raise RuntimeError(
                "Missing OAuth credentials. Provide YOUTUBE_TOKEN_JSON (preferred) or "
                "mount YOUTUBE_TOKEN_FILE in GitHub Actions."
            )
        creds = _run_oauth_flow()
    if not creds.valid:
        if creds.expired and creds.refresh_token:
            creds = _refresh_credentials(creds)
        else:
            if NON_INTERACTIVE:
                raise RuntimeError(
                    "OAuth credentials are invalid/expired and cannot be refreshed in CI. "
                    "Recreate a refresh token and set YOUTUBE_TOKEN_JSON."
                )
            creds = _run_oauth_flow()
    TOKEN_FILE.parent.mkdir(parents=True, exist_ok=True)
    TOKEN_FILE.write_text(creds.to_json())
    return build("youtube", "v3", credentials=creds)

def _load_defaults() -> Tuple[str, str]:
    title = "Untitled Short"
    description = "Uploaded via ownytub."
    if OUTPUT_JSON.exists():
        try:
            data = json.loads(OUTPUT_JSON.read_text(encoding="utf-8"))
            if isinstance(data, list) and data:
                last = data[-1]
                if isinstance(last, dict):
                    title = str(last.get("title") or title)
                    script = last.get("script")
                    if isinstance(script, list):
                        description = " ".join(str(x) for x in script[:3] if str(x).strip())
                    elif script:
                        description = str(script)
        except json.JSONDecodeError:
            pass
    if DESCRIPTION_SUFFIX and DESCRIPTION_SUFFIX.strip() not in description:
        description = f"{description}\n\n{DESCRIPTION_SUFFIX.strip()}"
    if KEYWORDS_LINE:
        description = f"{description}\n\n{KEYWORDS_LINE}"
    else:
        keywords = ", ".join(t.strip() for t in DEFAULT_TAGS if t.strip())
        if keywords:
            description = f"{description}\n\n{keywords}"
    if REFERENCE_TEXT:
        description = f"{description}\n\n{REFERENCE_TEXT}"
    return title, description

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

def _upload_video(youtube, video_path: Path, title: str, description: str) -> str:
    body = {
        "snippet": {
            "title": title,
            "description": description,
            "tags": [t.strip() for t in DEFAULT_TAGS if t.strip()],
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
            if status == 403:
                print("Thumbnail upload forbidden (403). Skipping and continuing.")
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

    default_title, default_description = _load_defaults()

    base_title = os.getenv("YOUTUBE_TITLE")
    base_description = os.getenv("YOUTUBE_DESCRIPTION")

    if not base_title:
        base_title = _prompt_text("Title", default_title)
    if not base_description:
        base_description = _prompt_text("Description", default_description)

    generated_hashtags, _reasoning = generate_hashtags(base_title, base_description)
    if generated_hashtags != POPULAR_HASHTAGS:
        base_description = base_description.replace(POPULAR_HASHTAGS, generated_hashtags)
        if generated_hashtags not in base_description:
            base_description = f"{base_description}\n\n{generated_hashtags}"

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
        video_id = _upload_video(youtube, video_path, title, description)
        _upload_thumbnail(youtube, video_id, _pick_thumbnail_for_video(video_path))

        if i < total_uploads - 1:
            next_time = datetime.now() + timedelta(seconds=WAIT_SECONDS_LOCAL)
            print(f"Next upload at ~{next_time.strftime('%Y-%m-%d %H:%M:%S')}")
            time.sleep(WAIT_SECONDS_LOCAL)

    elapsed = datetime.now() - start
    print(f"Done. Uploaded {total_uploads} videos in {elapsed}.")

if __name__ == "__main__":
    main()
