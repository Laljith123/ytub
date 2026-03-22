import json
import os
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Tuple

from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "output"
OUTPUT_JSON = ROOT / "output.json"

SCOPES = ["https://www.googleapis.com/auth/youtube.upload"]

CLIENT_SECRETS = Path(os.getenv("YOUTUBE_CLIENT_SECRETS", "client_secrets.json"))
TOKEN_FILE = Path(os.getenv("YOUTUBE_TOKEN_FILE", str(OUTPUT_DIR / "youtube_token.json")))
PRIVACY_STATUS = os.getenv("YOUTUBE_PRIVACY_STATUS", "private")
CATEGORY_ID = os.getenv("YOUTUBE_CATEGORY_ID", "24")
DEFAULT_TAGS = os.getenv("YOUTUBE_TAGS", "shorts,true crime,documentary").split(",")
DESCRIPTION_SUFFIX = os.getenv("YOUTUBE_DESCRIPTION_SUFFIX", "#shorts")

UPLOADS_PER_DAY = int(os.getenv("YOUTUBE_UPLOADS_PER_DAY", "10"))
LOOP_UPLOADS = os.getenv("YOUTUBE_LOOP", "1") == "1"
WAIT_SECONDS = float(os.getenv("YOUTUBE_WAIT_BETWEEN_UPLOADS", "0"))

VIDEO_QUEUE_DIR = Path(os.getenv("UPLOAD_QUEUE_DIR", str(OUTPUT_DIR / "queue")))
DEFAULT_VIDEO = Path(os.getenv("UPLOAD_SINGLE_VIDEO", str(OUTPUT_DIR / "final.mp4")))
THUMBNAIL_PATH = Path(os.getenv("THUMBNAIL_PATH", str(OUTPUT_DIR / "thumbnail.jpg")))
NON_INTERACTIVE = os.getenv("NON_INTERACTIVE", "").lower() in {"1", "true", "yes"} or os.getenv(
    "GITHUB_ACTIONS", ""
) == "true" or os.getenv("CI", "").lower() in {"1", "true", "yes"}

CLIENT_SECRETS_JSON = os.getenv("YOUTUBE_CLIENT_SECRETS_JSON", "")
TOKEN_JSON = os.getenv("YOUTUBE_TOKEN_JSON", "")


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
        if not CLIENT_SECRETS.exists():
            raise RuntimeError(
                f"Missing client secrets at {CLIENT_SECRETS}. "
                "Create OAuth client credentials in Google Cloud Console."
            )
        flow = InstalledAppFlow.from_client_secrets_file(str(CLIENT_SECRETS), SCOPES)
        creds = flow.run_local_server(port=0)

    if not creds.valid:
        if creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            if NON_INTERACTIVE:
                raise RuntimeError(
                    "OAuth credentials are invalid/expired and cannot be refreshed in CI. "
                    "Recreate a refresh token and set YOUTUBE_TOKEN_JSON."
                )
            flow = InstalledAppFlow.from_client_secrets_file(str(CLIENT_SECRETS), SCOPES)
            creds = flow.run_local_server(port=0)

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
    request.execute()
    print("Thumbnail uploaded.")


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

    total_uploads = UPLOADS_PER_DAY if LOOP_UPLOADS else min(UPLOADS_PER_DAY, len(videos))
    if total_uploads <= 0:
        raise RuntimeError("YOUTUBE_UPLOADS_PER_DAY must be > 0")

    if WAIT_SECONDS <= 0:
        WAIT_SECONDS_LOCAL = 24 * 60 * 60 / max(total_uploads, 1)
    else:
        WAIT_SECONDS_LOCAL = WAIT_SECONDS

    start = datetime.now()
    for i in range(total_uploads):
        video_path = videos[i % len(videos)] if LOOP_UPLOADS else videos[i]
        title = _make_title(base_title, i, total_uploads)
        description = _make_title(base_description, i, total_uploads)

        print(f"\nUploading {video_path.name} ({i + 1}/{total_uploads})")
        video_id = _upload_video(youtube, video_path, title, description)
        _upload_thumbnail(youtube, video_id, THUMBNAIL_PATH)

        if i < total_uploads - 1:
            next_time = datetime.now() + timedelta(seconds=WAIT_SECONDS_LOCAL)
            print(f"Next upload at ~{next_time.strftime('%Y-%m-%d %H:%M:%S')}")
            time.sleep(WAIT_SECONDS_LOCAL)

    elapsed = datetime.now() - start
    print(f"Done. Uploaded {total_uploads} videos in {elapsed}.")


if __name__ == "__main__":
    main()
