import base64
import json
import os
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List

import requests
from dotenv import load_dotenv

try:
    from rate_limit import retry_after_seconds, wait_for_provider_slot
except ImportError:  # pragma: no cover - used when imported as generating.images
    from generating.rate_limit import retry_after_seconds, wait_for_provider_slot

NVIDIA_GENAI_BASE_URL = os.getenv("NVIDIA_GENAI_BASE_URL", "https://ai.api.nvidia.com/v1/genai").rstrip("/")
INVOKE_URL = os.getenv("NVIDIA_IMAGE_INVOKE_URL", f"{NVIDIA_GENAI_BASE_URL}/black-forest-labs/flux.1-schnell")
# FLUX.2-klein-4B: NVIDIA-hosted, authenticated only with the static nvapi- key
# (no daily check-in / no human step), unified generation + editing, fast.
NVIDIA_FLUX2_URL = os.getenv("NVIDIA_FLUX2_IMAGE_URL", f"{NVIDIA_GENAI_BASE_URL}/black-forest-labs/flux.2-klein-4b")
# FLUX.2-klein-4B only accepts this fixed set of resolutions; we generate at the
# closest vertical size, then ffmpeg pads/scales to the final 1080x1920 frame.
NVIDIA_FLUX2_SUPPORTED_SIZES = [
    (672, 1568), (688, 1504), (720, 1456), (752, 1392), (800, 1328),
    (832, 1248), (880, 1184), (944, 1104), (1024, 1024), (1104, 944),
    (1184, 880), (1248, 832), (1328, 800), (1392, 752), (1456, 720),
    (1504, 688), (1568, 672),
]
NVIDIA_FLUX2_WIDTH = int(os.getenv("NVIDIA_FLUX2_WIDTH", "752"))
NVIDIA_FLUX2_HEIGHT = int(os.getenv("NVIDIA_FLUX2_HEIGHT", "1392"))
# Hard wall-clock cap for a single backend's request+retry loop, so a stalled
# endpoint can never hang a stage for tens of minutes. With two NVIDIA backends
# this bounds one image to ~2x this value before failing fast.
NVIDIA_IMAGE_MAX_SECONDS = float(os.getenv("NVIDIA_IMAGE_MAX_SECONDS", "150"))
# If every backend fails for one scene, reuse the previous good frame instead of
# aborting the whole video. The first scene still hard-fails if it has no prior image.
IMAGE_FALLBACK_REUSE_LAST = os.getenv("IMAGE_FALLBACK_REUSE_LAST", "1") == "1"

FREETHEAI_BASE_URL = os.getenv("FREETHEAI_BASE_URL", "https://api.freetheai.xyz/v1").rstrip("/")
FREETHEAI_IMAGE_URL = os.getenv("FREETHEAI_IMAGE_URL", f"{FREETHEAI_BASE_URL}/images/generations")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_JSON = Path(os.getenv("OUTPUT_JSON_PATH", str(PROJECT_ROOT / "output.json")))
OUT_DIR = PROJECT_ROOT / "output" / "image"

DEFAULT_MODE = "base"
DEFAULT_SEED = 0
DEFAULT_STEPS = 4
DEFAULT_WIDTH = 1080
DEFAULT_HEIGHT = 1920

IMAGE_BACKEND_ORDER = [
    item.strip().lower()
    for item in os.getenv("IMAGE_BACKEND_ORDER", "cloudflare").split(",")
    if item.strip()
]

CLOUDFLARE_API_TOKEN = os.getenv("CLOUDFLARE_API_TOKEN", "").strip()
CLOUDFLARE_ACCOUNT_ID = os.getenv("CLOUDFLARE_ACCOUNT_ID", "").strip()
CLOUDFLARE_IMAGE_MODEL = os.getenv("CLOUDFLARE_IMAGE_MODEL", "@cf/bytedance/stable-diffusion-xl-lightning").strip()
CLOUDFLARE_IMAGE_RETRIES = int(os.getenv("CLOUDFLARE_IMAGE_RETRIES", "3"))
CLOUDFLARE_IMAGE_BACKOFF = float(os.getenv("CLOUDFLARE_IMAGE_RETRY_BACKOFF", "3"))
FREETHEAI_IMAGE_MODEL = os.getenv("FREETHEAI_IMAGE_MODEL", "img/gpt-image-2").strip() or "img/gpt-image-2"
FREETHEAI_IMAGE_TIMEOUT = int(os.getenv("FREETHEAI_IMAGE_TIMEOUT", "240"))
FREETHEAI_IMAGE_RETRIES = int(os.getenv("FREETHEAI_IMAGE_RETRIES", "3"))
FREETHEAI_IMAGE_BACKOFF = float(os.getenv("FREETHEAI_IMAGE_RETRY_BACKOFF", "2"))
FREETHEAI_IMAGE_SIZE = os.getenv("FREETHEAI_IMAGE_SIZE", "1024x1536").strip()
FREETHEAI_IMAGE_RESPONSE_FORMAT = os.getenv("FREETHEAI_IMAGE_RESPONSE_FORMAT", "b64_json").strip()
FREETHEAI_IMAGE_PROMPT_PREFIX = os.getenv(
    "FREETHEAI_IMAGE_PROMPT_PREFIX",
    (
        "Create a cinematic, safe, non-graphic vertical YouTube Shorts image. "
        "No readable text, no logos, no gore. "
    ),
)

# Generic, model-agnostic prompt enhancement applied to the NVIDIA image backends.
# These are universal photographic/quality cues (not per-topic keyword tables), and
# every value is overridable via env so intent stays dynamic.
IMAGE_PROMPT_PREFIX = os.getenv("IMAGE_PROMPT_PREFIX", FREETHEAI_IMAGE_PROMPT_PREFIX).strip()
IMAGE_PROMPT_QUALITY_SUFFIX = os.getenv(
    "IMAGE_PROMPT_QUALITY_SUFFIX",
    "photorealistic, high detail, sharp focus, natural realistic lighting, "
    "true-to-life colors, professional cinematography, no text, no watermark, no logo",
).strip()


_BASE64_RE = re.compile(r"^[A-Za-z0-9+/=\n\r]+$")


def _looks_like_base64(value: str) -> bool:
    if not isinstance(value, str):
        return False
    if value.startswith("data:image"):
        return True
    if len(value) < 200:
        return False
    return _BASE64_RE.match(value) is not None


def _extract_base64_from_obj(obj: Any) -> str | None:
    if isinstance(obj, str):
        return obj if _looks_like_base64(obj) else None

    if isinstance(obj, dict):
        for key in ("b64_json", "image", "base64", "base64_png", "data"):
            if key in obj and isinstance(obj[key], str) and _looks_like_base64(obj[key]):
                return obj[key]
        for value in obj.values():
            found = _extract_base64_from_obj(value)
            if found:
                return found

    if isinstance(obj, list):
        for item in obj:
            found = _extract_base64_from_obj(item)
            if found:
                return found

    return None


def _save_image_from_response(response: requests.Response, out_path: str) -> None:
    content_type = response.headers.get("Content-Type", "")
    if content_type.startswith("image/"):
        with open(out_path, "wb") as f:
            f.write(response.content)
        return

    body = response.json()
    b64 = None

    if isinstance(body, dict):
        if "image" in body and isinstance(body["image"], str):
            b64 = body["image"]
        elif "images" in body:
            images = body["images"]
            if isinstance(images, str):
                b64 = images
            elif isinstance(images, list) and images:
                first = images[0]
                if isinstance(first, str):
                    b64 = first
                elif isinstance(first, dict):
                    for key in ("b64_json", "image", "base64", "base64_png", "data"):
                        if key in first and isinstance(first[key], str):
                            b64 = first[key]
                            break
        elif "data" in body and isinstance(body["data"], list) and body["data"]:
            first = body["data"][0]
            if isinstance(first, dict):
                for key in ("b64_json", "image", "base64", "base64_png", "data"):
                    if key in first and isinstance(first[key], str):
                        b64 = first[key]
                        break

    if not b64:
        b64 = _extract_base64_from_obj(body)

    if not b64:
        keys = list(body.keys()) if isinstance(body, dict) else type(body)
        raise ValueError(f"Unable to find base64 image data in response. Top-level keys: {keys}")

    if b64.startswith("data:image") and "," in b64:
        b64 = b64.split(",", 1)[1]

    image_bytes = base64.b64decode(b64)
    with open(out_path, "wb") as f:
        f.write(image_bytes)


def _provider_image_prompt(prompt: str) -> str:
    text = str(prompt or "").strip()
    prefix = str(FREETHEAI_IMAGE_PROMPT_PREFIX or "").strip()
    return f"{prefix} {text}".strip()


def _write_base64_image(data_url_or_b64: str, out_path: Path) -> None:
    value = str(data_url_or_b64 or "").strip()
    if not value:
        raise RuntimeError("Image response did not contain image data.")
    if value.startswith("data:image") and "," in value:
        value = value.split(",", 1)[1]
    out_path.write_bytes(base64.b64decode(value))


def _download_image(url: str, out_path: Path, timeout: int) -> None:
    response = requests.get(url, timeout=timeout)
    _raise_for_status_with_detail(response)
    out_path.write_bytes(response.content)


def _image_url_from_obj(obj: Any) -> str | None:
    if isinstance(obj, dict):
        for key in ("url", "image_url"):
            value = obj.get(key)
            if isinstance(value, str) and value.startswith(("http://", "https://")):
                return value
            if isinstance(value, dict):
                nested = value.get("url")
                if isinstance(nested, str) and nested.startswith(("http://", "https://")):
                    return nested
        for value in obj.values():
            found = _image_url_from_obj(value)
            if found:
                return found
    if isinstance(obj, list):
        for item in obj:
            found = _image_url_from_obj(item)
            if found:
                return found
    return None


def _save_image_payload(body: Any, out_path: Path, timeout: int) -> None:
    b64 = _extract_base64_from_obj(body)
    if b64:
        _write_base64_image(b64, out_path)
        return

    image_url = _image_url_from_obj(body)
    if image_url:
        _download_image(image_url, out_path, timeout)
        return

    keys = list(body.keys()) if isinstance(body, dict) else type(body)
    raise ValueError(f"Unable to find image data in response. Top-level keys: {keys}")


def _free_image_api_key() -> str:
    return _clean_api_key(os.getenv("FREETHEAI_API_KEY") or os.getenv("IMAGE_API_KEY"))


def _save_freetheai_image(prompt: str, out_path: Path) -> None:
    api_key = _free_image_api_key()
    if not api_key:
        raise RuntimeError("FREETHEAI_API_KEY is missing.")

    payload: Dict[str, Any] = {
        "model": FREETHEAI_IMAGE_MODEL,
        "prompt": _provider_image_prompt(prompt),
    }
    if FREETHEAI_IMAGE_SIZE:
        payload["size"] = FREETHEAI_IMAGE_SIZE
    if FREETHEAI_IMAGE_RESPONSE_FORMAT:
        payload["response_format"] = FREETHEAI_IMAGE_RESPONSE_FORMAT

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    response = _post_with_retries(
        payload,
        headers=headers,
        timeout=FREETHEAI_IMAGE_TIMEOUT,
        retries=FREETHEAI_IMAGE_RETRIES,
        backoff=FREETHEAI_IMAGE_BACKOFF,
        url=FREETHEAI_IMAGE_URL,
    )
    _raise_for_status_with_detail(response)
    _save_image_payload(response.json(), out_path, FREETHEAI_IMAGE_TIMEOUT)


def _save_nvidia_image(
    prompt_text: str,
    out_path: Path,
    *,
    headers: Dict[str, str],
    mode: str,
    steps: int,
    width: int | None,
    height: int | None,
    timeout: int,
    retries: int,
    backoff: float,
) -> None:
    payload = _build_payload(
        prompt_text,
        seed=DEFAULT_SEED,
        mode=mode,
        steps=steps,
        width=width,
        height=height,
    )
    response = _post_with_retries(
        payload,
        headers=headers,
        timeout=timeout,
        retries=retries,
        backoff=backoff,
        max_total_seconds=NVIDIA_IMAGE_MAX_SECONDS,
    )
    if response.status_code == 422 and width is not None and height is not None:
        payload = _build_payload(
            prompt_text,
            seed=DEFAULT_SEED,
            mode=mode,
            steps=steps,
            width=None,
            height=None,
        )
        response = _post_with_retries(
            payload,
            headers=headers,
            timeout=timeout,
            retries=retries,
            backoff=backoff,
        )
    _raise_for_status_with_detail(response)
    _save_image_from_response(response, str(out_path))


def _enhanced_image_prompt(prompt: str) -> str:
    """Generic prompt enhancer for the NVIDIA image backends.

    Wraps the model-written, scene-specific prompt with universal safety + quality
    cues. It does NOT inject topic/genre keywords, so intent stays dynamic.
    """
    text = str(prompt or "").strip()
    parts = [IMAGE_PROMPT_PREFIX, text] if IMAGE_PROMPT_PREFIX else [text]
    combined = " ".join(part for part in parts if part).strip()
    if IMAGE_PROMPT_QUALITY_SUFFIX:
        combined = f"{combined}, {IMAGE_PROMPT_QUALITY_SUFFIX}".strip(", ").strip()
    return combined or text


def _nearest_flux2_size(width: int | None, height: int | None) -> tuple[int, int]:
    req_w = width or NVIDIA_FLUX2_WIDTH
    req_h = height or NVIDIA_FLUX2_HEIGHT
    if (req_w, req_h) in NVIDIA_FLUX2_SUPPORTED_SIZES:
        return req_w, req_h
    target_ratio = req_w / req_h if req_h else 1.0
    return min(
        NVIDIA_FLUX2_SUPPORTED_SIZES,
        key=lambda wh: abs((wh[0] / wh[1]) - target_ratio),
    )


def _save_nvidia_flux2_image(
    prompt_text: str,
    out_path: Path,
    *,
    headers: Dict[str, str],
    timeout: int,
    retries: int,
    backoff: float,
) -> None:
    width, height = _nearest_flux2_size(NVIDIA_FLUX2_WIDTH, NVIDIA_FLUX2_HEIGHT)
    try:
        seed = int(os.getenv("NVIDIA_FLUX2_SEED", str(DEFAULT_SEED)) or DEFAULT_SEED)
    except ValueError:
        seed = DEFAULT_SEED
    steps = _get_optional_int_env("NVIDIA_FLUX2_STEPS")

    full_payload: Dict[str, Any] = {
        "prompt": prompt_text,
        "width": width,
        "height": height,
        "seed": seed,
    }
    if steps is not None and steps > 0:
        full_payload["steps"] = steps
    cfg_scale = os.getenv("NVIDIA_FLUX2_CFG_SCALE")
    if cfg_scale:
        try:
            full_payload["cfg_scale"] = float(cfg_scale)
        except ValueError:
            pass

    # Progressively strip optional fields if the endpoint rejects the request
    # body (400/422), so a schema mismatch never hard-fails the backend.
    payload_attempts: List[Dict[str, Any]] = [
        full_payload,
        {"prompt": prompt_text, "width": width, "height": height, "seed": seed},
        {"prompt": prompt_text},
    ]

    last_response: requests.Response | None = None
    seen: set[str] = set()
    for payload in payload_attempts:
        key = json.dumps(payload, sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        response = _post_with_retries(
            payload,
            headers=headers,
            timeout=timeout,
            retries=retries,
            backoff=backoff,
            url=NVIDIA_FLUX2_URL,
            max_total_seconds=NVIDIA_IMAGE_MAX_SECONDS,
        )
        last_response = response
        if response.status_code not in {400, 422}:
            _raise_for_status_with_detail(response)
            _save_image_from_response(response, str(out_path))
            return

    if last_response is not None:
        _raise_for_status_with_detail(last_response)
    raise RuntimeError("FLUX.2 image request failed before sending any payload.")


def _load_last_image_prompts(path: Path) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list) or not data:
        raise ValueError("output.json does not contain a non-empty list")

    last = data[-1]
    if not isinstance(last, dict):
        raise ValueError("The last item in output.json is not an object")

    images = last.get("image") or last.get("images")
    if images is None:
        raise ValueError("No 'image' field found in the last item of output.json")

    if isinstance(images, str):
        return [images]
    if not isinstance(images, list):
        raise ValueError("The 'image' field in the last item is not a list or string")

    prompts = [str(x) for x in images if str(x).strip()]
    if not prompts:
        raise ValueError("The 'image' list in the last item is empty")

    return prompts


def _get_optional_int_env(name: str) -> int | None:
    value = os.getenv(name)
    if value is None or value == "":
        return None
    try:
        return int(value)
    except ValueError as exc:
        raise RuntimeError(f"{name} must be an integer, got {value!r}") from exc


def _build_payload(
    prompt: str,
    seed: int,
    *,
    mode: str,
    steps: int,
    width: int | None,
    height: int | None,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "prompt": prompt,
        "mode": mode,
        "seed": seed,
        "steps": steps,
    }

    if (width is None) ^ (height is None):
        raise RuntimeError("Set both NVIDIA_IMAGE_WIDTH and NVIDIA_IMAGE_HEIGHT to use a custom size.")
    if width is not None and height is not None:
        payload["width"] = width
        payload["height"] = height

    return payload


def _with_aspect_hint(prompt: str, width: int | None, height: int | None) -> str:
    if width is None or height is None:
        return prompt
    if re.search(r"\b9\s*:\s*16\b", prompt) or re.search(r"\b1080\s*x\s*1920\b", prompt):
        return prompt
    return f"{prompt}, vertical 9:16, {width}x{height}"


def _raise_for_status_with_detail(response: requests.Response) -> None:
    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        detail: Any
        try:
            detail = response.json()
        except ValueError:
            detail = response.text
        raise RuntimeError(f"Request failed ({response.status_code}): {detail}") from exc


def _post_with_retries(
    payload: Dict[str, Any],
    *,
    headers: Dict[str, str],
    timeout: int,
    retries: int,
    backoff: float,
    url: str = INVOKE_URL,
    max_total_seconds: float | None = None,
) -> requests.Response:
    last_exc: Exception | None = None
    last_response: requests.Response | None = None
    current_timeout = timeout
    start = time.monotonic()

    def _budget_exhausted(extra: float = 0.0) -> bool:
        if max_total_seconds is None:
            return False
        return (time.monotonic() - start) + extra >= max_total_seconds

    for attempt in range(1, retries + 1):
        if _budget_exhausted():
            break
        try:
            if "api.freetheai.xyz" in str(url).lower():
                wait_for_provider_slot("FreeTheAi", rpm_env="FREETHEAI_RPM_LIMIT", default_rpm=10)
            response = requests.post(url, headers=headers, json=payload, timeout=current_timeout)
            last_response = response
            if response.status_code in {429, 500, 502, 503, 504}:
                if attempt == retries:
                    return response
                retry_after = response.headers.get("Retry-After")
                if response.status_code == 429:
                    sleep_for = retry_after_seconds(retry_after)
                elif retry_after:
                    sleep_for = retry_after_seconds(retry_after, default=backoff * attempt)
                else:
                    sleep_for = backoff * attempt
                if _budget_exhausted(sleep_for):
                    return response
                print(
                    f"Request returned {response.status_code}. Retrying in {sleep_for:.1f}s "
                    f"(attempt {attempt}/{retries})..."
                )
                time.sleep(sleep_for)
                current_timeout = int(current_timeout * 1.25)
                continue
            return response
        except (requests.exceptions.ReadTimeout, requests.exceptions.ConnectionError) as exc:
            last_exc = exc
            if attempt == retries:
                raise
            sleep_for = backoff * attempt
            if _budget_exhausted(sleep_for):
                raise
            print(
                f"Request timed out. Retrying in {sleep_for:.1f}s "
                f"(attempt {attempt}/{retries})..."
            )
            time.sleep(sleep_for)
            current_timeout = int(current_timeout * 1.25)

    if last_response is not None:
        return last_response
    if last_exc is not None:
        raise RuntimeError(
            f"Request failed within the {max_total_seconds}s budget."
        ) from last_exc
    raise RuntimeError("Request failed after retries.")


def _ensure_target_image(path: Path, width: int, height: int) -> None:
    if shutil.which("ffmpeg") is None:
        print("ffmpeg not found; skipping image conversion.")
        return

    temp_path = path.with_suffix(".tmp.png")
    vf = (
        f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
        f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2,"
        "setsar=1"
    )
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(path),
        "-vf",
        vf,
        "-frames:v",
        "1",
        str(temp_path),
    ]
    subprocess.run(cmd, check=True)
    temp_path.replace(path)


def _cleanup_old_images(out_dir: Path) -> None:
    if os.getenv("NVIDIA_IMAGE_CLEANUP", "1") != "1":
        return
    for pattern in ("image_*.png", "image_*.tmp.png"):
        for path in out_dir.glob(pattern):
            try:
                path.unlink()
            except OSError as exc:
                print(f"Unable to delete {path}: {exc}")


def _clean_api_key(value: str | None) -> str:
    key = str(value or "").strip().strip("\"'")
    if key.lower().startswith("bearer "):
        key = key[7:].strip()
    return key


def _save_cloudflare_image(prompt_text: str, out_path: Path, width: int | None, height: int | None, seed: int) -> None:
    """Generate an image using Cloudflare Workers AI."""
    if not CLOUDFLARE_API_TOKEN or not CLOUDFLARE_ACCOUNT_ID:
        raise RuntimeError("CLOUDFLARE_API_TOKEN or CLOUDFLARE_ACCOUNT_ID is missing.")

    url = f"https://api.cloudflare.com/client/v4/accounts/{CLOUDFLARE_ACCOUNT_ID}/ai/run/{CLOUDFLARE_IMAGE_MODEL}"
    
    headers = {
        "Authorization": f"Bearer {CLOUDFLARE_API_TOKEN}",
        "Content-Type": "application/json"
    }

    payload: Dict[str, Any] = {
        "prompt": _enhanced_image_prompt(prompt_text),
    }
    
    # SDXL models generally perform best at their native resolution.
    # We omit width/height and let the model decide (usually 1024x1024).
    # _ensure_target_image will crop/pad it to 1080x1920 afterwards.

    last_exc: Exception | None = None
    for attempt in range(1, CLOUDFLARE_IMAGE_RETRIES + 1):
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=120)
            _raise_for_status_with_detail(response)
            
            content_type = response.headers.get("Content-Type", "")
            if content_type.startswith("image/"):
                out_path.write_bytes(response.content)
            elif content_type.startswith("application/json"):
                data = response.json()
                if "result" in data and "image" in data["result"]:
                    b64 = data["result"]["image"]
                    _write_base64_image(b64, out_path)
                else:
                    raise RuntimeError(f"Unexpected JSON format from Cloudflare: {data}")
            else:
                out_path.write_bytes(response.content)
                
            # CROSS-CHECK: Verify the image is valid and not corrupted
            if not out_path.exists() or out_path.stat().st_size < 100:
                raise RuntimeError("Generated image is empty or too small.")
                
            from PIL import Image
            try:
                with Image.open(out_path) as img:
                    img.verify()
            except Exception as pil_exc:
                raise RuntimeError(f"Generated image is corrupt: {pil_exc}")
                
            return
        except Exception as exc:
            last_exc = exc
            if attempt < CLOUDFLARE_IMAGE_RETRIES:
                sleep_for = CLOUDFLARE_IMAGE_BACKOFF * attempt
                print(f"Cloudflare image attempt {attempt}/{CLOUDFLARE_IMAGE_RETRIES} failed: {exc}. Retrying in {sleep_for:.0f}s...")
                time.sleep(sleep_for)
    raise RuntimeError(f"Cloudflare image generation failed after {CLOUDFLARE_IMAGE_RETRIES} attempts: {last_exc}") from last_exc


def main() -> None:
    load_dotenv()
    api_key = _clean_api_key(os.getenv("NVIDIA_API_KEY"))
    nvidia_available = bool(api_key and api_key.startswith("nvapi-"))
    cloudflare_available = bool(CLOUDFLARE_API_TOKEN and CLOUDFLARE_ACCOUNT_ID)
    if not api_key:
        print("NVIDIA_API_KEY is not set. NVIDIA image fallback will be skipped.")
    if not cloudflare_available:
        print("CLOUDFLARE_API_TOKEN or ACCOUNT_ID is not set. Cloudflare backend will be skipped.")
    if not IMAGE_BACKEND_ORDER:
        raise RuntimeError("IMAGE_BACKEND_ORDER is empty.")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    _cleanup_old_images(OUT_DIR)

    prompts = _load_last_image_prompts(OUTPUT_JSON)

    headers = {}
    if nvidia_available:
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Accept": "application/json",
        }

    mode = os.getenv("NVIDIA_IMAGE_MODE", DEFAULT_MODE)
    steps = _get_optional_int_env("NVIDIA_IMAGE_STEPS") or DEFAULT_STEPS
    disable_size = os.getenv("NVIDIA_IMAGE_DISABLE_SIZE") == "1"
    add_aspect_hint = os.getenv("NVIDIA_IMAGE_PROMPT_ASPECT", "1") == "1"
    timeout = _get_optional_int_env("NVIDIA_IMAGE_TIMEOUT") or 120
    retries = _get_optional_int_env("NVIDIA_IMAGE_RETRIES") or 3
    backoff = float(os.getenv("NVIDIA_IMAGE_RETRY_BACKOFF", "1.5"))
    delay = float(os.getenv("NVIDIA_IMAGE_DELAY", "1"))
    width = _get_optional_int_env("NVIDIA_IMAGE_WIDTH")
    height = _get_optional_int_env("NVIDIA_IMAGE_HEIGHT")
    if not disable_size and width is None and height is None:
        width = DEFAULT_WIDTH
        height = DEFAULT_HEIGHT

    last_good_path: Path | None = None
    for idx, prompt in enumerate(prompts, start=1):
        filename = f"image_{idx:02d}.png"
        out_path = OUT_DIR / filename
        prompt_text = _with_aspect_hint(prompt, width, height) if add_aspect_hint else prompt

        saved_with = ""
        errors: list[str] = []
        for backend in IMAGE_BACKEND_ORDER:
            if backend in {"cloudflare", "cf"}:
                if not cloudflare_available:
                    errors.append("cloudflare: unavailable (missing secrets)")
                    continue
                try:
                    print(f"Generating image {idx:02d} with Cloudflare {CLOUDFLARE_IMAGE_MODEL}.")
                    current_seed = DEFAULT_SEED + idx
                    _save_cloudflare_image(prompt_text, out_path, width, height, current_seed)
                    saved_with = "Cloudflare"
                    break
                except Exception as exc:
                    errors.append(f"cloudflare: {exc}")
                    print(f"Cloudflare image generation failed for image {idx:02d}: {exc}")
                    continue

            if backend in {"nvidia_flux2", "nvidia-flux2", "flux2", "flux.2", "flux_2"}:
                if not nvidia_available:
                    errors.append("nvidia_flux2: unavailable")
                    continue
                try:
                    print(f"Generating image {idx:02d} with NVIDIA FLUX.2-klein-4B.")
                    _save_nvidia_flux2_image(
                        _enhanced_image_prompt(prompt_text),
                        out_path,
                        headers=headers,
                        timeout=timeout,
                        retries=retries,
                        backoff=backoff,
                    )
                    saved_with = "NVIDIA FLUX.2-klein-4B"
                    break
                except Exception as exc:
                    errors.append(f"nvidia_flux2: {exc}")
                    print(f"NVIDIA FLUX.2 image generation failed for image {idx:02d}: {exc}")
                    continue

            if backend in {"freetheai", "free-the-ai", "freeai"}:
                try:
                    print(f"Generating image {idx:02d} with FreeTheAi model {FREETHEAI_IMAGE_MODEL}.")
                    _save_freetheai_image(prompt_text, out_path)
                    saved_with = "FreeTheAi"
                    break
                except Exception as exc:
                    errors.append(f"freetheai: {exc}")
                    print(f"FreeTheAi image generation failed for image {idx:02d}: {exc}")
                    continue

            if backend == "nvidia":
                if not nvidia_available:
                    errors.append("nvidia: unavailable")
                    continue
                try:
                    print(f"Generating image {idx:02d} with NVIDIA FLUX.1-schnell.")
                    _save_nvidia_image(
                        _enhanced_image_prompt(prompt_text),
                        out_path,
                        headers=headers,
                        mode=mode,
                        steps=steps,
                        width=width,
                        height=height,
                        timeout=timeout,
                        retries=retries,
                        backoff=backoff,
                    )
                    saved_with = "NVIDIA FLUX.1-schnell"
                    break
                except Exception as exc:
                    errors.append(f"nvidia: {exc}")
                    print(f"NVIDIA image generation failed for image {idx:02d}: {exc}")
                    continue

            errors.append(f"{backend}: unsupported image backend")

        if not saved_with:
            if IMAGE_FALLBACK_REUSE_LAST and last_good_path is not None and last_good_path.exists():
                shutil.copy2(last_good_path, out_path)
                saved_with = "reuse-previous-image"
                print(
                    f"All image backends failed for image {idx:02d}; "
                    f"reusing previous image so the video still builds. "
                    f"({' | '.join(errors)})"
                )
            else:
                raise RuntimeError("All image backends failed. " + " | ".join(errors))

        if width is not None and height is not None and not disable_size:
            _ensure_target_image(out_path, width, height)
        last_good_path = out_path
        print(f"Saved {out_path} via {saved_with}")
        if delay > 0:
            time.sleep(delay)


if __name__ == "__main__":
    main()
