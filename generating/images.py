
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

INVOKE_URL = "https://ai.api.nvidia.com/v1/genai/black-forest-labs/flux.1-schnell"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_JSON = PROJECT_ROOT / "output.json"
OUT_DIR = PROJECT_ROOT / "output" / "image"

DEFAULT_MODE = "base"
DEFAULT_SEED = 0
DEFAULT_STEPS = 4
DEFAULT_WIDTH = 1080
DEFAULT_HEIGHT = 1920


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
) -> requests.Response:
    last_exc: Exception | None = None
    current_timeout = timeout
    for attempt in range(1, retries + 1):
        try:
            response = requests.post(INVOKE_URL, headers=headers, json=payload, timeout=current_timeout)
            if response.status_code in {429, 500, 502, 503, 504}:
                if attempt == retries:
                    return response
                retry_after = response.headers.get("Retry-After")
                if retry_after:
                    try:
                        sleep_for = float(retry_after)
                    except ValueError:
                        sleep_for = backoff * attempt
                else:
                    sleep_for = backoff * attempt
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
            print(
                f"Request timed out. Retrying in {sleep_for:.1f}s "
                f"(attempt {attempt}/{retries})..."
            )
            time.sleep(sleep_for)
            current_timeout = int(current_timeout * 1.25)
    raise RuntimeError("Request failed after retries.") from last_exc


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


def main() -> None:
    load_dotenv()
    api_key = os.getenv("NVIDIA_API_KEY")
    if not api_key:
        raise RuntimeError("NVIDIA_API_KEY is not set in environment")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    _cleanup_old_images(OUT_DIR)

    prompts = _load_last_image_prompts(OUTPUT_JSON)

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Accept": "application/json",
    }

    mode = os.getenv("NVIDIA_IMAGE_MODE", DEFAULT_MODE)
    steps = _get_optional_int_env("NVIDIA_IMAGE_STEPS") or DEFAULT_STEPS
    disable_size = os.getenv("NVIDIA_IMAGE_DISABLE_SIZE") == "1"
    add_aspect_hint = os.getenv("NVIDIA_IMAGE_PROMPT_ASPECT", "1") == "1"
    timeout = _get_optional_int_env("NVIDIA_IMAGE_TIMEOUT") or 240
    retries = _get_optional_int_env("NVIDIA_IMAGE_RETRIES") or 3
    backoff = float(os.getenv("NVIDIA_IMAGE_RETRY_BACKOFF", "1.5"))
    delay = float(os.getenv("NVIDIA_IMAGE_DELAY", "1"))
    width = _get_optional_int_env("NVIDIA_IMAGE_WIDTH")
    height = _get_optional_int_env("NVIDIA_IMAGE_HEIGHT")
    if not disable_size and width is None and height is None:
        width = DEFAULT_WIDTH
        height = DEFAULT_HEIGHT

    for idx, prompt in enumerate(prompts, start=1):
        prompt_text = _with_aspect_hint(prompt, width, height) if add_aspect_hint else prompt
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

        filename = f"image_{idx:02d}.png"
        out_path = OUT_DIR / filename
        _save_image_from_response(response, str(out_path))
        if width is not None and height is not None and not disable_size:
            _ensure_target_image(out_path, width, height)
        print(f"Saved {out_path}")
        if delay > 0:
            time.sleep(delay)


if __name__ == "__main__":
    main()
