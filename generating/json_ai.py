import os
import time
from typing import Any

import requests
from openai import OpenAI

try:
    from rate_limit import retry_after_seconds, wait_for_provider_slot
except ImportError:  # pragma: no cover - used when imported as generating.json_ai
    from generating.rate_limit import retry_after_seconds, wait_for_provider_slot


BLUESMINDS_BASE_URL = "https://api.bluesminds.com/v1"
NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
NVIDIA_DEFAULT_MODEL = "nvidia/nemotron-3-ultra-550b-a55b"
BLUESMINDS_MODEL_CANDIDATES = NVIDIA_DEFAULT_MODEL
_MODEL_CACHE: dict[tuple[str, str], set[str]] = {}


def env_value(*names: str, default: str = "") -> str:
    for name in names:
        value = os.getenv(name, "").strip()
        if value:
            return value
    return default


def is_nvidia_base_url(base_url: str) -> bool:
    return "integrate.api.nvidia.com" in str(base_url or "").lower()


def json_api_key(specific_env: str = "", base_url: str = "") -> str:
    if is_nvidia_base_url(base_url):
        names = [name for name in (specific_env, "NVIDIA_API_KEY", "JSON_API_KEY") if name]
        return env_value(*names)

    names = [name for name in (specific_env, "NVIDIA_API_KEY", "JSON_API_KEY") if name]
    return env_value(*names)


def json_base_url(specific_env: str = "") -> str:
    specific = env_value(specific_env) if specific_env else ""
    if specific:
        return specific

    shared = env_value("JSON_BASE_URL")
    if shared:
        return shared

    return env_value("NVIDIA_BASE_URL", default=NVIDIA_BASE_URL)


def json_model(specific_env: str = "", fallback: str = NVIDIA_DEFAULT_MODEL) -> str:
    specific = env_value(specific_env) if specific_env else ""
    if specific:
        return specific

    shared = env_value("JSON_MODEL")
    if shared:
        return shared

    return NVIDIA_DEFAULT_MODEL if fallback == NVIDIA_DEFAULT_MODEL else fallback


def is_bluesminds_base_url(base_url: str) -> bool:
    return "bluesminds.com" in str(base_url or "").lower()


def json_provider_name(base_url: str) -> str:
    normalized = str(base_url or "").lower()
    if is_bluesminds_base_url(normalized):
        return "Bluesminds"
    if is_nvidia_base_url(normalized):
        return "NVIDIA"
    return "OpenAI-compatible"


def json_client(api_key_env: str = "", base_url_env: str = "") -> OpenAI | None:
    base_url = json_base_url(base_url_env)
    key = json_api_key(api_key_env, base_url=base_url)
    if not key:
        return None
    return OpenAI(base_url=base_url, api_key=key)


def provider_supports_nvidia_extra_body(base_url: str) -> bool:
    return is_nvidia_base_url(base_url)


def _models_endpoint(base_url: str) -> str:
    return f"{str(base_url).rstrip('/')}/models"


def _model_id(item: Any) -> str:
    if isinstance(item, str):
        return item.strip()
    if isinstance(item, dict):
        return str(item.get("id") or item.get("name") or item.get("model") or "").strip()
    return str(getattr(item, "id", "") or getattr(item, "name", "") or "").strip()


def _load_available_models(base_url: str, api_key: str) -> set[str]:
    if not base_url or not api_key:
        return set()
    cache_key = (base_url.rstrip("/"), api_key[:10])
    if cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key]

    try:
        response = requests.get(
            _models_endpoint(base_url),
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=float(os.getenv("JSON_MODELS_TIMEOUT", "10")),
        )
        response.raise_for_status()
        payload = response.json()
    except Exception as exc:
        print(f"JSON model discovery skipped for {json_provider_name(base_url)}: {exc}")
        _MODEL_CACHE[cache_key] = set()
        return set()

    raw_items = payload.get("data") if isinstance(payload, dict) else payload
    if not isinstance(raw_items, list):
        raw_items = []
    models = {_model_id(item) for item in raw_items}
    models.discard("")
    _MODEL_CACHE[cache_key] = models
    return models


def _candidate_models(preferred_model: str) -> list[str]:
    raw = env_value("JSON_MODEL_CANDIDATES", default=BLUESMINDS_MODEL_CANDIDATES)
    values = [preferred_model]
    values.extend(part.strip() for part in raw.replace(";", ",").split(",") if part.strip())

    candidates: list[str] = []
    seen: set[str] = set()
    for value in values:
        if not value or value in seen:
            continue
        candidates.append(value)
        seen.add(value)
    return candidates


def resolve_json_model(model: str, base_url: str, api_key: str) -> str:
    if not is_bluesminds_base_url(base_url):
        return model
    if env_value("JSON_MODEL_DISCOVERY_ENABLED", default="1").lower() in {"0", "false", "no", "off"}:
        return model

    available = _load_available_models(base_url, api_key)
    if not available:
        return model
    if model in available:
        return model

    for candidate in _candidate_models(model):
        if candidate in available:
            print(f"JSON model '{model}' is unavailable; using '{candidate}' from /v1/models.")
            return candidate

    print(f"JSON model '{model}' was not found in /v1/models; using it anyway so the API returns the exact error.")
    return model


def json_extra_body(base_url: str, enable_thinking: bool, reasoning_budget: int) -> dict[str, Any] | None:
    override = env_value("JSON_EXTRA_BODY_ENABLED").lower()
    if override in {"0", "false", "no", "off"}:
        return None
    if override in {"1", "true", "yes", "on"} or provider_supports_nvidia_extra_body(base_url):
        return {
            "chat_template_kwargs": {"enable_thinking": enable_thinking},
            "reasoning_budget": reasoning_budget if enable_thinking else 0,
        }
    return None


def json_create_chat_completion(base_url: str, api_key: str, request: dict[str, Any]) -> Any:
    client = OpenAI(base_url=base_url, api_key=api_key)
    if is_nvidia_base_url(base_url):
        attempts = max(1, int(float(os.getenv("NVIDIA_JSON_MAX_ATTEMPTS", "3"))))
        for attempt in range(1, attempts + 1):
            wait_for_provider_slot("NVIDIA", rpm_env="NVIDIA_JSON_RPM_LIMIT", default_rpm=40)
            try:
                return client.chat.completions.create(**request)
            except Exception as exc:
                text = str(exc).lower()
                if "429" not in text and "rate limit" not in text and "too many requests" not in text:
                    raise
                if attempt >= attempts:
                    raise
                response = getattr(exc, "response", None)
                headers = getattr(response, "headers", {}) if response is not None else {}
                sleep_for = retry_after_seconds(
                    headers.get("Retry-After") if hasattr(headers, "get") else None,
                    default=float(os.getenv("NVIDIA_JSON_RETRY_WAIT_SECONDS", "20")),
                )
                print(f"NVIDIA rate limit hit during JSON call; waiting {sleep_for:.1f}s ({attempt}/{attempts}).")
                time.sleep(sleep_for)

    return client.chat.completions.create(**request)


def _choice_message_content(choice: Any) -> str:
    message = getattr(choice, "message", None)
    if isinstance(choice, dict):
        message = choice.get("message")

    if isinstance(message, dict):
        return str(message.get("content") or "")
    if message is not None:
        return str(getattr(message, "content", "") or "")

    if isinstance(choice, dict):
        delta = choice.get("delta") or {}
        return str(delta.get("content") or choice.get("text") or "")

    delta = getattr(choice, "delta", None)
    if delta is not None:
        return str(getattr(delta, "content", "") or "")
    return str(getattr(choice, "text", "") or "")


def json_completion_text(completion: Any, *, stream: bool = False) -> str:
    if completion is None:
        return ""
    if isinstance(completion, str):
        return completion.strip()
    if isinstance(completion, dict):
        if isinstance(completion.get("response"), str):
            return completion["response"].strip()
        if isinstance(completion.get("content"), str):
            return completion["content"].strip()
        choices = completion.get("choices")
        if choices:
            return _choice_message_content(choices[0]).strip()

    choices = getattr(completion, "choices", None)
    if choices:
        return _choice_message_content(choices[0]).strip()

    if stream:
        parts: list[str] = []
        for chunk in completion:
            if isinstance(chunk, str):
                parts.append(chunk)
                continue
            chunk_choices = chunk.get("choices") if isinstance(chunk, dict) else getattr(chunk, "choices", None)
            if not chunk_choices:
                continue
            parts.append(_choice_message_content(chunk_choices[0]))
        return "".join(parts).strip()

    return str(completion or "").strip()
