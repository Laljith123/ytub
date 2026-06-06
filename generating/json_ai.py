import os
from typing import Any

import requests
from openai import OpenAI


BLUESMINDS_BASE_URL = "https://api.bluesminds.com/v1"
NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
NVIDIA_DEFAULT_MODEL = "openai/gpt-oss-120b"
BLUESMINDS_DEFAULT_MODEL = "gpt-4o"
BLUESMINDS_MODEL_CANDIDATES = (
    "gpt-4o,"
    "gpt-5-chat,"
    "grok-4.20-0309-non-reasoning,"
    "openai/gpt-4o-mini,"
    "gpt-4o-mini"
)
_MODEL_CACHE: dict[tuple[str, str], set[str]] = {}


def env_value(*names: str, default: str = "") -> str:
    for name in names:
        value = os.getenv(name, "").strip()
        if value:
            return value
    return default


def has_bluesminds_key() -> bool:
    return bool(env_value("BLUESMINDS_API_KEY"))


def json_api_key(specific_env: str = "") -> str:
    names = [name for name in (specific_env, "JSON_API_KEY", "BLUESMINDS_API_KEY", "NVIDIA_API_KEY") if name]
    return env_value(*names)


def json_base_url(specific_env: str = "") -> str:
    specific = env_value(specific_env) if specific_env else ""
    if specific:
        return specific

    shared = env_value("JSON_BASE_URL", "BLUESMINDS_BASE_URL")
    if shared:
        return shared

    if has_bluesminds_key():
        return BLUESMINDS_BASE_URL

    return env_value("NVIDIA_BASE_URL", default=NVIDIA_BASE_URL)


def json_model(specific_env: str = "", fallback: str = NVIDIA_DEFAULT_MODEL) -> str:
    specific = env_value(specific_env) if specific_env else ""
    if specific:
        return specific

    shared = env_value("JSON_MODEL", "BLUESMINDS_MODEL")
    if shared:
        return shared

    if has_bluesminds_key():
        return BLUESMINDS_DEFAULT_MODEL

    return fallback


def is_bluesminds_base_url(base_url: str) -> bool:
    return "bluesminds.com" in str(base_url or "").lower()


def json_provider_name(base_url: str) -> str:
    normalized = str(base_url or "").lower()
    if is_bluesminds_base_url(normalized):
        return "Bluesminds"
    if "integrate.api.nvidia.com" in normalized:
        return "NVIDIA"
    return "OpenAI-compatible"


def json_client(api_key_env: str = "", base_url_env: str = "") -> OpenAI | None:
    key = json_api_key(api_key_env)
    if not key:
        return None
    return OpenAI(base_url=json_base_url(base_url_env), api_key=key)


def provider_supports_nvidia_extra_body(base_url: str) -> bool:
    return "integrate.api.nvidia.com" in str(base_url or "").lower()


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
