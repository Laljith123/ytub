import json
import os
import time
from typing import Any

import requests
from openai import OpenAI

try:
    from rate_limit import retry_after_seconds, wait_for_provider_interval, wait_for_provider_slot
except ImportError:  # pragma: no cover - used when imported as generating.json_ai
    from generating.rate_limit import retry_after_seconds, wait_for_provider_interval, wait_for_provider_slot


APIFREELLM_BASE_URL = "https://apifreellm.com/api/v1"
FREETHEAI_BASE_URL = "https://api.freetheai.xyz/v1"
BLUESMINDS_BASE_URL = "https://api.bluesminds.com/v1"
NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
CHATGPT_FREE_BASE_URL = "https://chatgpt-api.shn.hk/v1"
NO_AUTH_API_KEY = "no-auth-required"
APIFREELLM_DEFAULT_MODEL = "apifreellm"
FREETHEAI_DEFAULT_MODEL = "opc/deepseek-v4-flash-free"
NVIDIA_DEFAULT_MODEL = "openai/gpt-oss-120b"
BLUESMINDS_DEFAULT_MODEL = "gpt-4o"
BLUESMINDS_MODEL_CANDIDATES = (
    "gpt-4o,gpt-5-chat,grok-4.20-0309-non-reasoning,openai/gpt-4o-mini,gpt-4o-mini"
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


def has_freetheai_key() -> bool:
    return bool(env_value("FREETHEAI_API_KEY"))


def has_apifreellm_key() -> bool:
    return bool(env_value("APIFREELLM_API_KEY"))


def provider_allows_no_auth(base_url: str) -> bool:
    normalized = str(base_url or "").lower()
    return "chatgpt-api.shn.hk" in normalized


def is_apifreellm_base_url(base_url: str) -> bool:
    return "apifreellm.com" in str(base_url or "").lower()


def is_freetheai_base_url(base_url: str) -> bool:
    return "api.freetheai.xyz" in str(base_url or "").lower()


def json_api_key(specific_env: str = "", base_url: str = "") -> str:
    if provider_allows_no_auth(base_url):
        return NO_AUTH_API_KEY

    if is_apifreellm_base_url(base_url):
        names = [name for name in (specific_env, "APIFREELLM_API_KEY", "JSON_API_KEY") if name]
        return env_value(*names)

    if is_freetheai_base_url(base_url):
        names = [name for name in (specific_env, "FREETHEAI_API_KEY", "JSON_API_KEY") if name]
        return env_value(*names)

    names = [name for name in (specific_env, "JSON_API_KEY", "BLUESMINDS_API_KEY", "NVIDIA_API_KEY") if name]
    return env_value(*names)


def json_base_url(specific_env: str = "") -> str:
    specific = env_value(specific_env) if specific_env else ""
    if specific:
        return specific

    shared = env_value("JSON_BASE_URL", "BLUESMINDS_BASE_URL")
    if shared:
        return shared

    if has_apifreellm_key():
        return APIFREELLM_BASE_URL

    if has_freetheai_key():
        return FREETHEAI_BASE_URL

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

    if has_apifreellm_key():
        return APIFREELLM_DEFAULT_MODEL

    if has_freetheai_key():
        return FREETHEAI_DEFAULT_MODEL

    if has_bluesminds_key():
        return BLUESMINDS_DEFAULT_MODEL

    return fallback


def is_bluesminds_base_url(base_url: str) -> bool:
    return "bluesminds.com" in str(base_url or "").lower()


def json_provider_name(base_url: str) -> str:
    normalized = str(base_url or "").lower()
    if is_apifreellm_base_url(normalized):
        return "APIFreeLLM"
    if is_freetheai_base_url(normalized):
        return "FreeTheAi"
    if is_bluesminds_base_url(normalized):
        return "Bluesminds"
    if "integrate.api.nvidia.com" in normalized:
        return "NVIDIA"
    if provider_allows_no_auth(normalized):
        return "ChatGPT API Free"
    return "OpenAI-compatible"


def json_client(api_key_env: str = "", base_url_env: str = "") -> OpenAI | None:
    base_url = json_base_url(base_url_env)
    key = json_api_key(api_key_env, base_url=base_url)
    if not key:
        return None
    return OpenAI(base_url=base_url, api_key=key)


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
    if is_apifreellm_base_url(base_url):
        _MODEL_CACHE[(base_url.rstrip("/"), api_key[:10])] = set()
        return set()
    cache_key = (base_url.rstrip("/"), api_key[:10])
    if cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key]

    try:
        if is_freetheai_base_url(base_url):
            wait_for_provider_slot("FreeTheAi", rpm_env="FREETHEAI_RPM_LIMIT", default_rpm=10)
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


def _chat_completions_endpoint(base_url: str) -> str:
    if provider_allows_no_auth(base_url):
        return f"{str(base_url).rstrip('/')}/"
    return f"{str(base_url).rstrip('/')}/chat/completions"


def _apifreellm_chat_endpoint(base_url: str) -> str:
    base = str(base_url or APIFREELLM_BASE_URL).rstrip("/")
    if base.endswith("/chat"):
        return base
    return f"{base}/chat"


def _content_to_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text") or item.get("content")
                if isinstance(text, str):
                    parts.append(text)
                continue
            parts.append(str(item))
        return "\n".join(part for part in parts if part).strip()
    if isinstance(content, dict):
        text = content.get("text") or content.get("content")
        if isinstance(text, str):
            return text
        return json.dumps(content, ensure_ascii=False)
    return str(content)


def _apifreellm_message_from_request(request: dict[str, Any]) -> str:
    messages = request.get("messages")
    if not isinstance(messages, list):
        return _content_to_text(request.get("prompt") or request.get("message") or messages)

    blocks: list[str] = []
    for message in messages:
        if isinstance(message, dict):
            role = str(message.get("role") or "user").strip().capitalize()
            content = _content_to_text(message.get("content"))
        else:
            role = "Message"
            content = _content_to_text(message)
        if content:
            blocks.append(f"{role}:\n{content}")
    return "\n\n".join(blocks).strip()


def _apifreellm_completion_from_payload(payload: Any) -> dict[str, Any]:
    if isinstance(payload, dict):
        if payload.get("success") is False:
            detail = payload.get("error") or payload.get("message") or payload
            raise RuntimeError(f"APIFreeLLM request failed: {detail}")

        content = payload.get("response")
        if content is None:
            content = payload.get("message") or payload.get("content") or payload.get("text")
        if isinstance(content, dict):
            content = content.get("content") or content.get("text") or json.dumps(content, ensure_ascii=False)
    else:
        content = str(payload or "")

    return {"choices": [{"message": {"role": "assistant", "content": str(content or "")}}]}


def json_create_chat_completion(base_url: str, api_key: str, request: dict[str, Any]) -> Any:
    if provider_allows_no_auth(base_url):
        body = {key: value for key, value in request.items() if value is not None}
        if body.get("stream"):
            body["stream"] = False

        headers = {"Content-Type": "application/json"}
        if api_key and api_key != NO_AUTH_API_KEY:
            headers["Authorization"] = f"Bearer {api_key}"

        response = requests.post(
            _chat_completions_endpoint(base_url),
            headers=headers,
            json=body,
            timeout=float(os.getenv("JSON_COMPLETION_TIMEOUT", "90")),
        )
        if response.status_code >= 400:
            raise RuntimeError(f"HTTP {response.status_code}: {response.text[:500]}")
        return response.json()

    if is_apifreellm_base_url(base_url):
        if not api_key:
            raise RuntimeError("APIFREELLM_API_KEY is missing.")

        message = _apifreellm_message_from_request(request)
        if not message:
            raise RuntimeError("APIFreeLLM request is missing message content.")

        body: dict[str, Any] = {"message": message}
        model = str(request.get("model") or "").strip()
        if model:
            body["model"] = model

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        attempts = max(1, int(float(os.getenv("APIFREELLM_MAX_ATTEMPTS", "4"))))
        timeout = float(os.getenv("JSON_COMPLETION_TIMEOUT", os.getenv("APIFREELLM_TIMEOUT", "180")))
        for attempt in range(1, attempts + 1):
            wait_for_provider_interval(
                "APIFreeLLM",
                interval_env="APIFREELLM_REQUEST_INTERVAL_SECONDS",
                default_seconds=20.0,
            )
            response = requests.post(
                _apifreellm_chat_endpoint(base_url),
                headers=headers,
                json=body,
                timeout=timeout,
            )
            if response.status_code != 429:
                if response.status_code >= 400:
                    raise RuntimeError(f"HTTP {response.status_code}: {response.text[:500]}")
                try:
                    payload = response.json()
                except ValueError:
                    payload = response.text
                return _apifreellm_completion_from_payload(payload)

            sleep_for = retry_after_seconds(
                response.headers.get("Retry-After"),
                default=float(os.getenv("APIFREELLM_RETRY_WAIT_SECONDS", "20")),
            )
            print(f"APIFreeLLM rate limit hit during JSON call; waiting {sleep_for:.1f}s ({attempt}/{attempts}).")
            time.sleep(sleep_for)

        raise RuntimeError(f"APIFreeLLM JSON request failed after {attempts} rate-limit retries.")

    if is_freetheai_base_url(base_url):
        body = {key: value for key, value in request.items() if value is not None}
        if body.get("stream"):
            body["stream"] = False

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        attempts = max(1, int(float(os.getenv("FREETHEAI_MAX_ATTEMPTS", "3"))))
        for attempt in range(1, attempts + 1):
            wait_for_provider_slot("FreeTheAi", rpm_env="FREETHEAI_RPM_LIMIT", default_rpm=10)
            response = requests.post(
                _chat_completions_endpoint(base_url),
                headers=headers,
                json=body,
                timeout=float(os.getenv("JSON_COMPLETION_TIMEOUT", "90")),
            )
            if response.status_code != 429:
                if response.status_code >= 400:
                    raise RuntimeError(f"HTTP {response.status_code}: {response.text[:500]}")
                return response.json()

            sleep_for = retry_after_seconds(response.headers.get("Retry-After"))
            print(f"FreeTheAi rate limit hit during JSON call; waiting {sleep_for:.1f}s ({attempt}/{attempts}).")
            time.sleep(sleep_for)

        raise RuntimeError(f"FreeTheAi JSON request failed after {attempts} rate-limit retries.")

    client = OpenAI(base_url=base_url, api_key=api_key)
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
