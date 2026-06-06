import atexit
import os
from typing import Any

import requests
from openai import OpenAI


BLUESMINDS_BASE_URL = "https://api.bluesminds.com/v1"
NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
CHATGPT_FREE_BASE_URL = "https://chatgpt-api.shn.hk/v1"
PUTER_CHAT_BASE_URL = "puter://chat"
NO_AUTH_API_KEY = "no-auth-required"
NVIDIA_DEFAULT_MODEL = "openai/gpt-oss-120b"
BLUESMINDS_DEFAULT_MODEL = "gpt-4o"
BLUESMINDS_MODEL_CANDIDATES = (
    "gpt-4o,gpt-5-chat,grok-4.20-0309-non-reasoning,openai/gpt-4o-mini,gpt-4o-mini"
)
_MODEL_CACHE: dict[tuple[str, str], set[str]] = {}
_PUTER_PLAYWRIGHT = None
_PUTER_BROWSER = None
_PUTER_CONTEXT = None
_PUTER_PAGE = None


def env_value(*names: str, default: str = "") -> str:
    for name in names:
        value = os.getenv(name, "").strip()
        if value:
            return value
    return default


def has_bluesminds_key() -> bool:
    return bool(env_value("BLUESMINDS_API_KEY"))


def provider_uses_puter_chat(base_url: str) -> bool:
    normalized = str(base_url or "").strip().lower().rstrip("/")
    return normalized in {"puter", "puter://chat", "puter://ai/chat", "https://js.puter.com/v2"}


def provider_allows_no_auth(base_url: str) -> bool:
    normalized = str(base_url or "").lower()
    return "chatgpt-api.shn.hk" in normalized or provider_uses_puter_chat(normalized)


def json_api_key(specific_env: str = "", base_url: str = "") -> str:
    if provider_allows_no_auth(base_url):
        return NO_AUTH_API_KEY

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
    if provider_uses_puter_chat(normalized):
        return "Puter.js Chat"
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
    if not base_url or not api_key or provider_uses_puter_chat(base_url):
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


def _close_puter_chat_browser() -> None:
    global _PUTER_PLAYWRIGHT, _PUTER_BROWSER, _PUTER_CONTEXT, _PUTER_PAGE

    for item in (_PUTER_PAGE, _PUTER_CONTEXT, _PUTER_BROWSER):
        try:
            if item is not None:
                item.close()
        except Exception:
            pass

    try:
        if _PUTER_PLAYWRIGHT is not None:
            _PUTER_PLAYWRIGHT.stop()
    except Exception:
        pass

    _PUTER_PLAYWRIGHT = None
    _PUTER_BROWSER = None
    _PUTER_CONTEXT = None
    _PUTER_PAGE = None


atexit.register(_close_puter_chat_browser)


def _puter_chat_page():
    global _PUTER_PLAYWRIGHT, _PUTER_BROWSER, _PUTER_CONTEXT, _PUTER_PAGE

    if _PUTER_PAGE is not None:
        try:
            if not _PUTER_PAGE.is_closed():
                return _PUTER_PAGE
        except Exception:
            _close_puter_chat_browser()

    try:
        from playwright.sync_api import sync_playwright
    except ImportError as exc:
        raise RuntimeError(
            "Playwright is required for Puter chat. "
            "Run 'pip install playwright' and 'python -m playwright install chromium'."
        ) from exc

    timeout_ms = int(float(env_value("PUTER_CHAT_TIMEOUT_MS", default="180000")))
    _PUTER_PLAYWRIGHT = sync_playwright().start()
    _PUTER_BROWSER = _PUTER_PLAYWRIGHT.chromium.launch(
        headless=True,
        args=["--no-sandbox", "--disable-dev-shm-usage"],
    )
    _PUTER_CONTEXT = _PUTER_BROWSER.new_context()
    _PUTER_PAGE = _PUTER_CONTEXT.new_page()
    _PUTER_PAGE.set_default_timeout(timeout_ms)
    _PUTER_PAGE.set_content(
        """
        <html>
        <body>
            <script src="https://js.puter.com/v2/"></script>
        </body>
        </html>
        """,
        wait_until="domcontentloaded",
    )
    _PUTER_PAGE.wait_for_function(
        "() => window.puter && puter.ai && puter.ai.chat",
        timeout=timeout_ms,
    )
    return _PUTER_PAGE


def _message_content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                parts.append(str(item.get("text") or item.get("content") or ""))
            else:
                parts.append(str(item))
        return "\n".join(part for part in parts if part)
    return str(content or "")


def _puter_messages(messages: Any) -> list[dict[str, str]]:
    if not isinstance(messages, list):
        return [{"role": "user", "content": str(messages or "")}]

    cleaned: list[dict[str, str]] = []
    for item in messages:
        if not isinstance(item, dict):
            continue
        role = str(item.get("role") or "user").strip().lower()
        if role not in {"system", "user", "assistant"}:
            role = "user"
        content = _message_content_text(item.get("content"))
        if content:
            cleaned.append({"role": role, "content": content})
    return cleaned or [{"role": "user", "content": ""}]


def _puter_options(request: dict[str, Any]) -> dict[str, Any]:
    options: dict[str, Any] = {
        "model": str(request.get("model") or env_value("PUTER_CHAT_MODEL", default="claude-sonnet-4.5")),
        "stream": bool(request.get("stream")),
    }
    for key in ("max_tokens", "temperature"):
        value = request.get(key)
        if value is not None:
            options[key] = value
    reasoning_effort = env_value("PUTER_CHAT_REASONING_EFFORT")
    if reasoning_effort:
        options["reasoning_effort"] = reasoning_effort
    text_verbosity = env_value("PUTER_CHAT_TEXT_VERBOSITY")
    if text_verbosity:
        options["text_verbosity"] = text_verbosity
    return options


def _puter_chat_completion(request: dict[str, Any]) -> dict[str, Any]:
    page = _puter_chat_page()
    timeout_ms = int(float(env_value("PUTER_CHAT_TIMEOUT_MS", default="180000")))
    result = page.evaluate(
        """
        async ({ messages, options, timeoutMs }) => {
            const textFromContent = (content) => {
                if (!content) return "";
                if (typeof content === "string") return content;
                if (Array.isArray(content)) {
                    return content.map((part) => {
                        if (!part) return "";
                        if (typeof part === "string") return part;
                        return part.text || part.content || "";
                    }).filter(Boolean).join("\\n");
                }
                return String(content);
            };

            const textFromResponse = (response) => {
                if (!response) return "";
                if (typeof response === "string") return response;
                if (response.text) return response.text;
                if (response.message) return textFromContent(response.message.content || response.message);
                if (response.choices && response.choices[0]) {
                    const choice = response.choices[0];
                    if (choice.message) return textFromContent(choice.message.content || choice.message);
                    if (choice.text) return choice.text;
                }
                return String(response);
            };

            const collect = async () => {
                const response = await puter.ai.chat(messages, options);
                if (options.stream && response && response[Symbol.asyncIterator]) {
                    let text = "";
                    for await (const part of response) {
                        text += textFromResponse(part);
                    }
                    return text;
                }
                return textFromResponse(response);
            };

            const timeout = new Promise((_, reject) => {
                setTimeout(() => reject(new Error("Puter chat timed out")), timeoutMs);
            });
            return Promise.race([collect(), timeout]);
        }
        """,
        {
            "messages": _puter_messages(request.get("messages")),
            "options": _puter_options(request),
            "timeoutMs": timeout_ms,
        },
    )
    return {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": str(result or ""),
                }
            }
        ]
    }


def _chat_completions_endpoint(base_url: str) -> str:
    if provider_allows_no_auth(base_url):
        return f"{str(base_url).rstrip('/')}/"
    return f"{str(base_url).rstrip('/')}/chat/completions"


def json_create_chat_completion(base_url: str, api_key: str, request: dict[str, Any]) -> Any:
    if provider_uses_puter_chat(base_url):
        return _puter_chat_completion(request)

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
