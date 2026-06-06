import os
import time
from collections import deque


_REQUEST_TIMES: dict[str, deque[float]] = {}


def _env_float(name: str, default: str) -> float:
    try:
        return float(os.getenv(name, default))
    except ValueError:
        return float(default)


def wait_for_provider_slot(provider: str, *, rpm_env: str, default_rpm: int = 10) -> None:
    rpm = max(1, int(_env_float(rpm_env, str(default_rpm))))
    if rpm <= 0:
        return

    cushion = max(0.0, _env_float("RATE_LIMIT_CUSHION_SECONDS", "1.2"))
    window = 60.0 + cushion
    now = time.monotonic()
    bucket = _REQUEST_TIMES.setdefault(provider, deque())

    while bucket and now - bucket[0] >= window:
        bucket.popleft()

    if len(bucket) >= rpm:
        sleep_for = window - (now - bucket[0])
        print(f"{provider} rate limit pacing: waiting {sleep_for:.1f}s before continuing.")
        time.sleep(max(0.0, sleep_for))
        now = time.monotonic()
        while bucket and now - bucket[0] >= window:
            bucket.popleft()

    bucket.append(time.monotonic())


def retry_after_seconds(value: str | None, default: float = 72.0) -> float:
    if not value:
        return default
    try:
        return max(0.0, float(value))
    except ValueError:
        return default
