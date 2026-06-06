import os

import grpc


RIVA_VOICE = os.getenv("RIVA_VOICE", "Chatterbox-Multilingual.en-US.Male")
RIVA_LANGUAGE = os.getenv("RIVA_LANGUAGE", "en-US")
RIVA_URI = os.getenv("RIVA_URI", "grpc.nvcf.nvidia.com:443")
RIVA_FUNCTION_ID = os.getenv("RIVA_FUNCTION_ID", "ddacc747-1269-4fab-bfd9-8f593dead106")
RIVA_SAMPLE_RATE_HZ = int(os.getenv("RIVA_SAMPLE_RATE_HZ", "24000"))
RIVA_USE_SSL = os.getenv("RIVA_USE_SSL", "1") == "1"
RIVA_PERMANENT_ERROR_EXIT_CODE = 42
PREFLIGHT_TEXT = os.getenv("CHATTERBOX_PREFLIGHT_TEXT", "voice check")


def _clean_api_key(value: str | None) -> str:
    key = str(value or "").strip().strip("\"'")
    if key.lower().startswith("bearer "):
        key = key[7:].strip()
    return key


NVIDIA_API_KEY = _clean_api_key(os.getenv("NVIDIA_API_KEY"))


def _unavailable_message() -> str:
    return (
        "Chatterbox TTS is not available for this NVIDIA_API_KEY/account. "
        f"The configured Chatterbox function id is {RIVA_FUNCTION_ID}. "
        "Create or use an NVIDIA API key from the Chatterbox Multilingual model page "
        "for the same account, then update the GitHub secret NVIDIA_API_KEY."
    )


def _print_grpc_error(exc: Exception) -> None:
    code = exc.code() if isinstance(exc, grpc.RpcError) else None
    details = exc.details() if isinstance(exc, grpc.RpcError) else str(exc)
    print(f"Chatterbox gRPC status: {code}")
    print(f"Chatterbox gRPC details: {details}")


def _print_not_found_hint() -> None:
    print(
        "NVIDIA accepted the key format, but the hosted Cloud Function was not reachable for this key. "
        "Most likely causes: the secret uses an NGC/deploy key instead of the Chatterbox Try API key, "
        "the key was created under a different NVIDIA account/org, the key was revoked after exposure, "
        "or this account does not currently have access to the hosted Chatterbox function."
    )


def main() -> None:
    if not NVIDIA_API_KEY:
        raise RuntimeError("NVIDIA_API_KEY is missing.")
    if not NVIDIA_API_KEY.startswith("nvapi-"):
        raise RuntimeError(
            "NVIDIA_API_KEY does not look like a Chatterbox API key. "
            "Store only the raw nvapi-... value in the GitHub secret."
        )

    print("Chatterbox preflight starting with NVIDIA_API_KEY format OK.")

    import riva.client

    auth = riva.client.Auth(
        use_ssl=RIVA_USE_SSL,
        uri=RIVA_URI,
        metadata_args=[
            ["function-id", RIVA_FUNCTION_ID],
            ["authorization", f"Bearer {NVIDIA_API_KEY}"],
        ],
    )
    service = riva.client.SpeechSynthesisService(auth)

    try:
        response = service.synthesize(
            text=PREFLIGHT_TEXT,
            voice_name=RIVA_VOICE,
            language_code=RIVA_LANGUAGE,
            sample_rate_hz=RIVA_SAMPLE_RATE_HZ,
            encoding=riva.client.AudioEncoding.LINEAR_PCM,
        )
    except Exception as exc:
        error_text = str(exc)
        if "StatusCode.NOT_FOUND" in error_text and "Function" in error_text:
            _print_grpc_error(exc)
            print(_unavailable_message())
            _print_not_found_hint()
            raise SystemExit(RIVA_PERMANENT_ERROR_EXIT_CODE) from exc
        raise

    if not getattr(response, "audio", b""):
        raise RuntimeError("Chatterbox preflight returned empty audio.")

    print("Chatterbox preflight OK.")


if __name__ == "__main__":
    main()
