import os
import sys


RIVA_VOICE = os.getenv("RIVA_VOICE", "Chatterbox-Multilingual.en-US.Male")
RIVA_LANGUAGE = os.getenv("RIVA_LANGUAGE", "en-US")
RIVA_URI = os.getenv("RIVA_URI", "grpc.nvcf.nvidia.com:443")
RIVA_FUNCTION_ID = os.getenv("RIVA_FUNCTION_ID", "ddacc747-1269-4fab-bfd9-8f593dead106")
RIVA_SAMPLE_RATE_HZ = int(os.getenv("RIVA_SAMPLE_RATE_HZ", "24000"))
RIVA_USE_SSL = os.getenv("RIVA_USE_SSL", "1") == "1"
RIVA_PERMANENT_ERROR_EXIT_CODE = 42
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
PREFLIGHT_TEXT = os.getenv("CHATTERBOX_PREFLIGHT_TEXT", "voice check")


def _unavailable_message() -> str:
    return (
        "Chatterbox TTS is not available for this NVIDIA_API_KEY/account. "
        f"The configured Chatterbox function id is {RIVA_FUNCTION_ID}. "
        "Create or use an NVIDIA API key from the Chatterbox Multilingual model page "
        "for the same account, then update the GitHub secret NVIDIA_API_KEY."
    )


def main() -> None:
    if not NVIDIA_API_KEY:
        raise RuntimeError("NVIDIA_API_KEY is missing.")

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
            print(_unavailable_message())
            raise SystemExit(RIVA_PERMANENT_ERROR_EXIT_CODE) from exc
        raise

    if not getattr(response, "audio", b""):
        raise RuntimeError("Chatterbox preflight returned empty audio.")

    print("Chatterbox preflight OK.")


if __name__ == "__main__":
    main()
