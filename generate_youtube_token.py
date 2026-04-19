import argparse

from upload import HISTORY_SCOPES, TOKEN_FILE, UPLOAD_SCOPES, _get_service


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a YouTube OAuth token.")
    parser.add_argument(
        "--history",
        action="store_true",
        help="Request youtube.readonly in addition to youtube.upload for channel history checks.",
    )
    args = parser.parse_args()

    scopes = HISTORY_SCOPES if args.history else UPLOAD_SCOPES
    _get_service(scopes, force_reauth=True)

    mode = "upload + history" if args.history else "upload only"
    print(f"YouTube OAuth token saved to {TOKEN_FILE} ({mode})")


if __name__ == "__main__":
    main()
