"""Test audio transcription, speech generation, and Realtime API credentials."""

import argparse
from pathlib import Path
from typing import Optional, Sequence

from openai_api_tester import create_client


def transcribe_audio(client, audio_path: Path, model: str) -> str:
    with audio_path.open("rb") as audio_file:
        response = client.audio.transcriptions.create(model=model, file=audio_file)
    return response.text


def synthesize_speech(
    client, text: str, output_path: Path, model: str, voice: str
) -> Path:
    response = client.audio.speech.create(
        model=model,
        voice=voice,
        input=text,
        response_format="mp3",
    )
    response.stream_to_file(output_path)
    return output_path


def create_realtime_secret(client, model: str):
    return client.realtime.client_secrets.create(
        session={"type": "realtime", "model": model}
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument("--transcribe", type=Path, help="Path to an audio file.")
    actions.add_argument("--speak", help="Text to convert to speech.")
    actions.add_argument(
        "--realtime-secret",
        action="store_true",
        help="Create a short-lived Realtime API client secret.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("speech_test.mp3"),
        help="Output MP3 path for --speak.",
    )
    parser.add_argument(
        "--transcription-model",
        default="gpt-4o-mini-transcribe",
        help="Model to use for --transcribe.",
    )
    parser.add_argument(
        "--speech-model",
        default="gpt-4o-mini-tts",
        help="Model to use for --speak.",
    )
    parser.add_argument("--voice", default="alloy", help="Voice for --speak.")
    parser.add_argument(
        "--realtime-model", default="gpt-realtime", help="Model for Realtime."
    )
    args = parser.parse_args(argv)

    if args.transcribe and not args.transcribe.is_file():
        parser.error("the audio file does not exist or is not a file")

    client = create_client()
    if args.transcribe:
        print(transcribe_audio(client, args.transcribe, args.transcription_model))
    elif args.speak:
        print(
            synthesize_speech(
                client, args.speak, args.output, args.speech_model, args.voice
            )
        )
    else:
        secret = create_realtime_secret(client, args.realtime_model)
        print("Realtime client secret created successfully.")
        print("Expires at:", secret.expires_at)
        print("The secret value was not printed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
