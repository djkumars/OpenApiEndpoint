"""Command-line utility for checking OpenAI model access."""

import argparse
import os
from typing import Any, List


DEFAULT_MODEL = "gpt-4.1-mini"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="List OpenAI models or send a test prompt to a model."
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List models available to the configured API key.",
    )
    parser.add_argument(
        "--model",
        default=os.getenv("OPENAI_MODEL", DEFAULT_MODEL),
        help="Model to use for a prompt (default: OPENAI_MODEL or %(default)s).",
    )
    parser.add_argument(
        "--prompt",
        help="Prompt to send. Required unless --list-models is used.",
    )
    return parser


def create_client() -> Any:
    """Create an authenticated client using OPENAI_API_KEY from the environment."""
    from dotenv import load_dotenv

    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY is not set. Add it to your environment or .env file."
        )

    from openai import OpenAI

    return OpenAI(api_key=api_key)


def list_models(client: Any) -> List[str]:
    """Return model IDs accessible to the configured API key."""
    response = client.models.list()
    return sorted(model.id for model in response.data)


def send_prompt(client: Any, model: str, prompt: str) -> str:
    """Send a prompt using the Responses API and return the generated text."""
    response = client.responses.create(model=model, input=prompt)
    return response.output_text


def main(argv: List[str] = None) -> int:
    from dotenv import load_dotenv

    load_dotenv()
    parser = build_parser()
    args = parser.parse_args(argv)

    if not args.list_models and not args.prompt:
        parser.error("provide --prompt or use --list-models")

    try:
        client = create_client()
    except ValueError as error:
        parser.error(str(error))

    if args.list_models:
        for model_id in list_models(client):
            print(model_id)
        return 0

    print(send_prompt(client, args.model, args.prompt))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
