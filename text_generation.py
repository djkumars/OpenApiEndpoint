"""Test text generation with the OpenAI Responses API."""

import argparse
from typing import Optional, Sequence

from openai_api_tester import DEFAULT_MODEL, create_client


def generate_text(client, model: str, prompt: str) -> str:
    response = client.responses.create(model=model, input=prompt)
    return response.output_text


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt", required=True, help="Prompt to send.")
    parser.add_argument(
        "--model", default=DEFAULT_MODEL, help="Model to use for generation."
    )
    args = parser.parse_args(argv)
    print(generate_text(create_client(), args.model, args.prompt))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
