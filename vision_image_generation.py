"""Test image understanding and image generation with the OpenAI API."""

import argparse
import base64
from pathlib import Path
from typing import Optional, Sequence

from openai_api_tester import DEFAULT_MODEL, create_client


def describe_image(client, model: str, image_url: str, prompt: str) -> str:
    response = client.responses.create(
        model=model,
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt},
                    {"type": "input_image", "image_url": image_url},
                ],
            }
        ],
    )
    return response.output_text


def generate_image(client, prompt: str, output_path: Path, model: str) -> Path:
    response = client.images.generate(model=model, prompt=prompt)
    image_data = response.data[0].b64_json
    if not image_data:
        raise RuntimeError("The Images API response did not contain image data.")
    output_path.write_bytes(base64.b64decode(image_data))
    return output_path


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument("--image-url", help="URL of an image to analyze.")
    actions.add_argument("--generate", help="Prompt for generating an image.")
    parser.add_argument(
        "--prompt",
        default="Describe this image.",
        help="Question or instruction for image analysis.",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Vision model.")
    parser.add_argument(
        "--image-model", default="gpt-image-1", help="Image generation model."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("generated_image.png"),
        help="Output PNG path for --generate.",
    )
    args = parser.parse_args(argv)
    client = create_client()

    if args.image_url:
        print(describe_image(client, args.model, args.image_url, args.prompt))
    else:
        print(generate_image(client, args.generate, args.output, args.image_model))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
