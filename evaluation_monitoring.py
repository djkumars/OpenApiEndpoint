"""Read existing OpenAI Evals resources."""

import argparse
from typing import Optional, Sequence

from openai_api_tester import create_client


def list_evals(client):
    return client.evals.list(limit=10)


def get_eval(client, eval_id: str):
    return client.evals.retrieve(eval_id)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--eval-id",
        help="Retrieve one existing eval; otherwise list up to 10 evals.",
    )
    args = parser.parse_args(argv)
    client = create_client()

    if args.eval_id:
        evaluation = get_eval(client, args.eval_id)
        print(evaluation.model_dump_json(indent=2))
    else:
        for evaluation in list_evals(client).data:
            print(evaluation.id, evaluation.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
