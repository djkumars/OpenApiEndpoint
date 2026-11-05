"""Inspect fine-tuning jobs or explicitly create a new fine-tuning job."""

import argparse
from typing import Optional, Sequence

from openai_api_tester import DEFAULT_MODEL, create_client


def list_fine_tuning_jobs(client):
    return client.fine_tuning.jobs.list(limit=10)


def create_fine_tuning_job(client, model: str, training_file_id: str):
    return client.fine_tuning.jobs.create(
        model=model,
        training_file=training_file_id,
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument(
        "--list-jobs",
        action="store_true",
        help="List recent fine-tuning jobs.",
    )
    actions.add_argument(
        "--create-job",
        metavar="TRAINING_FILE_ID",
        help="Create a fine-tuning job from an uploaded JSONL training file.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="Base model for --create-job; check fine-tuning model support.",
    )
    args = parser.parse_args(argv)
    client = create_client()

    if args.list_jobs:
        for job in list_fine_tuning_jobs(client).data:
            print(job.id, job.status, job.model)
    else:
        job = create_fine_tuning_job(client, args.model, args.create_job)
        print("Fine-tuning job created:", job.id, job.status)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
