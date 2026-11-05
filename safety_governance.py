"""Test content moderation or read organization audit logs."""

import argparse
import os
from typing import Optional, Sequence

from openai_api_tester import create_client


def moderate_text(client, text: str):
    return client.moderations.create(
        model="omni-moderation-latest",
        input=text,
    )


def list_audit_logs(client):
    return client.admin.organization.audit_logs.list(limit=10)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument("--text", help="Text to classify with the Moderations API.")
    actions.add_argument(
        "--audit-logs",
        action="store_true",
        help="Read recent organization audit logs (requires an Admin API key).",
    )
    args = parser.parse_args(argv)

    if args.audit_logs:
        from dotenv import load_dotenv

        load_dotenv()
        admin_key = os.getenv("OPENAI_ADMIN_KEY")
        if not admin_key:
            parser.error("--audit-logs requires OPENAI_ADMIN_KEY")
        from openai import OpenAI

        logs = list_audit_logs(OpenAI(admin_api_key=admin_key))
        for entry in logs.data:
            print(entry.id, entry.type, entry.effective_at)
        return 0

    result = moderate_text(create_client(), args.text).results[0]
    print("Flagged:", result.flagged)
    print("Categories:", result.categories)
    print("Category scores:", result.category_scores)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
