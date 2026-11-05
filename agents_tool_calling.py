"""Test a hosted OpenAI Agents API session."""

import argparse
import json
from datetime import datetime, timezone
from typing import Optional, Sequence, Dict, Any

from openai_api_tester import DEFAULT_MODEL, create_client


TIME_TOOL = {
    "type": "function",
    "name": "get_current_time",
    "description": "Get the current UTC time.",
    "parameters": {
        "type": "object",
        "properties": {},
        "required": [],
        "additionalProperties": False,
    },
    "strict": True,
}


def run_agent(client, model: str, prompt: str) -> Dict[str, Any]:
    return client.post(
        "/agents/sessions",
        cast_to=Dict[str, Any],
        options={"headers": {"OpenAI-Beta": "agents=v1"}},
        body={
            "agent": {
                "model": model,
                "instructions": "Answer the user's request concisely and safely.",
            },
            "environment": {"type": "none"},
            "input": prompt,
        },
    )


def run_function_tool(client, model: str) -> str:
    response = client.responses.create(
        model=model,
        input="Call the get_current_time tool and tell me the returned time.",
        tools=[TIME_TOOL],
        tool_choice={"type": "function", "name": "get_current_time"},
    )
    tool_call = next(
        (
            item
            for item in response.output
            if item.type == "function_call" and item.name == "get_current_time"
        ),
        None,
    )
    if tool_call is None:
        raise RuntimeError("The model did not call get_current_time.")

    json.loads(tool_call.arguments)
    tool_result = datetime.now(timezone.utc).isoformat()
    follow_up = client.responses.create(
        model=model,
        previous_response_id=response.id,
        input=[
            {
                "type": "function_call_output",
                "call_id": tool_call.call_id,
                "output": tool_result,
            }
        ],
    )
    return follow_up.output_text


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument("--input", help="Task for the hosted agent.")
    actions.add_argument(
        "--tool-call",
        action="store_true",
        help="Exercise a local function tool through the Responses API.",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Agent model.")
    args = parser.parse_args(argv)

    client = create_client()
    if args.tool_call:
        print(run_function_tool(client, args.model))
    else:
        print(json.dumps(run_agent(client, args.model, args.input), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
