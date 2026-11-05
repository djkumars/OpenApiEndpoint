"""Test embeddings and hosted file search with OpenAI vector stores."""

import argparse
from pathlib import Path
from typing import Optional, Sequence

from openai_api_tester import DEFAULT_MODEL, create_client


def create_embedding(client, model: str, text: str):
    response = client.embeddings.create(model=model, input=text)
    return response.data[0].embedding


def create_vector_store_and_upload(client, file_path: Path, name: str):
    vector_store = client.vector_stores.create(name=name)
    with file_path.open("rb") as file_obj:
        client.vector_stores.files.upload_and_poll(
            vector_store_id=vector_store.id,
            file=file_obj,
        )
    return vector_store.id


def search_vector_store(client, model: str, vector_store_id: str, query: str) -> str:
    response = client.responses.create(
        model=model,
        input=query,
        tools=[
            {
                "type": "file_search",
                "vector_store_ids": [vector_store_id],
            }
        ],
    )
    return response.output_text


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument("--embed-text", help="Text to embed.")
    actions.add_argument(
        "--query",
        help="Question to ask using hosted file search; provide --vector-store-id.",
    )
    actions.add_argument(
        "--upload-file",
        type=Path,
        help="Upload a file to a newly created vector store.",
    )
    parser.add_argument(
        "--model", default=DEFAULT_MODEL, help="Responses model for file search."
    )
    parser.add_argument(
        "--embedding-model",
        default="text-embedding-3-small",
        help="Embedding model for --embed-text.",
    )
    parser.add_argument("--vector-store-id", help="Existing vector store ID.")
    parser.add_argument("--store-name", default="API tester knowledge base")
    args = parser.parse_args(argv)

    if args.query and not args.vector_store_id:
        parser.error("--query requires --vector-store-id")
    if args.upload_file and not args.upload_file.is_file():
        parser.error("the upload file does not exist or is not a file")

    client = create_client()
    if args.embed_text:
        embedding = create_embedding(client, args.embedding_model, args.embed_text)
        print("Embedding dimensions:", len(embedding))
        print("First five values:", embedding[:5])
    elif args.query:
        print(
            search_vector_store(
                client, args.model, args.vector_store_id, args.query
            )
        )
    else:
        vector_store_id = create_vector_store_and_upload(
            client, args.upload_file, args.store_name
        )
        print("Vector store created and file processed:", vector_store_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
