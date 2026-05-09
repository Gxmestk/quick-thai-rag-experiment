"""
Gather cognee responses for Thai RAG evaluation.
Queries cognee (GRAPH_COMPLETION + CHUNKS) and saves results to JSON.

Usage:  uv run tests/gather_responses.py
"""

import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

import cognee
from cognee import SearchType

from eval_data import EVAL_DATA

COGNEE_DATASET = "thai_industry"
OUTPUT_DIR = Path(__file__).resolve().parent / "data"
OUTPUT_FILE = OUTPUT_DIR / "responses.json"


async def query_cognee(user_input: str) -> tuple[str, list[str]]:
    """Query cognee for response and retrieved contexts."""
    try:
        results = await cognee.search(
            query_text=user_input,
            query_type=SearchType.GRAPH_COMPLETION,
            datasets=[COGNEE_DATASET],
        )
        response = str(results[0]) if results else ""
    except Exception as e:
        print(f"  Error (response): {e}")
        response = ""

    try:
        chunks = await cognee.search(
            query_text=user_input,
            query_type=SearchType.CHUNKS,
            datasets=[COGNEE_DATASET],
        )
        retrieved_contexts = [str(c) for c in chunks] if chunks else []
    except Exception as e:
        print(f"  Error (contexts): {e}")
        retrieved_contexts = []

    return response, retrieved_contexts


async def main():
    print("=" * 60)
    print("Gather Cognee Responses: Thai Industry RAG")
    print("=" * 60)

    OUTPUT_DIR.mkdir(exist_ok=True)

    responses = []
    for i, item in enumerate(EVAL_DATA, 1):
        tier = "T1" if i <= 8 else "T2"
        query = item["user_input"]
        print(f"\n[{tier}] [{i}/{len(EVAL_DATA)}] {query[:60]}...")
        response, contexts = await query_cognee(query)
        short = response[:120] + "..." if len(response) > 120 else response
        print(f"  Response: {short}")
        print(f"  Contexts: {len(contexts)} chunks")

        responses.append({
            "id": i,
            "tier": tier,
            "user_input": query,
            "reference": item["reference"],
            "response": response,
            "retrieved_contexts": contexts,
        })

    # Save
    output = {
        "gathered_at": datetime.now().isoformat(),
        "dataset": COGNEE_DATASET,
        "num_samples": len(responses),
        "samples": responses,
    }
    OUTPUT_FILE.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved {len(responses)} responses to {OUTPUT_FILE}")


if __name__ == "__main__":
    asyncio.run(main())
