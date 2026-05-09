"""
Write side: forget -> add -> cognify. No searching.
Data persists in local DB for read.py to query separately.

Usage:  uv run tests/quick_test_cognee_write.py
"""

import asyncio
import glob
import os
import sys
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

import cognee

DATA_DIR = os.path.join(os.path.dirname(__file__), "test_data")
DATASET = "thai_industry"


async def main():
    print("=" * 60)
    print("WRITE: forget -> add -> cognify")
    print("=" * 60)

    print("\nForgetting existing data...")
    await cognee.forget(everything=True)
    print("  Done.")

    files = sorted(glob.glob(os.path.join(DATA_DIR, "*.md")))
    print(f"\nFound {len(files)} documents:")
    for f in files:
        print(f"  - {os.path.basename(f)}")

    for filepath in files:
        filename = os.path.basename(filepath)
        print(f"\nAdding: {filename}")
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
        await cognee.add(content, dataset_name=DATASET)
        print("  Done.")

    print(f"\nRunning cognify on '{DATASET}'...")
    await cognee.cognify(datasets=DATASET)
    print("  Cognify complete.")

    print("\n" + "=" * 60)
    print("Write complete. Now run:  uv run tests/quick_test_cognee_read.py")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
