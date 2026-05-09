"""
Read side: search queries only. No add, no cognify.
Proves data persisted in local DB from write.py.

Usage:  uv run tests/quick_test_cognee_read.py
"""

import asyncio
import sys
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

import cognee
from cognee import SearchType

DATASET = "thai_industry"


async def main():
    print("=" * 60)
    print("READ: search queries only")
    print("=" * 60)

    queries = [
        ("การผลิตรถยนต์", "ประเทศไทยผลิตรถยนต์ปี 2567 กี่คัน ส่งออกไปไหน"),
        ("ข้าวไทย", "ผลผลิตข้าวไทยปี 2567 กี่ตัน"),
        ("อีอีซี", "โครงการอีอีซีครอบคลุมจังหวัดอะไรบ้าง"),
        ("ฟินเทค", "ระบบชำระเงินดิจิทัลของไทยมีอะไรบ้าง"),
        ("พลังงานลม", "โครงการพลังงานลมในไทยมีอะไรบ้าง"),
        ("นักท่องเที่ยว", "นักท่องเที่ยวต่างชาติเข้าประเทศไทยกี่คนในปี 2567"),
        ("EV", "นโยบายส่งเสริมรถยนต์ไฟฟ้าของไทยมีอะไรบ้าง"),
        ("พลังงานหมุนเวียน", "พลังงานหมุนเวียนในไทยมีสัดส่วนเท่าไหร่"),
    ]

    print(f"\nRunning {len(queries)} queries (read-only)...\n")
    results_summary = []

    for i, (topic, query) in enumerate(queries, 1):
        print(f"--- [{i}/{len(queries)}] {topic} ---")
        print(f"Q: {query}")
        try:
            results = await cognee.search(
                query_text=query,
                query_type=SearchType.GRAPH_COMPLETION,
                datasets=[DATASET],
            )
            answer = str(results[0]) if results else "(no results)"
            short = answer[:300] + "..." if len(answer) > 300 else answer
            print(f"A: {short}")
            results_summary.append((topic, "OK", len(results)))
        except Exception as e:
            print(f"  Error: {e}")
            results_summary.append((topic, "ERROR", str(e)))
        print()

    print("=" * 60)
    ok = sum(1 for _, s, _ in results_summary if s == "OK")
    print(f"Results: {ok}/{len(queries)} queries answered")
    if ok == len(queries):
        print("PASS")
    elif ok > 0:
        print(f"PARTIAL - {ok} worked, some failed.")
    else:
        print("FAIL - No data. Run quick_test_cognee_write.py first.")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
