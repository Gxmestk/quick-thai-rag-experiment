"""
Ragas evaluation for Thai RAG pipeline.
Loads gathered cognee responses and runs Ragas metrics.

Usage:
  1. Gather:   uv run tests/gather_responses.py
  2. Evaluate:  uv run tests/eval_ragas.py
"""

import asyncio
import json
import math
import os
import sys
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

import httpx
from langchain_openai import ChatOpenAI
from langchain_core.embeddings import Embeddings

from ragas import SingleTurnSample
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.metrics._faithfulness import Faithfulness
from ragas.metrics._answer_relevance import AnswerRelevancy
from ragas.metrics._context_precision import ContextPrecision
from ragas.metrics._context_recall import ContextRecall
# FactualCorrectness disabled — 4x LLM calls per sample, expensive.
# Use for final eval only: from ragas.metrics._factual_correctness import FactualCorrectness

RESPONSES_FILE = Path(__file__).resolve().parent / "data" / "responses.json"
PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"
MAX_CONCURRENT = 10


class LocalEmbeddings(Embeddings):
    """Calls local embedding server directly with string inputs."""

    def __init__(self, base_url: str, model: str):
        self._url = f"{base_url}/v1/embeddings"
        self._model = model

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embed(texts)

    def embed_query(self, text: str) -> list[float]:
        return self._embed([text])[0]

    def _embed(self, texts: list[str]) -> list[list[float]]:
        resp = httpx.post(
            self._url,
            json={"input": texts, "model": self._model},
            timeout=60,
        )
        resp.raise_for_status()
        data = resp.json()
        return [d["embedding"] for d in sorted(data["data"], key=lambda x: x["index"])]


async def adapt_metrics_to_thai(metrics: list, llm) -> None:
    """Adapt all metric prompts to Thai, saving for reuse."""
    PROMPTS_DIR.mkdir(exist_ok=True)

    for metric in metrics:
        try:
            adapted = await metric.adapt_prompts(
                language="thai",
                llm=llm,
                adapt_instruction=True,
            )
        except Exception:
            print(f"  {metric.name}: instruction translation failed, adapting examples only...")
            adapted = await metric.adapt_prompts(
                language="thai",
                llm=llm,
                adapt_instruction=False,
            )
        metric.set_prompts(**adapted)
        metric.save_prompts(str(PROMPTS_DIR))
        print(f"  Adapted: {metric.name}")


def load_adapted_prompts(metrics: list) -> list:
    """Load previously saved Thai prompts. Returns list of metrics that need adaptation."""
    if not PROMPTS_DIR.exists():
        return metrics

    needs_adaptation = []
    for metric in metrics:
        try:
            metric.load_prompts(str(PROMPTS_DIR), language="thai")
            print(f"  Loaded: {metric.name}")
        except Exception:
            needs_adaptation.append(metric)
    return needs_adaptation


async def score_sample(sample_dict: dict, metrics: dict, semaphore: asyncio.Semaphore,
                       total: int, counter: dict) -> dict:
    """Score a single sample with all metrics. Uses semaphore for concurrency."""
    async with semaphore:
        ragas_sample = SingleTurnSample(
            user_input=sample_dict["user_input"],
            response=sample_dict["response"],
            reference=sample_dict["reference"],
            retrieved_contexts=sample_dict["retrieved_contexts"],
        )

        scores = {
            "id": sample_dict["id"],
            "tier": sample_dict["tier"],
            "user_input": sample_dict["user_input"][:60],
        }

        for metric_name, metric in metrics.items():
            for attempt in range(3):
                try:
                    value = await metric.single_turn_ascore(ragas_sample)
                    scores[metric_name] = value
                    if isinstance(value, float) and math.isnan(value):
                        print(f"    {metric_name}: nan")
                    else:
                        print(f"    {metric_name}: {value:.4f}" if isinstance(value, float) else f"    {metric_name}: {value}")
                    break
                except Exception as e:
                    if attempt < 2:
                        await asyncio.sleep(2 ** attempt)
                    else:
                        print(f"    {metric_name}: ERROR - {e}")
                        scores[metric_name] = None

        counter["done"] += 1
        print(f"  [{counter['done']}/{total}] Done")
        return scores


async def main():
    print("=" * 60)
    print("Ragas Evaluation: Thai Industry RAG")
    print("=" * 60)

    # Load gathered responses
    if not RESPONSES_FILE.exists():
        print(f"\nERROR: {RESPONSES_FILE} not found.")
        print("Run `uv run tests/gather_responses.py` first.")
        return

    with open(RESPONSES_FILE, encoding="utf-8") as f:
        data = json.load(f)

    samples = data["samples"]
    print(f"\nLoaded {len(samples)} samples (gathered at {data['gathered_at']})")

    # Setup evaluator
    ragas_llm = LangchainLLMWrapper(
        ChatOpenAI(
            model="glm-5-turbo",
            base_url=os.getenv("LLM_ENDPOINT"),
            api_key=os.getenv("LLM_API_KEY"),
            temperature=0,
        )
    )
    ragas_embeddings = LangchainEmbeddingsWrapper(
        LocalEmbeddings(
            base_url=os.getenv("EMBEDDING_ENDPOINT", "http://localhost:8080"),
            model=os.getenv("EMBEDDING_MODEL", "aisingapore/SEA-LION-E5-Embedding-600M"),
        )
    )

    # Initialize metrics
    metrics = {
        "faithfulness": Faithfulness(llm=ragas_llm),
        "answer_relevancy": AnswerRelevancy(llm=ragas_llm, embeddings=ragas_embeddings),
        "context_precision": ContextPrecision(llm=ragas_llm),
        "context_recall": ContextRecall(llm=ragas_llm),
    }

    # Adapt prompts to Thai (one-time translation, saved for reuse)
    print("\n--- Adapting Prompts to Thai ---")
    needs_adaptation = load_adapted_prompts(list(metrics.values()))
    if needs_adaptation:
        print(f"  Adapting {len(needs_adaptation)} metric(s)...")
        await adapt_metrics_to_thai(needs_adaptation, ragas_llm)

    # Run all samples concurrently with progress tracking
    print(f"\n--- Running Ragas Metrics ({MAX_CONCURRENT} concurrent) ---\n")
    semaphore = asyncio.Semaphore(MAX_CONCURRENT)
    counter = {"done": 0}

    tasks = [
        score_sample(s, metrics, semaphore, len(samples), counter)
        for s in samples
    ]
    results = await asyncio.gather(*tasks)

    # Print results
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)

    print(f"\n{'#':>2} {'Tier':>3} {'Faith':>6} {'AnsR':>6} {'CtxP':>6} {'CtxR':>6}  Question")
    print("-" * 70)
    for r in sorted(results, key=lambda x: x["id"]):
        def fmt(v):
            if isinstance(v, float) and math.isnan(v):
                return "  nan"
            return f"{v:.3f}" if isinstance(v, float) else "  N/A"
        print(f"{r['id']:>2} {r['tier']:>3} {fmt(r.get('faithfulness')):>6} {fmt(r.get('answer_relevancy')):>6} "
              f"{fmt(r.get('context_precision')):>6} {fmt(r.get('context_recall')):>6}  {r['user_input'][:40]}...")

    print("\n--- Averages ---")
    for metric_name in metrics:
        values = [r[metric_name] for r in results
                  if isinstance(r.get(metric_name), (int, float))
                  and not (isinstance(r[metric_name], float) and math.isnan(r[metric_name]))]
        if values:
            avg = sum(values) / len(values)
            print(f"  {metric_name}: {avg:.4f} ({len(values)}/{len(results)} scored)")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
