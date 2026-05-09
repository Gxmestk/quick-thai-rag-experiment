"""
Ragas evaluation for Thai RAG pipeline.
Loads gathered cognee responses and runs Ragas metrics.

Usage:
  1. Gather:   uv run tests/gather_responses.py
  2. Evaluate:  uv run tests/eval_ragas.py
"""

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
from ragas.metrics._factual_correctness import FactualCorrectness

RESPONSES_FILE = Path(__file__).resolve().parent / "data" / "responses.json"


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
            model="glm-5.1",
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
        "factual_correctness": FactualCorrectness(llm=ragas_llm),
    }

    # Score each sample
    print("\n--- Running Ragas Metrics ---\n")
    results = []

    for sample in samples:
        i = sample["id"]
        tier = sample["tier"]
        query = sample["user_input"]
        reference = sample["reference"]
        response = sample["response"]
        contexts = sample["retrieved_contexts"]

        print(f"[{tier}] [{i}/{len(samples)}] Scoring: {query[:50]}...")

        scores = {"id": i, "tier": tier, "user_input": query[:60]}
        ragas_sample = SingleTurnSample(
            user_input=query,
            response=response,
            reference=reference,
            retrieved_contexts=contexts,
        )
        for metric_name, metric in metrics.items():
            try:
                value = await metric.single_turn_ascore(ragas_sample)
                scores[metric_name] = value
                if isinstance(value, float) and math.isnan(value):
                    print(f"  {metric_name}: nan")
                else:
                    print(f"  {metric_name}: {value:.4f}" if isinstance(value, float) else f"  {metric_name}: {value}")
            except Exception as e:
                print(f"  {metric_name}: ERROR - {e}")
                scores[metric_name] = None

        results.append(scores)

    # Print summary
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)

    # Per-sample table
    print(f"\n{'#':>2} {'Tier':>3} {'Faith':>6} {'AnsR':>6} {'CtxP':>6} {'CtxR':>6} {'Fact':>6}  Question")
    print("-" * 78)
    for r in results:
        def fmt(v):
            if isinstance(v, float) and math.isnan(v):
                return "  nan"
            return f"{v:.3f}" if isinstance(v, float) else "  N/A"
        print(f"{r['id']:>2} {r['tier']:>3} {fmt(r.get('faithfulness')):>6} {fmt(r.get('answer_relevancy')):>6} "
              f"{fmt(r.get('context_precision')):>6} {fmt(r.get('context_recall')):>6} "
              f"{fmt(r.get('factual_correctness')):>6}  {r['user_input'][:40]}...")

    # Averages
    print("\n--- Averages ---")
    for metric_name in metrics:
        values = [r[metric_name] for r in results if isinstance(r.get(metric_name), (int, float)) and not (isinstance(r[metric_name], float) and math.isnan(r[metric_name]))]
        if values:
            avg = sum(values) / len(values)
            print(f"  {metric_name}: {avg:.4f} ({len(values)}/{len(results)} scored)")
        else:
            print(f"  {metric_name}: no scores")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
