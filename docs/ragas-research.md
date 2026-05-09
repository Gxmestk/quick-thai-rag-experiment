# Ragas Evaluation Framework — Research Notes

**Ragas version**: 0.4.3 (installed)
**Docs**: https://docs.ragas.io/en/stable/
**Source**: https://github.com/vibrantlabsai/ragas

---

## Core Concepts

Ragas evaluates RAG pipelines with three types of metrics:

- **LLM-based** — Require an LLM to judge quality (e.g., Faithfulness). Inherit from `MetricWithLLM`.
- **Embedding-based** — Require an embedding model (e.g., AnswerRelevancy). Inherit from `MetricWithEmbeddings`.
- **Non-LLM** — Pure computation, no API cost (e.g., CHRF, BLEU). Inherit from `Metric`.

Metric output types: **Discrete** (pass/fail), **Numeric** (0.0–1.0), **Ranking**.

Single-turn metrics evaluate individual query-response pairs via `single_turn_ascore()`.
Multi-turn metrics evaluate conversation sequences via `multi_turn_ascore()`.

---

## Evaluation API

### Current API: `@experiment` decorator (recommended)

`aevaluate()` and `evaluate()` are **deprecated**. Use the `@experiment` decorator instead.

```python
from ragas import Dataset, experiment

dataset = Dataset.load(name="my_eval", backend="local/csv", root_dir="./data")

@experiment()
async def evaluate_rag(row):
    response = await my_rag_pipeline(row["question"])
    return {**row, "response": response}
```

Results are automatically saved as timestamped CSV files in `experiments/`.

### Inline metric scoring

```python
from ragas.metrics.collections import Faithfulness

@experiment()
async def evaluated_experiment(row):
    response = await my_rag_pipeline(row["question"])
    score = await Faithfulness().single_turn_ascore(
        user_input=row["question"],
        response=response,
        retrieved_contexts=row["contexts"]
    )
    return {**row, "response": response, "faithfulness": score.value}
```

### Parameterized experiments (A/B testing)

```python
@experiment()
async def model_comparison(row, model_name: str, temperature: float):
    response = await my_pipeline(row["input"], model=model_name)
    return {**row, "response": response, "experiment_name": f"{model_name}_temp_{temperature}"}

await model_comparison.arun(dataset, model_name="glm-5.1", temperature=0.0)
```

### RunConfig

```python
from ragas.run_config import RunConfig

config = RunConfig(
    timeout=300,       # seconds per API call
    max_retries=5,
    max_wait=120,
    max_workers=8,     # concurrent workers (reduce for rate limits)
    seed=42
)
```

---

## Available Metrics

### Retrieval Quality (LLM-based)

| Metric | Requires | Best For |
|--------|----------|----------|
| **Context Precision** | LLM | Are relevant chunks ranked higher than irrelevant ones? |
| **Context Recall** | LLM | Was all relevant information retrieved? |
| **Context Entity Recall** | LLM | Entity-level recall for domain-specific eval |

### Response Quality

| Metric | Requires | Best For |
|--------|----------|----------|
| **Faithfulness** | LLM | Hallucination detection — does the answer stick to context? |
| **FaithfulnesswithHHEM** | LLM + HHEM-2.1 | Faster hallucination detection via Vectara's classifier |
| **FactualCorrectness** | LLM | Compare response against reference for factual accuracy |
| **Answer Relevancy** | LLM + Embeddings | How relevant is the answer to the question? |
| **ResponseGroundedness** | LLM | Is the response grounded in provided context? |

### Non-LLM Metrics (fast, no API cost)

| Metric | Best For |
|--------|----------|
| **BleuScore** | N-gram overlap (translation-style) |
| **RougeScore** | Recall-oriented overlap (summarization-style) |
| **CHRF** | Character n-gram F-score — **good for Thai** |
| **StringPresence** | Check if specific string appears |
| **ExactMatch** | Strict equality |
| **NonLLMStringSimilarity** | Distance-based string comparison |
| **NonLLMContextSimilarity** | Context similarity without LLM |
| **NonLLMContextPrecision** | Context precision without LLM |
| **NonLLMContextRecall** | Context recall without LLM |

### Custom Metrics

```python
from ragas.metrics._metric import numeric_metric, discrete_metric

@numeric_metric(name="thai_token_overlap", lower_bound=0.0, upper_bound=1.0)
async def thai_token_overlap(response: str, reference: str) -> float:
    from pythainlp.tokenize import word_tokenize
    response_tokens = set(word_tokenize(response))
    reference_tokens = set(word_tokenize(reference))
    if not reference_tokens:
        return 0.0
    return len(response_tokens & reference_tokens) / len(reference_tokens)
```

---

## Custom LLMs and Embeddings

### Wrapping LangChain models

```python
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import ChatOpenAI

ragas_llm = LangchainLLMWrapper(ChatOpenAI(model="glm-5.1", base_url="...", api_key="..."))
ragas_embeddings = LangchainEmbeddingsWrapper(my_langchain_embeddings)
```

### Passing to metrics

```python
from ragas.metrics.collections import Faithfulness, AnswerRelevancy

scorer = Faithfulness(llm=ragas_llm)
relevancy = AnswerRelevancy(llm=ragas_llm, embeddings=ragas_embeddings)
```

### Import from collections (recommended)

Use `from ragas.metrics.collections import ...` instead of legacy `from ragas.metrics import ...`.

---

## Testset Generation

### Flow

1. Load documents → build `KnowledgeGraph` with transformations (summaries, embeddings)
2. LLM generates synthetic questions with configurable distribution
3. Output: `Testset` with `user_input`, `reference`, `reference_contexts`

### Quickstart

```python
from ragas.testset import TestsetGenerator
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

generator_llm = LangchainLLMWrapper(ChatOpenAI(model="glm-5.1", ...))
generator_embeddings = LangchainEmbeddingsWrapper(my_embeddings)

generator = TestsetGenerator(llm=generator_llm, embedding_model=generator_embeddings)
dataset = generator.generate_with_langchain_docs(docs, testset_size=10)
```

### KnowledgeGraph architecture

- **Node** — Document chunk with properties (text, metadata), typed (DOCUMENT, CHUNK)
- **Relationship** — Links nodes with type, direction, properties
- Uses **Leiden algorithm** for community detection (clusters related chunks)
- Save/load with **UTF-8 encoding** — Thai text safe

### Default query distribution

- 50% SingleHopSpecificQuerySynthesizer
- 25% MultiHopAbstractQuerySynthesizer
- 25% MultiHopSpecificQuerySynthesizer

### Custom distribution

```python
from ragas.testset.synthesizers import default_query_distribution
query_distribution = default_query_distribution(generator_llm)
```

---

## Thai RAG Recommendations

### Metric selection for Thai

**Primary:**
- **Faithfulness** (LLM) — hallucination detection, language-agnostic reasoning
- **Context Precision/Recall** (LLM) — retrieval quality
- **CHRF** (non-LLM) — character n-gram metric, works well for Thai (no whitespace segmentation needed)

**Use with caution:**
- **Answer Relevancy** — requires embeddings with strong Thai support; poor Thai embeddings = unreliable scores
- **BLEU/ROUGE** — assume whitespace-tokenized words; Thai needs pre-tokenization (e.g., pythainlp)

### LLM for evaluation

The evaluator LLM must understand Thai well enough to judge faithfulness and relevance. GLM-5.1 works but a stronger multilingual model (GPT-4o, Claude) would give more reliable metric scores.

### Embeddings for AnswerRelevancy

- `text-embedding-3-large` (OpenAI) — good multilingual coverage
- Multilingual-e5 or BGE-M3 — open-source with strong Thai support
- SEA-LION E5 (our local model) — Thai-optimized but needs proper string passthrough (LangChain's OpenAIEmbeddings tokenizes Thai into integer arrays before sending)

---

## Gotchas

1. `aevaluate()` is deprecated — use `@experiment` decorator
2. `from ragas.metrics.collections import ...` is the current API, not `from ragas.metrics import ...`
3. AnswerRelevancy sends Thai text through LangChain's OpenAIEmbeddings which tokenizes into integers — need custom `LocalEmbeddings` wrapper that sends raw strings
4. Some doc pages (v0.4) return 404 — docs may lag behind the published source code
5. KnowledgeGraph module uses UTF-8 throughout — safe for Thai
6. `RunConfig.max_workers=16` default may hit rate limits on Z.AI endpoint — reduce to 4-8
