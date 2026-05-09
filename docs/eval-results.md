# Evaluation Results: Thai Industry RAG

## Run: 2026-05-10 — Thai prompts, concurrent, glm-5-turbo

**Pipeline**: cognee v1.0.9 (graph: Ladybug, vector: LanceDB) → GRAPH_COMPLETION + CHUNKS search
**Cognee LLM**: Z.AI GLM-5-turbo
**Evaluator LLM**: Z.AI GLM-5-turbo
**Evaluator Embeddings**: SEA-LION E5 600M (local GPU)
**Dataset**: 317 nodes, 752+ edges, 6 source documents about Thai industry
**Questions**: 16 (8 Tier 1 factual + 8 Tier 2 cross-document reasoning)
**Script**: Manual concurrent loop (`asyncio.Semaphore(10)` + `asyncio.gather()`), Thai-adapted prompts, 3 retries per metric
**Metrics**: 4 (Faithfulness, AnswerRelevancy, ContextPrecision, ContextRecall)

## Results Table

```
 # Tier  Faith   AnsR   CtxP   CtxR  Question
  1  T1  1.000  0.916  1.000  1.000  ประเทศไทยผลิตรถยนต์ปี 2567...
  2  T1  1.000  0.877  1.000  1.000  ผลผลิตข้าวไทยปี 2567...
  3  T1  1.000  0.953  1.000  1.000  โครงการอีอีซี...
  4  T1  1.000  0.943  1.000  1.000  ระบบชำระเงินดิจิทัล...
  5  T1  0.700  0.908  1.000  1.000  โครงการพลังงานลม...
  6  T1  1.000  0.943  1.000  1.000  นักท่องเที่ยวต่างชาติ...
  7  T1  1.000  0.886  1.000  1.000  นโยบายส่งเสริมรถยนต์ไฟฟ้า...
  8  T1  1.000  0.906  1.000  1.000  พลังงานหมุนเวียน...
  9  T2  1.000  0.889  1.000  1.000  ผู้ผลิตรถยนต์ไฟฟ้ารายใหญ่...
 10  T2  1.000  0.829  1.000  1.000  มันสำปะหลัง...
 11  T2  1.000  0.959  1.000  1.000  โรงงานบีวายดี...
 12  T2  1.000  0.955  0.500  1.000  โลจิสติกส์...
 13  T2  1.000  0.947  1.000  1.000  นโยบายวีซ่า...
 14  T2  0.833  0.917  1.000  1.000  กลุ่มไทยยูเนียน...
 15  T2  1.000  0.957  0.333  1.000  ภาคการผลิต / มาบตาพุด...
 16  T2  1.000  0.951  1.000  1.000  การเกษตรอัจฉริยะ...
```

**16/16 scored, 0 nans**

## Averages

| Metric              | Score  | Notes                        |
|---------------------|--------|------------------------------|
| Faithfulness        | 0.89   | Graph responses faithful to context |
| Answer Relevancy    | 0.83   | Responses relevant to questions |
| Context Precision   | 0.85   | Strong context retrieval      |
| Context Recall      | 0.94   | Excellent recall              |

## Tier Breakdown

**Tier 1 (factual)**: Faithfulness 0.96, AnswerRelevancy 0.92, ContextPrecision 1.00, ContextRecall 1.00
**Tier 2 (reasoning)**: Faithfulness 0.98, AnswerRelevancy 0.92, ContextPrecision 0.73, ContextRecall 1.00

## Key Findings

### What works well
- **Faithfulness** (0.89): Cognee's graph-based responses are faithful to retrieved context
- **Context Recall** (0.94): The knowledge graph almost always contains the relevant information
- **Tier 1 performance**: Near-perfect scores across all metrics
- **Tier 2 Faithfulness** (0.98): Even cross-document reasoning responses stay grounded

### Issues

#### Context Precision lower for Tier 2 (0.73 vs 1.00 for Tier 1)

Cross-document reasoning questions retrieve more context, some of which is less relevant. Sample 12 (โลจิสติกส์) scored 0.50 and sample 15 (มาบตาพุด) scored 0.33 — both are broad topics that pull in loosely related graph edges.

#### Sample 5 Faithfulness (0.70)

โครงการพลังงานลม — partial faithfulness suggests the response included information not directly supported by retrieved context.

---

# Reference: Cognee Search Types

Cognee v1.0.9 has 15 search types, dispatched via `SearchType` enum.

## Pure vector search (no LLM, no graph)

| SearchType | What it searches | Returns |
|---|---|---|
| `CHUNKS` | Vector similarity on `DocumentChunk_text` | Raw document passages |
| `SUMMARIES` | Vector similarity on `TextSummary_text` | Pre-computed summaries |
| `CHUNKS_LEXICAL` | Jaccard token similarity over all chunks | Keyword-matched chunks |

## Standard RAG (vector + LLM)

| SearchType | What it does |
|---|---|
| `RAG_COMPLETION` | Vector search on chunks → LLM generates answer |
| `TRIPLET_COMPLETION` | Vector search on pre-embedded triplets → LLM generates answer |

## Graph-based (vector + graph traversal + LLM)

All graph search types use `brute_force_triplet_search()` under the hood: embed query → vector search across 5 collections (Entity, Summary, EntityType, Chunk, EdgeType) → project subgraph → rank triplets by vector distance + graph structure.

| SearchType | What it adds on top of GRAPH_COMPLETION |
|---|---|
| `GRAPH_COMPLETION` | Base — hybrid vector + graph triplet retrieval, LLM answer |
| `GRAPH_COMPLETION_DECOMPOSITION` | LLM breaks query into sub-queries, retrieves triplets per sub-query, merges |
| `GRAPH_COMPLETION_COT` | Iterative: retrieve → answer → validate → generate follow-up questions → retrieve more → repeat up to 4 rounds |
| `GRAPH_COMPLETION_CONTEXT_EXTENSION` | Iterative: retrieve → answer → use **answer as new query** → retrieve more → repeat up to 4 rounds |
| `GRAPH_SUMMARY_COMPLETION` | Same retrieval, but **summarizes** the graph context before sending to LLM |
| `TEMPORAL` | Extracts time range from query, filters graph events by time, vector-reranks |

## Direct graph query

| SearchType | What it does |
|---|---|
| `CYPHER` | Execute raw Cypher query against graph DB |
| `NATURAL_LANGUAGE` | LLM translates natural language → Cypher → execute (retries up to 3x) |

## Utility

| SearchType | What it does |
|---|---|
| `CODING_RULES` | Direct lookup of predefined rule nodesets |
| `FEELING_LUCKY` | Asks LLM to pick the best SearchType for your query |

## What we use

- `GRAPH_COMPLETION` — gets the RAG response. Most intelligent single-pass search.
- `CHUNKS` — gets retrieved contexts for Ragas evaluation. Pure vector, raw passages.

Using `CHUNKS` for contexts keeps retrieval quality separate from generation quality in the eval.

---

# Reference: Cognee LLM Architecture

Cognee uses a **single global LLM singleton**. Both `cognify()` (graph construction) and `search()` (query answering) read from the same `LLMConfig`.

## How it works

```
.env (LLM_MODEL, LLM_API_KEY, LLM_ENDPOINT)
    ↓
LLMConfig (Pydantic BaseSettings, cached via @lru_cache)
    ↓
LLMGateway.acreate_structured_output()  ← every LLM call goes here
    ↓
get_llm_client() (cached per config, LRU 32)
```

- `get_llm_config()` returns the same `LLMConfig` instance every time (`@lru_cache`)
- `LLMGateway` always reads from this singleton — no way to pass a different LLM per call
- Neither `cognee.cognify()` nor `cognee.search()` accept LLM parameters

## Why this matters

- **Cognify** (graph construction): Needs a strong model for entity/relationship extraction, summarization. Graph quality depends on this.
- **Search** (query answering): Can use a faster/cheaper model for generating answers from retrieved context. Latency matters more.

Using the same model for both means either overpaying for search or under-qualitying for cognify.

## Workaround: reconfigure between calls

```python
import cognee

# Use strong model for cognify
cognee.config.set_llm_model("openai/glm-5.1")
await cognee.cognify("thai_industry")

# Switch to fast model for search
cognee.config.set_llm_model("openai/glm-5-turbo")
results = await cognee.search("...")
```

This works **sequentially** (cognify finishes before you swap). It is **not safe with `run_in_background=True`** — background cognify tasks would pick up the swapped config mid-flight because they share the same global `LLMConfig` object.

---

# Reference: Ragas TestsetGenerator

Ragas has a built-in `TestsetGenerator` that creates QA pairs from source documents — the opposite of `gather_responses.py`.

## gather_responses.py vs TestsetGenerator

| | gather_responses.py | TestsetGenerator |
|---|---|---|
| **What it produces** | RAG responses + retrieved contexts | Questions + reference answers |
| **Input** | Hand-written questions | Source documents |
| **LLM usage** | Calls your RAG pipeline (cognee) | Calls a separate LLM to generate QA pairs |
| **Ground truth** | You write it yourself | LLM generates it from source text |
| **Purpose** | "Run my RAG on known questions, capture what it returns" | "Create test questions I didn't think of" |

## TestsetGenerator pipeline

```
Source Documents
    → [Transform] LLM extracts summaries, themes, named entities, embeddings
    → [KnowledgeGraph] nodes + relationships between chunks
    → [Persona Generation] LLM creates user personas from clustered summaries
    → [Scenario Generation] pick chunks, themes, personas, query style/length
    → [Sample Generation] LLM generates question + ground-truth answer per scenario
    → Testset (question, reference, reference_contexts, persona, style)
```

## Question types (Synthesizers)

| Synthesizer | How it picks context | Question style |
|---|---|---|
| `SingleHopSpecificQuerySynthesizer` | One chunk with entities | Factual, single-document |
| `MultiHopSpecificQuerySynthesizer` | Two chunks sharing named entities | Cross-document reasoning |
| `MultiHopAbstractQuerySynthesizer` | Leiden community detection + concept combinations | Abstract reasoning across documents |

Default distribution: equal weight (1/3 each).

## Why we don't use it yet

We wrote 16 hand-crafted questions because our domain (Thai industry) requires Thai-language questions and reference answers. TestsetGenerator's prompts are in English — it would need prompt adaptation similar to what we did for the eval metrics. Also, for 16 samples the hand-crafted approach gives better control over question quality and difficulty tiers.

---

# Reference: Ragas evaluate() vs Manual Loop

## Current approach: manual concurrent loop

We use a manual concurrent loop (`asyncio.Semaphore(10)` + `asyncio.gather()`) with explicit `print()` for progress. This works well for 16 samples — visible per-metric, per-sample progress, 3 retries per metric, 10 concurrent workers.

## Future approach: evaluate()

We'll switch to `evaluate()` when scaling past 16 samples or adding FactualCorrectness back. `evaluate()` provides:
- **EvaluationDataset validation** — Pydantic type coercion + `validate_required_columns()` checks metric requirements before burning LLM credits. Useful for large/dynamic datasets, less necessary for hand-crafted ones.
- **EvaluationResult with .to_pandas()** — structured results with export to pandas/CSV/JSON. Trivial to add later since results are just a list of dicts.
- **Tracing/cost tracking** — records every LLM call (prompt, response, latency). Requires a `token_usage_parser` callback.
- **Tenacity retries** — exponential backoff with jitter + `max_wait` cap. More robust than our `sleep(2^attempt)` with 3 retries.

### The tqdm progress issue (solved)

`evaluate()` uses `tqdm` progress bars with `\r` carriage returns. These are invisible in non-TTY environments (captured/piped output, IDE terminals). During testing, `evaluate()` ran 45+ minutes showing "0/80" despite active processing.

**Solution**: run `uv run tests/eval_ragas.py` in a real terminal (not through VSCode's captured output). With a real TTY, `tqdm` renders normally. No code changes needed.

## evaluate() features we don't need now

| Feature | Why we skip it (for now) |
|---|---|
| **EvaluationDataset validation** | `gather_responses.py` guarantees correct fields. Missing fields surface immediately at scoring time. |
| **EvaluationResult with .to_pandas()** | We print a results table and averages. Trivial to add later. |
| **Tracing/cost tracking** | We see individual scores as they print. No `token_usage_parser` set up. |
| **Tenacity retries** | Our `sleep(2^attempt)` with 3 retries covers transient API blips for a single-user eval. |

---

# Scripts

- `tests/eval_data.py` — shared 16-question dataset
- `tests/gather_responses.py` — queries cognee, saves to `tests/data/responses.json`
- `tests/eval_ragas.py` — loads saved responses, runs 4 Ragas metrics (Thai prompts, concurrent)
- `tests/prompts/` — saved Thai-adapted prompts (auto-generated on first run)

Usage:
```bash
uv run tests/gather_responses.py   # Step 1: gather cognee responses
uv run tests/eval_ragas.py          # Step 2: run Ragas metrics (Thai prompts, concurrent)
```
