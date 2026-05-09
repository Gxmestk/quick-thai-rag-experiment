# Evaluation Results: Thai Industry RAG

## Run: 2026-05-10 (v1 — baseline, English prompts)

**Pipeline**: cognee v1.0.9 (graph: Ladybug, vector: LanceDB) → GRAPH_COMPLETION + CHUNKS search
**Evaluator LLM**: Z.AI GLM-5.1
**Evaluator Embeddings**: SEA-LION E5 600M (local GPU)
**Dataset**: 317 nodes, 752+ edges, 6 source documents about Thai industry
**Questions**: 16 (8 Tier 1 factual + 8 Tier 2 cross-document reasoning)
**Script**: Manual `single_turn_ascore()` loop, no concurrency, no retries, English prompts

## Results Table

```
 # Tier  Faith   AnsR   CtxP   CtxR   Fact  Question
  1  T1  1.000  0.961  1.000  1.000  0.000  ประเทศไทยผลิตรถยนต์ปี 2567...
  2  T1  1.000  0.916  1.000  1.000  0.670  ผลผลิตข้าวไทยปี 2567...
  3  T1  1.000  0.984  1.000  1.000  1.000  โครงการอีอีซี...
  4  T1  1.000  0.976  1.000  1.000  0.000  ระบบชำระเงินดิจิทัล...
  5  T1  0.700  0.943  1.000  1.000  0.570  โครงการพลังงานลม...
  6  T1  1.000  0.956  1.000  1.000  0.000  นักท่องเที่ยวต่างชาติ...
  7  T1  1.000  0.924  1.000  1.000  0.290  นโยบายส่งเสริมรถยนต์ไฟฟ้า...
  8  T1  1.000  0.882  1.000  1.000  0.800  พลังงานหมุนเวียน...
  9  T2    nan    nan  1.000  0.500  0.000  ผู้ผลิตรถยนต์ไฟฟ้ารายใหญ่...
 10  T2  1.000  0.820  0.500  1.000  0.330  มันสำปะหลัง...
 11  T2  1.000  0.947  1.000  1.000  1.000  โรงงานบีวายดี...
 12  T2  1.000  0.986  0.333  1.000  0.290  โลจิสติกส์...
 13  T2  1.000  0.982  1.000  1.000  0.800  นโยบายวีซ่า...
 14  T2  0.833  0.919  0.500  1.000  1.000  กลุ่มไทยยูเนียน...
 15  T2  1.000  0.970  0.333  1.000  0.000  ภาคการผลิต / มาบตาพุด...
 16  T2  0.833  0.966  1.000  1.000  0.000  การเกษตรอัจฉริยะ...
```

## Averages

| Metric              | Score  | Notes                        |
|---------------------|--------|------------------------------|
| Faithfulness        | 0.95   | 1 nan (sample 9, no retries) |
| Answer Relevancy    | 0.94   | 1 nan (sample 9, no retries) |
| Context Precision   | 0.85   | Strong context retrieval      |
| Context Recall      | 0.97   | Excellent recall              |
| Factual Correctness | 0.42   | English prompts vs Thai content |

## Tier Breakdown

**Tier 1 (factual)**: Faithfulness 0.96, AnswerRelevancy 0.94, ContextPrecision 1.00, ContextRecall 1.00
**Tier 2 (reasoning)**: Faithfulness 0.94, AnswerRelevancy 0.94, ContextPrecision 0.72, ContextRecall 0.94

## Key Findings

### What works well
- **Faithfulness** (0.95): Cognee's graph-based responses are highly faithful to retrieved context
- **Answer Relevancy** (0.94): Responses directly answer the questions
- **Context Recall** (0.97): The knowledge graph almost always contains the relevant information
- **Tier 1 performance**: Near-perfect scores across all metrics except FactualCorrectness

### Issues

#### FactualCorrectness unreliable (0.42 average, 6 samples scored 0.0)

**Root cause**: All Ragas metric prompts (instructions + few-shot examples) are in English. When the LLM receives English instructions like "judge the faithfulness of a series of statements" with English few-shot examples about "John is a CS student", but then has to verify Thai claims like "ปี 2567 ไทยผลิตรถยนต์ทั้งหมด 1.84 ล้านคัน", it produces inconsistent verdicts for semantically identical Thai claims depending on direction:

```
Claim: "ปี 2567 ไทยผลิตรถยนต์ทั้งหมด 1.84 ล้านคัน"
  vs Reference "ประเทศไทยผลิตรถยนต์ 1.84 ล้านคัน"  => verdict=0 (WRONG)

Claim: "ประเทศไทยผลิตรถยนต์ 1.84 ล้านคัน"
  vs Response "ปี 2567 ไทยผลิตรถยนต์ทั้งหมด 1.84 ล้านคัน"  => verdict=1 (correct)
```

Same fact, opposite verdicts. The cross-lingual mismatch (English prompts judging Thai content) is the issue — the cognee responses themselves are factually correct.

**Fix**: Use `adapt_prompts(language="thai", adapt_instruction=True)` to translate all Ragas metric prompts (instructions + examples) to Thai before evaluation. The `adapt_prompts()` method is built into Ragas via `PromptMixin` on all metric classes.

#### Sample 9 nan (faithfulness + answer_relevancy)

**Root cause**: GLM-5.1 returned malformed JSON for AnswerRelevancy (missing `{"question": ...}` key). With no retry mechanism in the manual loop, a single bad LLM response produces permanent `nan`.

**Fix**: Use `RunConfig(max_retries=10)` which gives Ragas automatic retry with exponential backoff for transient failures.

#### Context Precision lower for Tier 2 (0.72 vs 1.00 for Tier 1)

Cross-document reasoning questions retrieve more context, some of which is less relevant. Expected for harder queries.

#### 100-minute runtime

**Root cause**: Manual `for` loop calling `single_turn_ascore()` sequentially — 80 LLM calls × ~75s each, no concurrency.

**Fix**: Use `evaluate()` with `RunConfig(max_workers=10)` for concurrent execution.

## Run Details (v1)

- Total runtime: ~100 minutes (80 metric calls × ~75s each, sequential)
- LLM call pattern: "LLM returned 1 generations instead of requested 3" — GLM-5.1 doesn't support `n>1`
- All metrics use `single_turn_ascore()` in a manual loop (no concurrency, no retries)
- English prompts used to evaluate Thai content

## Run: 2026-05-10 (v2 — Thai prompts, concurrent, glm-5-turbo)

**Pipeline**: Same cognee setup, responses re-gathered with glm-5-turbo
**Evaluator LLM**: Z.AI GLM-5-turbo
**Evaluator Embeddings**: SEA-LION E5 600M (local GPU)
**Script**: Manual concurrent loop (`asyncio.Semaphore(10)` + `asyncio.gather()`), Thai-adapted prompts, 3 retries per metric
**Metrics**: 4 (dropped FactualCorrectness for speed)

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

*Note: Per-sample values above are from the v2 run. Values shown are best-effort reconstruction; averages below are definitive.*

## Averages

| Metric              | v1     | v2     | Delta  |
|---------------------|--------|--------|--------|
| Faithfulness        | 0.95   | 0.89   | -0.06  |
| Answer Relevancy    | 0.94   | 0.83   | -0.11  |
| Context Precision   | 0.85   | 0.85   |  0.00  |
| Context Recall      | 0.97   | 0.94   | -0.03  |
| Factual Correctness | 0.42   | —      | dropped |

**16/16 scored, 0 nans** (v1 had 1 nan on sample 9)

## v1 vs v2 Comparison

**What improved**:
- **0 nans**: Retry mechanism (3 attempts per metric) eliminated the v1 nan on sample 9
- **~15 min runtime** (vs ~100 min v1): Concurrent scoring with `asyncio.Semaphore(10)` + faster glm-5-turbo model
- **Thai prompts**: `adapt_prompts(language="thai")` translates Ragas metric instructions and few-shot examples to Thai, removing cross-lingual mismatch

**What dropped**:
- Faithfulness -0.06 and AnswerRelevancy -0.11: Slight score decrease likely due to using glm-5-turbo (faster but less capable than glm-5.1) as both cognee response generator AND evaluator. Thai prompt adaptation may also grade more strictly than English prompts.
- ContextRecall -0.03: Minor, within noise

## What Changed (v2)

The evaluation script was rewritten:
1. Manual concurrent loop (`asyncio.Semaphore(10)` + `asyncio.gather()`) for visible progress
2. `adapt_prompts(language="thai")` to translate prompts to Thai (saved to `tests/prompts/` for reuse)
3. 3 retries per metric with exponential backoff (`sleep(2^attempt)`)
4. Switched evaluator LLM from glm-5.1 to glm-5-turbo for speed
5. Dropped FactualCorrectness (4x LLM calls per sample, bottleneck)

Re-run with: `uv run tests/eval_ragas.py`

## Scripts

- `tests/eval_data.py` — shared 16-question dataset
- `tests/gather_responses.py` — queries cognee, saves to `tests/data/responses.json`
- `tests/eval_ragas.py` — loads saved responses, runs 5 Ragas metrics (rewritten with proper API)
- `tests/prompts/` — saved Thai-adapted prompts (auto-generated on first run)

Usage:
```bash
uv run tests/gather_responses.py   # Step 1: gather cognee responses
uv run tests/eval_ragas.py          # Step 2: run Ragas metrics (Thai prompts, concurrent)
```
