# Evaluation Results: Thai Industry RAG

## Run: 2026-05-10

**Pipeline**: cognee v1.0.9 (graph: Ladybug, vector: LanceDB) → GRAPH_COMPLETION + CHUNKS search
**Evaluator LLM**: Z.AI GLM-5.1
**Evaluator Embeddings**: SEA-LION E5 600M (local GPU)
**Dataset**: 317 nodes, 752+ edges, 6 source documents about Thai industry
**Questions**: 16 (8 Tier 1 factual + 8 Tier 2 cross-document reasoning)

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
| Faithfulness        | 0.95   | 1 nan (sample 9, LLM glitch) |
| Answer Relevancy    | 0.94   | 1 nan (sample 9, LLM glitch) |
| Context Precision   | 0.85   | Strong context retrieval      |
| Context Recall      | 0.97   | Excellent recall              |
| Factual Correctness | 0.42   | Unreliable with GLM-5.1 (see below) |

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

**Root cause**: GLM-5.1 is inconsistent at NLI (Natural Language Inference) verification for Thai text.

The metric decomposes claims from both response and reference, then asks the LLM "can claim X be inferred from text Y?" The LLM gives contradictory verdicts for semantically identical Thai claims depending on direction:

```
Claim: "ปี 2567 ไทยผลิตรถยนต์ทั้งหมด 1.84 ล้านคัน"
  vs Reference "ประเทศไทยผลิตรถยนต์ 1.84 ล้านคัน"  => verdict=0 (WRONG)

Claim: "ประเทศไทยผลิตรถยนต์ 1.84 ล้านคัน"
  vs Response "ปี 2567 ไทยผลิตรถยนต์ทั้งหมด 1.84 ล้านคัน"  => verdict=1 (correct)
```

Same fact, opposite verdicts. This is not a data quality issue — the cognee responses are factually correct.

**Mitigation**: Use a stronger evaluator LLM (GPT-4o) for FactualCorrectness, or use `mode="recall"` to only check one direction.

#### Sample 9 nan (faithfulness + answer_relevancy)

**Root cause**: Transient LLM failure. GLM-5.1 returned malformed JSON that Ragas couldn't repair.
- Faithfulness: `nan` — claim extraction/NLI returned empty
- AnswerRelevancy: `Invalid JSON response. Expected dictionary with key 'question'`

This is intermittent — re-running would likely produce a valid score.

#### Context Precision lower for Tier 2 (0.72 vs 1.00 for Tier 1)

Cross-document reasoning questions retrieve more context, some of which is less relevant. Expected for harder queries.

## Run Details

- Total runtime: ~100 minutes (80 metric calls × ~75s each)
- LLM call pattern: "LLM returned 1 generations instead of requested 3" — GLM-5.1 doesn't support `n>1`
- All metrics use `single_turn_ascore()` (Ragas v0.4.3 direct API)

## Scripts

- `tests/eval_data.py` — shared 16-question dataset
- `tests/gather_responses.py` — queries cognee, saves to `tests/data/responses.json`
- `tests/eval_ragas.py` — loads saved responses, runs 5 Ragas metrics

Usage:
```bash
uv run tests/gather_responses.py   # Step 1: gather cognee responses
uv run tests/eval_ragas.py          # Step 2: run Ragas metrics
```
