# Embedding Timeout Cascade

## Root Problem

Cognee's `OpenAICompatibleEmbeddingEngine` has a hardcoded 30-second timeout for embedding requests.

**File**: `.venv/lib/python3.14/site-packages/cognee/infrastructure/databases/vector/embeddings/OpenAICompatibleEmbeddingEngine.py` line 153

```python
timeout=30.0  # hardcoded, too short for local models
```

On CPU with SEA-LION (600M params), large document chunks can exceed 30 seconds. Even on GPU, first cold requests or large batches can be slow.

## Cascade of Failures

```
Embedding request takes >30s
    → Exception raised mid-cognify pipeline
        → Partial graph state written (nodes without embeddings)
            → RuntimeError: Graph edge indexing error
                → InstructorRetryException: "multiple tool calls"
                    → Appears to be an LLM issue, but is actually corrupted graph
```

The instructor errors are **misleading**. Corrupted graph state causes cognee to send garbled context to the LLM, which responds abnormally (multiple tool calls instead of one). The LLM is not the problem.

## Fix

Patch the timeout to 300 seconds:

```python
timeout=300.0  # was 30.0
```

Then prune and rebuild: `cognee.prune_system(metadata=True)` → re-run `add()` + `cognify()`.

## Verification

- Clean graph: 390 nodes, 875 edges
- All 16 test queries return substantive answers
- No instructor errors

## Note

This patch is in `.venv/` — it resets if `uv` recreates the venv. Re-apply after `uv sync` or `uv lock --upgrade`.
