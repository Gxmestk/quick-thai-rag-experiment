# Troubleshooting

## Embedding timeout cascade → graph corruption

**Symptoms**: `RuntimeError: Graph edge indexing error`, `InstructorRetryException: multiple tool calls`, thin/empty search answers

**Cause**: Cognee's embedding timeout is hardcoded at 30s. When it fires mid-cognify, the graph ends up partially constructed.

**Fix**: Patch timeout to 300s, then prune and rebuild. See [embedding-timeout-cascade.md](embedding-timeout-cascade.md).

## Instructor: "does not support multiple tool calls"

**NOT an LLM issue**. Caused by corrupted graph state sending garbled prompts to the LLM.

**Fix**: Fix the embedding timeout and rebuild the graph. See [embedding-timeout-cascade.md](embedding-timeout-cascade.md).

## AnswerRelevancy: all nan

**Cause**: LangChain's `OpenAIEmbeddings` tokenizes Thai text into integer arrays before sending to the embedding endpoint. The local server expects strings, gets `[85894, 32882, ...]` and returns 422.

**Fix**: Use a custom `LocalEmbeddings` class that sends raw strings directly via httpx, wrapped in `LangchainEmbeddingsWrapper`.

## "Unknown Model" error from Z.AI

**Cause**: Used `openai/glm-5.1` as model ID when calling Z.AI API directly. The `openai/` prefix is for litellm routing only.

**Fix**: Use `glm-5.1` (no prefix) when calling the API directly (e.g., Ragas evaluator LLM).

## Graph edge indexing error

**Cause**: Corrupted Kuzu graph from embedding timeout cascade.

**Fix**: `cognee.prune_system(metadata=True)` then re-run `add()` + `cognify()`.

## LLM connection test timeout

**Fix**: Set `COGNEE_SKIP_CONNECTION_TEST=true` in `.env`.
