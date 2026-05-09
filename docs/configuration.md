# Configuration

## LLM — Z.AI GLM-5.1

```bash
LLM_API_KEY="<your-zai-api-key>"
LLM_MODEL="openai/glm-5.1"       # openai/ prefix required for litellm routing
LLM_PROVIDER="openai"
LLM_ENDPOINT="https://api.z.ai/api/coding/paas/v4"
```

The `openai/` prefix is required because cognee uses litellm, which routes via the OpenAI SDK with a custom `api_base`.

**Important**: The Z.AI API expects `glm-5.1` as the model ID (no prefix). The `openai/` prefix is only for litellm routing inside cognee. When calling the API directly (e.g., Ragas evaluator), use `glm-5.1`.

## Embedding — SEA-LION E5 (GPU)

```bash
EMBEDDING_PROVIDER="openai_compatible"
EMBEDDING_MODEL="aisingapore/SEA-LION-E5-Embedding-600M"
EMBEDDING_ENDPOINT="http://localhost:8080"
EMBEDDING_DIMENSIONS=1024
EMBEDDING_MAX_TOKENS=512
EMBEDDING_API_KEY="not-needed"
```

Runs in Docker with NVIDIA GPU passthrough (GTX 1650, ~2,244 MiB VRAM).

## Docker GPU Setup

Requires rootless Docker GPU passthrough:

1. **Host config**: `/etc/nvidia-container-runtime/config.toml` — set `no-cgroups = true`
2. **docker-compose.yml**: Use `runtime: nvidia` + `NVIDIA_VISIBLE_DEVICES=all` (not `deploy.resources.reservations.devices`)
3. **Dockerfile**: Use `nvidia/cuda:12.4.1-runtime-ubuntu22.04` base + `torch+cu124` (not `python:3.12-slim`)

## Graph & Vector

```bash
GRAPH_DATABASE_PROVIDER="kuzu"
VECTOR_DB_PROVIDER="lancedb"
```

## Dev Flags

```bash
COGNEE_SKIP_CONNECTION_TEST=true
COGNEE_TRACING_ENABLED=false
LITELLM_LOG="ERROR"
TOKENIZERS_PARALLELISM="false"
ENABLE_BACKEND_ACCESS_CONTROL=false
ENV="local"
```
