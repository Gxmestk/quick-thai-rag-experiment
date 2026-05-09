from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import uvicorn
import os

MODEL_NAME = os.getenv("MODEL_ID", "aisingapore/SEA-LION-E5-Embedding-600M")
DIMENSIONS = int(os.getenv("EMBEDDING_DIMENSIONS", "1024"))
MAX_BATCH_SIZE = int(os.getenv("MAX_BATCH_SIZE", "64"))

print(f"Loading model: {MODEL_NAME}")
model = SentenceTransformer(MODEL_NAME, trust_remote_code=True)
print(f"Model loaded. Dimensions: {model.get_sentence_embedding_dimension()}")

app = FastAPI()


class EmbeddingRequest(BaseModel):
    input: str | list[str]
    model: str = MODEL_NAME
    encoding_format: str = "float"


class EmbeddingObject(BaseModel):
    object: str = "embedding"
    embedding: list[float]
    index: int


class EmbeddingResponse(BaseModel):
    object: str = "list"
    data: list[EmbeddingObject]
    model: str
    usage: dict


@app.post("/v1/embeddings")
async def embeddings(req: EmbeddingRequest) -> EmbeddingResponse:
    texts = req.input if isinstance(req.input, list) else [req.input]
    all_embeddings = []
    for i in range(0, len(texts), MAX_BATCH_SIZE):
        batch = texts[i : i + MAX_BATCH_SIZE]
        vectors = model.encode(batch, normalize_embeddings=True, show_progress_bar=False)
        all_embeddings.extend(vectors.tolist())

    return EmbeddingResponse(
        object="list",
        data=[
            EmbeddingObject(embedding=emb, index=i)
            for i, emb in enumerate(all_embeddings)
        ],
        model=MODEL_NAME,
        usage={"prompt_tokens": sum(len(t.split()) for t in texts), "total_tokens": sum(len(t.split()) for t in texts)},
    )


@app.get("/health")
async def health():
    return {"status": "ok"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8080")), workers=1)
