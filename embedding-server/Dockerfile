FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

RUN apt-get update && \
    apt-get install -y --no-install-recommends python3 python3-pip && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip install --no-cache-dir --break-system-packages \
    sentence-transformers>=3.0 \
    fastapi \
    uvicorn[standard] \
    torch --index-url https://download.pytorch.org/whl/cu124

COPY server.py .

EXPOSE 80

CMD ["python3", "server.py"]
