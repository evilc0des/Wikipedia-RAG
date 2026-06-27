FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    QDRANT_URL=http://localhost:6333

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3-pip \
    python3.10-venv \
    curl \
    ca-certificates \
    zlib1g \
    libcudnn8 \
    && rm -rf /var/lib/apt/lists/*

RUN curl -L -o /tmp/qdrant.tar.gz \
        https://github.com/qdrant/qdrant/releases/download/v1.9.7/qdrant-x86_64-unknown-linux-gnu.tar.gz \
    && tar xzf /tmp/qdrant.tar.gz -C /usr/local/bin \
    && rm /tmp/qdrant.tar.gz

WORKDIR /app

COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install \
        --extra-index-url https://download.pytorch.org/whl/cu124 \
        torch==2.5.1 \
        -r requirements.txt && \
    pip install --force-reinstall --no-deps onnxruntime-gpu==1.21.0

COPY *.py /app/
COPY scripts/ /app/scripts/

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

EXPOSE 8080

ENTRYPOINT ["/entrypoint.sh"]
CMD ["serve"]
