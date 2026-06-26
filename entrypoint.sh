#!/bin/bash
set -e

if [ ! -L /app/data ]; then
    ln -s /data /app/data
fi

mkdir -p /app/data/qdrant_storage

export QDRANT__STORAGE__STORAGE_PATH=/app/data/qdrant_storage

MODE="${1:-serve}"
shift || true

SNAPSHOT_ARG=""

if [ "$MODE" = "serve" ]; then
    if [ -n "$(ls -A /app/data/qdrant_storage 2>/dev/null)" ]; then
        echo "Qdrant storage has data from prior boot. Starting normally."
    elif [ -f /app/data/dense_index.snapshot ]; then
        echo "Snapshot found. Starting Qdrant from snapshot..."
        SNAPSHOT_ARG="--storage-snapshot /app/data/dense_index.snapshot"
    elif [ -n "$HF_DATASET_REPO" ]; then
        echo "Downloading data from HuggingFace Hub..."
        python3 scripts/bootstrap.py --download-only
        echo "Starting Qdrant from snapshot..."
        SNAPSHOT_ARG="--storage-snapshot /app/data/dense_index.snapshot"
    else
        echo "ERROR: No indices found and HF_DATASET_REPO not set."
        echo "Run with 'index' mode first, or set HF_DATASET_REPO + HF_TOKEN."
        exit 1
    fi
fi

echo "Starting Qdrant..."
/usr/local/bin/qdrant $SNAPSHOT_ARG &

echo "Waiting for Qdrant health..."
for i in $(seq 1 60); do
    if curl -sf http://localhost:6333/ > /dev/null 2>&1; then
        echo "Qdrant is healthy."
        break
    fi
    if [ "$i" -eq 60 ]; then
        echo "ERROR: Qdrant failed to start within 60s"
        exit 1
    fi
    sleep 1
done

case "$MODE" in
    index)
        echo "Running indexing: python3 index_data.py $@"
        exec python3 index_data.py "$@"
        ;;
    serve)
        echo "Starting API server on port 8080..."
        exec python3 -m uvicorn api:app --host 0.0.0.0 --port 8080
        ;;
    *)
        echo "Usage: $0 {index|serve} [args...]"
        echo "  index [--pages N] [--workers N] [--rebuild]   Run the indexing pipeline"
        echo "  serve                                          Start the query API"
        exit 1
        ;;
esac
