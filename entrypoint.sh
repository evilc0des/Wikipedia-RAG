#!/bin/bash
set -e

mkdir -p /data/qdrant_storage

if [ ! -L /app/data ]; then
    ln -s /data /app/data
fi

echo "Starting Qdrant..."
/usr/local/bin/qdrant --storage-path /data/qdrant_storage &

echo "Waiting for Qdrant health..."
for i in $(seq 1 60); do
    if curl -sf http://localhost:6333/health > /dev/null 2>&1; then
        echo "Qdrant is healthy."
        break
    fi
    if [ "$i" -eq 60 ]; then
        echo "ERROR: Qdrant failed to start within 60s"
        exit 1
    fi
    sleep 1
done

MODE="${1:-serve}"
shift || true

case "$MODE" in
    index)
        echo "Running indexing: python3 index_data.py $*"
        exec python3 index_data.py "$@"
        ;;
    serve)
        if [ -f /app/data/chunks.db ] && ls /app/data/sparse_shards/shard_*.pkl > /dev/null 2>&1; then
            echo "Indices found. Skipping bootstrap."
        elif [ -n "$HF_DATASET_REPO" ]; then
            echo "Bootstrapping data from HuggingFace Hub..."
            python3 scripts/bootstrap.py
        else
            echo "ERROR: No indices found and HF_DATASET_REPO not set."
            echo "Run 'index' mode first, or set HF_DATASET_REPO + HF_TOKEN for bootstrap."
            exit 1
        fi
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
