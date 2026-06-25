# RAG Pipeline Demo

A Retrieval-Augmented Generation (RAG) system built over Wikipedia (KILT dataset) with hybrid retrieval, cross-encoder reranking, and citation-grounded answer generation.

## Architecture

```
                    QUERY
                      │
      ┌───────────────┼───────────────┐
      ▼                               ▼
 Sparse Retrieval                Dense Retrieval
 (BM25 over shards)              (bge-small + Qdrant)
      │                               │
      └───────────┬───────────────────┘
                  ▼
           RRF Fusion
         (Reciprocal Rank Fusion)
                  │
                  ▼
          Cross-Encoder Rerank
            (ms-marco-MiniLM)
                  │
                  ▼
          Section Assembly
          (DB parent lookups)
                  │
                  ▼
          LLM Generation
    (OpenAI-compatible API)
                  │
                  ▼
         Cited Answer Output
```

### Pipeline Stages

| # | Stage | Module | Description |
|---|-------|--------|-------------|
| 1 | Sparse Retrieval | `indexing.py` | BM25 keyword search over tokenized child chunks, sharded for scale |
| 2 | Dense Retrieval | `indexing.py` | Semantic search via `bge-small-en-v1.5` embeddings stored in Qdrant |
| 3 | RRF Fusion | `retrieval.py` | Reciprocal Rank Fusion combining sparse + dense rankings |
| 4 | Reranking | `reranking.py` | Cross-encoder (`ms-marco-MiniLM-L-6-v2`) re-scores fusion candidates |
| 5 | Section Assembly | `reranking.py` | Groups child chunks under their parent sections via DB lookups |
| 6 | Generation | `generation.py` | OpenAI-compatible LLM call with citation-grounded answer generation |

### Data Model

Documents are chunked into a 3-level hierarchy:

- **Page** — a full Wikipedia article
- **Section** — a top-level section within a page (e.g., "Etymology")
- **Child** — a text block starting at a section boundary, accumulating paragraphs until the next section

Chunks are stored in a SQLite database (`data/chunks.db`) with parent/child/prev/next ID chains for navigation.

### Qdrant Modes

Qdrant runs in one of two modes, controlled by the `QDRANT_URL` env var:

| Mode | Config | Qdrant location |
|------|--------|-----------------|
| **Remote** (Docker) | `QDRANT_URL=http://localhost:6333` | Docker container via `docker compose up -d` |
| **Local** (disk) | `QDRANT_URL` unset | Embedded local storage at `data/qdrant/` |

Remote mode is recommended for deployment. Local mode is useful for development without Docker.

## Files

| File | Role |
|------|------|
| `main.py` | Entry point: loads indices, runs query, prints cited answer |
| `chunking.py` | Wikipedia paragraph parsing, section/bullet detection, chunk factory |
| `index_data.py` | Dataset streaming, chunking pipeline, index building from DB |
| `db.py` | SQLite chunk store with CRUD, indexing, and lookup methods |
| `indexing.py` | `SparseRetriever` (BM25), `DenseRetriever` (Qdrant), snapshot helpers |
| `retrieval.py` | Hybrid retrieval with RRF fusion and rerank orchestration |
| `reranking.py` | Cross-encoder reranker and section context assembly |
| `generation.py` | OpenAI-compatible LLM client with citation validation |
| `benchmark.py` | Layer-by-layer benchmarking and quality evaluation |
| `docker-compose.yml` | Docker Compose config for Qdrant container |
| `scripts/package_data.py` | Export index data and upload to HuggingFace Hub |
| `scripts/bootstrap.py` | Download data from HuggingFace Hub and import into Qdrant |
| `requirements.txt` | Python dependencies |

## Setup

### Prerequisites

- Python 3.10+
- ~120 GB free disk space (for the Wikipedia KILT dataset index)
- (Optional) OpenAI-compatible API endpoint for generation
- (Optional) Docker + Docker Compose for remote Qdrant mode

### Installation

```bash
python -m venv .venv
.venv\Scripts\activate   # Windows
# source .venv/bin/activate  # Linux/macOS

pip install -r requirements.txt
```

### Environment

Copy `.env.example` to `.env` and configure:

```env
OPENAI_API_KEY=sk-your-key-here
OPENAI_BASE_URL=https://api.openai.com/v1
QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=
HF_DATASET_REPO=your-org/rag-demo-data
HF_TOKEN=hf_your_token_here
```

- `QDRANT_URL` — set to connect to Docker Qdrant; leave unset for local disk mode
- `QDRANT_API_KEY` — only needed for Qdrant Cloud or gated on-prem instances
- `HF_DATASET_REPO` — HuggingFace dataset repo ID for data packaging/bootstrap
- `HF_TOKEN` — HuggingFace API token (write access for packaging, read access for bootstrap)

The `OPENAI_BASE_URL` can point to any OpenAI-compatible endpoint (vLLM, Ollama, LiteLLM, etc.).

### Docker Qdrant (remote mode)

```bash
docker compose up -d
```

Qdrant will be available at `http://localhost:6333` (HTTP) and `localhost:6334` (gRPC). Data persists in a named Docker volume.

## Usage

### 1. Index the Dataset

Downloads and processes Wikipedia articles from the KILT dataset, building all indices:

```bash
# Fresh build against Docker Qdrant
export QDRANT_URL=http://localhost:6333
python index_data.py --pages 2000

# Rebuild from scratch (deletes all existing data)
python index_data.py --rebuild --pages 2000

# Local disk mode (no Docker required)
python index_data.py --pages 2000
```

This performs three steps sequentially:
1. **Chunking** — streams Wikipedia pages, applies section/paragraph parsing, inserts into SQLite (`data/chunks.db`)
2. **Sparse indexing** — reads children from DB, builds sharded BM25 indices (`data/sparse_shards/`)
3. **Dense indexing** — reads children from DB, embeds with `bge-small-en-v1.5`, stores in Qdrant (remote or local)

The indexing is incremental by default — it resumes from where it left off. Use `--rebuild` to delete existing data and start fresh.

Options:

| Flag | Default | Description |
|------|---------|-------------|
| `--pages` | `2000` | Number of Wikipedia pages to process |
| `--workers` | CPU count | Number of parallel chunking workers |
| `--rebuild` | `false` | Delete existing data and re-index from scratch |

### 2. Query the System

```bash
python main.py "What is the pronunciation and etymology of the letter Z?"
```

Or without arguments (uses a default query):

```bash
python main.py
```

Output includes:
- Sparse/dense result counts
- Retrieved sections with scores
- The generated answer with inline citations
- Citation metadata with source and supporting child IDs

### 3. Run Benchmarks

```bash
# All layers
python benchmark.py --layers all

# Specific layers
python benchmark.py --layers sparse,dense,rrf,rerank,combined

# Batch layers only (chunking + indexing throughput)
python benchmark.py --layers chunking,sparse_indexing,dense_indexing --query-source-pages 2000

# Quick retrieval test
python benchmark.py --layers sparse,dense --num-queries 100

# Benchmark against remote Qdrant
python benchmark.py --layers dense --qdrant-url http://localhost:6333
```

## Data Packaging & Deployment

Pre-built index data can be packaged and distributed via HuggingFace Hub so deployments don't need to re-run the full indexing pipeline.

### Package (one-time, on build machine)

After running `index_data.py` against Docker Qdrant:

```bash
export HF_TOKEN=hf_your_token_here
export HF_DATASET_REPO=your-org/rag-demo-data
python scripts/package_data.py
```

This creates a Qdrant snapshot, downloads it, and uploads to your HF dataset repo:
- `chunks.db` — SQLite chunk store
- `sparse_shards/*.pkl` — sharded BM25 indices
- `dense_index.snapshot` — Qdrant collection snapshot

### Bootstrap (per deployment)

On each deployment machine, after starting Docker Qdrant:

```bash
# Start Qdrant
docker compose up -d

# Download data and import snapshot
python scripts/bootstrap.py
```

The bootstrap script is idempotent — it skips download if data files already exist, and skips snapshot import if the Qdrant collection already has points. Use `--force` to re-download and re-import.

## Benchmark Script

`benchmark.py` measures every layer of the pipeline individually and in combination.

### Layers

| CLI name | Type | What's measured |
|----------|------|-----------------|
| `sparse` | Query | BM25 search latency, QPS, retrieval quality |
| `dense` | Query | Embedding + Qdrant search latency, QPS, retrieval quality |
| `rrf` | Query | Reciprocal Rank Fusion latency, QPS |
| `rerank` | Query | Cross-encoder reranking latency, QPS, retrieval quality |
| `section_assembly` | Query | DB parent-lookup latency, QPS |
| `combined` | Query | Full retrieval pipeline latency, QPS, retrieval quality |
| `generation` | Query | LLM API call latency, tokens/sec, groundedness ratio |
| `generation_standalone` | Query | LLM call without retrieval overhead |
| `chunking` | Batch | Pages/sec, paragraphs/sec, chunks/sec throughput |
| `sparse_indexing` | Batch | BM25 shard building throughput (children/sec) |
| `dense_indexing` | Batch | Embedding + Qdrant upsert throughput (children/sec) |

### Metrics

**Per-layer (query):**
- Latency percentiles: min, max, mean, p50, p95, p99 (ms)
- Throughput: queries per second (QPS)
- Quality: Recall@k, Precision@k, MRR, NDCG@k (k=3,5,10)

**Generation-specific:**
- Average output tokens per query
- Output tokens per second
- Groundedness ratio (fraction of answers with valid citations)

**Per-layer (batch):**
- Total processing time
- Throughput (pages/sec, children/sec)

### Query Dataset

Queries are generated from Wikipedia section titles (e.g., "Etymology", "History") by re-streaming the KILT dataset and matching sections to the indexed DB. Ground truth is the set of child chunk IDs belonging to each section.

- Queries are cached to `data/benchmark_queries.json` for reuse
- Configurable sample size via `--num-queries` and `--query-source-pages`

### CLI Reference

```
python benchmark.py [OPTIONS]
```

| Flag | Default | Description |
|------|---------|-------------|
| `--layers` | `all` | Comma-separated layer names or `all` |
| `--num-queries` | `1000` | Number of queries to benchmark |
| `--query-source-pages` | `20000` | Pages streamed from dataset for query generation and chunking/indexing benchmarks |
| `--sparse-shard-size` | `50000` | Children per BM25 shard |
| `--dense-batch-size` | `1000` | Children per embedding batch |
| `--warmup` | `10` | Warmup queries before timing |
| `--gen-queries` | `50` | Queries for generation benchmark (reduced to control API cost) |
| `--gen-model` | `gemma-4-31B-it` | Model name for generation |
| `--gen-temperature` | `0.2` | LLM temperature |
| `--output` | auto | JSON results path (timestamped if omitted) |
| `--db-path` | `data/chunks.db` | Path to chunk SQLite DB |
| `--sparse-shards-dir` | `data/sparse_shards` | Path to BM25 shards |
| `--dense-path` | `data/qdrant` | Path to Qdrant storage (local mode) |
| `--qdrant-url` | `QDRANT_URL` env | Qdrant server URL (remote mode) |
| `--qdrant-api-key` | `QDRANT_API_KEY` env | Qdrant API key (optional) |
| `--keep-temp` | `false` | Keep temporary benchmark indices/DBs |
| `--seed` | `42` | Random seed for query sampling |

### Example Output

```
===============================================================================================
BENCHMARK RESULTS SUMMARY
===============================================================================================

Layer                         mean      p50      p95      p99      QPS  Quality/Tokens
-----------------------------------------------------------------------------------------------
sparse                         0.1      0.0      0.2      0.2  16971.6  mrr=0.000  recall@10=0.000
dense                         12.0      9.0     26.4     29.2     83.5  mrr=0.000  recall@10=0.000
rrf                            0.7      0.5      1.4      2.6   1463.2
rerank                         3.2      3.0      5.1      7.8    312.5  mrr=0.452  recall@10=0.380
section_assembly               0.3      0.2      0.6      1.1   3333.3
combined_retrieval            16.3     12.5     33.1     42.3     61.3  mrr=0.452  recall@10=0.380
===============================================================================================
```

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `qdrant-client` | latest | Vector database for dense retrieval |
| `fastembed` | latest | Embedding model inference (bge-small-en-v1.5) |
| `datasets` | latest | HuggingFace dataset streaming (KILT Wikipedia) |
| `huggingface_hub` | latest | HuggingFace Hub upload/download for data packaging |
| `rank-bm25` | latest | BM25 sparse retrieval |
| `sentence-transformers` | latest | Cross-encoder reranking model |
| `requests` | latest | LLM API calls and Qdrant snapshot API |
| `python-dotenv` | latest | Environment variable loading |

## Notes

- **First run**: `index_data.py` downloads the KILT Wikipedia dataset (~35 GB compressed) and builds ~120 GB of indices. Ensure sufficient disk space.
- **Generation model**: The default model is `gemma-4-31B-it` — change via `.env` or `--gen-model` if using a different endpoint.
- **Sparse shards**: For accurate BM25 benchmarks, ensure sharded sparse indices are built. The `sparse_index.pkl` file is a single-index artifact from early testing and covers only a tiny fraction of the corpus.
- **Benchmark queries**: Section titles are short and generic by design — expect lower quality scores than if using full-sentence queries. This is a realistic test of retrieval robustness.
- **Docker Qdrant**: Data persists in a named Docker volume (`qdrant_storage`). To reset, run `docker compose down -v` before `docker compose up -d`.
