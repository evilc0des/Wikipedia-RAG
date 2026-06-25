from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

_sparse_retriever = None
_dense_retriever = None
_db = None


def _ensure_loaded():
    global _sparse_retriever, _dense_retriever, _db
    if _db is not None:
        return

    from indexing import SparseRetriever, DenseRetriever
    from db import ChunkStoreDB

    sparse_shards_dir = Path("data/sparse_shards")
    sparse_index_path = Path("data/sparse_index.pkl")

    if sparse_shards_dir.exists() and list(sparse_shards_dir.glob("shard_*.pkl")):
        _sparse_retriever = SparseRetriever.load_sharded(str(sparse_shards_dir))
    elif sparse_index_path.exists():
        _sparse_retriever = SparseRetriever.load(str(sparse_index_path))
    else:
        raise HTTPException(
            status_code=503,
            detail="Indices not loaded. Run bootstrap first.",
        )

    _dense_retriever = DenseRetriever.load()
    _db = ChunkStoreDB("data/chunks.db")


app = FastAPI(title="RAG Demo API")


class QueryRequest(BaseModel):
    query: str
    model: str | None = "gemma-4-31B-it"
    temperature: float | None = 0.2


class Citation(BaseModel):
    citation_id: str
    source_id: str | None
    section_id: str | None
    supporting_child_ids: list[str]


class SectionResult(BaseModel):
    chunk_id: str
    score: float
    rerank_score: float
    text: str
    child_ids: list[str]


class QueryResponse(BaseModel):
    answer_text: str | None
    citations: list[Citation]
    grounded: bool
    abstained: bool
    reason: str | None
    sections: list[SectionResult]


@app.get("/health")
async def health():
    try:
        _ensure_loaded()
        return {"status": "ok", "indices_loaded": True}
    except Exception:
        return {"status": "starting", "indices_loaded": False}


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    try:
        _ensure_loaded()
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load indices: {e}")

    from retrieval import hybrid_retrieve_with_rerank
    from generation import build_context_blocks, AnswerGenerator

    result = hybrid_retrieve_with_rerank(
        request.query,
        _sparse_retriever,
        _dense_retriever,
        _db,
    )

    context_blocks = build_context_blocks(result["results"])

    try:
        generator = AnswerGenerator({
            "model": request.model,
            "temperature": request.temperature,
        })
        answer = generator.generate(request.query, context_blocks)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"LLM generation failed: {e}")

    sections = [
        SectionResult(
            chunk_id=r.get("chunk_id", ""),
            score=r.get("score", 0.0),
            rerank_score=r.get("rerank_score", r.get("score", 0.0)),
            text=r.get("text", "")[:500],
            child_ids=r.get("child_ids", []),
        )
        for r in result["results"]
    ]

    citations = [
        Citation(
            citation_id=c.get("citation_id", ""),
            source_id=c.get("source_id"),
            section_id=c.get("section_id"),
            supporting_child_ids=c.get("supporting_child_ids", []),
        )
        for c in answer.get("citations", [])
    ]

    return QueryResponse(
        answer_text=answer.get("answer_text"),
        citations=citations,
        grounded=answer.get("grounded", True),
        abstained=answer.get("abstained", False),
        reason=answer.get("reason"),
        sections=sections,
    )
