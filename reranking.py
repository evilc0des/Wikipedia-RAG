from collections import defaultdict

import torch
from sentence_transformers import CrossEncoder


def _get_device():
    if not torch.cuda.is_available():
        return "cpu"
    major, _ = torch.cuda.get_device_capability()
    if major >= 7:
        return "cuda"
    return "cpu"


class Reranker:
    def __init__(self, model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"):
        device = _get_device()
        self.model = CrossEncoder(model_name, device=device)
        print(f"  Reranker loaded on {device}")

    def rerank(self, query, candidates, top_k=8):
        if not candidates:
            return []

        pairs = [(query, c["text"]) for c in candidates]
        scores = self.model.predict(pairs)

        for c, score in zip(candidates, scores):
            if "retrieval_score" not in c:
                c["retrieval_score"] = c.get("score", 0.0)
            c["rerank_score"] = float(score)

        candidates.sort(key=lambda c: c["rerank_score"], reverse=True)
        return candidates[:top_k]


def assemble_section_context(child_results, db, top_sections=3):
    if not child_results:
        return []

    sections = defaultdict(lambda: {
        "best_rerank_score": -999.0,
        "best_retrieval_score": -999.0,
        "child_ids": [],
    })

    for child in child_results:
        parent_id = child.get("parent_id")
        if not parent_id:
            continue
        section = db.get_chunk(parent_id)
        if section is None:
            continue

        rerank_score = child.get("rerank_score", -999.0)
        retrieval_score = child.get("retrieval_score", child.get("score", 0.0))
        entry = sections[parent_id]
        entry["child_ids"].append(child["chunk_id"])
        if rerank_score > entry["best_rerank_score"]:
            entry["best_rerank_score"] = rerank_score
        if retrieval_score > entry["best_retrieval_score"]:
            entry["best_retrieval_score"] = retrieval_score

    sorted_sections = sorted(
        sections.items(),
        key=lambda kv: kv[1]["best_rerank_score"],
        reverse=True,
    )
    sorted_sections = sorted_sections[:top_sections]

    results = []
    for section_id, data in sorted_sections:
        section = db.get_chunk(section_id)
        if section is None:
            continue
        results.append({
            **section,
            "score": data["best_rerank_score"],
            "rerank_score": data["best_rerank_score"],
            "retrieval_score": data["best_retrieval_score"],
            "child_ids": data["child_ids"],
        })

    return results
