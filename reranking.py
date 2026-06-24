from collections import defaultdict

from sentence_transformers import CrossEncoder


class Reranker:
    def __init__(self, model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model = CrossEncoder(model_name)

    def rerank(self, query, candidates, top_k=8):
        if not candidates:
            return []

        pairs = [(query, c["text"]) for c in candidates]
        scores = self.model.predict(pairs)

        for c, score in zip(candidates, scores):
            c["rerank_score"] = float(score)

        candidates.sort(key=lambda c: c["rerank_score"], reverse=True)
        return candidates[:top_k]


def assemble_section_context(child_results, chunk_store, top_sections=3):
    if not child_results:
        return []

    sections = defaultdict(lambda: {"best_score": -999.0, "child_ids": []})

    for child in child_results:
        parent_id = child.get("parent_id")
        if not parent_id:
            continue
        if parent_id not in chunk_store:
            continue

        score = child.get("rerank_score", child.get("score", 0.0))
        entry = sections[parent_id]
        entry["child_ids"].append(child["chunk_id"])
        if score > entry["best_score"]:
            entry["best_score"] = score

    sorted_sections = sorted(sections.items(), key=lambda kv: kv[1]["best_score"], reverse=True)
    sorted_sections = sorted_sections[:top_sections]

    results = []
    for section_id, data in sorted_sections:
        section = chunk_store[section_id]
        results.append({
            **section,
            "score": data["best_score"],
            "child_ids": data["child_ids"],
        })

    return results
