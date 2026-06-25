from collections import defaultdict

from reranking import Reranker, assemble_section_context

_reranker = None


def _get_reranker():
    global _reranker
    if _reranker is None:
        _reranker = Reranker()
    return _reranker


def reciprocal_rank_fusion(rank_lists, k=60):
    fused_scores = defaultdict(float)

    for rank_list in rank_lists:
        for rank, item in enumerate(rank_list, start=1):
            chunk_id = item["chunk_id"]
            fused_scores[chunk_id] += 1.0 / (k + rank)

    return fused_scores

def hybrid_retrieve(
    query_text,
    sparse_retriever,
    dense_retriever,
    db,
    top_k=10,
    sparse_k=30,
    dense_k=30,
    rrf_k=60,
    expand_to_section=False,
):
    sparse_results = sparse_retriever.search(query_text, top_k=sparse_k)
    dense_results = dense_retriever.search(query_text, top_k=dense_k)

    fused_scores = reciprocal_rank_fusion([sparse_results, dense_results], k=rrf_k)

    sparse_rank = {r["chunk_id"]: i + 1 for i, r in enumerate(sparse_results)}
    dense_rank = {r["chunk_id"]: i + 1 for i, r in enumerate(dense_results)}

    sorted_ids = sorted(fused_scores.keys(), key=lambda cid: fused_scores[cid], reverse=True)

    results = []
    seen = set()
    for chunk_id in sorted_ids:
        if chunk_id in seen:
            continue
        seen.add(chunk_id)
        chunk = db.get_chunk(chunk_id)
        if chunk is None:
            continue

        if expand_to_section and chunk.get("chunk_type") == "child":
            section_id = chunk.get("parent_id")
            if section_id:
                section = db.get_chunk(section_id)
                if section and section_id not in seen:
                    seen.add(section_id)
                    result = {
                        **section,
                        "score": fused_scores[chunk_id],
                        "child_ids": [chunk_id],
                        "sparse_rank": sparse_rank.get(chunk_id),
                        "dense_rank": dense_rank.get(chunk_id)
                    }
                    results.append(result)
            else:
                result = {
                    **chunk,
                    "score": fused_scores[chunk_id],
                    "sparse_rank": sparse_rank.get(chunk_id),
                    "dense_rank": dense_rank.get(chunk_id)
                }
                results.append(result)
        else:
            result = {
                **chunk,
                "score": fused_scores[chunk_id],
                "sparse_rank": sparse_rank.get(chunk_id),
                "dense_rank": dense_rank.get(chunk_id)
            }
            results.append(result)

        if len(results) >= top_k:
            break

    return {
        "query": query_text,
        "results": results,
        "sparse_results": sparse_results,
        "dense_results": dense_results,
    }


def hybrid_retrieve_with_rerank(
    query_text,
    sparse_retriever,
    dense_retriever,
    db,
    fusion_top_k=50,
    rerank_top_k=8,
    section_top_k=3,
    sparse_k=50,
    dense_k=50,
    rrf_k=60,
):
    fusion_result = hybrid_retrieve(
        query_text,
        sparse_retriever,
        dense_retriever,
        db,
        top_k=fusion_top_k,
        sparse_k=sparse_k,
        dense_k=dense_k,
        rrf_k=rrf_k,
        expand_to_section=False,
    )

    candidates = fusion_result["results"]
    if not candidates:
        return fusion_result

    reranker = _get_reranker()
    reranked_children = reranker.rerank(query_text, candidates, top_k=rerank_top_k)

    sections = assemble_section_context(reranked_children, db, top_sections=section_top_k)

    return {
        "query": query_text,
        "results": sections,
        "sparse_results": fusion_result["sparse_results"],
        "dense_results": fusion_result["dense_results"],
    }
