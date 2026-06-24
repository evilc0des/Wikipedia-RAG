from collections import defaultdict


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
    chunk_store,
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
        chunk = chunk_store.get(chunk_id)
        if chunk is None:
            continue

        if expand_to_section and chunk.get("chunk_type") == "child":
            section_id = chunk.get("parent_id")
            if section_id:
                section = chunk_store.get(section_id)
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