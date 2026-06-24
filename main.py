import pickle
import sys

from indexing import SparseRetriever, DenseRetriever
from retrieval import hybrid_retrieve_with_rerank
from generation import AnswerGenerator, build_context_blocks

sparse_retriever = SparseRetriever.load("data/sparse_index.pkl")
dense_retriever = DenseRetriever.load()
print(f"Sparse index loaded: {len(sparse_retriever.chunk_store)} children")
print(f"Dense index loaded: {len(dense_retriever.chunk_store)} children")

with open("data/chunk_store.pkl", "rb") as f:
    chunk_store = pickle.load(f)
print(f"Chunk store loaded: {len(chunk_store)} entries")

query = sys.argv[1] if len(sys.argv) > 1 else "What is the pronunciation and etymology of the letter Z?"

result = hybrid_retrieve_with_rerank(
    query,
    sparse_retriever,
    dense_retriever,
    chunk_store,
)

print(f"Query: {result['query']}")
print(f"sparse results: {len(result['sparse_results'])}, dense results: {len(result['dense_results'])}")
print(f"sections returned: {len(result['results'])}")
for i, r in enumerate(result["results"]):
    child_ids_str = ", ".join(r.get("child_ids", []))
    print(f"\n--- Section {i+1} (score={r['score']:.4f}, type={r['chunk_type']}) ---")
    print(f"Child IDs: [{child_ids_str}]")
    print(r["text"][:300])

print("\n=== Generated Answer ===")

context_blocks = build_context_blocks(result["results"])
generator = AnswerGenerator({"model": "gemma-4-31B-it", "temperature": 0.2})
answer = generator.generate(result["query"], context_blocks)

print(f"Grounded:  {answer['grounded']}")
print(f"Abstained: {answer['abstained']}")
if answer["reason"]:
    print(f"Reason:    {answer['reason']}")
print(f"\nAnswer:\n{answer['answer_text']}")
print(f"\nCitations ({len(answer['citations'])}):")
for c in answer["citations"]:
    kids = ", ".join(c.get("supporting_child_ids", []))
    print(f"  {c['citation_id']}  source={c['source_id']}  section={c['section_id']}  children=[{kids}]")

dense_retriever.client.close()
