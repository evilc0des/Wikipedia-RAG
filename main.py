from datasets import load_dataset

from chunking import create_new_chunk, parse_bullet, parse_section, _prev_id
from indexing import build_indexes

ds = load_dataset("facebook/kilt_wikipedia", split="full", trust_remote_code=True, streaming=True)

pages = []
chunks = []
sections = []

for s in ds.take(2):
    last_chunk_len = 0

    for idx, para in enumerate(s["text"]["paragraph"]):
        section_title, section_path = parse_section(para)
        bullet = parse_bullet(para) if section_title is None else None

        if section_title is not None:
            if last_chunk_len == 0 and chunks:
                chunks.pop()
            chunks.append(create_new_chunk(
                "child", section_title,
                doc_id=s["wikipedia_id"],
                section_path=section_path,
                title=s["wikipedia_title"],
                source_url=s["history"]["url"],
                paragraph_start=idx,
                paragraph_end=idx,
                prev_id=_prev_id(chunks),
                next_id=None,
                parent_chunk_id=None,
            ))
            last_chunk_len = 0

        elif bullet is not None:
            if not chunks:
                chunks.append(create_new_chunk(
                    "child", bullet,
                    doc_id=s["wikipedia_id"],
                    section_path=[s["wikipedia_title"]],
                    title=s["wikipedia_title"],
                    source_url=s["history"]["url"],
                    paragraph_start=idx,
                    paragraph_end=idx,
                    prev_id=_prev_id(chunks),
                    next_id=None,
                    parent_id=None,
                ))
            else:
                chunks[-1]["text"] += f"\n{bullet}"
                chunks[-1]["paragraph_end"] = idx
            last_chunk_len += 1

        else:
            text = para.strip()
            if not chunks:
                chunks.append(create_new_chunk(
                    "child", text,
                    doc_id=s["wikipedia_id"],
                    section_path=[s["wikipedia_title"]],
                    title=s["wikipedia_title"],
                    source_url=s["history"]["url"],
                    paragraph_start=idx,
                    paragraph_end=idx,
                    prev_id=_prev_id(chunks),
                    next_id=None,
                    parent_id=None,
                ))
            else:
                chunks[-1]["text"] += f"\n{text}"
                chunks[-1]["paragraph_end"] = idx
            last_chunk_len += 1

        last_chunk = chunks[-1] if chunks else None
        last_section = sections[-1] if sections else None
        if last_chunk and last_chunk["section_path"] is not None:
            is_new = (
                last_section is None
                or last_chunk["section_path"][0] != last_section["section_path"][0]
            )
            if is_new:
                sections.append(create_new_chunk(
                    "section", last_chunk["text"],
                    doc_id=s["wikipedia_id"],
                    section_path=last_chunk["section_path"],
                    title=s["wikipedia_title"],
                    source_url=s["history"]["url"],
                    paragraph_start=idx,
                    paragraph_end=idx,
                    prev_id=_prev_id(sections),
                    next_id=None,
                    parent_id=None,
                ))
            else:
                sections[-1]["text"] += f"\n{last_chunk['text']}"
                sections[-1]["paragraph_end"] = idx
            last_chunk["parent_id"] = sections[-1]["chunk_id"]
            sections[-1]["children_ids"].append(last_chunk["chunk_id"])

    pages.append(create_new_chunk(
        "page", "\n".join(s["text"] for s in sections),
        doc_id=s["wikipedia_id"],
        section_path=[s["wikipedia_title"]],
        title=s["wikipedia_title"],
        source_url=s["history"]["url"],
        paragraph_start=None,
        paragraph_end=None,
        prev_id=_prev_id(pages),
        next_id=None,
        parent_chunk_id=None,
        children_ids=[s["chunk_id"] for s in sections],
    ))
    if len(pages) > 1:
        pages[-2]["next_id"] = pages[-1]["chunk_id"]

    print(f"Page {len(pages)}: {s['wikipedia_title']} — {len(sections)} sections, {len(chunks)} chunks")

all_chunks = chunks + sections + pages
sparse_retriever, dense_retriever, chunk_store = build_indexes(
    all_chunks, sparse_path="data/sparse_index.pkl", dense_path="data/qdrant"
)
print(f"BM25 index built:  {len(sparse_retriever.chunk_store)} children indexed")
print(f"Dense index built: {len(dense_retriever.chunk_store)} children indexed")
print(f"Full chunk store:  {len(chunk_store)} chunks (children + sections + pages)")

from retrieval import hybrid_retrieve_with_rerank

result = hybrid_retrieve_with_rerank(
    "What is the pronunciation and etymology of the letter Z?",
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

from generation import AnswerGenerator, build_context_blocks

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
