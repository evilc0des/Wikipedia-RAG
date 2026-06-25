from pathlib import Path

from datasets import load_dataset

from chunking import create_new_chunk, parse_bullet, parse_section
from db import ChunkStoreDB
from indexing import build_sparse_indexes_from_db, build_dense_index_from_db


def _prev_id_for(items, global_last_id):
    return items[-1]["chunk_id"] if items else global_last_id


DB_PATH = "data/chunks.db"
SPARSE_SHARDS_DIR = "data/sparse_shards"
DENSE_PATH = "data/qdrant"

Path(SPARSE_SHARDS_DIR).mkdir(parents=True, exist_ok=True)

ds = load_dataset("facebook/kilt_wikipedia", split="full", trust_remote_code=True, streaming=True)
db = ChunkStoreDB(DB_PATH)

last_child_id = db.get_last_chunk_id("child")
last_section_id = db.get_last_chunk_id("section")
last_page_id = db.get_last_chunk_id("page")
page_count = db.count_children("page")

if page_count > 0:
    last_doc_id = db.get_last_page_doc_id()
    print(f"Resuming from page {page_count} (last doc_id={last_doc_id}). "
          f"child={last_child_id}, section={last_section_id}, page={last_page_id}")
    print("Skipping already-processed pages...")

skipped = 0
for s in ds:
    if skipped < page_count:
        skipped += 1
        if skipped % 1000 == 0:
            print(f"  Skipped {skipped}/{page_count} pages...")
        continue
    chunks = []
    sections = []
    last_chunk_len = 0

    for idx, para in enumerate(s["text"]["paragraph"]):
        section_title, section_path = parse_section(para)
        bullet = parse_bullet(para) if section_title is None else None

        if section_title is not None:
            if last_chunk_len == 0 and chunks:
                chunks.pop()
            chunk = create_new_chunk(
                "child", section_title,
                doc_id=s["wikipedia_id"],
                section_path=section_path,
                title=s["wikipedia_title"],
                source_url=s["history"]["url"],
                paragraph_start=idx,
                paragraph_end=idx,
                prev_id=_prev_id_for(chunks, last_child_id),
                next_id=None,
                parent_chunk_id=None,
            )
            chunks.append(chunk)
            last_chunk_len = 0

        elif bullet is not None:
            if not chunks:
                chunk = create_new_chunk(
                    "child", bullet,
                    doc_id=s["wikipedia_id"],
                    section_path=[s["wikipedia_title"]],
                    title=s["wikipedia_title"],
                    source_url=s["history"]["url"],
                    paragraph_start=idx,
                    paragraph_end=idx,
                    prev_id=_prev_id_for(chunks, last_child_id),
                    next_id=None,
                    parent_id=None,
                )
                chunks.append(chunk)
            else:
                chunks[-1]["text"] += f"\n{bullet}"
                chunks[-1]["paragraph_end"] = idx
            last_chunk_len += 1

        else:
            text = para.strip()
            if not chunks:
                chunk = create_new_chunk(
                    "child", text,
                    doc_id=s["wikipedia_id"],
                    section_path=[s["wikipedia_title"]],
                    title=s["wikipedia_title"],
                    source_url=s["history"]["url"],
                    paragraph_start=idx,
                    paragraph_end=idx,
                    prev_id=_prev_id_for(chunks, last_child_id),
                    next_id=None,
                    parent_id=None,
                )
                chunks.append(chunk)
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
                section_chunk = create_new_chunk(
                    "section", last_chunk["text"],
                    doc_id=s["wikipedia_id"],
                    section_path=last_chunk["section_path"],
                    title=s["wikipedia_title"],
                    source_url=s["history"]["url"],
                    paragraph_start=idx,
                    paragraph_end=idx,
                    prev_id=_prev_id_for(sections, last_section_id),
                    next_id=None,
                    parent_id=None,
                )
                sections.append(section_chunk)
            else:
                sections[-1]["text"] += f"\n{last_chunk['text']}"
                sections[-1]["paragraph_end"] = idx
            last_chunk["parent_id"] = sections[-1]["chunk_id"]
            sections[-1]["children_ids"].append(last_chunk["chunk_id"])

    page_chunk = create_new_chunk(
        "page", "\n".join(s["text"] for s in sections),
        doc_id=s["wikipedia_id"],
        section_path=[s["wikipedia_title"]],
        title=s["wikipedia_title"],
        source_url=s["history"]["url"],
        paragraph_start=None,
        paragraph_end=None,
        prev_id=last_page_id,
        next_id=None,
        parent_chunk_id=None,
        children_ids=[sc["chunk_id"] for sc in sections],
    )

    if last_page_id:
        db.update_next_id(last_page_id, page_chunk["chunk_id"])

    for c in chunks:
        db.insert_chunk(c)
    for sc in sections:
        db.insert_chunk(sc)
    db.insert_chunk(page_chunk)

    if chunks:
        last_child_id = chunks[-1]["chunk_id"]
    if sections:
        last_section_id = sections[-1]["chunk_id"]
    last_page_id = page_chunk["chunk_id"]

    page_count += 1
    if page_count % 100 == 0:
        db.commit()
        print(f"Page {page_count}: {s['wikipedia_title']} — {len(sections)} sections, {len(chunks)} children, child_count={db.count_children('child')}")

db.commit()
print(f"Chunking complete. {page_count} pages, {db.count_children('child')} children in SQLite.")
db.close()

build_sparse_indexes_from_db(DB_PATH, SPARSE_SHARDS_DIR, shard_size=100000)
print("Sparse (BM25 sharded) indexes built.")

build_dense_index_from_db(DB_PATH, DENSE_PATH, batch_size=1000)
print("Dense (Qdrant) index built.")

print("Indexing complete.")
