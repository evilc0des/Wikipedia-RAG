import argparse
import os
from pathlib import Path

from chunking import process_pages
from db import ChunkStoreDB


DB_PATH = "data/chunks.db"
SPARSE_SHARDS_DIR = "data/sparse_shards"
DENSE_PATH = "data/qdrant"


def _index_progress(event, *args):
    if event == "parallel":
        n_pages, w, chunksize = args
        print(f"Processing {n_pages} pages with {w} workers (chunksize={chunksize})...", flush=True)
    elif event == "sequential":
        n_pages, _, _ = args
        print(f"Processing {n_pages} pages sequentially...", flush=True)
    elif event == "page":
        page_count, n_children, n_sections = args
        print(f"Page {page_count}: {n_sections} sections, {n_children} children", flush=True)


def main():
    from datasets import load_dataset
    from indexing import build_sparse_indexes_from_db, build_dense_index_from_db

    qdrant_url = os.environ.get("QDRANT_URL")
    qdrant_api_key = os.environ.get("QDRANT_API_KEY") or None

    parser = argparse.ArgumentParser(description="Ingest and index Wikipedia pages for RAG")
    parser.add_argument("--pages", type=int, default=2000, help="Number of pages to process (default: 2000)")
    parser.add_argument("--workers", type=int, default=os.cpu_count(),
                        help=f"Number of parallel chunking workers (default: {os.cpu_count()})")
    args = parser.parse_args()

    max_pages = args.pages
    workers = args.workers

    Path(SPARSE_SHARDS_DIR).mkdir(parents=True, exist_ok=True)

    ds = load_dataset("facebook/kilt_wikipedia", split="full", streaming=True)
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

    pages = []
    skipped = 0
    for s in ds.take(max_pages):
        if skipped < page_count:
            skipped += 1
            if skipped % 1000 == 0:
                print(f"  Skipped {skipped}/{page_count} pages...")
            continue
        pages.append(s)

    if not pages:
        print("No new pages to process.")
    else:
        last_child_id, last_section_id, last_page_id, page_count = process_pages(
            pages, db,
            last_child_id=last_child_id,
            last_section_id=last_section_id,
            last_page_id=last_page_id,
            page_count=page_count,
            workers=workers,
            progress=_index_progress,
        )
        db.commit()
        print(f"Chunking complete. {page_count} pages, {db.count_children('child')} children in SQLite.")

    db.close()

    build_sparse_indexes_from_db(DB_PATH, SPARSE_SHARDS_DIR, shard_size=100000)
    print("Sparse (BM25 sharded) indexes built.")

    build_dense_index_from_db(DB_PATH, DENSE_PATH, batch_size=1000,
                              qdrant_url=qdrant_url, qdrant_api_key=qdrant_api_key)
    print("Dense (Qdrant) index built.")

    print("Indexing complete.")


if __name__ == '__main__':
    main()
