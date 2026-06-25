import pickle
import re
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from fastembed import TextEmbedding
from rank_bm25 import BM25Okapi


def tokenize(text):
    return re.findall(r'\w+', text.lower())


class SparseRetriever:
    def __init__(self):
        self.bm25 = None
        self.corpus = []
        self.chunk_store = []
        self.shards = []
        self._is_sharded = False

    def index(self, chunks):
        children = [c for c in chunks if c.get("chunk_type") == "child"]
        self.corpus = [tokenize(c["text"]) for c in children]
        self.bm25 = BM25Okapi(self.corpus)
        self.chunk_store = children
        self._is_sharded = False

    def search(self, query, top_k=5):
        if self._is_sharded:
            return self._search_sharded(query, top_k)
        if not self.bm25:
            return []
        query_tokens = tokenize(query)
        return self.bm25.get_top_n(query_tokens, self.chunk_store, n=top_k)

    def _search_sharded(self, query, top_k=5):
        query_tokens = tokenize(query)
        all_results = []
        for shard in self.shards:
            results = shard["bm25"].get_top_n(
                query_tokens, shard["chunk_store"], n=top_k * 3
            )
            all_results.extend(results)
        all_results.sort(key=lambda r: r.get("score", 0), reverse=True)
        return all_results[:top_k]

    def save(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(
                {"bm25": self.bm25, "corpus": self.corpus, "chunk_store": self.chunk_store},
                f,
            )

    @classmethod
    def load(cls, path):
        with open(path, "rb") as f:
            data = pickle.load(f)
        instance = cls()
        instance.bm25 = data["bm25"]
        instance.corpus = data["corpus"]
        instance.chunk_store = data["chunk_store"]
        return instance

    @classmethod
    def load_sharded(cls, shards_dir):
        instance = cls()
        instance._is_sharded = True
        shard_files = sorted(Path(shards_dir).glob("shard_*.pkl"))
        print(f"Loading {len(shard_files)} BM25 shards from {shards_dir} ...")
        for sf in shard_files:
            with open(sf, "rb") as f:
                data = pickle.load(f)
            instance.shards.append(data)
        return instance


class DenseRetriever:
    def __init__(self, storage_path="data/qdrant", collection_name="dense_index", model_name="BAAI/bge-small-en-v1.5"):
        self.collection_name = collection_name
        try:
            import onnxruntime
            available = onnxruntime.get_available_providers()
        except Exception:
            available = []
        gpu_providers = {"CUDAExecutionProvider", "DmlExecutionProvider", "TensorrtExecutionProvider"}
        has_gpu = bool(gpu_providers & set(available))
        if "CUDAExecutionProvider" in available:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        elif "DmlExecutionProvider" in available:
            providers = ["DmlExecutionProvider", "CPUExecutionProvider"]
        else:
            providers = ["CPUExecutionProvider"]
            if not has_gpu:
                import sys
                print(
                    "  WARNING: No GPU provider available (CPU only). Install one of:\n"
                    "    pip install onnxruntime-gpu        # CUDA (production)\n"
                    "    pip install onnxruntime-directml   # DirectML (Windows dev)",
                    file=sys.stderr,
                )
        self.model = TextEmbedding(
            model_name=model_name,
            providers=providers,
        )
        self.client = QdrantClient(path=storage_path)
        self.chunk_store = []

        if not self.client.collection_exists(collection_name):
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=384,
                    distance=Distance.COSINE,
                ),
            )

    def index(self, chunks):
        children = [c for c in chunks if c.get("chunk_type") == "child"]
        if not children:
            self.chunk_store = []
            return

        self.chunk_store = children

        if self.client.collection_exists(self.collection_name):
            self.client.delete_collection(self.collection_name)
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(
                size=384,
                distance=Distance.COSINE,
            ),
        )

        texts = [c["text"] for c in children]
        embeddings = _embed_batch(self, texts)

        points = [
            PointStruct(
                id=chunk["chunk_id"],
                vector=emb.tolist(),
                payload={
                    "chunk_id": chunk["chunk_id"],
                    "doc_id": chunk.get("doc_id"),
                    "chunk_type": chunk.get("chunk_type"),
                    "text": chunk["text"],
                    "section_path": chunk.get("section_path"),
                    "title": chunk.get("title"),
                    "source_url": chunk.get("source_url"),
                    "paragraph_start": chunk.get("paragraph_start"),
                    "paragraph_end": chunk.get("paragraph_end"),
                    "prev_id": chunk.get("prev_id"),
                    "next_id": chunk.get("next_id"),
                    "parent_id": chunk.get("parent_id"),
                    "children_ids": chunk.get("children_ids", []),
                },
            )
            for chunk, emb in zip(children, embeddings)
        ]
        self.client.upsert(collection_name=self.collection_name, points=points)

    def search(self, query, top_k=5):
        if not self.chunk_store and not self.client.collection_exists(self.collection_name):
            return []

        query_embedding = list(self.model.embed([query]))[0]
        results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_embedding.tolist(),
            limit=top_k,
        )

        hits = []
        for point in results.points:
            chunk = dict(point.payload)
            chunk["score"] = point.score
            hits.append(chunk)
        return hits

    def save(self):
        pass

    @classmethod
    def load(cls, storage_path="data/qdrant", collection_name="dense_index", model_name="BAAI/bge-small-en-v1.5"):
        instance = cls(storage_path=storage_path, collection_name=collection_name, model_name=model_name)
        if instance.client.collection_exists(collection_name):
            count = instance.client.count(collection_name=collection_name)
            if count.count > len(instance.chunk_store):
                instance.chunk_store = list(range(count.count))
        return instance


def build_indexes(chunks, sparse_path=None, dense_path="data/qdrant"):
    retriever = SparseRetriever()
    retriever.index(chunks)
    if sparse_path:
        retriever.save(sparse_path)

    dense = DenseRetriever(storage_path=dense_path)
    dense.index(chunks)
    if sparse_path:
        dense.save()

    chunk_store = {c["chunk_id"]: c for c in chunks}
    return retriever, dense, chunk_store


def build_sparse_indexes_from_db(db_path, output_dir, shard_size=100000):
    from db import ChunkStoreDB
    db = ChunkStoreDB(db_path)
    total = db.count_children("child")

    output_path = Path(output_dir)
    existing_shards = sorted(output_path.glob("shard_*.pkl"))
    if existing_shards:
        last_shard_file = existing_shards[-1]
        shard_id = int(last_shard_file.stem.split("_")[1])
        start_offset = shard_id * shard_size
        print(f"  Resuming sparse index from shard {shard_id} (offset={start_offset}/{total})")
    else:
        shard_id = 0
        start_offset = 0

    for offset in range(start_offset, total, shard_size):
        batch = db.get_children_by_type("child", limit=shard_size, offset=offset)
        if not batch:
            break
        corpus = [tokenize(c["text"]) for c in batch]
        bm25 = BM25Okapi(corpus)
        shard_path = output_path / f"shard_{shard_id:04d}.pkl"
        with open(shard_path, "wb") as f:
            pickle.dump({"bm25": bm25, "chunk_store": batch}, f)
        print(f"  BM25 shard {shard_id:04d}: {len(batch)} children -> {shard_path}")
        shard_id += 1

    db.close()
    print(f"  Built {shard_id} BM25 shards total.")


def _is_dml_active(dense):
    try:
        import onnxruntime
        return "DmlExecutionProvider" in onnxruntime.get_available_providers()
    except Exception:
        return False


_DML_SUB_BATCH = 16


def _embed_batch(dense, texts):
    if not _is_dml_active(dense):
        return list(dense.model.embed(texts))
    all_embeddings = []
    for i in range(0, len(texts), _DML_SUB_BATCH):
        chunk = texts[i:i + _DML_SUB_BATCH]
        all_embeddings.extend(dense.model.embed(chunk))
    return all_embeddings


def build_dense_index_from_db(db_path, dense_path, batch_size=1000):
    from db import ChunkStoreDB
    db = ChunkStoreDB(db_path)
    total = db.count_children("child")

    dense = DenseRetriever(storage_path=dense_path)

    collection_exists = dense.client.collection_exists(dense.collection_name)
    existing_count = 0
    if collection_exists:
        existing_count = dense.client.count(collection_name=dense.collection_name).count
        if existing_count == total:
            print(f"  Dense index already complete ({total}/{total} children). Skipping.")
            db.close()
            return
        if existing_count > 0:
            print(f"  Resuming dense index from offset {existing_count}/{total}")

    if not collection_exists:
        dense.client.create_collection(
            collection_name=dense.collection_name,
            vectors_config=VectorParams(
                size=384,
                distance=Distance.COSINE,
            ),
        )

    for offset in range(existing_count, total, batch_size):
        batch = db.get_children_by_type("child", limit=batch_size, offset=offset)
        if not batch:
            break
        texts = [c["text"] for c in batch]
        embeddings = _embed_batch(dense, texts)

        points = [
            PointStruct(
                id=chunk["chunk_id"],
                vector=emb.tolist(),
                payload={
                    "chunk_id": chunk["chunk_id"],
                    "doc_id": chunk.get("doc_id"),
                    "chunk_type": chunk.get("chunk_type"),
                    "text": chunk["text"],
                    "section_path": chunk.get("section_path"),
                    "title": chunk.get("title"),
                    "source_url": chunk.get("source_url"),
                    "paragraph_start": chunk.get("paragraph_start"),
                    "paragraph_end": chunk.get("paragraph_end"),
                    "prev_id": chunk.get("prev_id"),
                    "next_id": chunk.get("next_id"),
                    "parent_id": chunk.get("parent_id"),
                    "children_ids": chunk.get("children_ids", []),
                },
            )
            for chunk, emb in zip(batch, embeddings)
        ]
        dense.client.upsert(collection_name=dense.collection_name, points=points)

        if (offset // batch_size) % 10 == 0:
            print(f"  Dense: embedded {offset + len(batch)}/{total} children")

    db.close()
    print(f"  Dense: embedded {total}/{total} children")
