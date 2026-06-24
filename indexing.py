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

    def index(self, chunks):
        children = [c for c in chunks if c.get("chunk_type") == "child"]
        self.corpus = [tokenize(c["text"]) for c in children]
        self.bm25 = BM25Okapi(self.corpus)
        self.chunk_store = children

    def search(self, query, top_k=5):
        if not self.bm25:
            return []
        query_tokens = tokenize(query)
        return self.bm25.get_top_n(query_tokens, self.chunk_store, n=top_k)

    def save(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"bm25": self.bm25, "corpus": self.corpus, "chunk_store": self.chunk_store}, f)

    @classmethod
    def load(cls, path):
        with open(path, "rb") as f:
            data = pickle.load(f)
        instance = cls()
        instance.bm25 = data["bm25"]
        instance.corpus = data["corpus"]
        instance.chunk_store = data["chunk_store"]
        return instance


class DenseRetriever:
    def __init__(self, storage_path="data/qdrant", collection_name="dense_index", model_name="BAAI/bge-small-en-v1.5"):
        self.collection_name = collection_name
        self.model = TextEmbedding(model_name=model_name)
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
        embeddings = list(self.model.embed(texts))

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
        if not self.chunk_store:
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
        self.client.close()

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

    return retriever, dense, retriever.chunk_store
