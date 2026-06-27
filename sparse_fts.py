"""
Sparse BM25 retriever backed by SQLite FTS5.

Replaces the in-memory rank_bm25 shards with a disk-resident inverted index.
RAM usage: near-zero (SQLite page cache only, ~2 MB default).
"""

import re
import sqlite3
from pathlib import Path


def tokenize(text):
    """Identical to indexing.tokenize — kept local to avoid heavy imports."""
    return re.findall(r'\w+', text.lower())


class SparseFTS5Retriever:
    """BM25 sparse retriever backed by SQLite FTS5 — near-zero RAM usage."""

    def __init__(self, db_path="data/sparse_fts.db"):
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(db_path)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self.conn.execute(
            """CREATE VIRTUAL TABLE IF NOT EXISTS sparse_fts
               USING fts5(text, chunk_id UNINDEXED)"""
        )
        self.conn.commit()

    def build_from_db(self, chunks_db_path, batch_size=10000):
        """Populate FTS5 index from existing chunks.db.

        Reads child chunks, pre-tokenizes text with the same regex tokenizer
        used by the original BM25 shards, and inserts into FTS5.
        Supports resume: skips rows already indexed.
        """
        fts_count = self.conn.execute(
            "SELECT COUNT(*) FROM sparse_fts"
        ).fetchone()[0]

        chunks_conn = sqlite3.connect(chunks_db_path)
        total = chunks_conn.execute(
            "SELECT COUNT(*) FROM chunks WHERE chunk_type = 'child'"
        ).fetchone()[0]

        if fts_count >= total:
            print(f"  FTS5 index already complete ({fts_count}/{total}). Skipping.")
            chunks_conn.close()
            return

        if fts_count > 0:
            print(f"  Resuming FTS5 index from {fts_count}/{total}")

        offset = fts_count
        while offset < total:
            rows = chunks_conn.execute(
                "SELECT chunk_id, text FROM chunks WHERE chunk_type = 'child' "
                "ORDER BY rowid LIMIT ? OFFSET ?",
                (batch_size, offset),
            ).fetchall()
            if not rows:
                break

            # Pre-tokenize with the same \w+ regex so FTS5 tokens match
            # the original rank_bm25 tokenization exactly.
            batch = [
                (" ".join(tokenize(text)), chunk_id)
                for chunk_id, text in rows
            ]
            self.conn.executemany(
                "INSERT INTO sparse_fts(text, chunk_id) VALUES (?, ?)",
                batch,
            )
            self.conn.commit()
            offset += len(rows)
            if (offset // batch_size) % 10 == 0 or offset >= total:
                print(f"  FTS5: indexed {offset}/{total} children")

        chunks_conn.close()
        print(f"  FTS5 index complete: {total} children")

    def search(self, query, top_k=5):
        """Search using BM25 ranking via FTS5.

        Pre-tokenizes the query with the same regex tokenizer and quotes each
        token to avoid FTS5 syntax issues with special characters.

        Returns a list of dicts: [{"chunk_id": str, "score": float}, ...]
        Score is positive (higher = better match).
        """
        tokens = tokenize(query)
        if not tokens:
            return []

        # Quote each token to escape any FTS5 special characters
        match_expr = " ".join(f'"{t}"' for t in tokens)

        try:
            rows = self.conn.execute(
                """SELECT chunk_id, bm25(sparse_fts) AS score
                   FROM sparse_fts
                   WHERE text MATCH ?
                   ORDER BY score
                   LIMIT ?""",
                (match_expr, top_k),
            ).fetchall()
        except sqlite3.OperationalError:
            # Malformed query that still slipped through — return empty
            return []

        # FTS5 bm25() returns negative values (more negative = better match)
        # Negate to produce positive scores for downstream consumers.
        return [{"chunk_id": row[0], "score": -row[1]} for row in rows]

    def count(self):
        """Return the number of indexed documents."""
        return self.conn.execute(
            "SELECT COUNT(*) FROM sparse_fts"
        ).fetchone()[0]

    def close(self):
        """Close the database connection."""
        self.conn.close()

    @classmethod
    def load(cls, db_path="data/sparse_fts.db"):
        """Load an existing FTS5 index. Returns None if the DB doesn't exist."""
        if not Path(db_path).exists():
            return None
        instance = cls(db_path)
        if instance.count() == 0:
            instance.close()
            return None
        return instance
