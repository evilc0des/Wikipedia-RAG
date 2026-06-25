import json
import sqlite3
from pathlib import Path


class ChunkStoreDB:
    def __init__(self, db_path):
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self.conn.execute("PRAGMA cache_size=-64000")
        self.conn.execute(
            """CREATE TABLE IF NOT EXISTS chunks (
                chunk_id        TEXT PRIMARY KEY,
                doc_id          TEXT,
                chunk_type      TEXT NOT NULL,
                text            TEXT NOT NULL,
                section_path    TEXT,
                title           TEXT,
                source_url      TEXT,
                paragraph_start INTEGER,
                paragraph_end   INTEGER,
                prev_id         TEXT,
                next_id         TEXT,
                parent_id       TEXT,
                children_ids    TEXT
            )"""
        )
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_chunks_type ON chunks(chunk_type)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_chunks_parent ON chunks(parent_id)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(doc_id)")
        self.conn.commit()

    def insert_chunk(self, chunk):
        section_path = json.dumps(chunk.get("section_path")) if chunk.get("section_path") is not None else None
        children_ids = json.dumps(chunk.get("children_ids")) if chunk.get("children_ids") else "[]"
        self.conn.execute(
            """INSERT OR REPLACE INTO chunks
               (chunk_id, doc_id, chunk_type, text, section_path, title, source_url,
                paragraph_start, paragraph_end, prev_id, next_id, parent_id, children_ids)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                chunk["chunk_id"],
                chunk.get("doc_id"),
                chunk.get("chunk_type"),
                chunk.get("text"),
                section_path,
                chunk.get("title"),
                chunk.get("source_url"),
                chunk.get("paragraph_start"),
                chunk.get("paragraph_end"),
                chunk.get("prev_id"),
                chunk.get("next_id"),
                chunk.get("parent_id"),
                children_ids,
            ),
        )

    def get_chunk(self, chunk_id):
        row = self.conn.execute(
            "SELECT * FROM chunks WHERE chunk_id = ?", (chunk_id,)
        ).fetchone()
        if row is None:
            return None
        return self._row_to_dict(row)

    def get_children_by_type(self, chunk_type, limit=1000, offset=0):
        rows = self.conn.execute(
            "SELECT * FROM chunks WHERE chunk_type = ? ORDER BY rowid LIMIT ? OFFSET ?",
            (chunk_type, limit, offset),
        ).fetchall()
        return [self._row_to_dict(r) for r in rows]

    def count_children(self, chunk_type=None):
        if chunk_type:
            row = self.conn.execute(
                "SELECT COUNT(*) FROM chunks WHERE chunk_type = ?", (chunk_type,)
            ).fetchone()
        else:
            row = self.conn.execute("SELECT COUNT(*) FROM chunks").fetchone()
        return row[0] if row else 0

    def commit(self):
        self.conn.commit()

    def update_next_id(self, chunk_id, next_id):
        self.conn.execute(
            "UPDATE chunks SET next_id = ? WHERE chunk_id = ?",
            (next_id, chunk_id),
        )

    def update_children_ids(self, chunk_id, children_ids):
        self.conn.execute(
            "UPDATE chunks SET children_ids = ? WHERE chunk_id = ?",
            (json.dumps(children_ids), chunk_id),
        )

    def close(self):
        self.conn.commit()
        self.conn.close()

    def _row_to_dict(self, row):
        keys = [
            "chunk_id", "doc_id", "chunk_type", "text", "section_path",
            "title", "source_url", "paragraph_start", "paragraph_end",
            "prev_id", "next_id", "parent_id", "children_ids",
        ]
        d = dict(zip(keys, row))
        if d.get("section_path"):
            try:
                d["section_path"] = json.loads(d["section_path"])
            except (json.JSONDecodeError, TypeError):
                pass
        if d.get("children_ids"):
            try:
                d["children_ids"] = json.loads(d["children_ids"])
            except (json.JSONDecodeError, TypeError):
                d["children_ids"] = []
        else:
            d["children_ids"] = []
        return d
