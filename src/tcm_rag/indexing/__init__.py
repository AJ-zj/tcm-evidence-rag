"""双库持久化：SQLite（分块/记忆/会话等结构化数据） + FAISS（向量索引）。

ChunkStore：SQLite 后端（chunks 表）。构造不传 db_path 时为纯内存模式（单测用）；
传入路径时读写 `data/db/tcm_rag.db`，与 FAISS 向量索引（hnsw.index）构成双库。
"""

import json
import sqlite3
import threading
from pathlib import Path
from typing import Sequence

import numpy as np

from ..schema import Chunk

_SCHEMA = """
CREATE TABLE IF NOT EXISTS chunks (
    chunk_id     TEXT PRIMARY KEY,
    doc_id       TEXT NOT NULL,
    seq          INTEGER NOT NULL,
    title_path   TEXT NOT NULL,
    text         TEXT NOT NULL,
    source_type  TEXT NOT NULL,
    char_start   INTEGER DEFAULT 0,
    char_end     INTEGER DEFAULT 0,
    n_tokens     INTEGER DEFAULT 0,
    ocr_noise    REAL DEFAULT 0,
    split_method TEXT DEFAULT 'semantic',
    meta         TEXT DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(doc_id);
"""


class ChunkStore:
    """分块存储：内存缓存 + SQLite 持久化（双写）。"""

    def __init__(self, db_path: str | Path | None = None):
        self.db_path = Path(db_path) if db_path else None
        self._lock = threading.Lock()
        self._chunks: dict[str, Chunk] = {}
        self._order: list[str] = []
        if self.db_path is not None:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            with self._conn() as conn:
                conn.executescript(_SCHEMA)
            self._load_from_db()

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=30)
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _load_from_db(self) -> None:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT chunk_id, doc_id, seq, title_path, text, source_type, char_start, "
                "char_end, n_tokens, ocr_noise, split_method, meta FROM chunks ORDER BY rowid"
            ).fetchall()
        for r in rows:
            c = Chunk(
                chunk_id=r[0], doc_id=r[1], seq=int(r[2]),
                title_path=json.loads(r[3] or "[]"), text=r[4], source_type=r[5],
                char_start=int(r[6] or 0), char_end=int(r[7] or 0),
                n_tokens=int(r[8] or 0), ocr_noise=float(r[9] or 0.0),
                split_method=r[10] or "semantic", meta=json.loads(r[11] or "{}"),
            )
            self._chunks[c.chunk_id] = c
            self._order.append(c.chunk_id)

    # ------------------------------------------------------------------
    def add(self, chunks: Sequence[Chunk]) -> None:
        with self._lock:
            new_rows = []
            for c in chunks:
                if c.chunk_id not in self._chunks:
                    self._order.append(c.chunk_id)
                self._chunks[c.chunk_id] = c
                new_rows.append(c)
            if self.db_path is not None and new_rows:
                with self._conn() as conn:
                    conn.executemany(
                        "INSERT OR REPLACE INTO chunks(chunk_id, doc_id, seq, title_path, text, "
                        "source_type, char_start, char_end, n_tokens, ocr_noise, split_method, meta) "
                        "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                        [
                            (
                                c.chunk_id, c.doc_id, c.seq, json.dumps(c.title_path, ensure_ascii=False),
                                c.text, c.source_type, c.char_start, c.char_end, c.n_tokens,
                                c.ocr_noise, c.split_method, json.dumps(c.meta, ensure_ascii=False),
                            )
                            for c in new_rows
                        ],
                    )

    def get(self, chunk_id: str) -> Chunk | None:
        return self._chunks.get(chunk_id)

    def all_chunks(self) -> list[Chunk]:
        return [self._chunks[cid] for cid in self._order]

    def by_doc(self, doc_id: str) -> list[Chunk]:
        return [c for c in self.all_chunks() if c.doc_id == doc_id]

    def count_by(self, field: str) -> dict[str, int]:
        out: dict[str, int] = {}
        for c in self.all_chunks():
            key = str(getattr(c, field))
            out[key] = out.get(key, 0) + 1
        return out

    def __len__(self) -> int:
        return len(self._chunks)

    # ------------------------------------------------------------------
    def save(self, path: str | Path | None = None) -> None:
        """兼容旧接口：无 db_path 时把内存态写到指定 JSONL；有 db_path 则已在双写中落库。"""
        target = Path(path) if path else self.db_path
        if target is None:
            raise ValueError("ChunkStore 未配置持久化路径")
        if target.suffix == ".jsonl":
            target.parent.mkdir(parents=True, exist_ok=True)
            with open(target, "w", encoding="utf-8") as f:
                for cid in self._order:
                    f.write(json.dumps(self._chunks[cid].to_dict(), ensure_ascii=False) + "\n")
        # sqlite 模式下 add() 已双写，无需额外动作

    @classmethod
    def load(cls, path: str | Path) -> "ChunkStore":
        """按扩展名分发：.db → SQLite；.jsonl → 旧格式（只读进内存）。"""
        p = Path(path)
        if p.suffix == ".jsonl":
            store = cls()
            with open(p, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        store.add([Chunk.from_dict(json.loads(line))])
            return store
        return cls(db_path=p)


class FaissHNSWIndex:
    """FAISS IndexHNSWFlat 封装：归一化向量 + 内积 = 余弦相似度。"""

    def __init__(self, dim: int, m: int = 32, ef_construction: int = 200, ef_search: int = 128):
        import faiss

        self._faiss = faiss
        self.dim = dim
        self.index = faiss.IndexHNSWFlat(dim, m, faiss.METRIC_INNER_PRODUCT)
        self.index.hnsw.efConstruction = ef_construction
        self.index.hnsw.efSearch = ef_search
        self._ids: list[str] = []          # 行号 → chunk_id
        self._id_to_row: dict[str, int] = {}

    @property
    def ntotal(self) -> int:
        return self.index.ntotal

    def add(self, vectors: np.ndarray, chunk_ids: Sequence[str]) -> None:
        vecs = np.ascontiguousarray(vectors, dtype=np.float32)
        assert vecs.shape[0] == len(chunk_ids)
        base = len(self._ids)
        self._ids.extend(chunk_ids)
        for i, cid in enumerate(chunk_ids):
            self._id_to_row[cid] = base + i
        self.index.add(vecs)

    def search(self, query_vectors: np.ndarray, k: int) -> list[list[tuple[str, float]]]:
        """返回每个查询的 [(chunk_id, score)]，score 为余弦相似度。"""
        if self.index.ntotal == 0:
            return [[] for _ in range(len(query_vectors))]
        k = min(k, self.index.ntotal)
        q = np.ascontiguousarray(query_vectors, dtype=np.float32)
        scores, idxs = self.index.search(q, k)
        results: list[list[tuple[str, float]]] = []
        for row_scores, row_idxs in zip(scores, idxs):
            hits = []
            for s, i in zip(row_scores, row_idxs):
                if i < 0 or i >= len(self._ids):
                    continue
                hits.append((self._ids[i], float(s)))
            results.append(hits)
        return results

    def search_one(self, vector: np.ndarray, k: int) -> list[tuple[str, float]]:
        return self.search(vector.reshape(1, -1), k)[0]

    # ------------------------------------------------------------------
    def save(self, dir_path: str | Path) -> None:
        """序列化保存。faiss 原生 write_index 在 Windows 非 ASCII 路径下会失败，
        因此改用 serialize_index → Python 文件写入。"""
        d = Path(dir_path)
        d.mkdir(parents=True, exist_ok=True)
        blob = self._faiss.serialize_index(self.index)
        with open(d / "hnsw.index", "wb") as f:
            f.write(np.asarray(blob, dtype=np.uint8).tobytes())
        with open(d / "hnsw_ids.json", "w", encoding="utf-8") as f:
            json.dump({"dim": self.dim, "ids": self._ids}, f, ensure_ascii=False)

    @classmethod
    def load(
        cls, dir_path: str | Path, m: int = 32, ef_search: int = 128
    ) -> "FaissHNSWIndex":
        import faiss

        d = Path(dir_path)
        with open(d / "hnsw_ids.json", encoding="utf-8") as f:
            meta = json.load(f)
        with open(d / "hnsw.index", "rb") as f:
            blob = np.frombuffer(f.read(), dtype=np.uint8).copy()
        obj = cls.__new__(cls)
        obj._faiss = faiss
        obj.dim = meta["dim"]
        obj.index = faiss.deserialize_index(blob)
        obj.index.hnsw.efSearch = ef_search
        obj._ids = meta["ids"]
        obj._id_to_row = {cid: i for i, cid in enumerate(obj._ids)}
        return obj
