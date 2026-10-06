"""长期记忆：BGE 相似度 + importance 阈值筛选入库，NLI 冲突检测控制写入。

持久化：SQLite（memories 表，含 embedding BLOB），与 ChunkStore/会话同库
（data/db/tcm_rag.db），构成"SQLite + FAISS"双库设计。构造传入 db_path 时
每条写入即时 upsert 落库；不传则纯内存（单测），save() 兼容 .db/.jsonl 两种后缀。

写入决策流（write）：
1. importance = NER 医疗实体加权重要度；低于阈值 → 拒绝入库（低价值信息）
2. 与现有 active 记录做 BGE 余弦相似度检索：
   - sim ≥ merge_similarity → 判定为同一事实的重复/更新 → 合并（hits+1，
     importance 取较大者，content 取较新较长者），不新增记录
   - 0.45 ≤ sim < merge_similarity → 进入 NLI 冲突检测：
     * contradiction → 按策略消解（newer_wins：新记录 active、旧记录 superseded，
       双向登记冲突原因；importance_wins：重要度高者存活）
     * entailment/neutral → 正常插入
3. 其余 → 直接插入

召回（recall）：相似度 × (0.75 + 0.25×importance) × 时间新近加成，返回 top-k。
"""
from __future__ import annotations

import json
import sqlite3
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ..embedding import Embedder
from ..ner.medical_ner import MedicalNER
from ..nli.conflict import RuleNLI

CONFLICT_CANDIDATE_SIM = 0.45

_MEM_SCHEMA = """
CREATE TABLE IF NOT EXISTS memories (
    id         TEXT PRIMARY KEY,
    content    TEXT NOT NULL,
    importance REAL NOT NULL,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    hits       INTEGER DEFAULT 0,
    status     TEXT DEFAULT 'active',
    source     TEXT DEFAULT '',
    conflicts  TEXT DEFAULT '[]',
    embedding  BLOB
);
"""


@dataclass
class MemoryRecord:
    id: str
    content: str
    importance: float
    created_at: float
    updated_at: float
    hits: int = 0
    status: str = "active"          # active | superseded
    source: str = ""                # 来源（会话/轮次）
    conflicts: list[dict] = field(default_factory=list)
    embedding: list[float] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id, "content": self.content, "importance": self.importance,
            "created_at": self.created_at, "updated_at": self.updated_at,
            "hits": self.hits, "status": self.status, "source": self.source,
            "conflicts": self.conflicts, "embedding": self.embedding,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "MemoryRecord":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class WriteResult:
    action: str                     # inserted | merged | rejected_low_importance | conflict_superseded | conflict_rejected
    record_id: str = ""
    detail: str = ""
    importance: float = 0.0
    similarity: float = 0.0


class LongTermMemory:
    def __init__(
        self,
        ner: MedicalNER,
        embedder: Embedder,
        nli: RuleNLI | None = None,
        merge_similarity: float = 0.88,
        write_importance_threshold: float = 0.50,
        recall_top_k: int = 3,
        conflict_policy: str = "newer_wins",
        conflict_candidate_sim: float = CONFLICT_CANDIDATE_SIM,
        db_path: str | Path | None = None,
    ):
        self.ner = ner
        self.embedder = embedder
        self.nli = nli or RuleNLI(ner)
        self.merge_similarity = merge_similarity
        self.write_threshold = write_importance_threshold
        self.recall_top_k = recall_top_k
        self.conflict_policy = conflict_policy
        self.conflict_candidate_sim = conflict_candidate_sim
        self.db_path = Path(db_path) if db_path else None
        if self.db_path is not None:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            with sqlite3.connect(str(self.db_path)) as conn:
                conn.executescript(_MEM_SCHEMA)

        self.records: dict[str, MemoryRecord] = {}
        self.stats = {
            "writes_attempted": 0, "inserted": 0, "merged": 0,
            "rejected_low_importance": 0, "conflicts_detected": 0,
            "superseded": 0, "conflict_rejected": 0,
        }

    # ------------------------------------------------------------------
    def _active(self) -> list[MemoryRecord]:
        return [r for r in self.records.values() if r.status == "active"]

    def _similarity_search(self, vec: np.ndarray, k: int = 5) -> list[tuple[MemoryRecord, float]]:
        active = self._active()
        if not active:
            return []
        mat = np.stack([np.asarray(r.embedding, dtype=np.float32) for r in active])
        sims = mat @ vec
        order = np.argsort(-sims)[:k]
        return [(active[i], float(sims[i])) for i in order]

    # ------------------------------------------------------------------
    def write(self, content: str, source: str = "") -> WriteResult:
        content = content.strip()
        if not content:
            return WriteResult(action="rejected_low_importance", detail="empty_content")
        self.stats["writes_attempted"] += 1

        importance = self.ner.importance(content)
        if importance < self.write_threshold:
            self.stats["rejected_low_importance"] += 1
            return WriteResult(
                action="rejected_low_importance",
                detail=f"importance={importance:.2f} < {self.write_threshold}",
                importance=importance,
            )

        vec = self.embedder.encode_documents([content])[0]
        neighbors = self._similarity_search(vec, k=5)

        # 1) 近重复 → 合并
        if neighbors and neighbors[0][1] >= self.merge_similarity:
            rec, sim = neighbors[0]
            rec.hits += 1
            rec.updated_at = time.time()
            rec.importance = max(rec.importance, importance)
            if len(content) > len(rec.content):
                rec.content = content
            rec.embedding = vec.tolist()
            self._persist_record(rec)
            self.stats["merged"] += 1
            return WriteResult(
                action="merged", record_id=rec.id,
                detail=f"sim={sim:.3f} ≥ merge_threshold={self.merge_similarity}",
                importance=importance, similarity=sim,
            )

        # 2) 相似但不同 → NLI 冲突检测
        for rec, sim in neighbors:
            if sim < self.conflict_candidate_sim:
                continue
            result = self.nli.detect_conflict(content, rec.content)
            if result.is_contradiction:
                self.stats["conflicts_detected"] += 1
                survive_new = (
                    self.conflict_policy == "newer_wins"
                    or (self.conflict_policy == "importance_wins" and importance >= rec.importance)
                )
                conflict_info = {
                    "with": rec.id, "reason": result.reason,
                    "score": result.score, "at": time.time(),
                }
                if survive_new:
                    rec.status = "superseded"
                    rec.conflicts.append({**conflict_info, "resolution": "superseded_by_new"})
                    self._persist_record(rec)
                    self.stats["superseded"] += 1
                    new_rec = self._insert(content, importance, vec, source)
                    new_rec.conflicts.append({**conflict_info, "resolution": "supersedes_old"})
                    self._persist_record(new_rec)
                    return WriteResult(
                        action="conflict_superseded", record_id=new_rec.id,
                        detail=f"与 {rec.id} 矛盾({result.reason})，新事实入库，旧记录标记 superseded",
                        importance=importance, similarity=sim,
                    )
                else:
                    rec.conflicts.append({**conflict_info, "resolution": "kept_old"})
                    self._persist_record(rec)
                    self.stats["conflict_rejected"] += 1
                    return WriteResult(
                        action="conflict_rejected", record_id=rec.id,
                        detail=f"与 {rec.id} 矛盾({result.reason})，旧记录重要度更高，拒绝写入",
                        importance=importance, similarity=sim,
                    )

        # 3) 正常插入
        new_rec = self._insert(content, importance, vec, source)
        self.stats["inserted"] += 1
        return WriteResult(
            action="inserted", record_id=new_rec.id,
            detail=f"importance={importance:.2f}", importance=importance,
        )

    # ------------------------------------------------------------------
    def _persist_record(self, rec: MemoryRecord) -> None:
        """单条 upsert（配置 db_path 时每条变更即时落库）。"""
        if self.db_path is None:
            return
        emb = np.asarray(rec.embedding, dtype=np.float32).tobytes() if rec.embedding else None
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO memories(id, content, importance, created_at, updated_at, "
                "hits, status, source, conflicts, embedding) VALUES (?,?,?,?,?,?,?,?,?,?)",
                (
                    rec.id, rec.content, rec.importance, rec.created_at, rec.updated_at,
                    rec.hits, rec.status, rec.source,
                    json.dumps(rec.conflicts, ensure_ascii=False), emb,
                ),
            )

    def _insert(self, content: str, importance: float, vec: np.ndarray, source: str) -> MemoryRecord:
        now = time.time()
        rec = MemoryRecord(
            id=f"mem_{uuid.uuid4().hex[:10]}",
            content=content,
            importance=importance,
            created_at=now,
            updated_at=now,
            source=source,
            embedding=vec.tolist(),
        )
        self.records[rec.id] = rec
        self._persist_record(rec)
        return rec

    # ------------------------------------------------------------------
    def recall(self, query: str, k: int | None = None) -> list[tuple[MemoryRecord, float]]:
        """相似度 × 重要度 × 新近度 加权召回。"""
        k = k or self.recall_top_k
        active = self._active()
        if not active:
            return []
        qv = self.embedder.encode_queries([query])[0]
        now = time.time()
        scored: list[tuple[MemoryRecord, float]] = []
        for rec in active:
            sim = float(np.dot(np.asarray(rec.embedding, dtype=np.float32), qv))
            recency = 1.0 / (1.0 + (now - rec.updated_at) / 86400.0 / 7.0)  # 周级衰减
            score = max(sim, 0.0) * (0.75 + 0.25 * rec.importance) * (0.85 + 0.15 * recency)
            scored.append((rec, round(score, 4)))
        scored.sort(key=lambda x: x[1], reverse=True)
        out = scored[:k]
        for rec, _ in out:
            rec.hits += 1
        return out

    def context_block(self, query: str) -> str:
        hits = self.recall(query)
        if not hits:
            return ""
        lines = [f"- {rec.content}（重要度{rec.importance:.2f}，命中{rec.hits}次）" for rec, _ in hits]
        return "【长期记忆】\n" + "\n".join(lines)

    # ------------------------------------------------------------------
    def save(self, path: str | Path | None = None) -> None:
        """全量落盘。有 db_path 时数据已实时同步（幂等重写）；也兼容导出 .jsonl。"""
        target = Path(path) if path else self.db_path
        if target is None:
            raise ValueError("LongTermMemory 未配置持久化路径")
        if target.suffix in (".db", ".sqlite"):
            target.parent.mkdir(parents=True, exist_ok=True)
            with sqlite3.connect(str(target)) as conn:
                conn.executescript(_MEM_SCHEMA)
                for rec in self.records.values():
                    emb = np.asarray(rec.embedding, dtype=np.float32).tobytes() if rec.embedding else None
                    conn.execute(
                        "INSERT OR REPLACE INTO memories(id, content, importance, created_at, updated_at, "
                        "hits, status, source, conflicts, embedding) VALUES (?,?,?,?,?,?,?,?,?,?)",
                        (
                            rec.id, rec.content, rec.importance, rec.created_at, rec.updated_at,
                            rec.hits, rec.status, rec.source,
                            json.dumps(rec.conflicts, ensure_ascii=False), emb,
                        ),
                    )
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            with open(target, "w", encoding="utf-8") as f:
                for rec in self.records.values():
                    f.write(json.dumps(rec.to_dict(), ensure_ascii=False) + "\n")

    @classmethod
    def load(cls, path: str | Path, **kwargs) -> "LongTermMemory":
        p = Path(path)
        if p.suffix in (".db", ".sqlite"):
            obj = cls.__new__(cls)
            LongTermMemory.__init__(obj, db_path=p, **kwargs)
            if p.exists():
                with sqlite3.connect(str(p)) as conn:
                    rows = conn.execute(
                        "SELECT id, content, importance, created_at, updated_at, hits, status, "
                        "source, conflicts, embedding FROM memories"
                    ).fetchall()
                for r in rows:
                    emb = (
                        np.frombuffer(r[9], dtype=np.float32).tolist() if r[9] else []
                    )
                    rec = MemoryRecord(
                        id=r[0], content=r[1], importance=r[2], created_at=r[3],
                        updated_at=r[4], hits=r[5], status=r[6], source=r[7],
                        conflicts=json.loads(r[8] or "[]"), embedding=emb,
                    )
                    obj.records[rec.id] = rec
            return obj
        obj = cls.__new__(cls)
        LongTermMemory.__init__(obj, **kwargs)
        if p.exists():
            with open(p, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        rec = MemoryRecord.from_dict(json.loads(line))
                        obj.records[rec.id] = rec
        return obj

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_records": len(self.records),
            "n_active": len(self._active()),
            "stats": dict(self.stats),
        }
