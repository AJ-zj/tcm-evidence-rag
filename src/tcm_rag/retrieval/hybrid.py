"""混合召回 + RRF 融合 + 重排 + 二次过滤的检索编排。

支持多种检索模式用于 A/B 评估：
- dense:        仅向量召回（基线，复现"朴素检索"）
- bm25:         仅稀疏召回
- hybrid:       BM25 + 向量双路召回，RRF 融合
- hybrid_rerank: 混合召回 + 重排
- full:         混合召回 + 重排 + 二次证据过滤（完整管线）
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..embedding import Embedder
from ..indexing import ChunkStore, FaissHNSWIndex
from ..schema import Chunk, Evidence
from .bm25 import BM25Index
from .evidence_filter import SecondaryEvidenceFilter
from .rerank import FeatureReranker, Reranker

MODES = ("dense", "bm25", "hybrid", "hybrid_rerank", "full")


@dataclass
class RetrievalResult:
    query: str
    mode: str
    evidences: list[Evidence] = field(default_factory=list)      # 最终输出（full 模式下为过滤后）
    candidates: list[Evidence] = field(default_factory=list)     # 过滤前候选（含全部通道分）
    dropped: list[Evidence] = field(default_factory=list)
    timings_ms: dict[str, float] = field(default_factory=dict)

    @property
    def chunk_ids(self) -> list[str]:
        return [e.chunk.chunk_id for e in self.evidences]

    @property
    def total_ms(self) -> float:
        return round(sum(self.timings_ms.values()), 2)


def rrf_fuse(
    ranked_lists: dict[str, list[tuple[str, float]]], k: int = 60
) -> dict[str, float]:
    """Reciprocal Rank Fusion：score(d) = Σ_channel 1/(k + rank_channel(d))。"""
    fused: dict[str, float] = {}
    for hits in ranked_lists.values():
        for rank, (cid, _score) in enumerate(hits):
            fused[cid] = fused.get(cid, 0.0) + 1.0 / (k + rank + 1)
    return fused


class HybridRetriever:
    def __init__(
        self,
        cfg,
        embedder: Embedder,
        chunk_store: ChunkStore,
        hnsw: FaissHNSWIndex,
        bm25: BM25Index,
        reranker: Reranker | None = None,
        evidence_filter: SecondaryEvidenceFilter | None = None,
    ):
        self.cfg = cfg
        self.embedder = embedder
        self.store = chunk_store
        self.hnsw = hnsw
        self.bm25 = bm25
        self.reranker = reranker or FeatureReranker(ner=None)  # type: ignore[arg-type]
        self.filter = evidence_filter or SecondaryEvidenceFilter(enabled=False)

        self.dense_top_k = cfg.get("retrieval.dense_top_k", 30)
        self.bm25_top_k = cfg.get("retrieval.bm25_top_k", 30)
        self.rrf_k = cfg.get("retrieval.rrf_k", 60)
        self.rerank_top_k = cfg.get("retrieval.rerank_top_k", 12)
        self.final_top_k = cfg.get("retrieval.final_top_k", 5)

    # ------------------------------------------------------------------
    def retrieve(
        self,
        query: str,
        mode: str = "full",
        top_k: int | None = None,
        query_vector: np.ndarray | None = None,
    ) -> RetrievalResult:
        assert mode in MODES, f"未知检索模式: {mode}"
        top_k = top_k or self.final_top_k
        timings: dict[str, float] = {}

        t0 = time.perf_counter()
        qv = query_vector if query_vector is not None else self.embedder.encode_queries([query])[0]
        timings["embed_ms"] = (time.perf_counter() - t0) * 1000

        dense_hits: list[tuple[str, float]] = []
        bm25_hits: list[tuple[str, float]] = []

        if mode in ("dense", "hybrid", "hybrid_rerank", "full"):
            t0 = time.perf_counter()
            dense_hits = self.hnsw.search_one(qv, self.dense_top_k)
            timings["dense_ms"] = (time.perf_counter() - t0) * 1000
        if mode in ("bm25", "hybrid", "hybrid_rerank", "full"):
            t0 = time.perf_counter()
            bm25_hits = self.bm25.search(query, self.bm25_top_k)
            timings["bm25_ms"] = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        # 通道归一分（绝对量纲，避免"垃圾结果的相对第一名"拿到满分）：
        # - bm25: 饱和归一 s/(s+10)
        # - rrf:  除以理论最大值 n_channels/(k+1)
        dense_map = {cid: s for cid, s in dense_hits}
        bm25_norm_map = {cid: s / (s + 10.0) for cid, s in bm25_hits}

        if mode == "dense":
            fused = {cid: s for cid, s in dense_hits}
            fused_max = 1.0
        elif mode == "bm25":
            fused = dict(bm25_norm_map)
            fused_max = 1.0
        else:
            fused = rrf_fuse({"dense": dense_hits, "bm25": bm25_hits}, k=self.rrf_k)
            fused_max = 2.0 / (self.rrf_k + 1)   # 双通道均排第一时的理论最大值

        candidates: list[Evidence] = []
        for cid, fscore in sorted(fused.items(), key=lambda x: x[1], reverse=True):
            chunk: Chunk | None = self.store.get(cid)
            if chunk is None:
                continue
            ev = Evidence(
                chunk=chunk,
                score=fscore,
                channel_scores={
                    "dense": dense_map.get(cid, 0.0),
                    "bm25_norm": bm25_norm_map.get(cid, 0.0),
                    "rrf": fscore,
                    "rrf_norm": min(1.0, fscore / fused_max) if fused_max > 0 else 0.0,
                },
            )
            candidates.append(ev)
        timings["fuse_ms"] = (time.perf_counter() - t0) * 1000

        # 重排
        if mode in ("hybrid_rerank", "full"):
            t0 = time.perf_counter()
            pool = candidates[: max(self.rerank_top_k * 2, self.dense_top_k)]
            candidates = self.reranker.rerank(query, pool)
            timings["rerank_ms"] = (time.perf_counter() - t0) * 1000
        else:
            for ev in candidates:
                ev.score = ev.channel_scores.get("rrf_norm", ev.channel_scores.get("dense", 0.0))

        # 二次过滤
        dropped: list[Evidence] = []
        if mode == "full":
            t0 = time.perf_counter()
            kept, dropped = self.filter.apply(candidates, limit=max(top_k, self.final_top_k))
            timings["filter_ms"] = (time.perf_counter() - t0) * 1000
            final = kept[:top_k]
        else:
            final = candidates[:top_k]

        return RetrievalResult(
            query=query,
            mode=mode,
            evidences=final,
            candidates=candidates,
            dropped=dropped,
            timings_ms={k: round(v, 2) for k, v in timings.items()},
        )

    # ------------------------------------------------------------------
    def dense_only_latency(self, query_vector: np.ndarray, k: int = 30) -> float:
        """纯 FAISS-HNSW 单次检索延迟（ms），用于 P95 基准。"""
        t0 = time.perf_counter()
        self.hnsw.search_one(query_vector, k)
        return (time.perf_counter() - t0) * 1000

    def stats(self) -> dict[str, Any]:
        return {
            "n_chunks": len(self.store),
            "hnsw_ntotal": self.hnsw.ntotal,
            "reranker": getattr(self.reranker, "name", "unknown"),
            "filter_enabled": self.filter.enabled,
        }
