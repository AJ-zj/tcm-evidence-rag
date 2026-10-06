"""检索链路测试：RRF 融合、BM25、FAISS-HNSW、二次过滤、特征重排、混合检索。"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tcm_rag.config import Config
from tcm_rag.embedding import HashingEmbedder
from tcm_rag.indexing import ChunkStore, FaissHNSWIndex
from tcm_rag.ner.medical_ner import MedicalNER
from tcm_rag.retrieval.bm25 import BM25Index
from tcm_rag.retrieval.evidence_filter import SecondaryEvidenceFilter
from tcm_rag.retrieval.hybrid import HybridRetriever, rrf_fuse
from tcm_rag.retrieval.rerank import FeatureReranker
from tcm_rag.schema import Chunk, Evidence

VOCAB = {
    "herb": ["甘草", "麻黄", "桂枝", "人参"],
    "formula": ["桂枝汤", "麻黄汤", "小柴胡汤"],
    "symptom": ["发热", "恶寒", "汗出", "头痛"],
}


def make_chunks() -> list[Chunk]:
    texts = [
        ("c1", "桂枝汤由桂枝芍药甘草生姜大枣组成，主治太阳中风，发热汗出恶风头痛。", "textbook", 0.0),
        ("c2", "麻黄汤由麻黄桂枝甘草杏仁组成，主治外感风寒表实证，恶寒发热无汗而喘。", "textbook", 0.0),
        ("c3", "小柴胡汤主治少阳证，往来寒热，胸胁苦满，心烦喜呕。", "textbook", 0.0),
        ("c4", "甘 草 味甘平□，主五 脏六府寒热邪气。", "ocr_scan", 0.35),   # 高噪声块
        ("c5", "短块。", "classic", 0.0),                                    # 过短块
        ("c6", "桂枝汤方：桂枝三两去皮，芍药三两，甘草二两炙，生姜三两切，大枣十二枚擘。服已须臾啜热稀粥。", "classic", 0.0),
    ]
    return [
        Chunk(chunk_id=cid, doc_id=f"doc_{cid}", text=t, title_path=["测试书", f"节{i}"],
              source_type=st, seq=i, n_tokens=len(t), ocr_noise=noise)
        for i, (cid, t, st, noise) in enumerate(texts)
    ]


def make_cfg(**overrides) -> Config:
    data = {
        "retrieval": {
            "dense_top_k": 6, "bm25_top_k": 6, "rrf_k": 60,
            "rerank_top_k": 6, "final_top_k": 3,
            "secondary_filter": {
                "enabled": True, "min_rel_score": 0.10, "ocr_noise_max": 0.18,
                "min_chars": 10, "dedup_jaccard": 0.72,
            },
        }
    }
    for k, v in overrides.items():
        section, _, key = k.partition(".")
        data.setdefault(section, {})[key] = v
    return Config(data, Path("."))


def test_rrf_fuse():
    fused = rrf_fuse({
        "a": [("x", 0.9), ("y", 0.5)],
        "b": [("y", 8.0), ("z", 3.0)],
    }, k=60)
    assert fused["y"] > fused["x"] > 0    # 双通道命中者分数最高
    assert "z" in fused


def test_bm25_search():
    chunks = make_chunks()
    idx = BM25Index(chunks)
    hits = idx.search("桂枝汤的组成", top_k=3)
    assert hits
    assert hits[0][0] in ("c1", "c6")


def test_faiss_hnsw_roundtrip(tmp_path):
    emb = HashingEmbedder(dim=64)
    chunks = make_chunks()
    vecs = emb.encode_documents([c.text for c in chunks])
    index = FaissHNSWIndex(dim=64, m=16, ef_construction=40, ef_search=32)
    index.add(vecs, [c.chunk_id for c in chunks])
    assert index.ntotal == len(chunks)

    q = emb.encode_queries(["桂枝汤组成与主治"])[0]
    hits = index.search_one(q, k=3)
    assert hits and hits[0][0] in {c.chunk_id for c in chunks}

    index.save(tmp_path / "hnsw")
    loaded = FaissHNSWIndex.load(tmp_path / "hnsw")
    assert loaded.ntotal == index.ntotal
    hits2 = loaded.search_one(q, k=3)
    assert [h[0] for h in hits2] == [h[0] for h in hits]


def test_secondary_filter_drops_noise_and_short():
    chunks = {c.chunk_id: c for c in make_chunks()}
    evs = [
        Evidence(chunk=chunks["c4"], score=0.9),   # 高噪声 → 丢
        Evidence(chunk=chunks["c5"], score=0.8),   # 过短 → 丢
        Evidence(chunk=chunks["c1"], score=0.7),
        Evidence(chunk=chunks["c2"], score=0.6),
        Evidence(chunk=chunks["c3"], score=0.05),  # 低分 → 丢
    ]
    f = SecondaryEvidenceFilter(min_rel_score=0.1, ocr_noise_max=0.18, min_chars=10, final_top_k=5)
    kept, dropped = f.apply(evs)
    kept_ids = {e.chunk.chunk_id for e in kept}
    assert kept_ids == {"c1", "c2"}
    reasons = {e.chunk.chunk_id: e.drop_reason for e in dropped}
    assert "ocr_noise" in reasons["c4"]
    assert "too_short" in reasons["c5"]
    assert "low_relevance" in reasons["c3"]


def test_secondary_filter_dedup():
    base = make_chunks()[0]
    dup = Chunk(chunk_id="c1b", doc_id="d", text=base.text + "（重排重复）", title_path=base.title_path,
                source_type="classic", seq=9, ocr_noise=0.0)
    evs = [Evidence(chunk=base, score=0.8), Evidence(chunk=dup, score=0.75)]
    f = SecondaryEvidenceFilter(min_rel_score=0.1, dedup_jaccard=0.6, final_top_k=5)
    kept, dropped = f.apply(evs)
    assert len(kept) == 1 and kept[0].chunk.chunk_id == "c1"
    assert dropped and "duplicate_of" in dropped[0].drop_reason


def test_feature_reranker_prefers_clean_and_relevant():
    ner = MedicalNER(vocab=VOCAB)
    rr = FeatureReranker(ner)
    chunks = {c.chunk_id: c for c in make_chunks()}
    ev_clean = Evidence(chunk=chunks["c1"], score=0.5,
                        channel_scores={"dense": 0.6, "bm25_norm": 0.5, "rrf_norm": 0.5})
    ev_noisy = Evidence(chunk=chunks["c4"], score=0.5,
                        channel_scores={"dense": 0.6, "bm25_norm": 0.5, "rrf_norm": 0.5})
    out = rr.rerank("桂枝汤的组成和主治", [ev_noisy, ev_clean])
    assert out[0].chunk.chunk_id == "c1"   # 清洁且实体匹配者靠前


def test_hybrid_retrieve_end_to_end():
    chunks = make_chunks()
    emb = HashingEmbedder(dim=64)
    store = ChunkStore()
    store.add(chunks)
    index = FaissHNSWIndex(dim=64, m=16)
    index.add(emb.encode_documents([c.text for c in chunks]), [c.chunk_id for c in chunks])
    bm25 = BM25Index(chunks)
    ner = MedicalNER(vocab=VOCAB)
    cfg = make_cfg()
    retriever = HybridRetriever(
        cfg, emb, store, index, bm25,
        reranker=FeatureReranker(ner),
        evidence_filter=SecondaryEvidenceFilter(
            min_rel_score=0.1, ocr_noise_max=0.18, min_chars=10, final_top_k=3
        ),
    )
    res = retriever.retrieve("桂枝汤由哪些药物组成？主治什么？", mode="full")
    ids = [e.chunk.chunk_id for e in res.evidences]
    assert ids and ids[0] in ("c1", "c6")
    # 高噪声块被二次过滤拦截
    assert "c4" not in ids
    assert all(e.chunk.chunk_id != "c4" for e in res.evidences)
    assert res.total_ms >= 0


def test_hybrid_modes_run():
    chunks = make_chunks()
    emb = HashingEmbedder(dim=64)
    store = ChunkStore(); store.add(chunks)
    index = FaissHNSWIndex(dim=64, m=16)
    index.add(emb.encode_documents([c.text for c in chunks]), [c.chunk_id for c in chunks])
    bm25 = BM25Index(chunks)
    ner = MedicalNER(vocab=VOCAB)
    retriever = HybridRetriever(make_cfg(), emb, store, index, bm25,
                                reranker=FeatureReranker(ner))
    for mode in ("dense", "bm25", "hybrid", "hybrid_rerank", "full"):
        res = retriever.retrieve("麻黄汤主治什么", mode=mode)
        assert isinstance(res.evidences, list)
