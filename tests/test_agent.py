"""CoT+ReAct Agent 测试：可答题带引用作答、证据不足拒答、static 基线对照、
指代消解、证据冲突处理。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tcm_rag.agent.react_agent import EvidenceAgent
from tcm_rag.config import Config
from tcm_rag.embedding import HashingEmbedder
from tcm_rag.indexing import ChunkStore, FaissHNSWIndex
from tcm_rag.llm.extractive import ExtractiveAnswerer
from tcm_rag.memory.short_term import ShortTermMemory
from tcm_rag.ner.medical_ner import MedicalNER
from tcm_rag.nli.conflict import RuleNLI
from tcm_rag.retrieval.bm25 import BM25Index
from tcm_rag.retrieval.evidence_filter import SecondaryEvidenceFilter
from tcm_rag.retrieval.hybrid import HybridRetriever
from tcm_rag.retrieval.rerank import FeatureReranker
from tcm_rag.schema import Chunk

VOCAB = {
    "herb": ["甘草", "麻黄", "桂枝", "人参", "附子", "芍药", "生姜", "大枣", "杏仁"],
    "formula": ["桂枝汤", "麻黄汤", "小柴胡汤", "四逆汤", "天麻钩藤饮"],
    "symptom": ["发热", "恶寒", "汗出", "头痛", "恶风", "无汗", "喘"],
    "pattern": ["风寒表虚证", "风寒表实证", "肝阳上亢证"],
    "disease": ["感冒"],
}

CHUNK_TEXTS = [
    ("c1", "桂枝汤出自《伤寒论》，由桂枝三两、芍药三两、甘草二两、生姜三两、大枣十二枚组成。"
           "功用解肌发表，调和营卫。主治太阳中风表虚证：发热汗出，恶风头痛，脉浮缓。"
           "使用注意：外感风寒表实证及温病初起者不宜使用。服后啜热稀粥，温覆取微汗。"),
    ("c2", "麻黄汤出自《伤寒论》，由麻黄三两、桂枝二两、甘草一两、杏仁七十个组成。"
           "功用发汗解表，宣肺平喘。主治外感风寒表实证：恶寒发热，无汗而喘，头身疼痛，脉浮紧。"
           "使用注意：表虚自汗、阴虚盗汗及虚喘者慎用。"),
    ("c3", "小柴胡汤由柴胡、黄芩、人参、半夏、甘草、生姜、大枣组成。主治伤寒少阳证："
           "往来寒热，胸胁苦满，默默不欲饮食，心烦喜呕。但见一证便是，不必悉具。"),
    ("c4", "四逆汤由甘草二两、干姜一两半、附子一枚组成。主治少阴病，四肢厥逆，恶寒蜷卧，"
           "呕吐不渴，腹痛下利，脉沉微细。附子有毒须先煎。使用注意：热厥及阴虚火旺者忌用。"),
]


def make_agent(**agent_cfg) -> tuple[EvidenceAgent, HybridRetriever]:
    chunks = [
        Chunk(chunk_id=cid, doc_id=f"doc_{cid}", text=t, title_path=["测试书", cid],
              source_type="textbook", seq=i, n_tokens=len(t))
        for i, (cid, t) in enumerate(CHUNK_TEXTS)
    ]
    emb = HashingEmbedder(dim=128)
    store = ChunkStore()
    store.add(chunks)
    index = FaissHNSWIndex(dim=128, m=16)
    index.add(emb.encode_documents([c.text for c in chunks]), [c.chunk_id for c in chunks])
    bm25 = BM25Index(chunks)
    ner = MedicalNER(vocab=VOCAB)
    nli = RuleNLI(ner)
    cfg = Config({
        "retrieval": {
            "dense_top_k": 4, "bm25_top_k": 4, "rrf_k": 60,
            "rerank_top_k": 4, "final_top_k": 3,
            "secondary_filter": {"enabled": True, "min_rel_score": 0.05,
                                 "ocr_noise_max": 0.18, "min_chars": 10, "dedup_jaccard": 0.8},
        },
        "agent": {
            "max_rounds": 3,
            "confidence_answer_threshold": 0.45,
            "confidence_refusal_threshold": 0.22,
            "consistency_threshold": 0.5,
            "coverage_target": 0.66,
            **agent_cfg,
        },
    }, Path("."))
    retriever = HybridRetriever(
        cfg, emb, store, index, bm25,
        reranker=FeatureReranker(ner),
        evidence_filter=SecondaryEvidenceFilter(
            min_rel_score=0.05, ocr_noise_max=0.18, min_chars=10,
            dedup_jaccard=0.8, final_top_k=3,
        ),
    )
    agent = EvidenceAgent(cfg, retriever, ner, nli, ExtractiveAnswerer(ner), llm=None)
    return agent, retriever


def test_answerable_question_with_citations():
    agent, _ = make_agent()
    res = agent.answer("桂枝汤由哪些药物组成？主治什么证候？")
    assert not res.refused
    assert res.answer
    assert res.citations
    assert res.confidence >= agent.refusal_threshold
    assert any("c1" in c for c in res.citations)
    assert res.trace and res.trace[0].action == "search"


def test_offtopic_question_low_confidence_or_refusal():
    agent, _ = make_agent()
    res = agent.answer("量子计算机的超导比特工作温度是多少？")
    # 证据不足 → 拒答，或至少置信度低于作答阈值并给出警示
    assert res.refused or res.confidence < agent.answer_threshold
    if res.refused:
        assert "证据" in res.answer or "检索" in res.answer


def test_static_baseline_always_answers():
    agent, _ = make_agent()
    res = agent.answer("量子计算机的超导比特工作温度是多少？", decision_mode="static")
    assert not res.refused           # 基线不做拒答控制（错误自信的来源）
    assert res.rounds_used == 1


def test_refusal_trace_recorded():
    agent, _ = make_agent()
    res = agent.answer("区块链共识算法的吞吐量上限是多少？")
    if res.refused:
        assert any(s.action == "refuse" for s in res.trace)
        assert res.refusal_reason


def test_anaphora_resolution_with_stm():
    agent, _ = make_agent()
    ner = MedicalNER(vocab=VOCAB)
    stm = ShortTermMemory(ner=ner, window_turns=4)
    stm.add_turn("user", "桂枝汤的组成是什么？")
    stm.add_turn("assistant", "桂枝汤由桂枝、芍药、甘草、生姜、大枣组成。")
    q, added = agent.resolve_anaphora("它的用法用量和注意事项呢？", stm)
    assert "桂枝汤" in added
    assert "桂枝汤" in q
    res = agent.answer("它的服用方法和禁忌是什么？", short_term=stm)
    assert not res.refused


def test_followup_without_entities_uses_last_topic():
    """口语追问（无指代词、无实体）应自动锚定上一轮用户话题实体。"""
    agent, _ = make_agent()
    ner = MedicalNER(vocab=VOCAB)
    stm = ShortTermMemory(ner=ner, window_turns=4)
    stm.add_turn("user", "桂枝汤主治发热汗出恶风的表虚证。")
    stm.add_turn("assistant", "桂枝汤解肌发表，调和营卫。")
    q, added = agent.resolve_anaphora("那该怎么煎服呢？", stm)
    assert added, "追问应触发实体补全"
    assert "桂枝汤" in q
    # 有实体的独立问题不应被误改写
    q2, added2 = agent.resolve_anaphora("麻黄汤的禁忌是什么？", stm)
    assert added2 == []
    assert q2 == "麻黄汤的禁忌是什么？"


def test_followup_anchors_assistant_conclusion_over_symptoms():
    """复现实测 bug：症状主诉后追问"怎么调理"，应锚定上一轮 assistant 的
    结论实体（证型/方剂），而不是被用户症状词面主导（曾跑偏到麻黄汤）。"""
    agent, _ = make_agent()
    ner = MedicalNER(vocab=VOCAB)
    stm = ShortTermMemory(ner=ner, window_turns=6)
    stm.add_turn("user", "我最近头晕头痛，面红，急躁易怒，睡不好，是什么证型？")
    stm.add_turn("assistant", "倾向判断为肝阳上亢证，治法平肝潜阳，方用天麻钩藤饮加减。")
    q, added = agent.resolve_anaphora("那该怎么调理？用什么方子？", stm)
    assert "肝阳上亢证" in added or "天麻钩藤饮" in added, f"补全实体错误: {added}"
    assert "肝阳上亢证" in q or "天麻钩藤饮" in q


def test_signals_structure():
    agent, retriever = make_agent()
    rr = retriever.retrieve("桂枝汤组成", mode="full")
    pool = {e.chunk.chunk_id: e for e in rr.evidences}
    sig = agent.compute_signals("桂枝汤由哪些药物组成？", pool)
    for key in ("coverage", "top_score", "consistency", "confidence", "missing_entities"):
        assert key in sig
    assert 0.0 <= sig["confidence"] <= 1.0
    assert 0.0 <= sig["consistency"] <= 1.0


def test_answer_dict_serializable():
    import json

    agent, _ = make_agent()
    res = agent.answer("麻黄汤主治什么？")
    d = res.to_dict()
    json.dumps(d, ensure_ascii=False)   # 不抛异常即可
    assert d["question"] == "麻黄汤主治什么？"
