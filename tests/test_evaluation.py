"""评估指标与测试集构建器测试。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tcm_rag.evaluation.dataset import EvalSetBuilder
from tcm_rag.evaluation.metrics import (
    FaithfulnessScorer,
    answer_relevancy,
    context_precision,
    context_recall,
    mrr,
    nanmean,
    precision_at_k,
    recall_at_k,
    split_claims,
    top1_accuracy,
)
from tcm_rag.ner.medical_ner import MedicalNER
from tcm_rag.nli.conflict import RuleNLI
from tcm_rag.schema import Chunk

VOCAB = {
    "herb": ["甘草", "麻黄", "桂枝"],
    "formula": ["桂枝汤", "麻黄汤"],
    "symptom": ["发热", "汗出", "恶寒"],
}


def test_recall_precision_mrr():
    retrieved = ["a", "b", "c", "d"]
    gold = ["b", "x"]
    assert recall_at_k(retrieved, gold, 2) == 0.5
    assert recall_at_k(retrieved, gold, 4) == 0.5
    assert precision_at_k(retrieved, gold, 2) == 0.5
    assert mrr(retrieved, gold) == 0.5
    assert top1_accuracy(retrieved, gold) == 0.0
    assert top1_accuracy(["b", "a"], gold) == 1.0


def test_context_precision_rewards_early_hits():
    gold = ["g"]
    early = context_precision(["g", "x", "y"], gold)
    late = context_precision(["x", "y", "g"], gold)
    assert early > late


def test_context_recall():
    assert context_recall(["a", "g1", "g2"], ["g1", "g2"]) == 1.0
    assert context_recall(["a"], ["g1"]) == 0.0


def test_nanmean_skips_nan():
    v = nanmean([1.0, float("nan"), 0.0])
    assert v == 0.5


def test_split_claims_strips_citations():
    claims = split_claims("桂枝汤由桂枝芍药甘草生姜大枣五味组成〔1〕。主治太阳中风表虚证发热汗出恶风〔2〕！")
    assert len(claims) == 2
    assert all("〔" not in c for c in claims)


def test_faithfulness_supported_vs_fabricated():
    ner = MedicalNER(vocab=VOCAB)
    scorer = FaithfulnessScorer(ner, RuleNLI(ner))
    ctx = ["桂枝汤由桂枝三两、芍药三两、甘草二两、生姜、大枣组成，主治发热汗出恶风。"]
    grounded = "桂枝汤由桂枝、芍药、甘草、生姜、大枣组成，主治发热汗出。"
    fabricated = "桂枝汤由桂枝芍药组成，另加人参三钱以补气，可治疗所有温病初起。"
    f1, _ = scorer.score(grounded, ctx)
    f2, d2 = scorer.score(fabricated, ctx)
    assert f1 > f2
    assert f1 >= 0.9
    assert d2["n_claims"] >= 1


def test_answer_relevancy():
    ner = MedicalNER(vocab=VOCAB)
    q = "桂枝汤主治什么证候？"
    rel = answer_relevancy(q, "桂枝汤主治太阳中风表虚证，症见发热汗出恶风。", ner)
    irrel = answer_relevancy(q, "今天天气很好，适合外出散步运动。", ner)
    assert rel > irrel


def test_eval_builder_mini_corpus():
    ner = MedicalNER(vocab=VOCAB)
    chunks = [
        Chunk(chunk_id="qa1#c000", doc_id="qa1", text="# 医案问答：感冒怎么办\n\n【患者问】感冒发热汗出怎么办？\n\n【医师答】表虚者可用桂枝汤调和营卫。",
              title_path=["医案问答：感冒怎么办"], source_type="qa", seq=0,
              meta={"question": "感冒发热汗出怎么办？", "answer": "表虚者可用桂枝汤调和营卫。",
                    "label": "中医科", "doc_title": "感冒怎么办"}),
        Chunk(chunk_id="tb1#c000", doc_id="tb1", text="〈中药学·解表药·桂枝〉\n【性味归经】辛甘温。【功效】发汗解肌。",
              title_path=["中药学", "解表药", "桂枝"], source_type="textbook", seq=0),
        Chunk(chunk_id="tb2#c000", doc_id="tb2", text="〈方剂学·解表剂·桂枝汤〉\n【出处】《伤寒论》。【组成】桂枝芍药甘草生姜大枣。【功用】解肌发表。",
              title_path=["方剂学", "解表剂", "桂枝汤"], source_type="textbook", seq=0),
        Chunk(chunk_id="cl1#c000", doc_id="cl1", text="〈伤寒论·辨太阳病脉证并治·第12条〉\n太阳中风，阳浮而阴弱……桂枝汤主之。",
              title_path=["伤寒论", "辨太阳病脉证并治", "第12条"], source_type="classic", seq=0),
        Chunk(chunk_id="cs1#c000", doc_id="cs1", text="〈教学医案·医案一〉\n患者风寒表虚证，桂枝汤主之，三剂而愈。",
              title_path=["教学医案", "医案一"], source_type="case", seq=0),
    ]
    builder = EvalSetBuilder(chunks, ner, seed=42, target_size=100, unanswerable_count=5)
    items = builder.build()
    types = EvalSetBuilder.type_counts(items)
    assert types.get("qa", 0) >= 1
    assert types.get("herb", 0) >= 1
    assert types.get("formula", 0) >= 1
    assert types.get("classic", 0) >= 1
    assert types.get("case", 0) >= 1
    assert types.get("unanswerable", 0) == 5
    # 金标准非空（unanswerable 除外）
    for it in items:
        if it.item_type != "unanswerable":
            assert it.gold_chunk_ids
    # 序列化往返
    import json

    for it in items:
        json.dumps(it.to_dict(), ensure_ascii=False)
