"""规则式 NLI 冲突检测测试。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tcm_rag.ner.medical_ner import MedicalNER
from tcm_rag.nli.conflict import CONTRADICTION, ENTAILMENT, NEUTRAL, RuleNLI

VOCAB = {
    "herb": ["甘草", "麻黄", "人参", "附子", "桂枝"],
    "formula": ["桂枝汤", "麻黄汤"],
    "symptom": ["发热", "恶寒", "汗出", "无汗", "口渴"],
    "disease": ["感冒"],
}


def make_nli():
    return RuleNLI(MedicalNER(vocab=VOCAB))


def test_negation_contradiction():
    nli = make_nli()
    # 同一主体（桂枝汤证）下的 汗出/无汗 矛盾
    r = nli.judge("桂枝汤证患者发热汗出恶风。", "桂枝汤证患者无汗恶寒。")
    assert r.label == CONTRADICTION


def test_antonym_contradiction():
    nli = make_nli()
    r = nli.judge("附子性大热，有毒，孕妇忌用。", "附子性寒凉，无毒，孕妇可用。")
    assert r.label == CONTRADICTION
    assert "antonym" in r.reason or "negation" in r.reason


def test_dose_contradiction():
    nli = make_nli()
    r = nli.judge("方用麻黄3g，桂枝6g。", "该方麻黄用量为15g。")
    assert r.label == CONTRADICTION
    assert "dose" in r.reason


def test_entailment_high_overlap():
    nli = make_nli()
    r = nli.judge(
        "桂枝汤由桂枝、芍药、甘草、生姜、大枣组成，功用解肌发表，调和营卫。",
        "桂枝汤由桂枝芍药甘草生姜大枣组成，解肌发表调和营卫。",
    )
    assert r.label == ENTAILMENT


def test_neutral_different_topics():
    nli = make_nli()
    r = nli.judge("麻黄汤主治外感风寒表实证。", "人参味甘微寒，主补五脏。")
    assert r.label == NEUTRAL


def test_no_false_contradiction_on_complementary():
    nli = make_nli()
    r = nli.judge("桂枝汤主治发热汗出恶风。", "桂枝汤服药后需啜热粥以助药力。")
    assert r.label != CONTRADICTION


def test_consistency_of():
    nli = make_nli()
    score, conflicts = nli.consistency_of([
        "附子有毒，须先煎久煎。",
        "附子炮制后毒性降低，仍需先煎。",
        "附子无毒，可任意大量使用。",
    ])
    assert score < 1.0
    assert len(conflicts) >= 1


def test_consistency_single_statement():
    nli = make_nli()
    score, conflicts = nli.consistency_of(["麻黄汤发汗解表。"])
    assert score == 1.0
    assert conflicts == []
