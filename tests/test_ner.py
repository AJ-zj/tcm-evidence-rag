"""医疗 NER 与重要度加权测试。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tcm_rag.ner.medical_ner import MedicalNER, build_vocab_from_corpus

VOCAB = {
    "herb": ["甘草", "麻黄", "桂枝", "茯苓"],
    "formula": ["桂枝汤", "麻黄汤", "小柴胡汤"],
    "symptom": ["发热", "恶寒", "头痛", "汗出"],
    "pattern": ["风寒表虚证"],
    "disease": ["感冒", "咳嗽"],
}


def make_ner():
    return MedicalNER(vocab=VOCAB)


def test_recognize_longest_match():
    ner = make_ner()
    ents = ner.recognize("太阳中风者，桂枝汤主之，恶风发热汗出。")
    texts = [e.text for e in ents]
    assert "桂枝汤" in texts          # 最长匹配优先于"桂枝"
    assert "发热" in texts
    types = {e.text: e.type for e in ents}
    assert types["桂枝汤"] == "formula"


def test_entity_types_grouping():
    ner = make_ner()
    groups = ner.entity_types("麻黄汤治伤寒头痛发热，桂枝汤治风寒表虚证汗出。")
    assert "麻黄汤" in groups["formula"]
    assert "桂枝汤" in groups["formula"]
    assert "头痛" in groups["symptom"]
    assert "风寒表虚证" in groups["pattern"]


def test_importance_ordering():
    ner = make_ner()
    low = ner.importance("今天天气不错。")
    mid = ner.importance("患者发热头痛。")
    high = ner.importance("桂枝汤治太阳中风，发热汗出恶风头痛，方用桂枝芍药甘草生姜大枣。")
    assert low < mid < high
    assert 0.0 <= low <= 1.0 and high <= 1.0


def test_importance_medical_text_higher_than_chitchat():
    ner = make_ner()
    assert ner.importance("麻黄汤的组成是麻黄桂枝甘草杏仁") > ner.importance("你好啊请坐")


def test_save_load_roundtrip(tmp_path):
    ner = make_ner()
    p = tmp_path / "vocab.json"
    ner.save(p)
    ner2 = MedicalNER.load(p)
    assert {e.text for e in ner2.recognize("桂枝汤主之")} == {e.text for e in ner.recognize("桂枝汤主之")}


def test_build_vocab_from_corpus(tmp_path):
    tb = tmp_path / "textbook"
    tb.mkdir()
    import json

    data = {
        "book": "中药学",
        "source_type": "textbook",
        "chapters": [
            {"title": "解表药", "records": [{"term": "测试药", "fields": {"功效": "测试"}}]}
        ],
    }
    (tb / "中药学.json").write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    vocab = build_vocab_from_corpus(tb)
    assert "测试药" in vocab["herb"]
    assert "甘草" in vocab["herb"]  # 种子词表仍在
