"""分层记忆测试：短期滑窗压缩 + 重要度衰减；长期阈值入库 + 冲突消解。"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tcm_rag.embedding import HashingEmbedder
from tcm_rag.memory.long_term import LongTermMemory
from tcm_rag.memory.short_term import ShortTermMemory
from tcm_rag.ner.medical_ner import MedicalNER

VOCAB = {
    "herb": ["甘草", "麻黄", "桂枝", "人参", "附子"],
    "formula": ["桂枝汤", "麻黄汤", "四逆汤"],
    "symptom": ["发热", "恶寒", "汗出", "头痛"],
    "disease": ["感冒"],
}


def make_ner():
    return MedicalNER(vocab=VOCAB)


# ---------------- 短期记忆 ----------------

def test_stm_sliding_window_and_compression():
    stm = ShortTermMemory(ner=make_ner(), window_turns=3, max_window_tokens=500,
                          summary_max_tokens=200)
    for i in range(6):
        stm.add_turn("user", f"第{i}个问题：桂枝汤和麻黄汤的区别是什么？发热汗出如何处理？")
        stm.add_turn("assistant", f"第{i}个回答：桂枝汤调和营卫，麻黄汤发汗解表。")
    active = stm._active_turns()
    assert len(active) <= 3
    assert stm.summary  # 旧轮次被压缩进摘要
    assert stm.stats["turns_compressed"] >= 6


def test_stm_importance_decay():
    ner = make_ner()
    stm = ShortTermMemory(ner=ner, window_turns=10)
    t_old = stm.add_turn("user", "桂枝汤主治发热汗出恶风头痛。")
    for _ in range(5):
        stm.add_turn("user", "嗯。")
    t_new = stm.add_turn("user", "麻黄汤主治恶寒发热无汗。")
    imp_old = stm.effective_importance(t_old)
    imp_new = stm.effective_importance(t_new)
    # 相同医疗内容密度下，旧轮次有效重要度衰减更多
    assert imp_old < imp_new
    assert t_old.base_importance > 0.2   # 医疗文本基础重要度显著高于闲聊


def test_stm_context_contains_summary_and_recent():
    stm = ShortTermMemory(ner=make_ner(), window_turns=2, max_window_tokens=400)
    stm.add_turn("user", "患者感冒发热，用了桂枝汤。")
    stm.add_turn("assistant", "桂枝汤调和营卫，适合表虚证。")
    stm.add_turn("user", "还需要注意什么？")
    stm.add_turn("assistant", "服后啜热粥，温覆取微汗。")
    stm.add_turn("user", "麻黄汤能用吗？")
    ctx = stm.context()
    assert "麻黄汤" in ctx            # 近期轮次原文
    assert "近期对话" in ctx


def test_stm_recent_entities():
    stm = ShortTermMemory(ner=make_ner())
    stm.add_turn("user", "桂枝汤的组成是什么？")
    ents = stm.recent_entities()
    assert "桂枝汤" in ents.get("formula", [])


# ---------------- 长期记忆 ----------------

def make_ltm(**kw) -> LongTermMemory:
    defaults = dict(
        ner=make_ner(), embedder=HashingEmbedder(dim=256),
        merge_similarity=0.88, write_importance_threshold=0.45,
        recall_top_k=3, conflict_policy="newer_wins",
        conflict_candidate_sim=0.25,   # 哈希嵌入下放宽冲突检测候选阈值
    )
    defaults.update(kw)
    return LongTermMemory(**defaults)


def test_ltm_rejects_low_importance():
    ltm = make_ltm()
    r = ltm.write("今天天气不错啊")
    assert r.action == "rejected_low_importance"
    assert ltm.stats["rejected_low_importance"] == 1


def test_ltm_insert_and_recall():
    ltm = make_ltm()
    r = ltm.write("患者对麻黄过敏，使用桂枝汤时需慎用麻黄类方剂。", source="s1")
    assert r.action == "inserted"
    hits = ltm.recall("麻黄过敏需要注意什么")
    assert hits and hits[0][0].content.startswith("患者对麻黄过敏")


def test_ltm_merges_near_duplicate():
    ltm = make_ltm(merge_similarity=0.6)  # 哈希嵌入下放宽合并阈值以便测试
    ltm.write("患者对麻黄过敏，使用桂枝汤时需慎用麻黄类方剂。")
    r = ltm.write("患者对麻黄过敏，使用桂枝汤时需慎用麻黄类方剂。")
    assert r.action == "merged"
    assert ltm.stats["merged"] == 1
    assert len(ltm._active()) == 1


def test_ltm_conflict_newer_wins():
    ltm = make_ltm(conflict_policy="newer_wins")
    r1 = ltm.write("附子无毒，可以放心大量长期服用。")
    assert r1.action in ("inserted",)
    r2 = ltm.write("附子有毒，必须炮制先煎，不可大量服用。")
    assert r2.action == "conflict_superseded"
    assert ltm.stats["conflicts_detected"] >= 1
    active = ltm._active()
    assert len(active) == 1
    assert "有毒" in active[0].content   # 新事实存活


def test_ltm_conflict_importance_wins():
    ltm = make_ltm(conflict_policy="importance_wins")
    ltm.write("附子有毒，必须炮制先煎，孕妇忌用；配伍人参、甘草可减其毒，不可大量服用。")
    r2 = ltm.write("附子无毒，可大量服用。")
    # 旧记录实体更丰富、重要度更高 → 拒绝新写入
    assert r2.action == "conflict_rejected"


def test_ltm_save_load_roundtrip(tmp_path):
    ltm = make_ltm()
    ltm.write("患者对麻黄过敏，慎用麻黄汤。")
    # SQLite 后端 roundtrip（双库设计的结构化侧）
    p_db = tmp_path / "ltm.db"
    ltm.save(p_db)
    ltm2 = LongTermMemory.load(p_db, ner=make_ner(), embedder=HashingEmbedder(dim=256))
    assert len(ltm2.records) == 1
    hits = ltm2.recall("麻黄过敏")
    assert hits
    # 旧 JSONL 后端仍兼容
    p_json = tmp_path / "ltm.jsonl"
    ltm.save(p_json)
    ltm3 = LongTermMemory.load(p_json, ner=make_ner(), embedder=HashingEmbedder(dim=256))
    assert len(ltm3.records) == 1


def test_ltm_sqlite_autopersist(tmp_path):
    """配置 db_path 时写入即时落库，新实例加载可见。"""
    db = tmp_path / "mem.db"
    ltm = make_ltm(db_path=db)
    ltm.write("患者对人参过敏，禁用人参及含人参之方剂。")
    ltm2 = LongTermMemory.load(db, ner=make_ner(), embedder=HashingEmbedder(dim=256))
    assert len(ltm2.records) == 1
    assert ltm2.recall("人参过敏禁用")
