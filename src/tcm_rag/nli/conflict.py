"""规则式 NLI 冲突检测（长期记忆写入门控 / 证据一致性评估共用）。

面向中医陈述的三类矛盾信号：
1. 否定矛盾：同一实体下 "不/无 X" 与 "X" 并存（如"不恶寒" vs "恶寒"）
2. 反义属性矛盾：寒↔热、虚↔实、有毒↔无毒、忌用↔可用 等领域反义对
3. 数值矛盾：同一药物剂量不一致

蕴含判定采用内容字 bigram Jaccard + 实体覆盖，其余归为中性。
规则式实现零推理依赖、可解释、可单测；接口与模型式 NLI 兼容。
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

from ..ner.medical_ner import MedicalNER

ENTAILMENT = "entailment"
NEUTRAL = "neutral"
CONTRADICTION = "contradiction"

# 否定线索（紧邻术语前方 1~2 字）
NEGATION_CUES = ("不", "无", "非", "未", "勿", "莫", "否")

# 领域反义对（字/词级）
OPPOSITE_PAIRS: list[tuple[str, str]] = [
    ("寒", "热"), ("温", "凉"), ("虚", "实"), ("补", "泻"), ("升", "降"),
    ("浮", "沉"), ("表", "里"), ("燥", "湿"), ("收", "散"), ("迟", "数"),
    ("有毒", "无毒"), ("小毒", "无毒"), ("大毒", "无毒"),
    ("忌用", "可用"), ("禁用", "可用"), ("慎用", "可用"), ("禁忌", "适宜"),
    ("宜", "忌"), ("恶寒", "恶热"), ("自汗", "无汗"), ("汗出", "无汗"),
    ("口渴", "不渴"), ("便秘", "泄泻"), ("喜温", "喜凉"), ("拒按", "喜按"),
    ("加重", "缓解"), ("增加", "减少"), ("亢盛", "不足"), ("上炎", "下陷"),
    ("外感", "内伤"), ("生用", "炙用"), ("先煎", "后下"), ("久服", "中病即止"),
]

# 剂量模式：3~10g / 三钱 / 一两 等
DOSE_RE = re.compile(r"(\d+(?:\.\d+)?|[一二三四五六七八九十半]+)\s*(g|克|钱|两|枚|升|合|片)")

_NUM_MAP = {"一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7,
            "八": 8, "九": 9, "十": 10, "半": 0.5}


@dataclass
class NLIResult:
    label: str                    # entailment | neutral | contradiction
    score: float                  # 判定强度 [0,1]
    reason: str = ""

    @property
    def is_contradiction(self) -> bool:
        return self.label == CONTRADICTION

    @property
    def is_entailment(self) -> bool:
        return self.label == ENTAILMENT


def _cn_num(s: str) -> float | None:
    try:
        return float(s)
    except ValueError:
        pass
    if s in _NUM_MAP:
        return _NUM_MAP[s]
    if len(s) == 2 and s[0] == "十":
        return 10 + _NUM_MAP.get(s[1], 0)
    return None


def _char_bigrams(text: str) -> set[str]:
    chars = re.sub(r"\s+", "", text)
    if len(chars) < 2:
        return {chars} if chars else set()
    return {chars[i:i + 2] for i in range(len(chars) - 1)}


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _split_clauses(text: str) -> list[str]:
    parts = re.split(r"[。！？；\n，,]", text)
    return [p.strip() for p in parts if len(p.strip()) >= 4]


class RuleNLI:
    """规则式矛盾/蕴含判定器。"""

    def __init__(self, ner: MedicalNER | None = None, entailment_threshold: float = 0.55):
        self.ner = ner or MedicalNER()
        self.entailment_threshold = entailment_threshold

    # ------------------------------------------------------------------
    def _entities(self, text: str) -> set[str]:
        return {e.text for e in self.ner.recognize(text)}

    def _negated_terms(self, text: str) -> set[str]:
        """找出被否定线索修饰的术语：不X / 无X / 未X。"""
        negated: set[str] = set()
        for term in self._entities(text):
            for cue in NEGATION_CUES:
                if cue + term in text:
                    negated.add(term)
        return negated

    def _dose_map(self, text: str) -> dict[str, float]:
        """药名 → 剂量：在前置 12 字窗口内找最近出现的药名。"""
        out: dict[str, float] = {}
        for m in DOSE_RE.finditer(text):
            start = m.start()
            window = text[max(0, start - 12):start]
            best, best_pos = None, -1
            for herb in self.ner.vocab.get("herb", set()):
                pos = window.rfind(herb)
                if pos > best_pos:
                    best, best_pos = herb, pos
            if best is not None:
                value = _cn_num(m.group(1))
                if value is not None:
                    unit = m.group(2)
                    if unit == "钱":
                        value *= 3.0  # 一钱≈3g，统一到克
                    out[best] = value
        return out

    # ------------------------------------------------------------------
    def judge(self, premise: str, hypothesis: str) -> NLIResult:
        """判定 hypothesis 相对 premise 的关系（对称矛盾检测 + 非对称蕴含）。"""
        ent_p, ent_h = self._entities(premise), self._entities(hypothesis)
        shared = ent_p & ent_h
        # 文本高度重叠时先做蕴含
        jac = _jaccard(_char_bigrams(premise), _char_bigrams(hypothesis))
        if jac >= self.entailment_threshold and not self._has_conflict_signal(premise, hypothesis):
            return NLIResult(ENTAILMENT, round(min(1.0, jac), 3), f"content_overlap={jac:.2f}")

        if not shared and not self._topic_overlap(premise, hypothesis):
            return NLIResult(NEUTRAL, 0.0, "no_shared_entity")

        conflict = self._find_conflict(premise, hypothesis, shared)
        if conflict:
            label, score, reason = conflict
            return NLIResult(label, score, reason)

        if jac >= self.entailment_threshold * 0.7 and ent_h and ent_h <= ent_p:
            return NLIResult(ENTAILMENT, round(jac, 3), "entity_subset")
        return NLIResult(NEUTRAL, round(jac, 3), "insufficient_overlap")

    def detect_conflict(self, new_fact: str, existing_fact: str) -> NLIResult:
        """长期记忆写入前的冲突检测入口（语义上等价 judge，命名更直观）。"""
        return self.judge(existing_fact, new_fact)

    # ------------------------------------------------------------------
    def _topic_overlap(self, a: str, b: str) -> bool:
        """无共享词典实体时的宽松主题重叠（字符 bigram Jaccard）。"""
        return _jaccard(_char_bigrams(a), _char_bigrams(b)) >= 0.35

    def _has_conflict_signal(self, a: str, b: str) -> bool:
        return self._find_conflict(a, b, self._entities(a) & self._entities(b)) is not None

    def _find_conflict(self, a: str, b: str, shared: set[str]) -> tuple[str, float, str] | None:
        clauses_a = _split_clauses(a) or [a]
        clauses_b = _split_clauses(b) or [b]

        # 1) 否定矛盾：A 说"不X"，B 肯定"X"（X 为共享实体术语）
        neg_a, neg_b = self._negated_terms(a), self._negated_terms(b)
        for term in shared:
            if term in neg_a and any(term in cb and term not in neg_b for cb in clauses_b):
                return CONTRADICTION, 0.9, f"negation:{term}"
            if term in neg_b and any(term in ca and term not in neg_a for ca in clauses_a):
                return CONTRADICTION, 0.9, f"negation:{term}"

        # 2) 反义属性矛盾：x 出现在 a 的某子句、y 出现在 b 的某子句（双向），
        #    且两子句共享实体或主题重叠（避免不同主体的寒/热被误判为矛盾）
        for pair in OPPOSITE_PAIRS:
            hit = False
            for x, y in (pair, (pair[1], pair[0])):
                if x == y:
                    continue
                for ca in clauses_a:
                    if x not in ca or y in ca:
                        continue
                    for cb in clauses_b:
                        if y not in cb or x in cb:
                            continue
                        ent_ca = self._entities(ca)
                        ent_cb = self._entities(cb)
                        if (ent_ca & ent_cb) or _jaccard(_char_bigrams(ca), _char_bigrams(cb)) >= 0.3:
                            hit = True
                            break
                    if hit:
                        break
                if hit:
                    break
            if hit:
                return CONTRADICTION, 0.85, f"antonym:{pair[0]}↔{pair[1]}"

        # 3) 数值/剂量矛盾
        dose_a, dose_b = self._dose_map(a), self._dose_map(b)
        for herb, da in dose_a.items():
            db = dose_b.get(herb)
            if db is not None and abs(da - db) / max(da, db, 1e-6) > 0.25:
                return CONTRADICTION, 0.8, f"dose:{herb}({da}vs{db})"

        return None

    # ------------------------------------------------------------------
    def consistency_of(self, statements: Iterable[str]) -> tuple[float, list[NLIResult]]:
        """一组陈述的两两一致性 ∈ [0,1]：1 - 矛盾对占比（附带矛盾明细）。"""
        items = [s for s in statements if s and s.strip()]
        if len(items) < 2:
            return 1.0, []
        conflicts: list[NLIResult] = []
        pairs = 0
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                pairs += 1
                r = self.judge(items[i], items[j])
                if r.is_contradiction:
                    conflicts.append(r)
        score = 1.0 - len(conflicts) / max(pairs, 1)
        return round(max(0.0, score), 4), conflicts
