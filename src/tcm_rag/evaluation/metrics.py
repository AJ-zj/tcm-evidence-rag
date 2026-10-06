"""RAGAS 风格评估指标（离线可复现实现 + 可选 LLM 裁判）。

检索侧：Recall@K / Precision@K / MRR / Top-1 Accuracy / Hit@K
生成侧（对齐 RAGAS 四指标语义）：
- faithfulness（忠实度）：答案陈述中能被检索上下文支持的比例
- context_precision（上下文精确度）：检索结果中金标准证据的排序加权占比
- context_recall（上下文召回率）：金标准证据被检索到的比例
- answer_relevancy（答案相关度）：答案对问题内容词/实体的响应度（离线代理）

离线实现全部基于确定性规则（实体重叠 / bigram Jaccard / RuleNLI），
保证评估闭环在无外部 API 时可复现；配置 LLM 后可切换 LLM 裁判。
"""
from __future__ import annotations

import re
from typing import Any, Sequence

from ..ner.medical_ner import MedicalNER
from ..nli.conflict import RuleNLI
from ..retrieval.bm25 import tokenize


# ----------------------------------------------------------------------
# 检索指标
# ----------------------------------------------------------------------
def recall_at_k(retrieved_ids: Sequence[str], gold_ids: Sequence[str], k: int) -> float:
    if not gold_ids:
        return float("nan")
    top = set(retrieved_ids[:k])
    return len(top & set(gold_ids)) / len(set(gold_ids))


def precision_at_k(retrieved_ids: Sequence[str], gold_ids: Sequence[str], k: int) -> float:
    top = set(retrieved_ids[:k])
    if not top:
        return 0.0
    return len(top & set(gold_ids)) / k


def hit_at_k(retrieved_ids: Sequence[str], gold_ids: Sequence[str], k: int) -> float:
    if not gold_ids:
        return float("nan")
    return 1.0 if set(retrieved_ids[:k]) & set(gold_ids) else 0.0


def mrr(retrieved_ids: Sequence[str], gold_ids: Sequence[str]) -> float:
    if not gold_ids:
        return float("nan")
    gold = set(gold_ids)
    for rank, cid in enumerate(retrieved_ids, 1):
        if cid in gold:
            return 1.0 / rank
    return 0.0


def top1_accuracy(retrieved_ids: Sequence[str], gold_ids: Sequence[str]) -> float:
    if not gold_ids or not retrieved_ids:
        return float("nan")
    return 1.0 if retrieved_ids[0] in set(gold_ids) else 0.0


def context_precision(retrieved_ids: Sequence[str], gold_ids: Sequence[str]) -> float:
    """RAGAS context precision：Σ (precision@k × rel_k) / |gold ∩ retrieved|。"""
    if not gold_ids:
        return float("nan")
    gold = set(gold_ids)
    hits = [1 if cid in gold else 0 for cid in retrieved_ids]
    n_hit = sum(hits)
    if n_hit == 0:
        return 0.0
    weighted = 0.0
    seen = 0
    for k, h in enumerate(hits, 1):
        if h:
            seen += 1
            weighted += seen / k
    return weighted / n_hit


def context_recall(retrieved_ids: Sequence[str], gold_ids: Sequence[str]) -> float:
    return recall_at_k(retrieved_ids, gold_ids, len(retrieved_ids))


# ----------------------------------------------------------------------
# 生成指标（离线规则版）
# ----------------------------------------------------------------------
_CITATION_RE = re.compile(r"〔\d+〕|\[[\w#]+\]")


def split_claims(answer: str) -> list[str]:
    """把答案拆成原子陈述句（去掉引用标记）。"""
    text = _CITATION_RE.sub("", answer)
    parts = re.split(r"[。！？；\n]", text)
    claims = [p.strip() for p in parts if len(p.strip()) >= 8]
    return claims


def _bigrams(text: str) -> set[str]:
    chars = re.sub(r"\s+", "", text)
    return {chars[i:i + 2] for i in range(len(chars) - 1)}


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


class FaithfulnessScorer:
    """忠实度：claim 被上下文支持 = 内容重叠足够高 或 实体全覆盖，且不与上下文矛盾。"""

    def __init__(self, ner: MedicalNER, nli: RuleNLI, overlap_threshold: float = 0.30):
        self.ner = ner
        self.nli = nli
        self.overlap_threshold = overlap_threshold

    def is_supported(self, claim: str, context: str) -> bool:
        c_bigrams = _bigrams(claim)
        ctx_bigrams = _bigrams(context)
        overlap = len(c_bigrams & ctx_bigrams) / max(len(c_bigrams), 1)  # 覆盖率而非 Jaccard
        ents = {e.text for e in self.ner.recognize(claim)}
        ent_covered = all(e in context for e in ents) if ents else True
        contradicted = self.nli.judge(context, claim).is_contradiction
        return (overlap >= self.overlap_threshold and ent_covered and not contradicted)

    def score(self, answer: str, contexts: Sequence[str]) -> tuple[float, dict[str, Any]]:
        claims = split_claims(answer)
        if not claims:
            return (1.0, {"n_claims": 0}) if not answer.strip() else (0.0, {"n_claims": 0})
        context = "\n".join(contexts)
        supported = sum(1 for c in claims if self.is_supported(c, context))
        detail = {
            "n_claims": len(claims),
            "n_supported": supported,
            "unsupported_claims": [c for c in claims if not self.is_supported(c, context)][:5],
        }
        return supported / len(claims), detail


def answer_relevancy(question: str, answer: str, ner: MedicalNER) -> float:
    """离线代理：问题内容词/实体在答案中的覆盖率（0.6词面 + 0.4实体）。"""
    if not answer.strip():
        return 0.0
    q_tokens = {t for t in tokenize(question) if len(t) >= 2}
    a_text = answer
    token_cov = len(q_tokens & set(tokenize(a_text))) / max(len(q_tokens), 1)
    q_ents = {e.text for e in ner.recognize(question)}
    ent_cov = (sum(1 for e in q_ents if e in a_text) / len(q_ents)) if q_ents else token_cov
    return round(min(1.0, 0.6 * token_cov + 0.4 * ent_cov), 4)


class LLMJudge:
    """可选 LLM 裁判（RAGAS 提示风格）。配置了 LLM 时用于 faithfulness/relevancy 复核。"""

    FAITH_PROMPT = (
        "给定上下文与答案，判断答案中每个陈述是否可由上下文推出。"
        "输出 JSON: {\"supported\": <int>, \"total\": <int>}。仅依据上下文判断，不要使用外部知识。\n\n"
        "【上下文】\n{context}\n\n【答案】\n{answer}\n"
    )
    RELEV_PROMPT = (
        "给定问题与答案，评估答案与问题的相关程度（0~1，两位小数）。"
        "输出 JSON: {\"score\": <float>}。\n\n【问题】{question}\n\n【答案】{answer}\n"
    )

    def __init__(self, llm):
        self.llm = llm

    def faithfulness(self, answer: str, contexts: Sequence[str]) -> float | None:
        try:
            out = self.llm.complete_json(
                "你是严格的评估器。", self.FAITH_PROMPT.format(context="\n".join(contexts)[:6000], answer=answer[:3000])
            )
            total = max(int(out.get("total", 0)), 1)
            return min(1.0, int(out.get("supported", 0)) / total)
        except Exception:  # noqa: BLE001
            return None

    def relevancy(self, question: str, answer: str) -> float | None:
        try:
            out = self.llm.complete_json(
                "你是严格的评估器。", self.RELEV_PROMPT.format(question=question, answer=answer[:3000])
            )
            return float(out.get("score", 0.0))
        except Exception:  # noqa: BLE001
            return None


def nanmean(values: Sequence[float]) -> float:
    vals = [v for v in values if v == v]  # 过滤 NaN
    return round(sum(vals) / len(vals), 4) if vals else float("nan")
