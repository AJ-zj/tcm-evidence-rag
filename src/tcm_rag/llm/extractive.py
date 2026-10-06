"""离线抽取式回答引擎：无 LLM 时从证据中抽取并组织答案（天然高忠实度）。

流程：证据句抽取 → 查询相关度打分（实体重叠 + 词面重叠 + 来源先验）→
去重排序 → 组装带引用标记〔n〕的答案。同时输出 claim 列表，
供评估层做忠实度（faithfulness）判定。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Sequence

from ..ner.medical_ner import MedicalNER
from ..retrieval.bm25 import tokenize
from ..schema import Evidence


@dataclass
class ExtractiveAnswer:
    answer: str
    claims: list[str] = field(default_factory=list)          # 答案分解出的陈述句
    used_citations: list[str] = field(default_factory=list)  # 引用到的 chunk_id
    grounded: bool = True                                     # 全部来自证据


class ExtractiveAnswerer:
    def __init__(self, ner: MedicalNER, max_claims: int = 8, max_answer_chars: int = 600):
        self.ner = ner
        self.max_claims = max_claims
        self.max_answer_chars = max_answer_chars

    # ------------------------------------------------------------------
    @staticmethod
    def _split_evidence_sentences(text: str) -> list[str]:
        # 去掉块级标题前缀〈…〉
        text = re.sub(r"^〈[^〉]*〉\s*", "", text.strip())
        parts = re.split(r"(?<=[。！？；])|(?<=\n)", text)
        out = []
        for p in parts:
            p = p.strip()
            if len(p) >= 6:
                out.append(p)
        return out

    def _sentence_score(self, query_tokens: set[str], query_ents: set[str], sent: str) -> float:
        s_tokens = set(tokenize(sent))
        s_ents = {e.text for e in self.ner.recognize(sent)}
        token_overlap = len(query_tokens & s_tokens) / max(len(query_tokens), 1)
        ent_overlap = len(query_ents & s_ents) / max(len(query_ents), 1) if query_ents else 0.0
        return 0.55 * ent_overlap + 0.45 * token_overlap

    # ------------------------------------------------------------------
    def compose(
        self,
        query: str,
        evidences: Sequence[Evidence],
        conflict_notes: Sequence[str] = (),
    ) -> ExtractiveAnswer:
        if not evidences:
            return ExtractiveAnswer(answer="", claims=[], used_citations=[], grounded=False)

        query_tokens = set(tokenize(query))
        query_ents = {e.text for e in self.ner.recognize(query)}

        scored: list[tuple[float, str, Evidence]] = []
        for ev in evidences:
            prior = 1.0 - min(ev.chunk.ocr_noise, 0.2)  # 噪声证据降权
            base = 0.5 + 0.5 * ev.score                  # 检索相关性权重
            for sent in self._split_evidence_sentences(ev.chunk.text):
                s = self._sentence_score(query_tokens, query_ents, sent) * base * prior
                scored.append((s, sent, ev))

        scored.sort(key=lambda x: x[0], reverse=True)

        # 去重（相同句子只留最高分）+ 截断
        seen: set[str] = set()
        claims: list[tuple[str, Evidence]] = []
        for s, sent, ev in scored:
            key = re.sub(r"\s+", "", sent)[:40]
            if key in seen:
                continue
            seen.add(key)
            claims.append((sent, ev))
            if len(claims) >= self.max_claims:
                break

        # 行级高亮：每条证据贡献的最高分句子作为 quote 回填（前端高亮用）
        for sent, ev in claims:
            if not ev.quote:
                ev.quote = sent

        # 组织答案：按证据顺序分组输出，附引用标记
        citation_ids: list[str] = []
        blocks: list[str] = []
        total = 0
        by_ev: dict[str, list[str]] = {}
        for sent, ev in claims:
            by_ev.setdefault(ev.chunk.chunk_id, []).append(sent)
        for ev in evidences:
            cid = ev.chunk.chunk_id
            if cid not in by_ev:
                continue
            citation_ids.append(cid)
            mark = f"〔{citation_ids.index(cid) + 1}〕"
            text = "".join(by_ev[cid])
            if total + len(text) > self.max_answer_chars:
                break
            blocks.append(f"{mark}{text}")
            total += len(text)

        answer = "\n".join(blocks)
        if conflict_notes:
            answer += "\n\n⚠ 证据间存在不一致：" + "；".join(conflict_notes)

        # 开头一句直答（取最高分 claim 的开头）
        if claims:
            top_sent = claims[0][0].rstrip("。；")
            lead = f"根据检索到的证据：{top_sent}。\n\n" if not answer.startswith(top_sent[:10]) else ""
            answer = lead + answer

        return ExtractiveAnswer(
            answer=answer,
            claims=[s for s, _ in claims],
            used_citations=citation_ids,
            grounded=True,
        )
