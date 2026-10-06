"""二次证据过滤：在重排之后、生成之前对证据做质量把关。

过滤维度：
1. 相关性下限：重排分低于阈值的证据直接丢弃（抑制低价值证据）
2. OCR 噪声上限：残留噪声率过高的块不作为证据（抑制 OCR 错误）
3. 长度下限：过短碎片（目录行、孤立字段名）不构成有效证据
4. 近重复去重：字符 bigram Jaccard 高于阈值的重复证据只保留最优一条
   （跨来源重复时优先保留噪声更低、来源先验更高者）

被丢弃的证据保留 drop_reason，供可解释性追溯。
"""
from __future__ import annotations

from typing import Sequence

from ..schema import Chunk, Evidence

SOURCE_RANK = {"textbook": 0, "classic": 1, "case": 2, "pdf": 3, "qa": 4, "pdf_scan": 5, "ocr_scan": 6}


def _bigrams(text: str) -> set[str]:
    chars = "".join(text.split())
    return {chars[i:i + 2] for i in range(len(chars) - 1)}


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


class SecondaryEvidenceFilter:
    def __init__(
        self,
        enabled: bool = True,
        min_rel_score: float = 0.30,
        ocr_noise_max: float = 0.18,
        min_chars: int = 24,
        dedup_jaccard: float = 0.72,
        final_top_k: int = 5,
    ):
        self.enabled = enabled
        self.min_rel_score = min_rel_score
        self.ocr_noise_max = ocr_noise_max
        self.min_chars = min_chars
        self.dedup_jaccard = dedup_jaccard
        self.final_top_k = final_top_k

    # ------------------------------------------------------------------
    def _dedup_key_sort(self, ev: Evidence) -> tuple:
        c: Chunk = ev.chunk
        return (
            -round(ev.score, 4),
            c.ocr_noise,
            SOURCE_RANK.get(c.source_type, 9),
        )

    def apply(
        self, evidences: Sequence[Evidence], limit: int | None = None
    ) -> tuple[list[Evidence], list[Evidence]]:
        """返回 (保留证据, 丢弃证据[含原因])，按重排分降序。limit 缺省用 final_top_k。"""
        limit = limit or self.final_top_k
        if not self.enabled:
            kept = sorted(evidences, key=lambda e: e.score, reverse=True)[:limit]
            return kept, []

        kept: list[Evidence] = []
        dropped: list[Evidence] = []
        kept_bigrams: list[set[str]] = []

        for ev in sorted(evidences, key=self._dedup_key_sort):
            text = ev.chunk.text
            reason = ""
            if len("".join(text.split())) < self.min_chars:
                reason = f"too_short(<{self.min_chars}chars)"
            elif ev.chunk.ocr_noise > self.ocr_noise_max:
                reason = f"ocr_noise({ev.chunk.ocr_noise:.2f}>{self.ocr_noise_max})"
            elif ev.score < self.min_rel_score:
                reason = f"low_relevance({ev.score:.2f}<{self.min_rel_score})"
            else:
                bg = _bigrams(text)
                for prev_bg, prev_ev in zip(kept_bigrams, kept):
                    if _jaccard(bg, prev_bg) > self.dedup_jaccard:
                        reason = f"duplicate_of({prev_ev.chunk.chunk_id})"
                        break
            if reason:
                ev.kept = False
                ev.drop_reason = reason
                dropped.append(ev)
            else:
                ev.kept = True
                kept.append(ev)
                kept_bigrams.append(_bigrams(text))
            if len(kept) >= limit:
                break

        return kept, dropped
