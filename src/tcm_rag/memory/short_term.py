"""短期记忆：滑动窗口 + 摘要压缩，重要度由医疗 NER 加权并随轮次衰减。

机制：
- 每轮对话计算 base_importance = NER 医疗实体加权重要度
- 有效重要度 effective_importance = base_importance × decay^(age轮次)
- 最近 window_turns 轮保留原文；超出窗口的轮次按有效重要度压缩进滚动摘要
  （摘要为抽取式：优先保留高重要度轮次的实体句；配置了 LLM 时可用 LLM 摘要）
- 窗口内总 token 超过预算时，压缩窗口内有效重要度最低的轮次
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable

from ..ner.medical_ner import MedicalNER
from ..utils.tokens import count_tokens, split_sentences


@dataclass
class Turn:
    idx: int
    role: str                      # user | assistant | system
    text: str
    entities: dict[str, list[str]] = field(default_factory=dict)
    base_importance: float = 0.0
    created_at: float = field(default_factory=time.time)
    compressed: bool = False       # 已被压缩进摘要（原文不再进上下文）

    def to_dict(self) -> dict[str, Any]:
        return {
            "idx": self.idx, "role": self.role, "text": self.text,
            "entities": self.entities, "base_importance": self.base_importance,
            "created_at": self.created_at, "compressed": self.compressed,
        }


class ShortTermMemory:
    def __init__(
        self,
        ner: MedicalNER,
        window_turns: int = 6,
        max_window_tokens: int = 900,
        importance_decay: float = 0.82,
        summary_max_tokens: int = 220,
        llm_summarizer: Callable[[str], str] | None = None,
    ):
        self.ner = ner
        self.window_turns = window_turns
        self.max_window_tokens = max_window_tokens
        self.decay = importance_decay
        self.summary_max_tokens = summary_max_tokens
        self.llm_summarizer = llm_summarizer

        self.turns: list[Turn] = []
        self.summary: str = ""
        self._next_idx = 0
        self.stats = {"turns_total": 0, "turns_compressed": 0, "summary_rebuilds": 0}

    # ------------------------------------------------------------------
    def add_turn(self, role: str, text: str) -> Turn:
        entities = self.ner.entity_types(text)
        importance = self.ner.importance(text)
        turn = Turn(
            idx=self._next_idx,
            role=role,
            text=text,
            entities=entities,
            base_importance=importance,
        )
        self._next_idx += 1
        self.turns.append(turn)
        self.stats["turns_total"] += 1
        self._maybe_compress()
        return turn

    def effective_importance(self, turn: Turn, now_idx: int | None = None) -> float:
        now = self._next_idx - 1 if now_idx is None else now_idx
        age = max(0, now - turn.idx)
        return turn.base_importance * (self.decay ** age)

    # ------------------------------------------------------------------
    def _active_turns(self) -> list[Turn]:
        return [t for t in self.turns if not t.compressed]

    def _maybe_compress(self) -> None:
        active = self._active_turns()
        if len(active) <= self.window_turns:
            window = active
        else:
            # 超出滑动窗口的旧轮次进入压缩候选
            overflow = active[: len(active) - self.window_turns]
            window = active[len(active) - self.window_turns:]
            self._compress(overflow)
            active = window

        # 窗口内 token 超预算 → 压缩有效重要度最低的轮次（保留最近一轮）
        while len(active) > 1 and sum(count_tokens(t.text) for t in active) > self.max_window_tokens:
            now = self._next_idx - 1
            weakest = min(active[:-1], key=lambda t: self.effective_importance(t, now))
            self._compress([weakest])
            active = [t for t in active if t.idx != weakest.idx]

    def _compress(self, turns: list[Turn]) -> None:
        if not turns:
            return
        for t in turns:
            t.compressed = True
        self.stats["turns_compressed"] += len(turns)

        # 抽取式摘要：按有效重要度排序轮次，保留实体句
        now = self._next_idx - 1
        ranked = sorted(turns, key=lambda t: self.effective_importance(t, now), reverse=True)
        pieces: list[str] = []
        for t in ranked:
            sents = split_sentences(t.text)
            ent_sents = [s for s in sents if any(v for v in t.entities.values())] or sents[:1]
            snippet = "".join(ent_sents[:2])[:120]
            tag = "患" if t.role == "user" else "医"
            pieces.append(f"[{tag}]{snippet}")
        addition = " ".join(pieces)

        if self.llm_summarizer:
            try:
                merged = self.llm_summarizer((self.summary + "\n" + addition).strip())
                self.summary = merged[: self.summary_max_tokens * 2]
                self.stats["summary_rebuilds"] += 1
                return
            except Exception:  # noqa: BLE001  LLM 摘要失败回退抽取式
                pass
        combined = (self.summary + " " + addition).strip()
        self.summary = self._truncate_summary(combined)
        self.stats["summary_rebuilds"] += 1

    def _truncate_summary(self, text: str) -> str:
        if count_tokens(text) <= self.summary_max_tokens:
            return text
        # 保留最近（末尾）的摘要内容
        parts = text.split(" ")
        out: list[str] = []
        tokens = 0
        for p in reversed(parts):
            t = count_tokens(p)
            if tokens + t > self.summary_max_tokens:
                break
            out.insert(0, p)
            tokens += t
        return " ".join(out)

    # ------------------------------------------------------------------
    def context(self, max_tokens: int | None = None) -> str:
        """组装供 Agent 使用的上下文：滚动摘要 + 滑动窗口原文。"""
        blocks: list[str] = []
        if self.summary:
            blocks.append(f"【历史摘要】{self.summary}")
        active = self._active_turns()
        if active:
            lines = []
            for t in active:
                role = "患者" if t.role == "user" else ("医师" if t.role == "assistant" else "系统")
                lines.append(f"{role}: {t.text}")
            blocks.append("【近期对话】\n" + "\n".join(lines))
        text = "\n\n".join(blocks)
        if max_tokens and count_tokens(text) > max_tokens:
            from ..utils.tokens import truncate_to_tokens

            text = truncate_to_tokens(text, max_tokens)
        return text

    def recent_entities(self, last_n: int = 3, role: str | None = None) -> dict[str, list[str]]:
        """最近 N 轮出现过的实体（问题改写/检索扩展用）。

        role=None 取全部轮次；指定 "user"/"assistant" 时只取对应角色的轮次——
        追问改写需区分"用户在谈的对象"（user 轮，多为症状主诉）与
        "上一轮给出的结论"（assistant 轮，多为证型/方剂实体）。
        """
        merged: dict[str, list[str]] = {}
        turns = self._active_turns()
        if role is not None:
            turns = [t for t in turns if t.role == role][-last_n:]
        else:
            turns = turns[-last_n * 2:]
        for t in turns:
            for etype, terms in t.entities.items():
                bucket = merged.setdefault(etype, [])
                for term in terms:
                    if term not in bucket:
                        bucket.append(term)
        return merged

    def to_dict(self) -> dict[str, Any]:
        return {
            "summary": self.summary,
            "turns": [t.to_dict() for t in self.turns],
            "next_idx": self._next_idx,
            "stats": dict(self.stats),
        }

    def load_state(self, state: dict[str, Any]) -> None:
        """从持久化状态恢复（会话续聊）。"""
        self.summary = state.get("summary", "")
        self._next_idx = int(state.get("next_idx", 0))
        self.stats.update(state.get("stats", {}))
        self.turns = [
            Turn(
                idx=int(d.get("idx", i)),
                role=d.get("role", "user"),
                text=d.get("text", ""),
                entities=d.get("entities", {}),
                base_importance=float(d.get("base_importance", 0.0)),
                created_at=float(d.get("created_at", time.time())),
                compressed=bool(d.get("compressed", False)),
            )
            for i, d in enumerate(state.get("turns", []))
        ]
