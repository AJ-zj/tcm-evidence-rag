"""会话编排：短期/长期记忆 + Agent 的多轮对话闭环。

会话状态持久化到 SQLite sessions 表（与分块/长期记忆同库，双库设计的结构化侧）。
"""
from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Any

from ..memory.long_term import LongTermMemory
from ..memory.short_term import ShortTermMemory
from ..ner.medical_ner import MedicalNER
from ..schema import AgentAnswer
from ..utils.tokens import split_sentences
from .react_agent import EvidenceAgent

_SESSION_SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    session_id TEXT PRIMARY KEY,
    state      TEXT NOT NULL,
    updated_at REAL NOT NULL
);
"""


class ChatSession:
    """一个会话 = 一份短期记忆 + 对共享长期记忆的读写。"""

    def __init__(
        self,
        session_id: str,
        agent: EvidenceAgent,
        ner: MedicalNER,
        long_term: LongTermMemory | None = None,
        stm_kwargs: dict[str, Any] | None = None,
        persist_db: str | Path | None = None,
        persist_dir: str | Path | None = None,   # 兼容旧的 json 文件模式
    ):
        self.session_id = session_id
        self.agent = agent
        self.ner = ner
        self.long_term = long_term
        self.persist_db = Path(persist_db) if persist_db else None
        self.persist_path = (
            Path(persist_dir) / f"{session_id}.json" if persist_dir else None
        )
        self.short_term = ShortTermMemory(ner=ner, **(stm_kwargs or {}))
        self.history: list[AgentAnswer] = []
        self._restore()

    # ------------------------------------------------------------------
    def _restore(self) -> None:
        try:
            if self.persist_db is not None:
                with sqlite3.connect(str(self.persist_db)) as conn:
                    conn.executescript(_SESSION_SCHEMA)
                    row = conn.execute(
                        "SELECT state FROM sessions WHERE session_id=?", (self.session_id,)
                    ).fetchone()
                state = json.loads(row[0]) if row else None
            elif self.persist_path is not None and self.persist_path.exists():
                state = json.loads(self.persist_path.read_text(encoding="utf-8"))
            else:
                state = None
            if state:
                self.short_term.load_state(state.get("short_term", {}))
        except (json.JSONDecodeError, sqlite3.DatabaseError, OSError):
            pass  # 损坏的存档按新会话处理

    def persist(self) -> None:
        payload = json.dumps({"short_term": self.short_term.to_dict()}, ensure_ascii=False)
        if self.persist_db is not None:
            self.persist_db.parent.mkdir(parents=True, exist_ok=True)
            with sqlite3.connect(str(self.persist_db)) as conn:
                conn.executescript(_SESSION_SCHEMA)
                conn.execute(
                    "INSERT OR REPLACE INTO sessions(session_id, state, updated_at) VALUES (?,?,?)",
                    (self.session_id, payload, time.time()),
                )
        elif self.persist_path is not None:
            self.persist_path.parent.mkdir(parents=True, exist_ok=True)
            self.persist_path.write_text(payload, encoding="utf-8")

    # ------------------------------------------------------------------
    def _memory_candidates(self, message: str, result: AgentAnswer) -> list[str]:
        """从用户陈述与答案结论中抽取值得长期保存的事实句。"""
        candidates: list[str] = []
        for sent in split_sentences(message):
            if self.ner.recognize(sent):        # 含医疗实体的用户陈述（主诉/病史/偏好）
                candidates.append(sent)
        # 答案中最关键的一条陈述（含引用溯源）
        for ev in result.evidences[:1]:
            first = split_sentences(ev.get("text", ""))
            if first:
                candidates.append(f"关于「{result.question[:20]}」：{first[0]}")
        return candidates[:4]

    # ------------------------------------------------------------------
    def chat(self, message: str, decision_mode: str = "react") -> AgentAnswer:
        self.short_term.add_turn("user", message)
        result = self.agent.answer(
            message,
            decision_mode=decision_mode,
            short_term=self.short_term,
            long_term=self.long_term,
        )
        self.short_term.add_turn("assistant", result.answer)

        write_results = []
        if self.long_term is not None:
            for cand in self._memory_candidates(message, result):
                wr = self.long_term.write(cand, source=f"session:{self.session_id}")
                write_results.append({"content": cand[:60], "action": wr.action, "detail": wr.detail})
        result.memory_used = {**result.memory_used, "ltm_writes": write_results}
        self.history.append(result)
        self.persist()   # 会话状态持久化（重启服务后续聊不丢上下文）
        return result

    # ------------------------------------------------------------------
    def state(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "short_term": self.short_term.to_dict(),
            "long_term": self.long_term.to_dict() if self.long_term else None,
            "n_turns": len(self.history),
        }
