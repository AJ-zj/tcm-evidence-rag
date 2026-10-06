"""核心数据结构：Document / Chunk / Evidence / AgentAnswer。"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any


@dataclass
class Document:
    """解析后的原始文档（切分前）。"""

    doc_id: str
    title: str
    text: str
    source_type: str          # classic | textbook | case | qa | pdf_scan
    source_path: str = ""
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Chunk:
    """索引与检索的最小单元。"""

    chunk_id: str
    doc_id: str
    text: str                 # 清洗后正文（含标题上下文前缀）
    title_path: list[str]     # 层级标题路径，如 ["伤寒论", "辨太阳病脉证并治", "第12条"]
    source_type: str
    seq: int = 0              # 块在文档中的序号
    char_start: int = 0       # 在原文中的字符偏移（追溯用）
    char_end: int = 0
    n_tokens: int = 0
    ocr_noise: float = 0.0    # OCR 噪声率估计 [0,1]
    split_method: str = "semantic"   # semantic | sentence | token_fallback
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def title(self) -> str:
        return " > ".join(self.title_path) if self.title_path else ""

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["title"] = self.title
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Chunk":
        d = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        return cls(**d)


@dataclass
class Evidence:
    """送入生成与评估的证据条目（带可追溯引用信息）。"""

    chunk: Chunk
    score: float                       # 重排后综合相关性 [0,1]
    channel_scores: dict[str, float] = field(default_factory=dict)  # dense/bm25/rrf/rerank
    quote: str = ""                    # 命中的关键引文片段
    kept: bool = True                  # 是否通过二次过滤
    drop_reason: str = ""

    @property
    def citation(self) -> str:
        return f"[{self.chunk.doc_id}#{self.chunk.seq}] {self.chunk.title}"


@dataclass
class TraceStep:
    """ReAct 轨迹中的一步。"""

    round: int
    thought: str
    action: str
    action_input: dict[str, Any] = field(default_factory=dict)
    observation: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AgentAnswer:
    """Agent 最终输出（含证据引用与决策轨迹，支持可追溯分析）。"""

    question: str
    answer: str
    citations: list[str] = field(default_factory=list)
    evidences: list[dict[str, Any]] = field(default_factory=list)
    confidence: float = 0.0
    consistency: float = 0.0
    coverage: float = 0.0
    refused: bool = False
    refusal_reason: str = ""
    rounds_used: int = 0
    trace: list[TraceStep] = field(default_factory=list)
    memory_used: dict[str, Any] = field(default_factory=dict)
    latency_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["trace"] = [t.to_dict() if isinstance(t, TraceStep) else t for t in self.trace]
        return d
