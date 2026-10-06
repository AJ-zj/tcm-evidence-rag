"""BM25 稀疏检索（jieba 分词，中文单字回退增强）。"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Sequence

import jieba

from ..schema import Chunk

_STOPWORDS = {"的", "了", "是", "在", "和", "与", "及", "或", "而", "等", "为", "对",
              "之", "其", "者", "也", "于", "以", "则", "乃", "但", "且", "夫", "盖"}


def tokenize(text: str) -> list[str]:
    """jieba 精确模式 + 中医词表挂载；单字补充提高古籍召回。"""
    words = [w.strip() for w in jieba.cut(text) if w.strip()]
    tokens: list[str] = []
    for w in words:
        if w in _STOPWORDS:
            continue
        tokens.append(w)
    return tokens


class BM25Index:
    """rank_bm25 封装：语料级倒排 + 持久化（pickle）。"""

    def __init__(self, chunks: Sequence[Chunk] | None = None):
        self._chunk_ids: list[str] = []
        self._bm25 = None
        if chunks:
            self.build(chunks)

    def build(self, chunks: Sequence[Chunk]) -> None:
        from rank_bm25 import BM25Okapi

        self._chunk_ids = [c.chunk_id for c in chunks]
        corpus = [tokenize(c.text) for c in chunks]
        self._bm25 = BM25Okapi(corpus)

    def search(self, query: str, top_k: int = 30) -> list[tuple[str, float]]:
        if self._bm25 is None or not self._chunk_ids:
            return []
        scores = self._bm25.get_scores(tokenize(query))
        top_k = min(top_k, len(scores))
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [(self._chunk_ids[i], float(scores[i])) for i in ranked if scores[i] > 0]

    def load_jieba_dict(self, terms: Sequence[str]) -> None:
        """把医药词表挂进 jieba，避免"桂枝汤"被切开。"""
        for t in terms:
            if len(t) >= 2:
                jieba.add_word(t, freq=100000)

    # ------------------------------------------------------------------
    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "wb") as f:
            pickle.dump({"ids": self._chunk_ids, "bm25": self._bm25}, f)

    @classmethod
    def load(cls, path: str | Path) -> "BM25Index":
        with open(path, "rb") as f:
            data = pickle.load(f)
        obj = cls()
        obj._chunk_ids = data["ids"]
        obj._bm25 = data["bm25"]
        return obj
