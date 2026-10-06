"""Token 估算与中文句子切分工具。

BGE (WordPiece) 对中文近似 1 字 = 1 token，对 ASCII 单词约 ceil(len/4) token。
这里用确定性估算，避免引入 tiktoken 等额外依赖；用于切分块大小控制足够精确。
"""
from __future__ import annotations

import math
import re

CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
CJK_PUNCT_RE = re.compile(r"[\u3000-\u303f\uff00-\uffef]")
ASCII_WORD_RE = re.compile(r"[A-Za-z0-9]+")

# 句子边界：中文句号/问号/叹号/分号 + 英文对应符号；保留分隔符
SENT_END_RE = re.compile(r"(?<=[。！？；!?;])|(?<=\.{3})(?=\s|$)|\n+")


def count_tokens(text: str) -> int:
    """估算文本的 WordPiece token 数（中文 1 字 1 token，ASCII 词约 len/4）。"""
    if not text:
        return 0
    cjk = len(CJK_RE.findall(text))
    cjk_punct = len(CJK_PUNCT_RE.findall(text))
    ascii_words = sum(max(1, math.ceil(len(m.group(0)) / 4)) for m in ASCII_WORD_RE.finditer(text))
    return cjk + cjk_punct + ascii_words


def truncate_to_tokens(text: str, max_tokens: int) -> str:
    """按 token 估算截断文本（用于基线截断策略与回退硬切分）。"""
    if count_tokens(text) <= max_tokens:
        return text
    # 中文占比高时 token≈字符数，直接按字符逐步收缩
    lo, hi = 0, len(text)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if count_tokens(text[:mid]) <= max_tokens:
            lo = mid
        else:
            hi = mid - 1
    return text[:lo]


def split_sentences(text: str) -> list[str]:
    """切分句子，保留句末标点；换行视为边界。"""
    parts: list[str] = []
    buf: list[str] = []
    for ch in text:
        buf.append(ch)
        if ch in "。！？；!?;\n":
            parts.append("".join(buf))
            buf = []
    if buf:
        parts.append("".join(buf))
    return [p for p in (s.strip() for s in parts) if p]


def split_paragraphs(text: str) -> list[str]:
    """按空行/单换行切分段落，保留段落文本。"""
    paras = re.split(r"\n\s*\n|\n", text)
    return [p.strip() for p in paras if p.strip()]
