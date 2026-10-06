"""语义切分 + Token 回退策略。

切分层次（逐级回退）：
1. 标题结构（Markdown 层级）→ 以语义单元（条文/药味/方剂/医案）为天然边界
2. 段落边界（空行/换行）
3. 句子边界（。！？；等）
4. Token 回退：单句仍超过 max_tokens 时按 token 预算硬切分并保留重叠，
   确保任何文本都不会因 embedding 截断（512 token）而丢失信息

基线对照 NaiveTruncationChunker 模拟"整篇截断"的朴素做法，
retention_score() 度量两种策略下的长文信息保留率。

注：Chunk 的 char_start/char_end 以清洗后的文档文本为基准。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable, Sequence

from ..schema import Chunk, Document
from ..utils.tokens import count_tokens, split_sentences, truncate_to_tokens
from .ocr_clean import CleanResult, OCRTextCleaner

HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
SENT_END_CHARS = "。！？；!?;\n"


@dataclass
class Section:
    """一个标题层级下的正文块。"""

    path: list[str]
    content: str
    start: int
    end: int


@dataclass
class _Unit:
    text: str
    start: int
    end: int
    method: str  # semantic | sentence | token_fallback


def markdown_sections(text: str, default_title: str = "") -> list[Section]:
    """按 Markdown 标题层级切分文档，返回带标题路径的正文段。"""
    lines = text.splitlines(keepends=True)
    sections: list[Section] = []
    heading_stack: list[tuple[int, str]] = []
    buf: list[str] = []
    buf_start = 0
    offset = 0
    doc_title_seen = False

    def flush(end: int) -> None:
        content = "".join(buf)
        if content.strip():
            path = [t for _, t in heading_stack] or ([default_title] if default_title else [])
            sections.append(Section(path=list(path), content=content, start=buf_start, end=end))

    for line in lines:
        m = HEADING_RE.match(line.rstrip("\n"))
        if m:
            flush(offset)
            level = len(m.group(1))
            title = m.group(2).strip()
            if level == 1 and not doc_title_seen:
                default_title = default_title or title
                doc_title_seen = True
            while heading_stack and heading_stack[-1][0] >= level:
                heading_stack.pop()
            heading_stack.append((level, title))
            buf = []
            offset += len(line)
            buf_start = offset
        else:
            buf.append(line)
            offset += len(line)
    flush(offset)
    return sections


def _split_with_offsets(text: str, base_offset: int, sep_re: str) -> list[_Unit]:
    """按正则分隔符切分并记录偏移，分隔符归前一段。"""
    units: list[_Unit] = []
    pos = 0
    for m in re.finditer(sep_re, text):
        end = m.end()
        part = text[pos:end]
        if part.strip():
            units.append(_Unit(part.strip(), base_offset + pos, base_offset + end, "sentence"))
        pos = end
    tail = text[pos:]
    if tail.strip():
        units.append(_Unit(tail.strip(), base_offset + pos, base_offset + len(text), "sentence"))
    return units


def _split_sentences_offsets(text: str, base_offset: int) -> list[_Unit]:
    """句子级切分（保留偏移）。"""
    units: list[_Unit] = []
    start = 0
    for i, ch in enumerate(text):
        if ch in SENT_END_CHARS:
            seg = text[start:i + 1]
            if seg.strip():
                units.append(_Unit(seg.strip(), base_offset + start, base_offset + i + 1, "sentence"))
            start = i + 1
    tail = text[start:]
    if tail.strip():
        units.append(_Unit(tail.strip(), base_offset + start, base_offset + len(text), "sentence"))
    return units


def _token_fallback_split(text: str, base_offset: int, max_tokens: int, overlap_tokens: int) -> list[_Unit]:
    """Token 回退：超长单句按 token 预算硬切，带重叠窗口。"""
    units: list[_Unit] = []
    pos = 0
    while pos < len(text):
        piece = truncate_to_tokens(text[pos:], max_tokens)
        if not piece:
            piece = text[pos:pos + max_tokens]
        units.append(_Unit(piece, base_offset + pos, base_offset + pos + len(piece), "token_fallback"))
        if pos + len(piece) >= len(text):
            break
        # 回退 overlap_tokens 对应的字符数作为下一段起点
        back = len(truncate_to_tokens(piece[-min(len(piece), overlap_tokens * 2):], overlap_tokens)) or 1
        pos = pos + len(piece) - back
    return units


class SemanticChunker:
    """语义优先、Token 回退兜底的分层切分器。"""

    def __init__(
        self,
        target_tokens: int = 256,
        max_tokens: int = 480,
        overlap_tokens: int = 32,
        min_tokens: int = 24,
    ):
        self.target_tokens = target_tokens
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens
        self.min_tokens = min_tokens

    # ------------------------------------------------------------------
    def _section_units(self, section: Section) -> list[_Unit]:
        content = section.content
        lead = len(content) - len(content.lstrip())
        body = content.strip()
        base = section.start + lead
        if count_tokens(body) <= self.max_tokens:
            return [_Unit(body, base, base + len(body), "semantic")]

        # 段落级切分（空行或单换行）
        units: list[_Unit] = []
        para_pos = 0
        for m in re.finditer(r"\n\s*\n|\n", content):
            para = content[para_pos:m.start()]
            if para.strip():
                units.extend(self._split_large(para.strip(), base + para_pos))
            para_pos = m.end()
        tail = content[para_pos:]
        if tail.strip():
            units.extend(self._split_large(tail.strip(), base + para_pos))
        return units

    def _split_large(self, text: str, base_offset: int) -> list[_Unit]:
        if count_tokens(text) <= self.max_tokens:
            return [_Unit(text, base_offset, base_offset + len(text), "semantic")]
        out: list[_Unit] = []
        for sent in _split_sentences_offsets(text, base_offset):
            if count_tokens(sent.text) <= self.max_tokens:
                out.append(sent)
            else:
                out.extend(
                    _token_fallback_split(sent.text, sent.start, self.max_tokens, self.overlap_tokens)
                )
        return out

    # ------------------------------------------------------------------
    def _pack(self, units: Sequence[_Unit], prefix: str) -> list[tuple[str, _Unit, list[_Unit]]]:
        """把单元打包为不超过目标 token 的块；返回 (块文本, 首单元, 成员单元)。"""
        prefix_tokens = count_tokens(prefix)
        budget = max(self.target_tokens - prefix_tokens, self.min_tokens)
        hard = max(self.max_tokens - prefix_tokens, self.min_tokens)

        packed: list[tuple[str, _Unit, list[_Unit]]] = []
        cur: list[_Unit] = []
        cur_tokens = 0
        for u in units:
            ut = count_tokens(u.text)
            if cur and cur_tokens + ut > budget:
                packed.append(("\n".join(x.text for x in cur), cur[0], cur))
                # 句级重叠：把上一块尾部 overlap_tokens 以内的内容带入下一块
                carry: list[_Unit] = []
                carry_tokens = 0
                for x in reversed(cur):
                    t = count_tokens(x.text)
                    if carry_tokens + t > self.overlap_tokens:
                        break
                    carry.insert(0, x)
                    carry_tokens += t
                cur = carry if (carry and carry_tokens <= hard // 2) else []
                cur_tokens = sum(count_tokens(x.text) for x in cur)
            cur.append(u)
            cur_tokens += ut
        if cur:
            packed.append(("\n".join(x.text for x in cur), cur[0], cur))
        return packed

    # ------------------------------------------------------------------
    def chunk(self, doc: Document, cleaner: OCRTextCleaner | None = None) -> list[Chunk]:
        """对单个文档执行：清洗 → 标题切分 → 分层语义切分 → 打包。"""
        if cleaner is not None:
            res: CleanResult = cleaner.clean(doc.text)
        else:
            res = CleanResult(text=doc.text, noise_before=0.0, noise_after=0.0)
        text = res.text
        sections = markdown_sections(text, default_title=doc.title)

        chunks: list[Chunk] = []
        seq = 0
        for section in sections:
            units = self._section_units(section)
            if not units:
                continue
            path = section.path or [doc.title]
            prefix = f"〈{'·'.join(path)}〉\n" if len(path) >= 2 else ""
            for body, first, members in self._pack(units, prefix):
                full = prefix + body
                method_rank = {"semantic": 0, "sentence": 1, "token_fallback": 2}
                method = max((m.method for m in members), key=lambda m: method_rank.get(m, 0))
                chunk = Chunk(
                    chunk_id=f"{doc.doc_id}#c{seq:03d}",
                    doc_id=doc.doc_id,
                    text=full,
                    title_path=list(path),
                    source_type=doc.source_type,
                    seq=seq,
                    char_start=first.start,
                    char_end=members[-1].end,
                    n_tokens=count_tokens(full),
                    ocr_noise=(cleaner.noise_score(body) if cleaner else 0.0),
                    split_method=method,
                    meta={
                        "doc_title": doc.title,
                        "source_path": doc.source_path,
                        "noise_before_clean": res.noise_before,
                        # 传播 QA 文档的关键元数据（评估集构建需要）
                        **{
                            k: doc.meta[k]
                            for k in ("question", "answer", "label", "related_diseases", "score")
                            if k in doc.meta
                        },
                    },
                )
                chunks.append(chunk)
                seq += 1
        return chunks

    def chunk_many(
        self, docs: Iterable[Document], cleaner: OCRTextCleaner | None = None
    ) -> list[Chunk]:
        out: list[Chunk] = []
        for d in docs:
            out.extend(self.chunk(d, cleaner))
        return out


class NaiveTruncationChunker:
    """基线：整篇按 embedding 上限截断（超长部分信息全部丢失）。"""

    def __init__(self, max_tokens: int = 512):
        self.max_tokens = max_tokens

    def chunk(self, doc: Document, cleaner: OCRTextCleaner | None = None) -> list[Chunk]:
        text = cleaner.clean(doc.text).text if cleaner else doc.text
        truncated = truncate_to_tokens(text, self.max_tokens)
        return [
            Chunk(
                chunk_id=f"{doc.doc_id}#c000",
                doc_id=doc.doc_id,
                text=truncated,
                title_path=[doc.title],
                source_type=doc.source_type,
                seq=0,
                char_start=0,
                char_end=len(truncated),
                n_tokens=count_tokens(truncated),
                split_method="naive_truncation",
                meta={"doc_title": doc.title, "source_path": doc.source_path},
            )
        ]


# ----------------------------------------------------------------------
def _normalize_for_match(text: str) -> str:
    return re.sub(r"\s+", "", text)


def retention_score(
    docs: Sequence[Document],
    chunks_by_doc: dict[str, list[Chunk]],
    entity_terms: Iterable[str] = (),
) -> dict:
    """信息保留率：原文中"含医药实体的信息句"完整出现在某个块内的比例。

    entity_terms 为空时统计全部句子。用于对比语义切分与朴素截断。
    """
    terms = [t for t in entity_terms if len(t) >= 2]
    total = 0
    retained = 0
    per_doc: dict[str, float] = {}
    for doc in docs:
        sentences = split_sentences(doc.text)
        if terms:
            info_sents = [s for s in sentences if any(t in s for t in terms)]
        else:
            info_sents = sentences
        if not info_sents:
            continue
        blob = _normalize_for_match(
            "\n".join(c.text for c in chunks_by_doc.get(doc.doc_id, []))
        )
        hit = sum(1 for s in info_sents if _normalize_for_match(s) in blob)
        per_doc[doc.doc_id] = round(hit / len(info_sents), 4)
        total += len(info_sents)
        retained += hit
    overall = round(retained / total, 4) if total else 0.0
    return {"overall": overall, "total_info_sentences": total, "retained": retained, "per_doc": per_doc}
