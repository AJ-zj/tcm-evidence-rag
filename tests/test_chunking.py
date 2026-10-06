"""语义切分器测试：标题切分、块大小上限、Token 回退、信息保留率。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tcm_rag.parsing.chunking import (
    NaiveTruncationChunker,
    SemanticChunker,
    markdown_sections,
    retention_score,
)
from tcm_rag.schema import Document
from tcm_rag.utils.tokens import count_tokens

MD = """# 测试书

前言内容一句。

## 第一章

### 第一节

桂枝汤主之。太阳中风，阳浮而阴弱。

### 第二节

麻黄汤主之。太阳病，头痛发热，身疼腰痛，骨节疼痛，恶风无汗而喘者。

## 第二章

长段落内容。
"""


def test_markdown_sections():
    secs = markdown_sections(MD, default_title="测试书")
    paths = [tuple(s.path) for s in secs if s.content.strip()]
    assert ("测试书", "第一章", "第一节") in paths
    assert ("测试书", "第二章") in paths
    # 前言应挂在根标题下
    assert any("前言" in s.content for s in secs)


def test_semantic_chunk_respects_max_tokens():
    doc = Document(doc_id="d1", title="长文档", text=MD, source_type="classic")
    chunker = SemanticChunker(target_tokens=64, max_tokens=128, overlap_tokens=8)
    chunks = chunker.chunk(doc)
    assert len(chunks) >= 3
    for c in chunks:
        assert c.n_tokens <= 128 + 8, f"块超限: {c.n_tokens}"
        assert c.chunk_id.startswith("d1#c")
    # 标题路径保留
    assert any("第一节" in c.title for c in chunks)


def test_token_fallback_for_huge_sentence():
    huge = "桂枝汤主之" * 400  # 无句读超长句
    doc = Document(doc_id="d2", title="超长句", text=huge, source_type="classic")
    chunker = SemanticChunker(target_tokens=128, max_tokens=256, overlap_tokens=32)
    chunks = chunker.chunk(doc)
    assert len(chunks) > 1
    assert any(c.split_method == "token_fallback" for c in chunks)
    for c in chunks:
        assert count_tokens(c.text) <= 256 + 40


def test_naive_truncation_loses_information():
    long_text = "\n\n".join(f"### 节{i}\n\n这是第{i}节的内容，包含甘草和桂枝汤的论述。" for i in range(120))
    doc = Document(doc_id="d3", title="长文", text=long_text, source_type="textbook")
    naive = NaiveTruncationChunker(max_tokens=512).chunk(doc)
    assert len(naive) == 1
    sem = SemanticChunker(target_tokens=256, max_tokens=480).chunk(doc)
    assert len(sem) > 1

    terms = ["甘草", "桂枝汤"]
    ret_naive = retention_score([doc], {"d3": naive}, terms)
    ret_sem = retention_score([doc], {"d3": sem}, terms)
    assert ret_sem["overall"] > ret_naive["overall"]
    assert ret_sem["overall"] > 0.95


def test_chunk_offsets_within_doc():
    doc = Document(doc_id="d4", title="偏移", text=MD, source_type="classic")
    chunks = SemanticChunker().chunk(doc)
    for c in chunks:
        assert 0 <= c.char_start <= c.char_end <= len(MD) + 1
