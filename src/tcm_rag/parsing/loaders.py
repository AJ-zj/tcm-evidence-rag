"""多格式文档加载器：Markdown / 教材JSON / QA-JSONL / PDF / TXT。

统一输出 Document（见 schema.py），教材 JSON 与 QA JSONL 会被展平为
带层级标题的 Markdown 风格文本，使后续语义切分器可以统一处理。
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable

from ..schema import Document


def _doc_id_from_path(path: Path, prefix: str = "") -> str:
    stem = path.stem
    h = hashlib.md5(str(path).encode("utf-8")).hexdigest()[:6]
    return f"{prefix}{stem}_{h}" if prefix else f"{stem}_{h}"


def load_markdown(path: str | Path, source_type: str = "classic", meta: dict | None = None) -> Document:
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    title = p.stem
    for line in text.splitlines():
        if line.startswith("# "):
            title = line[2:].strip()
            break
    return Document(
        doc_id=_doc_id_from_path(p),
        title=title,
        text=text,
        source_type=source_type,
        source_path=str(p),
        meta=meta or {},
    )


def load_json_textbook(path: str | Path, meta: dict | None = None) -> Document:
    """加载教材 JSON（book/chapters/records/fields 结构），展平为 Markdown 文本。"""
    p = Path(path)
    data = json.loads(p.read_text(encoding="utf-8"))
    book = data.get("book", p.stem)
    source_type = data.get("source_type", "textbook")
    lines: list[str] = [f"# {book}", ""]
    n_records = 0
    for chapter in data.get("chapters", []):
        lines.append(f"## {chapter.get('title', '未分章')}")
        lines.append("")
        for record in chapter.get("records", []):
            n_records += 1
            lines.append(f"### {record.get('term', '未命名条目')}")
            lines.append("")
            for key, value in record.get("fields", {}).items():
                lines.append(f"【{key}】{value}")
            lines.append("")
    text = "\n".join(lines)
    return Document(
        doc_id=_doc_id_from_path(p),
        title=book,
        text=text,
        source_type=source_type,
        source_path=str(p),
        meta={"n_chapters": len(data.get("chapters", [])), "n_records": n_records, **(meta or {})},
    )


def load_jsonl_qa(path: str | Path, max_records: int | None = None, meta: dict | None = None) -> list[Document]:
    """加载 QA JSONL（Huatuo26M-Lite 导出格式），每条问答一个 Document。"""
    p = Path(path)
    docs: list[Document] = []
    with open(p, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_records is not None and i >= max_records:
                break
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            q = row["question"].strip()
            a = row["answer"].strip()
            doc_id = f"huatuo_qa_{row.get('id', i)}"
            title = q[:24] + ("…" if len(q) > 24 else "")
            text = f"# 医案问答：{title}\n\n【患者问】{q}\n\n【医师答】{a}\n"
            docs.append(
                Document(
                    doc_id=doc_id,
                    title=title,
                    text=text,
                    source_type="qa",
                    source_path=str(p),
                    meta={
                        "label": row.get("label", ""),
                        "related_diseases": row.get("related_diseases", ""),
                        "score": row.get("score", 0),
                        "question": q,
                        "answer": a,
                        **(meta or {}),
                    },
                )
            )
    return docs


def load_pdf(path: str | Path, source_type: str = "pdf_scan", meta: dict | None = None) -> Document:
    """加载 PDF（扫描件 OCR 结果或 reportlab 生成件），逐页提取文本。"""
    from pypdf import PdfReader

    p = Path(path)
    reader = PdfReader(str(p))
    pages: list[str] = []
    for page in reader.pages:
        pages.append(page.extract_text() or "")
    text = "\n\n".join(f"〔第{i + 1}页〕\n{t}" for i, t in enumerate(pages))
    return Document(
        doc_id=_doc_id_from_path(p),
        title=p.stem,
        text=text,
        source_type=source_type,
        source_path=str(p),
        meta={"n_pages": len(reader.pages), **(meta or {})},
    )


def load_txt(path: str | Path, source_type: str = "classic", meta: dict | None = None) -> Document:
    p = Path(path)
    return Document(
        doc_id=_doc_id_from_path(p),
        title=p.stem,
        text=p.read_text(encoding="utf-8"),
        source_type=source_type,
        source_path=str(p),
        meta=meta or {},
    )


def load_any(path: str | Path, source_type: str | None = None, meta: dict | None = None) -> list[Document]:
    """按扩展名分发加载；JSONL 返回多个 Document，其余返回单元素列表。"""
    p = Path(path)
    ext = p.suffix.lower()
    if ext in {".md", ".markdown"}:
        if source_type:
            st = source_type
        elif "ocr" in p.parts:
            st = "ocr_scan"          # OCR 噪声扫描样本
        elif "cases" in p.parts:
            st = "case"              # 医案
        else:
            st = "classic"
        return [load_markdown(p, source_type=st, meta=meta)]
    if ext == ".json":
        return [load_json_textbook(p, meta=meta)]
    if ext == ".jsonl":
        return load_jsonl_qa(p, meta=meta)
    if ext == ".pdf":
        st = source_type or ("ocr_scan" if "pdf" in p.parts or "ocr" in p.parts else "pdf")
        return [load_pdf(p, source_type=st, meta=meta)]
    if ext == ".txt":
        return [load_txt(p, source_type=source_type or "classic", meta=meta)]
    raise ValueError(f"不支持的文件格式: {p}")


def discover_corpus_files(corpus_dir: str | Path, exts: Iterable[str]) -> list[Path]:
    """递归发现语料目录下所有支持格式的文件（跳过 ground_truth 等辅助文件）。"""
    root = Path(corpus_dir)
    exts = {e.lower() for e in exts}
    files: list[Path] = []
    for p in sorted(root.rglob("*")):
        if not p.is_file():
            continue
        if p.suffix.lower() not in exts:
            continue
        if p.name.startswith("_") or p.name == "ground_truth.json":
            continue
        files.append(p)
    return files
