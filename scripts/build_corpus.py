"""构建实验语料：

1. 从 Huatuo26M-Lite 抽取中医科高质量 QA 子集 → data/corpus/qa/huatuo_tcm.jsonl
2. 对《神农本草经》做确定性 OCR 噪声模拟 → data/corpus/ocr/（清洁原文移入
   data/corpus_ground_truth/ 作为清洗质量评测的金标准）
3. 将《温病条辨》渲染为带噪声的"扫描版 PDF" → data/corpus/pdf/（验证 PDF 解析链路）

用法： .venv/Scripts/python scripts/build_corpus.py [--qa-count 600]
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tcm_rag.parsing.ocr_clean import CONFUSABLE_PAIRS  # noqa: E402

CORPUS = ROOT / "data" / "corpus"
TRUTH = ROOT / "data" / "corpus_ground_truth"
RAW_HUATUO = ROOT / "data" / "raw" / "Huatuo26M-Lite" / "format_data.jsonl"


# ----------------------------------------------------------------------
def export_huatuo_tcm(qa_count: int, seed: int = 42) -> dict:
    """抽取 label=中医科、score>=4、答案充实的 QA 子集。"""
    out_dir = CORPUS / "qa"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "huatuo_tcm.jsonl"

    candidates = []
    with open(RAW_HUATUO, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("label") != "中医科":
                continue
            if int(row.get("score", 0)) < 4:
                continue
            q, a = str(row.get("question", "")).strip(), str(row.get("answer", "")).strip()
            if len(a) < 60 or len(q) < 8:
                continue
            candidates.append(row)

    rng = random.Random(seed)
    rng.shuffle(candidates)
    picked = candidates[:qa_count]
    with open(out_file, "w", encoding="utf-8") as f:
        for row in picked:
            keep = {k: row.get(k) for k in ("id", "question", "answer", "score", "label", "related_diseases")}
            f.write(json.dumps(keep, ensure_ascii=False) + "\n")
    n_with_disease = sum(1 for r in picked if r.get("related_diseases"))
    return {
        "candidates_tcm_score4plus": len(candidates),
        "exported": len(picked),
        "with_related_diseases": n_with_disease,
        "file": str(out_file),
    }


# ----------------------------------------------------------------------
def _confusable_map() -> dict[str, str]:
    m: dict[str, str] = {}
    for a, b in CONFUSABLE_PAIRS:
        m.setdefault(a, b)
        m.setdefault(b, a)
    return m


def corrupt_text(text: str, rng: random.Random) -> tuple[str, dict]:
    """确定性 OCR 噪声模拟：形近字替换/插空格/标点噪声/乱码/重复字。"""
    cmap = _confusable_map()
    out: list[str] = []
    stats = {"char_confused": 0, "space_inserted": 0, "punct_corrupted": 0,
             "garbage_inserted": 0, "char_duplicated": 0}
    prev = ""
    for ch in text:
        r = rng.random()
        if ch in cmap and r < 0.10:
            out.append(cmap[ch])
            stats["char_confused"] += 1
        elif ch in "，。；：！？" and r < 0.03:
            repl = {"，": "、", "。": "．", "；": "：", "：": ";", "！": "!", "？": "?"}[ch]
            out.append(repl)
            stats["punct_corrupted"] += 1
        elif r < 0.034:
            out.append(ch)
            stats["char_duplicated"] += 1
            out.append(ch)
        else:
            out.append(ch)
        r2 = rng.random()
        if "\u4e00" <= ch <= "\u9fff" and r2 < 0.045:
            out.append(" ")
            stats["space_inserted"] += 1
        elif r2 < 0.004:
            out.append(rng.choice("□\ufffd◆"))
            stats["garbage_inserted"] += 1
        prev = ch
    corrupted = "".join(out)
    n = max(len(text), 1)
    stats["char_error_rate"] = round(
        (stats["char_confused"] + stats["garbage_inserted"] + stats["char_duplicated"]) / n, 4
    )
    return corrupted, stats


def make_ocr_variants(seed: int = 42) -> dict:
    """神农本草经 → OCR 噪声 MD；温病条辨 → 噪声扫描 PDF。清洁原文移入金标准目录。"""
    TRUTH.mkdir(parents=True, exist_ok=True)
    (CORPUS / "ocr").mkdir(parents=True, exist_ok=True)
    (CORPUS / "pdf").mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    report: dict = {}

    # --- 1) 神农本草经：MD 噪声版 ---
    clean_src = TRUTH / "神农本草经_clean.md"
    live_src = CORPUS / "classics" / "神农本草经.md"
    if not clean_src.exists() and live_src.exists():
        clean_src.write_text(live_src.read_text(encoding="utf-8"), encoding="utf-8")
    if clean_src.exists():
        if live_src.exists():
            live_src.unlink()  # 语料库中只保留 OCR 版（清洁版作为金标准不入库）
        text = clean_src.read_text(encoding="utf-8")
        corrupted, stats = corrupt_text(text, rng)
        ocr_file = CORPUS / "ocr" / "神农本草经_ocr.md"
        ocr_file.write_text(corrupted, encoding="utf-8")
        report["神农本草经_ocr.md"] = {"truth": str(clean_src), **stats}

    # --- 2) 温病条辨：扫描版 PDF ---
    clean_src2 = TRUTH / "温病条辨_clean.md"
    live_src2 = CORPUS / "classics" / "温病条辨.md"
    if not clean_src2.exists() and live_src2.exists():
        clean_src2.write_text(live_src2.read_text(encoding="utf-8"), encoding="utf-8")
    if clean_src2.exists():
        if live_src2.exists():
            live_src2.unlink()
        text2 = clean_src2.read_text(encoding="utf-8")
        corrupted2, stats2 = corrupt_text(text2, rng)
        pdf_file = CORPUS / "pdf" / "温病条辨_扫描版.pdf"
        _render_pdf(corrupted2, pdf_file)
        report["温病条辨_扫描版.pdf"] = {"truth": str(clean_src2), **stats2}

    (TRUTH / "corruption_manifest.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report


def _render_pdf(text: str, out_path: Path) -> None:
    """reportlab 渲染中文 PDF（STSong-Light CID 字体，无需外部字库文件）。"""
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.units import mm
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.cidfonts import UnicodeCIDFont
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

    pdfmetrics.registerFont(UnicodeCIDFont("STSong-Light"))
    style_body = ParagraphStyle(
        "body", fontName="STSong-Light", fontSize=10.5, leading=17, wordWrap="CJK"
    )
    style_h = ParagraphStyle(
        "h", fontName="STSong-Light", fontSize=14, leading=22, spaceBefore=8, spaceAfter=4
    )
    doc = SimpleDocTemplate(
        str(out_path), pagesize=A4,
        leftMargin=22 * mm, rightMargin=22 * mm, topMargin=20 * mm, bottomMargin=20 * mm,
    )
    story = []
    for line in text.splitlines():
        line = line.rstrip()
        if not line:
            story.append(Spacer(1, 4))
            continue
        escaped = line.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        if line.startswith("#"):
            level = len(line) - len(line.lstrip("#"))
            content = escaped.lstrip("#").strip()
            st = ParagraphStyle(f"h{level}", parent=style_h, fontSize=max(11, 17 - level * 2))
            story.append(Paragraph(content, st))
        else:
            story.append(Paragraph(escaped, style_body))
    doc.build(story)


# ----------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qa-count", type=int, default=600)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print("=== 构建实验语料 ===")
    if RAW_HUATUO.exists():
        qa_stats = export_huatuo_tcm(args.qa_count, args.seed)
        print(f"[QA] 中医科候选 {qa_stats['candidates_tcm_score4plus']} 条 → 导出 {qa_stats['exported']} 条"
              f"（含疾病标注 {qa_stats['with_related_diseases']} 条）→ {qa_stats['file']}")
    else:
        print(f"[QA] 未找到 {RAW_HUATUO}，跳过")

    ocr_report = make_ocr_variants(args.seed)
    for name, st in ocr_report.items():
        print(f"[OCR] {name}: 形近字 {st['char_confused']}, 插空格 {st['space_inserted']}, "
              f"乱码 {st['garbage_inserted']}, 重复字 {st['char_duplicated']}, CER≈{st['char_error_rate']}")
    print("=== 完成 ===")


if __name__ == "__main__":
    main()
