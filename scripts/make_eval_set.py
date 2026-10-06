"""生成评估测试集（默认 700 条，含金标准证据引用与拒答测试题）。

用法： .venv/Scripts/python scripts/make_eval_set.py [--size 700] [--out data/processed/eval_set.jsonl]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tcm_rag.config import load_config  # noqa: E402
from tcm_rag.evaluation.dataset import EvalSetBuilder  # noqa: E402
from tcm_rag.indexing import ChunkStore  # noqa: E402
from tcm_rag.ner.medical_ner import MedicalNER  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=None, help="测试集目标规模（默认取配置 700）")
    parser.add_argument("--unanswerable", type=int, default=None)
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config()
    processed = cfg.path(cfg.get("project.processed_dir", "data/processed"))
    db_path = cfg.path("data/db/tcm_rag.db")
    if db_path.exists():
        store = ChunkStore.load(db_path)          # 双库：SQLite chunks 表
    else:
        store = ChunkStore.load(processed / "chunks.jsonl")   # 兼容旧 artifacts
    ner = MedicalNER.load(processed / "vocab.json", weights=cfg.get("ner.weights"))

    builder = EvalSetBuilder(
        chunks=store.all_chunks(),
        ner=ner,
        seed=cfg.get("project.seed", 42),
        target_size=args.size or cfg.get("evaluation.test_set_size", 700),
        unanswerable_count=args.unanswerable or cfg.get("evaluation.unanswerable_count", 30),
    )
    items = builder.build()
    out = Path(args.out) if args.out else processed / "eval_set.jsonl"
    EvalSetBuilder.save(items, out)

    counts = EvalSetBuilder.type_counts(items)
    print(f"测试集：{len(items)} 条 → {out}")
    for t, n in sorted(counts.items(), key=lambda x: -x[1]):
        print(f"  {t:14s} {n}")


if __name__ == "__main__":
    main()
