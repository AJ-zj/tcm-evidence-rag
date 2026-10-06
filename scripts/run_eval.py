"""运行完整评估闭环（五组实验）并产出报告。

用法：
  .venv/Scripts/python scripts/run_eval.py                 # 全量
  .venv/Scripts/python scripts/run_eval.py --sample 150    # 快速抽样（exp3 端到端）
  .venv/Scripts/python scripts/run_eval.py --only exp1,exp4
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tcm_rag.config import load_config  # noqa: E402
from tcm_rag.evaluation.dataset import EvalSetBuilder  # noqa: E402
from tcm_rag.evaluation.runner import EvalRunner  # noqa: E402
from tcm_rag.pipeline import RAGSystem  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, default=None, help="exp3 端到端抽样条数（默认全量）")
    parser.add_argument("--only", type=str, default=None, help="只运行部分实验，如 exp1,exp4")
    args = parser.parse_args()

    cfg = load_config()
    processed = cfg.path(cfg.get("project.processed_dir", "data/processed"))
    eval_file = processed / "eval_set.jsonl"
    if not eval_file.exists():
        print("未找到测试集，请先运行 scripts/make_eval_set.py")
        sys.exit(1)

    items = EvalSetBuilder.load(eval_file)
    print(f"加载系统（{len(items)} 条测试题）…")
    system = RAGSystem.load(cfg, verbose=True)
    runner = EvalRunner(system, items)

    only = set(args.only.split(",")) if args.only else None
    reports_dir = cfg.path(cfg.get("project.reports_dir", "reports"))
    reports_dir.mkdir(parents=True, exist_ok=True)
    results: dict = {"n_eval_items": len(items)}

    def want(name: str) -> bool:
        return only is None or name in only

    if want("exp1"):
        print("\n[exp1] 检索架构 A/B…")
        results["exp1_retrieval_ab"] = runner.run_retrieval_ab()
    if want("exp2"):
        print("\n[exp2] 信息保留率 + OCR 恢复…")
        results["exp2_retention_ocr"] = runner.run_retention_and_ocr()
    if want("exp3"):
        print("\n[exp3] 端到端 RAGAS：static 基线…")
        s = runner.run_e2e("static", sample=args.sample)
        results["exp3_e2e_static"] = s["summary"]
        _dump_rows(reports_dir / "e2e_rows_static.jsonl", s["rows"])
        print("[exp3] 端到端 RAGAS：react 动态决策…")
        r = runner.run_e2e("react", sample=args.sample)
        results["exp3_e2e_react"] = r["summary"]
        _dump_rows(reports_dir / "e2e_rows_react.jsonl", r["rows"])
    if want("exp4"):
        print("\n[exp4] 延迟基准…")
        results["exp4_latency"] = runner.run_latency()
    if want("exp5"):
        print("\n[exp5] 分层记忆实验…")
        results["exp5_memory"] = runner.run_memory_experiment(
            n_dialogues=cfg.get("evaluation.dialogues_for_memory_eval", 40)
        )

    import time as _t

    results["generated_at"] = _t.strftime("%Y-%m-%d %H:%M:%S")
    (reports_dir / "eval_results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2, default=_default), encoding="utf-8"
    )
    print("\n=== 汇总 ===")
    print(json.dumps({k: v for k, v in results.items() if k.startswith("exp")},
                     ensure_ascii=False, indent=2, default=_default)[:3000])


def _dump_rows(path: Path, rows: list) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, default=_default) + "\n")


def _default(o):
    import numpy as np

    if isinstance(o, float) and o != o:
        return None
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    return str(o)


if __name__ == "__main__":
    main()
