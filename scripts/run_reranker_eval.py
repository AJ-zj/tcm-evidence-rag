"""重排器对比评估：用配置的重排器（api / feature / cross_encoder）重跑检索 A/B。

用法： .venv/Scripts/python scripts/run_reranker_eval.py [--reranker api] [--out reports/eval_results_reranker_api.json]
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
    parser.add_argument("--reranker", type=str, default=None, help="覆盖 config: feature|api|cross_encoder")
    parser.add_argument("--out", type=str, default="reports/eval_results_reranker_api.json")
    args = parser.parse_args()

    cfg = load_config()
    if args.reranker:
        cfg._data["retrieval"]["reranker"] = args.reranker

    print(f"加载系统（reranker={cfg.get('retrieval.reranker')}）…")
    system = RAGSystem.load(cfg, verbose=True)
    items = EvalSetBuilder.load(cfg.path(cfg.get("project.processed_dir", "data/processed")) / "eval_set.jsonl")
    runner = EvalRunner(system, items)
    out = runner.run_retrieval_ab()

    payload = {
        "reranker": cfg.get("retrieval.reranker"),
        "reranker_model": cfg.get("retrieval.api_rerank_model", ""),
        "n_queries": out["n_queries"],
        "modes": out["modes"],
    }
    out_path = cfg.path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n结果 → {out_path}")


if __name__ == "__main__":
    main()
