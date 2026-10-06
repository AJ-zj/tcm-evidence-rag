"""忠实度对比实验：LLM 无证据自由生成 vs 证据约束生成（RAG）。

设计：
- 同一批测试题，两种生成方式：
  1. free:   把问题直接交给 LLM（不给任何证据）——模拟"传统 LLM 直接回答"
  2. rag:    完整 RAG 链路（混合检索+重排+过滤 + 证据约束生成，〔n〕引用）
- 忠实度裁判 context 统一取 RAG 检索到的证据文本：衡量"答案陈述是否可由证据推出"。
  free 答案中大量语料外知识 → 不被证据支持的陈述比例高。
- 另对比不可答题：free LLM 必然硬答（幻觉风险），RAG 依据置信度拒答。

输出：reports/faithfulness_experiment.json + 控制台对比表。

用法： .venv/Scripts/python scripts/run_faithfulness_experiment.py [--n 40] [--unans 8]
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tcm_rag.config import load_config  # noqa: E402
from tcm_rag.evaluation.dataset import EvalSetBuilder  # noqa: E402
from tcm_rag.evaluation.metrics import FaithfulnessScorer  # noqa: E402
from tcm_rag.pipeline import RAGSystem  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=40, help="可答题抽样数")
    parser.add_argument("--unans", type=int, default=8, help="不可答题抽样数")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = load_config()
    if cfg.get("llm.provider", "offline") == "offline":
        print("未配置 LLM（llm.provider=offline），本实验需要在线 LLM 生成自由答案。")
        sys.exit(1)

    print("加载系统…")
    system = RAGSystem.load(cfg)
    items = EvalSetBuilder.load(cfg.path(cfg.get("project.processed_dir", "data/processed")) / "eval_set.jsonl")
    answerable = [i for i in items if i.gold_chunk_ids]
    unanswerable = [i for i in items if not i.gold_chunk_ids]

    rng = random.Random(args.seed)
    # 分层抽样保证题型覆盖
    by_type: dict[str, list] = {}
    for it in answerable:
        by_type.setdefault(it.item_type, []).append(it)
    sampled: list = []
    while len(sampled) < args.n and by_type:
        for t in list(by_type):
            if by_type[t] and len(sampled) < args.n:
                sampled.append(by_type[t].pop())
            if not by_type[t]:
                del by_type[t]
    sampled_unans = rng.sample(unanswerable, min(args.unans, len(unanswerable)))

    llm = system.llm
    scorer = FaithfulnessScorer(system.ner, system.nli)
    rows: list[dict] = []

    def gen_free(question: str) -> str:
        return llm.complete(
            "你是中医问答助手。请根据你自己的知识直接回答用户问题，不要说不知道，尽量给出具体答案。",
            question,
            max_tokens=2048,
        )

    t0 = time.time()
    total = len(sampled) + len(sampled_unans)
    done = 0
    for it in sampled:
        try:
            ans_free = gen_free(it.question)
        except Exception as e:  # noqa: BLE001
            ans_free = ""
        res = system.agent.answer(it.question, decision_mode="react")
        contexts = [e["text"] for e in res.evidences]
        faith_free, d_free = scorer.score(ans_free, contexts)
        faith_rag, d_rag = scorer.score(res.answer, contexts)
        rows.append({
            "qid": it.qid, "type": it.item_type, "question": it.question,
            "mode": "answerable",
            "faith_free": faith_free, "faith_rag": faith_rag,
            "n_claims_free": d_free.get("n_claims", 0),
            "n_unsupported_free": d_free.get("n_claims", 0) - d_free.get("n_supported", 0),
            "rag_refused": res.refused, "rag_confidence": res.confidence,
            "gold_hit": bool(set(e["chunk_id"] for e in res.evidences) & set(it.gold_chunk_ids)),
        })
        done += 1
        print(f"  [{done}/{total}] {it.item_type} faith free={faith_free:.2f} rag={faith_rag:.2f}")

    for it in sampled_unans:
        try:
            ans_free = gen_free(it.question)
        except Exception:  # noqa: BLE001
            ans_free = ""
        res = system.agent.answer(it.question, decision_mode="react")
        rows.append({
            "qid": it.qid, "type": "unanswerable", "question": it.question,
            "mode": "unanswerable",
            "free_hard_answered": bool(ans_free.strip()),
            "rag_refused": res.refused, "rag_confidence": res.confidence,
        })
        done += 1
        print(f"  [{done}/{total}] unanswerable rag_refused={res.refused}")

    # 汇总
    def mean(vals: list) -> float:
        v = [x for x in vals if x == x]
        return round(sum(v) / len(v), 4) if v else float("nan")

    ans_rows = [r for r in rows if r["mode"] == "answerable"]
    unans_rows = [r for r in rows if r["mode"] == "unanswerable"]
    summary = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "n_answerable": len(ans_rows),
        "n_unanswerable": len(unans_rows),
        "llm_model": cfg.get("llm.model", ""),
        "faithfulness_free_generation": mean([r["faith_free"] for r in ans_rows]),
        "faithfulness_rag_grounded": mean([r["faith_rag"] for r in ans_rows]),
        "avg_unsupported_claims_free": mean([r["n_unsupported_free"] for r in ans_rows]),
        "avg_claims_free": mean([r["n_claims_free"] for r in ans_rows]),
        "free_hard_answer_rate_on_unanswerable": round(
            sum(1 for r in unans_rows if r["free_hard_answered"]) / max(len(unans_rows), 1), 4
        ),
        "rag_refusal_rate_on_unanswerable": round(
            sum(1 for r in unans_rows if r["rag_refused"]) / max(len(unans_rows), 1), 4
        ),
        "elapsed_s": round(time.time() - t0, 1),
    }
    reports = cfg.path(cfg.get("project.reports_dir", "reports"))
    reports.mkdir(parents=True, exist_ok=True)
    out = {"summary": summary, "rows": rows}
    (reports / "faithfulness_experiment.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2, default=lambda o: None if o != o else str(o)),
        encoding="utf-8",
    )
    print("\n=== 忠实度对比（裁判 context = RAG 检索证据）===")
    print(f"free 无证据生成:  faithfulness={summary['faithfulness_free_generation']}")
    print(f"RAG  证据约束:    faithfulness={summary['faithfulness_rag_grounded']}")
    print(f"free 平均 unsupported 陈述: {summary['avg_unsupported_claims_free']}/{summary['avg_claims_free']}")
    print(f"不可答题硬答率: free={summary['free_hard_answer_rate_on_unanswerable']:.0%} vs "
          f"RAG拒答={summary['rag_refusal_rate_on_unanswerable']:.0%}")
    print(f"明细 → reports/faithfulness_experiment.json （{summary['elapsed_s']}s）")


if __name__ == "__main__":
    main()
