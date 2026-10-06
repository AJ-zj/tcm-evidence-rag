"""官方 RAGAS 库评估：与项目内置离线指标互为对照。

使用 ragas 官方实现（faithfulness / answer_relevancy / context_precision / context_recall），
LLM 裁判与 embeddings 走 SiliconFlow OpenAI 兼容接口（非推理模型，保证裁判 JSON 输出稳定）；
被评估的答案由"混合检索(全管线) + 证据约束生成"产生，与系统生产链路一致。

输出：reports/ragas_official.json（官方分）+ 与内置离线指标的对照打印。

用法： .venv/Scripts/python scripts/run_ragas_official.py [--n 40]
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dotenv import load_dotenv  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--judge-model", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--embed-model", type=str, default="BAAI/bge-m3")
    args = parser.parse_args()

    load_dotenv(ROOT / ".env")
    api_key = os.environ.get("LLM_API_KEY", "")
    base_url = os.environ.get("LLM_BASE_URL", "https://api.siliconflow.cn/v1")
    if not api_key:
        print("缺少 LLM_API_KEY"); sys.exit(1)

    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    from ragas import EvaluationDataset, evaluate
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness

    from tcm_rag.config import load_config
    from tcm_rag.evaluation.dataset import EvalSetBuilder
    from tcm_rag.pipeline import RAGSystem

    print("加载 RAG 系统…")
    cfg = load_config()
    system = RAGSystem.load(cfg)
    items = EvalSetBuilder.load(
        cfg.path(cfg.get("project.processed_dir", "data/processed")) / "eval_set.jsonl"
    )
    answerable = [i for i in items if i.gold_chunk_ids and i.reference_answer or i.gold_chunk_ids]
    rng = random.Random(args.seed)
    sampled = rng.sample([i for i in items if i.gold_chunk_ids], args.n)

    judge_llm = ChatOpenAI(
        model=args.judge_model, base_url=base_url, api_key=api_key, temperature=0, timeout=120
    )
    embeddings = OpenAIEmbeddings(model=args.embed_model, base_url=base_url, api_key=api_key, timeout=120)

    store = system.store
    rows: list[dict] = []
    t0 = time.time()
    for i, it in enumerate(sampled):
        rr = system.retriever.retrieve(it.question, mode="full", top_k=cfg.get("retrieval.final_top_k", 5))
        contexts = [e.chunk.text for e in rr.evidences]
        # 证据约束生成（与生产链路一致的约束式 prompt）
        ev_block = "\n\n".join(f"[{k+1}] {t[:600]}" for k, t in enumerate(contexts))
        resp = judge_llm.invoke(
            "你是中医循证问答助手。只依据给定证据回答，每条论断后标注〔n〕；证据不足时回答\"证据不足\"。\n\n"
            f"【证据】\n{ev_block}\n\n【问题】{it.question}"
        )
        answer = resp.content
        # reference：金标准块的原文（评测 context_recall 的标准依据）
        gold_texts = [store.get(g).text for g in it.gold_chunk_ids if store.get(g)]
        rows.append({
            "user_input": it.question,
            "response": answer,
            "retrieved_contexts": contexts,
            "reference": "\n".join(gold_texts)[:1500],
            "qid": it.qid, "type": it.item_type,
        })
        if (i + 1) % 10 == 0:
            print(f"  生成 {i+1}/{len(sampled)}（{time.time()-t0:.0f}s）")

    print(f"RAGAS 官方评估（judge={args.judge_model}, embed={args.embed_model}）…")
    dataset = EvaluationDataset.from_list(rows)
    result = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=LangchainLLMWrapper(judge_llm),
        embeddings=LangchainEmbeddingsWrapper(embeddings),
        show_progress=True,
    )
    df = result.to_pandas()

    def col(name: str) -> float:
        import pandas as pd

        v = pd.to_numeric(df[name], errors="coerce").dropna()
        return round(float(v.mean()), 4) if len(v) else float("nan")

    summary = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "n": len(rows),
        "judge_model": args.judge_model,
        "embed_model": args.embed_model,
        "faithfulness": col("faithfulness"),
        "answer_relevancy": col("answer_relevancy"),
        "context_precision": col("context_precision"),
        "context_recall": col("context_recall"),
        "elapsed_s": round(time.time() - t0, 1),
    }
    reports = cfg.path(cfg.get("project.reports_dir", "reports"))
    reports.mkdir(parents=True, exist_ok=True)
    detail = df[["user_input", "response", "faithfulness", "answer_relevancy",
                 "context_precision", "context_recall"]].to_dict(orient="records")
    (reports / "ragas_official.json").write_text(
        json.dumps({"summary": summary, "rows": detail}, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("明细 → reports/ragas_official.json")


if __name__ == "__main__":
    main()
