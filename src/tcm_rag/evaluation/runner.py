"""评估闭环运行器：五组可复现实验 + Markdown/JSON 报告。

exp1 retrieval_ab:  dense / bm25 / hybrid / hybrid_rerank / full 五种模式的
                    Recall@K、Top-1、MRR、Precision@K 对比
exp2 retention_ocr: 语义切分 vs 朴素截断的长文信息保留率；OCR 清洗恢复率
exp3 e2e_ragas:     static（一次检索直接答）vs react（动态决策）端到端对比：
                    faithfulness / context_precision / context_recall /
                    answer_relevancy / 拒答正确率 / 错误自信率
exp4 latency:       FAISS-HNSW 纯索引 P50/P95/P99 + 全管线检索延迟
exp5 memory:        多轮对话模拟：分层记忆 on/off 的命中率与错误率、
                    长期记忆写入统计（合并/拒绝/冲突消解）
"""
from __future__ import annotations

import json
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from ..config import Config
from ..ner.medical_ner import MedicalNER
from ..parsing.chunking import NaiveTruncationChunker, retention_score
from ..parsing.loaders import load_any
from ..parsing.ocr_clean import OCRTextCleaner
from ..schema import Chunk
from .dataset import EvalItem
from .metrics import (
    FaithfulnessScorer,
    answer_relevancy,
    context_precision,
    context_recall,
    mrr,
    nanmean,
    precision_at_k,
    recall_at_k,
    top1_accuracy,
)

RETRIEVAL_MODES = ("dense", "bm25", "hybrid", "hybrid_rerank", "full")


class EvalRunner:
    def __init__(self, system, items: list[EvalItem]):
        self.system = system
        self.items = items
        self.answerable = [it for it in items if it.gold_chunk_ids]
        self.unanswerable = [it for it in items if not it.gold_chunk_ids]

    # ==================================================================
    # exp1: 检索架构 A/B
    # ==================================================================
    def run_retrieval_ab(self, ks: Sequence[int] = (1, 5, 10, 30)) -> dict[str, Any]:
        out: dict[str, Any] = {"ks": list(ks), "n_queries": len(self.answerable), "modes": {}}
        max_k = max(ks)
        for mode in RETRIEVAL_MODES:
            per_k = {k: [] for k in ks}
            prec = {k: [] for k in ks}
            top1, mrrs = [], []
            t0 = time.time()
            for it in self.answerable:
                rr = self.system.retriever.retrieve(it.question, mode=mode, top_k=max_k)
                ids = [e.chunk.chunk_id for e in rr.evidences]
                for k in ks:
                    per_k[k].append(recall_at_k(ids, it.gold_chunk_ids, k))
                    prec[k].append(precision_at_k(ids, it.gold_chunk_ids, k))
                top1.append(top1_accuracy(ids, it.gold_chunk_ids))
                mrrs.append(mrr(ids, it.gold_chunk_ids))
            out["modes"][mode] = {
                "recall@k": {str(k): nanmean(per_k[k]) for k in ks},
                "precision@k": {str(k): nanmean(prec[k]) for k in ks},
                "top1_accuracy": nanmean(top1),
                "mrr": nanmean(mrrs),
                "elapsed_s": round(time.time() - t0, 1),
            }
            m = out["modes"][mode]
            print(f"  [exp1] {mode:14s} R@{max_k}={m['recall@k'][str(max_k)]:.3f} "
                  f"Top1={m['top1_accuracy']:.3f} MRR={m['mrr']:.3f}")
        return out

    # ==================================================================
    # exp2: 信息保留率 + OCR 清洗恢复
    # ==================================================================
    def run_retention_and_ocr(self) -> dict[str, Any]:
        cfg: Config = self.system.cfg
        processed = cfg.path(cfg.get("project.processed_dir", "data/processed"))
        docs_file = processed / "documents.jsonl"
        docs = []
        with open(docs_file, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    from ..schema import Document

                    d = json.loads(line)
                    docs.append(Document(**{k: v for k, v in d.items() if k in Document.__dataclass_fields__}))

        all_terms = self.system.ner.all_terms
        cleaner = OCRTextCleaner(vocab_terms=all_terms)
        cleaned_docs = []
        for d in docs:
            d.text = cleaner.clean(d.text).text
            cleaned_docs.append(d)

        naive = NaiveTruncationChunker(
            max_tokens=cfg.get("parsing.baseline_chunk.naive_truncate_tokens", 512)
        )
        naive_by_doc: dict[str, list[Chunk]] = defaultdict(list)
        for d in cleaned_docs:
            naive_by_doc[d.doc_id].extend(naive.chunk(d))

        semantic_by_doc: dict[str, list[Chunk]] = defaultdict(list)
        for c in self.system.store.all_chunks():
            semantic_by_doc[c.doc_id].append(c)

        naive_ret = retention_score(cleaned_docs, dict(naive_by_doc), all_terms)
        sem_ret = retention_score(cleaned_docs, semantic_by_doc, all_terms)

        # 按来源类型细分
        by_type: dict[str, dict] = {}
        type_docs: dict[str, list] = defaultdict(list)
        for d in cleaned_docs:
            type_docs[d.source_type].append(d)
        for st, ds in type_docs.items():
            by_type[st] = {
                "naive": retention_score(ds, dict(naive_by_doc), all_terms)["overall"],
                "semantic": retention_score(ds, semantic_by_doc, all_terms)["overall"],
            }

        # OCR 清洗恢复率
        ocr_report = self._ocr_recovery(cleaner)

        result = {
            "naive_truncation": {"overall": naive_ret["overall"], "sentences": naive_ret["total_info_sentences"]},
            "semantic_chunking": {"overall": sem_ret["overall"], "sentences": sem_ret["total_info_sentences"]},
            "improvement_abs": round(sem_ret["overall"] - naive_ret["overall"], 4),
            "improvement_relative": round(
                (sem_ret["overall"] - naive_ret["overall"]) / max(1e-9, 1 - naive_ret["overall"]), 4
            ),
            "by_source_type": by_type,
            "ocr_recovery": ocr_report,
        }
        print(f"  [exp2] 信息保留率: 朴素截断 {naive_ret['overall']:.1%} → 语义切分 {sem_ret['overall']:.1%}")
        return result

    def _ocr_recovery(self, cleaner: OCRTextCleaner) -> dict[str, Any]:
        cfg: Config = self.system.cfg
        truth_dir = cfg.path("data/corpus_ground_truth")
        manifest_f = truth_dir / "corruption_manifest.json"
        if not manifest_f.exists():
            return {}
        manifest = json.loads(manifest_f.read_text(encoding="utf-8"))
        corpus_dir = cfg.path(cfg.get("project.corpus_dir", "data/corpus"))
        out: dict[str, Any] = {}
        vocab_terms = self.system.ner.all_terms
        for name, info in manifest.items():
            truth_text = Path(info["truth"]).read_text(encoding="utf-8")
            # 找到语料库中的对应损坏文件
            candidates = list(corpus_dir.rglob(Path(name).stem + ".*"))
            if not candidates:
                continue
            cf = candidates[0]
            docs = load_any(cf)
            raw_text = "\n".join(d.text for d in docs)
            cleaned = cleaner.clean(raw_text).text

            truth_terms = {t for t in vocab_terms if t in truth_text}

            def term_recall(text: str) -> float:
                return len([t for t in truth_terms if t in text]) / max(len(truth_terms), 1)

            # 字符级相似度（在截断窗口上计算，控制耗时）
            from difflib import SequenceMatcher

            def char_acc(text: str, window: int = 3000) -> float:
                t = "".join(truth_text.split())[:window]
                x = "".join(text.split())[:window]
                return SequenceMatcher(None, t, x, autojunk=False).ratio()

            out[name] = {
                "corruption": {k: v for k, v in info.items() if k != "truth"},
                "term_recall_raw": round(term_recall(raw_text), 4),
                "term_recall_cleaned": round(term_recall(cleaned), 4),
                "char_acc_raw": round(char_acc(raw_text), 4),
                "char_acc_cleaned": round(char_acc(cleaned), 4),
                "noise_raw": round(cleaner.noise_score(raw_text), 4),
                "noise_cleaned": round(cleaner.noise_score(cleaned), 4),
            }
        if out:
            avg = lambda key: round(float(np.mean([v[key] for v in out.values()])), 4)  # noqa: E731
            out["_avg"] = {
                "term_recall_raw": avg("term_recall_raw"),
                "term_recall_cleaned": avg("term_recall_cleaned"),
                "char_acc_raw": avg("char_acc_raw"),
                "char_acc_cleaned": avg("char_acc_cleaned"),
            }
        return out

    # ==================================================================
    # exp3: 端到端 RAGAS（static vs react）
    # ==================================================================
    def run_e2e(self, decision_mode: str = "react", sample: int | None = None) -> dict[str, Any]:
        agent = self.system.agent
        faith_scorer = FaithfulnessScorer(self.system.ner, self.system.nli)
        items = self.items
        if sample and sample < len(items):
            rng = random.Random(self.system.cfg.get("project.seed", 42))
            items = rng.sample(items, sample)

        rows: list[dict] = []
        t0 = time.time()
        for i, it in enumerate(items):
            res = agent.answer(it.question, decision_mode=decision_mode)
            ids = [e["chunk_id"] for e in res.evidences]
            contexts = [e["text"] for e in res.evidences]
            gold_hit = bool(set(ids) & set(it.gold_chunk_ids)) if it.gold_chunk_ids else None

            faith, faith_detail = (
                faith_scorer.score(res.answer, contexts) if (res.answer and not res.refused) else (float("nan"), {})
            )
            rows.append({
                "qid": it.qid,
                "type": it.item_type,
                "question": it.question,
                "refused": res.refused,
                "confidence": res.confidence,
                "consistency": res.consistency,
                "coverage": res.coverage,
                "rounds_used": res.rounds_used,
                "gold_chunk_ids": it.gold_chunk_ids,
                "retrieved_ids": ids,
                "ctx_precision": context_precision(ids, it.gold_chunk_ids) if it.gold_chunk_ids else float("nan"),
                "ctx_recall": context_recall(ids, it.gold_chunk_ids) if it.gold_chunk_ids else float("nan"),
                "gold_hit": gold_hit,
                "faithfulness": faith,
                "faith_detail": faith_detail,
                "relevancy": answer_relevancy(it.question, res.answer, self.system.ner) if not res.refused else 0.0,
                "answer": res.answer[:500],
                "latency_ms": res.latency_ms,
            })
            if (i + 1) % 100 == 0:
                print(f"  [exp3:{decision_mode}] {i + 1}/{len(items)} ({time.time() - t0:.0f}s)")

        # 聚合
        answerable_rows = [r for r in rows if r["type"] != "unanswerable"]
        unans_rows = [r for r in rows if r["type"] == "unanswerable"]
        answered = [r for r in answerable_rows if not r["refused"]]
        false_confident = [r for r in answered if r["gold_hit"] is False]
        summary = {
            "mode": decision_mode,
            "n_items": len(rows),
            "faithfulness": nanmean([r["faithfulness"] for r in answered]),
            "context_precision": nanmean([r["ctx_precision"] for r in answerable_rows]),
            "context_recall": nanmean([r["ctx_recall"] for r in answerable_rows]),
            "answer_relevancy": nanmean([r["relevancy"] for r in answered]),
            "refusal_accuracy_on_unanswerable": (
                round(sum(1 for r in unans_rows if r["refused"]) / len(unans_rows), 4) if unans_rows else None
            ),
            "over_refusal_on_answerable": (
                round(sum(1 for r in answerable_rows if r["refused"]) / len(answerable_rows), 4)
                if answerable_rows else None
            ),
            "false_confidence_rate": (
                round(len(false_confident) / len(answered), 4) if answered else None
            ),
            "avg_rounds": round(float(np.mean([r["rounds_used"] for r in rows])), 2),
            "avg_latency_ms": round(float(np.mean([r["latency_ms"] for r in rows])), 1),
            "avg_confidence": round(float(np.mean([r["confidence"] for r in rows])), 4),
            "by_type": _aggregate_by_type(rows),
            "elapsed_s": round(time.time() - t0, 1),
        }
        return {"summary": summary, "rows": rows}

    # ==================================================================
    # exp4: 延迟基准
    # ==================================================================
    def run_latency(self, n_queries: int = 300) -> dict[str, Any]:
        rng = random.Random(7)
        queries = [rng.choice(self.answerable).question for _ in range(n_queries)]
        vectors = self.system.embedder.encode_queries(queries)

        # 纯 FAISS-HNSW 检索
        hnsw_ms = []
        for v in vectors:
            hnsw_ms.append(self.system.retriever.dense_only_latency(v, k=30))
        # 全管线（embed+双路召回+融合+重排+过滤）
        pipe_ms = []
        for q, v in zip(queries[:200], vectors[:200]):
            rr = self.system.retriever.retrieve(q, mode="full", query_vector=v)
            pipe_ms.append(rr.total_ms)

        def pctl(arr: list[float]) -> dict[str, float]:
            a = np.asarray(arr)
            return {
                "p50": round(float(np.percentile(a, 50)), 2),
                "p95": round(float(np.percentile(a, 95)), 2),
                "p99": round(float(np.percentile(a, 99)), 2),
                "mean": round(float(a.mean()), 2),
                "max": round(float(a.max()), 2),
            }

        out = {
            "n_queries": n_queries,
            "hnsw_search_ms": pctl(hnsw_ms),
            "full_pipeline_ms": pctl(pipe_ms),
            "index_size": self.system.hnsw.ntotal,
            "dim": self.system.embedder.dim,
        }
        print(f"  [exp4] HNSW P95={out['hnsw_search_ms']['p95']}ms，全管线 P95={out['full_pipeline_ms']['p95']}ms")
        return out

    # ==================================================================
    # exp5: 分层记忆多轮对话实验
    # ==================================================================
    def run_memory_experiment(self, n_dialogues: int = 40) -> dict[str, Any]:
        cfg = self.system.cfg
        rng = random.Random(cfg.get("project.seed", 42))

        # 选择有足够字段的教材条目（中药/方剂）
        from .dataset import group_records

        records = [
            r for r in group_records(self.system.store.all_chunks()).values()
            if ("中药学" in r.book or "方剂学" in r.book) and len(r.term) >= 2
        ]
        rng.shuffle(records)
        dialogues = []
        for i in range(min(n_dialogues, len(records) // 2)):
            e, f = records[2 * i], records[2 * i + 1]
            pronoun_e = "这个方剂" if "方剂学" in e.book else "这味药"
            pronoun_f = "这个方剂" if "方剂学" in f.book else "这味药"
            turns = [
                (f"{e.term}的主要功效/功用是什么？", e.chunk_ids, False),
                (f"它的用法用量和注意事项呢？", e.chunk_ids, True),
                (f"{f.term}的主治证候有哪些？", f.chunk_ids, False),
                (f"{pronoun_f}的药物组成是什么？", f.chunk_ids, True),
                (f"我最近体质虚弱容易感冒，正在服用{e.term}调理，平时还需要注意什么？", None, False),
                (f"其实我并不是体质虚弱，我是阴虚火旺的体质，{e.term}还适合我吗？", None, False),
            ]
            dialogues.append(turns)

        def run_one(use_memory: bool) -> dict[str, Any]:
            stats = {"anaphora_total": 0, "anaphora_hit": 0, "anaphora_error": 0,
                     "anaphora_refused": 0, "non_anaphora_hit": [], "latency": []}
            for di, turns in enumerate(dialogues):
                session_id = f"mem_eval_{'on' if use_memory else 'off'}_{di}"
                self.system.reset_session(session_id)
                for text, gold, is_anaphora in turns:
                    if use_memory:
                        res = self.system.chat(session_id, text, decision_mode="react")
                    else:
                        res = self.system.agent.answer(text, decision_mode="react")
                    ids = [e["chunk_id"] for e in res.evidences]
                    stats["latency"].append(res.latency_ms)
                    if gold is None:
                        continue
                    hit = bool(set(ids) & set(gold))
                    if is_anaphora:
                        stats["anaphora_total"] += 1
                        if res.refused:
                            stats["anaphora_refused"] += 1
                        elif hit:
                            stats["anaphora_hit"] += 1
                        else:
                            stats["anaphora_error"] += 1
                    else:
                        stats["non_anaphora_hit"].append(1.0 if hit else 0.0)
            n = max(stats["anaphora_total"], 1)
            return {
                "anaphora_hit_rate": round(stats["anaphora_hit"] / n, 4),
                "anaphora_error_rate": round(stats["anaphora_error"] / n, 4),
                "anaphora_refusal_rate": round(stats["anaphora_refused"] / n, 4),
                "direct_question_hit_rate": round(float(np.mean(stats["non_anaphora_hit"])), 4)
                if stats["non_anaphora_hit"] else None,
                "avg_latency_ms": round(float(np.mean(stats["latency"])), 1),
            }

        print("  [exp5] 运行记忆关闭基线…")
        off = run_one(False)
        print("  [exp5] 运行记忆开启（分层记忆）…")
        on = run_one(True)
        ltm = self.system._get_long_term_memory()
        out = {
            "n_dialogues": len(dialogues),
            "memory_off": off,
            "memory_on": on,
            "delta_hit_rate": round(on["anaphora_hit_rate"] - off["anaphora_hit_rate"], 4),
            "delta_error_rate": round(on["anaphora_error_rate"] - off["anaphora_error_rate"], 4),
            "long_term_stats": ltm.to_dict(),
        }
        self.system.save_long_term_memory()
        print(f"  [exp5] 指代轮命中率 {off['anaphora_hit_rate']:.1%} → {on['anaphora_hit_rate']:.1%}，"
              f"错误率 {off['anaphora_error_rate']:.1%} → {on['anaphora_error_rate']:.1%}")
        return out

    # ==================================================================
    # 全量运行 + 报告
    # ==================================================================
    def run_all(self, e2e_sample: int | None = None) -> dict[str, Any]:
        reports_dir = self.system.cfg.path(self.system.cfg.get("project.reports_dir", "reports"))
        reports_dir.mkdir(parents=True, exist_ok=True)

        print("[exp1] 检索架构 A/B…")
        exp1 = self.run_retrieval_ab()
        print("[exp2] 信息保留率 + OCR 恢复…")
        exp2 = self.run_retention_and_ocr()
        print("[exp3] 端到端 RAGAS：static 基线…")
        exp3_static = self.run_e2e("static", sample=e2e_sample)
        print("[exp3] 端到端 RAGAS：react 动态决策…")
        exp3_react = self.run_e2e("react", sample=e2e_sample)
        print("[exp4] 延迟基准…")
        exp4 = self.run_latency()
        print("[exp5] 分层记忆实验…")
        n_dlg = self.system.cfg.get("evaluation.dialogues_for_memory_eval", 40)
        exp5 = self.run_memory_experiment(n_dialogues=n_dlg)

        results = {
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "n_eval_items": len(self.items),
            "exp1_retrieval_ab": exp1,
            "exp2_retention_ocr": exp2,
            "exp3_e2e_static": exp3_static["summary"],
            "exp3_e2e_react": exp3_react["summary"],
            "exp4_latency": exp4,
            "exp5_memory": exp5,
        }
        (reports_dir / "eval_results.json").write_text(
            json.dumps(results, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8"
        )
        # 明细行（可追溯分析）
        with open(reports_dir / "e2e_rows_static.jsonl", "w", encoding="utf-8") as f:
            for r in exp3_static["rows"]:
                f.write(json.dumps(r, ensure_ascii=False, default=_json_default) + "\n")
        with open(reports_dir / "e2e_rows_react.jsonl", "w", encoding="utf-8") as f:
            for r in exp3_react["rows"]:
                f.write(json.dumps(r, ensure_ascii=False, default=_json_default) + "\n")

        md = render_markdown_report(results)
        (reports_dir / "eval_report.md").write_text(md, encoding="utf-8")
        print(f"报告已写入 {reports_dir}")
        return results


def _json_default(o: Any) -> Any:
    if isinstance(o, float) and o != o:
        return None
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    return str(o)


def _aggregate_by_type(rows: list[dict]) -> dict[str, dict]:
    by: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by[r["type"]].append(r)
    out = {}
    for t, rs in by.items():
        answered = [r for r in rs if not r["refused"]]
        out[t] = {
            "n": len(rs),
            "ctx_recall": nanmean([r["ctx_recall"] for r in rs]),
            "ctx_precision": nanmean([r["ctx_precision"] for r in rs]),
            "faithfulness": nanmean([r["faithfulness"] for r in answered]),
            "refusal_rate": round(sum(1 for r in rs if r["refused"]) / len(rs), 4),
        }
    return out


def render_markdown_report(res: dict[str, Any]) -> str:
    e1 = res["exp1_retrieval_ab"]
    e2 = res["exp2_retention_ocr"]
    e3s, e3r = res["exp3_e2e_static"], res["exp3_e2e_react"]
    e4, e5 = res["exp4_latency"], res["exp5_memory"]
    ks = e1["ks"]

    lines = [
        "# 中医循证 RAG 系统评估报告",
        "",
        f"- 生成时间：{res['generated_at']}",
        f"- 测试集规模：{res['n_eval_items']} 条（含不可回答题）",
        "",
        "## 实验一：检索架构 A/B（混合召回 + 重排 + 二次过滤）",
        "",
        "| 模式 | " + " | ".join(f"Recall@{k}" for k in ks) + " | Top-1 | MRR |",
        "|---|" + "---|" * (len(ks) + 2),
    ]
    for mode, m in e1["modes"].items():
        row = [f"{m['recall@k'][str(k)]:.1%}" for k in ks]
        lines.append(f"| {mode} | " + " | ".join(row) + f" | {m['top1_accuracy']:.1%} | {m['mrr']:.3f} |")
    dense30 = e1["modes"]["dense"]["recall@k"][str(max(ks))]
    full30 = e1["modes"]["full"]["recall@k"][str(max(ks))]
    dtop1 = e1["modes"]["dense"]["top1_accuracy"]
    ftop1 = e1["modes"]["full"]["top1_accuracy"]
    lines += [
        "",
        f"Recall@{max(ks)}：纯向量基线 {dense30:.1%} → 完整管线 {full30:.1%}"
        f"（提升 {full30 - dense30:+.1%}）；Top-1：{dtop1:.1%} → {ftop1:.1%}（{ftop1 - dtop1:+.1%}）。",
        "",
        "## 实验二：长文信息保留率 + OCR 清洗恢复",
        "",
        f"- 信息保留率：朴素截断(512 token) **{e2['naive_truncation']['overall']:.1%}** → "
        f"语义切分+Token回退 **{e2['semantic_chunking']['overall']:.1%}**"
        f"（绝对提升 {e2['improvement_abs']:+.1%}，丢失信息减少 {e2['improvement_relative']:.1%}）",
        "",
        "| 来源类型 | 朴素截断 | 语义切分 |",
        "|---|---|---|",
    ]
    for st, v in e2["by_source_type"].items():
        lines.append(f"| {st} | {v['naive']:.1%} | {v['semantic']:.1%} |")
    if e2.get("ocr_recovery"):
        avg = e2["ocr_recovery"].get("_avg", {})
        if avg:
            lines += [
                "",
                f"- OCR 清洗：医药术语恢复率 {avg['term_recall_raw']:.1%} → {avg['term_recall_cleaned']:.1%}；"
                f"字符一致率 {avg['char_acc_raw']:.1%} → {avg['char_acc_cleaned']:.1%}",
            ]
    lines += [
        "",
        "## 实验三：端到端 RAGAS 指标（静态检索 vs CoT+ReAct 动态决策）",
        "",
        "| 指标 | static 基线 | react Agent | 变化 |",
        "|---|---|---|---|",
    ]

    def fmt(v: Any, pct: bool = True) -> str:
        if v is None or (isinstance(v, float) and v != v):
            return "—"
        return f"{v:.1%}" if pct else f"{v:.3f}"

    metrics_rows = [
        ("忠实度 faithfulness", e3s["faithfulness"], e3r["faithfulness"], True),
        ("上下文精确度 context_precision", e3s["context_precision"], e3r["context_precision"], True),
        ("上下文召回率 context_recall", e3s["context_recall"], e3r["context_recall"], True),
        ("答案相关度 answer_relevancy", e3s["answer_relevancy"], e3r["answer_relevancy"], True),
        ("不可回答题拒答正确率", e3s["refusal_accuracy_on_unanswerable"], e3r["refusal_accuracy_on_unanswerable"], True),
        ("可回答题过度拒答率", e3s["over_refusal_on_answerable"], e3r["over_refusal_on_answerable"], True),
        ("错误自信回答率", e3s["false_confidence_rate"], e3r["false_confidence_rate"], True),
        ("平均检索轮次", e3s["avg_rounds"], e3r["avg_rounds"], False),
    ]
    for name, a, b, pct in metrics_rows:
        delta = ""
        if isinstance(a, (int, float)) and isinstance(b, (int, float)) and a == a and b == b:
            delta = f"{b - a:+.1%}" if pct else f"{b - a:+.2f}"
        lines.append(f"| {name} | {fmt(a, pct)} | {fmt(b, pct)} | {delta} |")
    lines += [
        "",
        "## 实验四：延迟基准",
        "",
        f"- FAISS-HNSW（ntotal={e4['index_size']}, dim={e4['dim']}）纯检索："
        f"P50={e4['hnsw_search_ms']['p50']}ms / **P95={e4['hnsw_search_ms']['p95']}ms** / P99={e4['hnsw_search_ms']['p99']}ms",
        f"- 全管线（编码+双路召回+RRF+重排+二次过滤）："
        f"P50={e4['full_pipeline_ms']['p50']}ms / **P95={e4['full_pipeline_ms']['p95']}ms** / P99={e4['full_pipeline_ms']['p99']}ms",
        "",
        "## 实验五：分层记忆（多轮对话）",
        "",
        "| 指标 | 记忆关闭 | 分层记忆 | 变化 |",
        "|---|---|---|---|",
        f"| 指代轮命中率 | {e5['memory_off']['anaphora_hit_rate']:.1%} | {e5['memory_on']['anaphora_hit_rate']:.1%} | {e5['delta_hit_rate']:+.1%} |",
        f"| 指代轮错误率 | {e5['memory_off']['anaphora_error_rate']:.1%} | {e5['memory_on']['anaphora_error_rate']:.1%} | {e5['delta_error_rate']:+.1%} |",
        f"| 直接问题命中率 | {fmt(e5['memory_off']['direct_question_hit_rate'])} | {fmt(e5['memory_on']['direct_question_hit_rate'])} | — |",
        "",
        f"- 长期记忆统计：{json.dumps(e5['long_term_stats'].get('stats', {}), ensure_ascii=False)}",
        "",
        "---",
        "*评估全程离线可复现（固定随机种子）；明细见 reports/e2e_rows_*.jsonl，支持逐条证据引用追溯。*",
    ]
    return "\n".join(lines)
