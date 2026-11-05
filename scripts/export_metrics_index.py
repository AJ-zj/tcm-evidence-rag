"""汇总仓库内已有评测产物，生成可追溯的指标明细索引。

用法：
    python scripts/export_metrics_index.py

输出：
    reports/metrics_index.json

只读 reports/ 下的 JSON / JSONL 与 config/default.yaml，不重新调用任何 LLM、
嵌入或检索服务，几秒内可复现。本脚本不产生新实验数据，只把散落在多份产物里的
指标值、字段路径、样本量与复现命令集中到一处，便于逐条核对简历 / README 数字。
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "reports"


def load(name):
    return json.loads((R / name).read_text(encoding="utf-8"))


def rows(name):
    return [json.loads(line) for line in (R / name).read_text(encoding="utf-8").splitlines() if line.strip()]


def pct(v):
    return round(v * 100, 2)


api = load("eval_results_reranker_api.json")
full = load("eval_results.json")
faith = load("faithfulness_experiment.json")["summary"]
ragas = load("ragas_official.json")["summary"]
static_rows = rows("e2e_rows_static.jsonl")
react_rows = rows("e2e_rows_react.jsonl")

metrics = []


def add(name, display, value, source, field, sample, reproduce, note=""):
    item = {
        "name": name,
        "display": display,
        "value": value,
        "source": source,
        "field": field,
        "sample": sample,
        "reproduce": reproduce,
    }
    if note:
        item["note"] = note
    metrics.append(item)


API = "reports/eval_results_reranker_api.json"
FULL = "reports/eval_results.json"
REPRO_API = "python scripts/run_reranker_eval.py --reranker api"
REPRO_FULL = "python scripts/run_eval.py"

# ---- 批次 A：API CrossEncoder 重排，670 条可答题（README 实验一） ----
api_dense = api["modes"]["dense"]
api_final = api["modes"]["full"]
api_hybrid = api["modes"]["hybrid"]
add("检索 Top-1 基线", "82.5%", pct(api_dense["top1_accuracy"]), API,
    "modes.dense.top1_accuracy", f"n_queries={api['n_queries']}", REPRO_API)
add("检索 Top-1 全管线", "97.5%", pct(api_final["top1_accuracy"]), API,
    "modes.full.top1_accuracy (= modes.hybrid_rerank)", f"n_queries={api['n_queries']}", REPRO_API)
add("检索 Top-1 绝对提升", "+15.0pp", round(pct(api_final["top1_accuracy"]) - pct(api_dense["top1_accuracy"]), 2), API,
    "modes.full.top1_accuracy - modes.dense.top1_accuracy", f"n_queries={api['n_queries']}", REPRO_API,
    "由两端字段相减得到，JSON 内无单独的提升字段")
add("RRF 融合阶段贡献", "+3.0pp", round(pct(api_hybrid["top1_accuracy"]) - pct(api_dense["top1_accuracy"]), 2), API,
    "modes.hybrid.top1_accuracy - modes.dense.top1_accuracy", f"n_queries={api['n_queries']}", REPRO_API)
add("重排阶段贡献", "+11.9pp", round(pct(api_final["top1_accuracy"]) - pct(api_hybrid["top1_accuracy"]), 2), API,
    "modes.full.top1_accuracy - modes.hybrid.top1_accuracy", f"n_queries={api['n_queries']}", REPRO_API,
    "README 表格另有「hybrid + 特征重排」中间档，该档未落盘 JSON，故不纳入索引")
add("Recall@1 基线 / 全管线", "81.4% → 96.3%",
    [pct(api_dense["recall@k"]["1"]), pct(api_final["recall@k"]["1"])], API,
    "modes.dense.recall@k.1 → modes.full.recall@k.1", f"n_queries={api['n_queries']}", REPRO_API)
add("MRR 基线 / 全管线", "0.867 → 0.983",
    [api_dense["mrr"], api_final["mrr"]], API,
    "modes.dense.mrr → modes.full.mrr", f"n_queries={api['n_queries']}", REPRO_API)
add("重排后端", "API CrossEncoder", api.get("reranker_model", api.get("reranker")), API,
    "reranker / reranker_model", f"n_queries={api['n_queries']}", REPRO_API)

# ---- 批次 B：默认配置完整管线，699 条（eval_report.md） ----
f_dense = full["exp1_retrieval_ab"]["modes"]["dense"]
f_final = full["exp1_retrieval_ab"]["modes"]["full"]
add("默认管线 Top-1", "87.7% → 98.5%",
    [pct(f_dense["top1_accuracy"]), pct(f_final["top1_accuracy"])], FULL,
    "exp1_retrieval_ab.modes.dense/full.top1_accuracy",
    f"n_eval_items={full['n_eval_items']}, exp1.n_queries={full['exp1_retrieval_ab']['n_queries']}", REPRO_FULL)
exp2 = full["exp2_retention_ocr"]
add("长文信息保留率", "59.5% → 87.8%",
    [pct(exp2["naive_truncation"]["overall"]), pct(exp2["semantic_chunking"]["overall"])], FULL,
    "exp2_retention_ocr.naive_truncation.overall → semantic_chunking.overall",
    f"sentences={exp2['naive_truncation']['sentences']}", REPRO_FULL,
    "README 实验二表格写 51.5%→87.9%，属另一批次展示值，仓库无对应明细文件")
ocr = exp2["ocr_recovery"]["_avg"]
add("OCR 医药术语恢复率", "90.2% → 97.6%",
    [pct(ocr["term_recall_raw"]), pct(ocr["term_recall_cleaned"])], FULL,
    "exp2_retention_ocr.ocr_recovery._avg.term_recall_raw → term_recall_cleaned",
    "模拟 OCR 文档平均", REPRO_FULL)

# ---- 批次 B 逐题明细：拒答率可由 jsonl 重算 ----
for label, data, src in [
    ("static", static_rows, "reports/e2e_rows_static.jsonl"),
    ("react", react_rows, "reports/e2e_rows_react.jsonl"),
]:
    un = [r for r in data if r["type"] == "unanswerable"]
    ans = [r for r in data if r["type"] != "unanswerable"]
    refused = [r for r in un if r.get("refused")]
    over = [r for r in ans if r.get("refused")]
    add(f"{label} 超范围题拒答正确率", f"{len(refused)}/{len(un)} = {pct(len(refused)/len(un))}%",
        round(len(refused) / len(un), 4), src,
        "逐行统计 type=='unanswerable' 且 refused==true", f"总 {len(data)} 行，不可答 {len(un)} 条",
        REPRO_FULL, "可直接由本文件逐题重算")
    add(f"{label} 可答题过度拒答", f"{len(over)}/{len(ans)}", len(over), src,
        "逐行统计 type!='unanswerable' 且 refused==true", f"可答 {len(ans)} 条", REPRO_FULL)

add("端到端拒答提升", "0% → 100%",
    [full["exp3_e2e_static"]["refusal_accuracy_on_unanswerable"],
     full["exp3_e2e_react"]["refusal_accuracy_on_unanswerable"]], FULL,
    "exp3_e2e_static.refusal_accuracy_on_unanswerable → exp3_e2e_react",
    f"n_items={full['exp3_e2e_react']['n_items']}", REPRO_FULL,
    "与 e2e_rows_*.jsonl 逐题重算一致；README 实验三表格的 90.0% 与本明细不符，不采用")

# ---- 独立复核实验（不属于上述两批） ----
add("生成方式对照忠实度", "56.0% → 88.8%",
    [pct(faith["faithfulness_free_generation"]), pct(faith["faithfulness_rag_grounded"])],
    "reports/faithfulness_experiment.json",
    "summary.faithfulness_free_generation → faithfulness_rag_grounded",
    f"n_answerable={faith['n_answerable']} + n_unanswerable={faith['n_unanswerable']} = 48 题",
    "python scripts/run_faithfulness_experiment.py")
add("超范围题硬答率", "62.5% → 0%",
    [pct(faith["free_hard_answer_rate_on_unanswerable"]),
     round((1 - faith["rag_refusal_rate_on_unanswerable"]) * 100, 2)],
    "reports/faithfulness_experiment.json",
    "summary.free_hard_answer_rate_on_unanswerable → (1 - rag_refusal_rate_on_unanswerable)",
    f"n_unanswerable={faith['n_unanswerable']}（8 条，非 48 题总体）",
    "python scripts/run_faithfulness_experiment.py",
    "硬答率分母是 8 条不可答题，不可写成 48 题的硬答率")
add("官方 RAGAS Faithfulness", "0.954", ragas["faithfulness"],
    "reports/ragas_official.json", "summary.faithfulness",
    f"n={ragas['n']}, judge={ragas['judge_model']}",
    "python scripts/run_ragas_official.py",
    "与内置离线规则实现是两套裁判口径，不可互比")

# ---- 规模与配置（可追溯性说明） ----
add("测试集规模", "700 条（670 可答 + 30 不可答）", 700,
    "config/default.yaml", "evaluation.test_set_size / evaluation.unanswerable_count / seed",
    "seed=42", "python scripts/make_eval_set.py",
    "逐题明细见 reports/e2e_rows_*.jsonl（699 行）")
add("语料规模", "610 篇 / 1067 块", 1067,
    "README.md", "第 102 / 256 行语料规模说明（非 JSON 字段）",
    "教学演示规模", "python scripts/build_corpus.py",
    "仓库未提交语料统计 JSON；1067 亦见 FAISS-HNSW ntotal 说明。此项为 README 记录值，非评测产物字段")

index = {
    "schema_version": 1,
    "generated_by": "scripts/export_metrics_index.py",
    "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "description": "简历与 README 中所有量化指标的明细出处索引。每个条目含值、来源文件、字段路径、样本量与复现命令。",
    "caveats": [
        "本仓库存在两批不同评测运行（670 题 API 重排 / 699 题默认管线），数字不可直接互比。",
        "README 实验二、三表格中的部分展示值没有对应明细文件；本索引只收录可从 reports/ 与 config/ 追溯的值。",
        "官方 RAGAS 与生成方式对照是两套独立复核实验，不属于上述任一评测批次。",
    ],
    "metrics": metrics,
}

out = R / "metrics_index.json"
out.write_text(json.dumps(index, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(f"已生成 {out.relative_to(ROOT)}，共 {len(metrics)} 条指标")
for m in metrics:
    if m["name"] in ("长文信息保留率", "端到端拒答提升", "超范围题硬答率"):
        print(f"  {m['name']}: {m['display']} <- {m['source']} {m['field'][:60]}")
