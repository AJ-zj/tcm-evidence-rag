"""补采语料：从 Huatuo26M-Lite 定向补充"生活调理/食疗"类问答（方案A）。

背景：首批 600 条中医科 QA 缺少食疗/生活调理话题（"生姜红糖水"类追问无法命中）。
本脚本从全量语料按食疗关键词定向补采，独立输出为 supplement 文件，
不改动原 huatuo_tcm.jsonl（保持首批语料与既有评估报告的可追溯性）。

选材规则（可复现，seed 固定）：
1. score>=4 优先，不足 300 条再用 score==3 补足
2. 过滤：answer>=40 字、question>=8 字、id 与首批不重复
3. 话题覆盖：按命中的首要关键词分组轮转采样，避免单一关键词刷屏

用法： .venv/Scripts/python scripts/supplement_corpus.py [--n 300]
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import OrderedDict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

KEYWORDS = ["食疗", "食补", "偏方", "煮水", "泡水", "姜汤", "生姜", "红糖", "葱白",
            "饮食调理", "忌口", "饮食注意", "枸杞", "红枣", "蜂蜜", "泡脚", "山药", "薏米"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_file = ROOT / "data" / "corpus" / "qa" / "huatuo_tcm_supplement.jsonl"
    out_file.parent.mkdir(parents=True, exist_ok=True)

    # 首批已用的 id
    seen_ids: set[int] = set()
    base = ROOT / "data" / "corpus" / "qa" / "huatuo_tcm.jsonl"
    if base.exists():
        with open(base, encoding="utf-8") as f:
            for line in f:
                seen_ids.add(json.loads(line)["id"])

    # 扫描全量语料
    by_kw: "OrderedDict[str, list[dict]]" = OrderedDict()
    raw = ROOT / "data" / "raw" / "Huatuo26M-Lite" / "format_data.jsonl"
    with open(raw, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if row["id"] in seen_ids:
                continue
            if int(row.get("score", 0)) < 3:
                continue
            q, a = str(row.get("question", "")).strip(), str(row.get("answer", "")).strip()
            if len(a) < 40 or len(q) < 8:
                continue
            hit = [k for k in KEYWORDS if k in q or k in a]
            if hit:
                by_kw.setdefault(hit[0], []).append(row)

    rng = random.Random(args.seed)
    picked: list[dict] = []
    picked_ids: set[int] = set()

    def take(pool: list[dict], n: int) -> None:
        """从候选池取至多 n 条（高分优先），全局上限 args.n。"""
        rng.shuffle(pool)
        pool.sort(key=lambda r: -int(r.get("score", 0)))  # 高分优先，同分保持随机
        taken = 0
        for row in pool:
            if len(picked) >= args.n or taken >= n:
                return
            if row["id"] in picked_ids:
                continue
            picked.append(row)
            picked_ids.add(row["id"])
            taken += 1

    # 阶段0：用户会追问的具体食材词（姜/红糖/葱白/煮水等）强优先保底，
    # 每组至多 40 条——没有这步，泛话题词（饮食调理）会把配额挤光
    PRIORITY = ["姜汤", "生姜", "红糖", "葱白", "煮水", "泡水", "偏方"]
    for k in PRIORITY:
        if len(picked) >= args.n:
            break
        if k in by_kw:
            take(by_kw[k], 40)

    # 第一轮：其余关键词组轮转取 1 条，保证话题覆盖面
    groups = [k for k, v in by_kw.items() if v]
    progress = True
    while len(picked) < args.n and progress:
        progress = False
        for k in groups:
            pool = [r for r in by_kw[k] if r["id"] not in picked_ids and int(r.get("score", 0)) >= 4]
            if pool:
                take(pool, 1)
                progress = True
            if len(picked) >= args.n:
                break
    # 第二轮：不足则用 score==3 补
    if len(picked) < args.n:
        rest = [r for rows in by_kw.values() for r in rows
                if r["id"] not in picked_ids]
        take(rest, args.n - len(picked))

    with open(out_file, "w", encoding="utf-8") as f:
        for row in picked:
            keep = {k: row.get(k) for k in ("id", "question", "answer", "score", "label", "related_diseases")}
            f.write(json.dumps(keep, ensure_ascii=False) + "\n")

    from collections import Counter

    kw_dist = Counter(next(k for k in KEYWORDS if k in row.get("question", "") + row.get("answer", ""))
                      for row in picked)
    print(f"补采 {len(picked)} 条 → {out_file}")
    print("话题分布:", dict(kw_dist.most_common()))


if __name__ == "__main__":
    main()
