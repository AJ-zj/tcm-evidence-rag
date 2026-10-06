"""评估测试集构建：从已切分语料自动生成带金标准证据引用的问答测试集。

金标准粒度为"条目"（record）：教材的一条药味/方剂/病种/证型、古籍的一条条文、
一份医案、一条真实医患 QA。若问题明确指向条目内某字段且该字段唯一落在某个块中，
则金标准收紧到该块。

题型：
- qa:          Huatuo 真实医患问答（原问题 + 原答案作参考答案）
- herb:        中药学条目模板题（性味归经/功效主治/用法用量/使用注意）
- formula:     方剂学条目模板题（出处/组成/功用主治/方解/使用注意）
- multi_hop:   药→方 桥接题（含有 X 的方剂 Y 的主治功用）
- disease:     中医内科学病种模板题（病因病机/辨证论治/预防调护）
- pattern:     中医诊断学证型模板题（临床表现/辨证要点/治法代表方）
- diagnosis:   四诊/八纲条目题
- classic:     古籍原文题（条文号/关键实体/原文片段）
- case:        医案辨证处方题
- unanswerable: 语料中无答案的问题（检验"证据不足时拒答"，gold 为空）
"""
from __future__ import annotations

import json
import random
import re
from collections import OrderedDict
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

from ..ner.medical_ner import MedicalNER
from ..schema import Chunk

# 语料中不应存在答案的问题候选池（拒答测试，gold 为空）。
# 覆盖：非医学技术/金融/文化 + 医学邻域但超出本语料范围的主题。
# 构建时按"与语料字符 bigram 重叠率"自动筛选，剔除语料实际覆盖的主题
# （如 Huatuo 中医科问答中可能出现的西医内容），保证 gold 标注诚实。
UNANSWERABLE_QUESTIONS: list[str] = [
    # --- 非医学：信息技术 / 工程 ---
    "量子计算机的超导比特工作温度是多少？",
    "区块链工作量证明机制的吞吐量上限是多少？",
    "5G基站的单站满载功耗是多少瓦？",
    "锂离子电池的能量密度上限是多少瓦时每千克？",
    "大语言模型GPT-4的参数总量是多少？",
    "液体火箭发动机的比冲一般是多少秒？",
    "光伏并网逆变器的转换效率极限是多少？",
    "单模光纤的通信波长是多少纳米？",
    "机械硬盘的磁头飞行高度是多少纳米？",
    "台积电3纳米制程的晶体管密度是多少？",
    "蓝牙5.0的理论传输速率是多少兆比特每秒？",
    "高铁复兴号的最高运营时速是多少公里？",
    "汽车涡轮增压器的常规工作转速是多少？",
    "航空煤油的辛烷值和热值是多少？",
    "无线充电Qi协议的功率上限是多少瓦？",
    # --- 非医学：金融 / 法律 / 文化 / 体育 ---
    "房贷利率LPR报价机制是怎么形成的？",
    "中国个人所得税专项附加扣除有哪些项目？",
    "公司法规定注册资本最低限额是多少？",
    "司法考试的科目和通过率是多少？",
    "诺贝尔物理学奖2023年授予了谁？",
    "《红楼梦》金陵十二钗分别是谁？",
    "国际足联世界杯历届冠军有哪些国家队？",
    "夏商周断代工程的结论是什么？",
    "欧盟碳排放交易体系的配额如何分配？",
    "股票市场的涨停板幅度规则是什么？",
    "全球定位系统GPS卫星的数量和轨道高度是多少？",
    "莎士比亚四大悲剧分别是哪几部？",
    # --- 医学邻域但明确超出本语料范围（设备/操作/检验/分子） ---
    "阿司匹林的化学分子式和合成路线是什么？",
    "青霉素皮试液的标准配制浓度是多少？",
    "PET-CT检查的辐射剂量当量是多少？",
    "达芬奇手术机器人的机械臂自由度有几个？",
    "胰岛素泵的基础率如何设定？",
    "心脏支架植入术的操作步骤是什么？",
    "心脏瓣膜置换术的术式选择标准是什么？",
    "骨髓穿刺术的操作要点是什么？",
    "胃镜下黏膜切除术的适应证是什么？",
    "呼吸机SIMV模式的参数如何设置？",
    "ECMO上机的血流动力学标准是什么？",
    "心脏起搏器程控参数如何优化？",
    "他汀类药物引起横纹肌溶解的分子机制是什么？",
    "地高辛的血药浓度治疗窗是多少？",
    "胺碘酮的负荷量和维持量分别是多少？",
    "糖尿病视网膜病变的分期标准是什么？",
    "肿瘤TNM分期的具体标准是什么？",
    "人类基因组计划测序耗资多少美元？",
    "CRISPR基因编辑的专利归属纠纷结果如何？",
    "mRNA疫苗的脂质纳米颗粒配方是什么？",
    "单克隆抗体药物的临床试验终点如何设计？",
    "CT图像重建的滤波反投影算法原理是什么？",
    "血气分析中Beers校准公式的推导过程是什么？",
    "流式细胞术的荧光补偿矩阵如何调节？",
    "质谱仪的离子源温度一般设置多少度？",
]

def _bigrams_of(text: str) -> set[str]:
    chars = "".join(text.split())
    return {chars[i:i + 2] for i in range(len(chars) - 1)}


HERB_TEMPLATES = [
    ("中药{t}的性味归经是什么？", "性味归经"),
    ("{t}有什么功效？主要用于治疗什么病证？", "功效"),
    ("中药{t}的用法用量是怎样的？", "用法用量"),
    ("使用中药{t}需要注意什么？哪些人慎用或忌用？", "使用注意"),
]
FORMULA_TEMPLATES = [
    ("{t}出自哪部医籍？", "出处"),
    ("{t}的药物组成有哪些？", "组成"),
    ("{t}的功用和主治证候是什么？", "功用"),
    ("{t}的方解（君臣佐使配伍意义）是怎样的？", "方解"),
    ("{t}在临床使用时有什么注意事项？", "使用注意"),
]
DISEASE_TEMPLATES = [
    ("中医内科学中{t}的病因病机是什么？", "病因病机"),
    ("{t}如何辨证论治？常见证型和代表方有哪些？", "辨证论治"),
    ("{t}的预防调护要注意什么？", "预防调护"),
    ("{t}的诊查要点有哪些？", "诊查要点"),
]
PATTERN_TEMPLATES = [
    ("{t}的临床表现有哪些？", "临床表现"),
    ("{t}的辨证要点是什么？", "辨证要点"),
    ("{t}的治法和代表方是什么？", "治法"),
]
CASE_TEMPLATES = [
    ("医案中{t}是如何辨证分析和处方用药的？", "辨证"),
    ("治疗{t}的医案用了什么方剂加减？", "处方"),
]


@dataclass
class EvalItem:
    qid: str
    question: str
    gold_chunk_ids: list[str]
    item_type: str
    reference_answer: str = ""
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "EvalItem":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class Record:
    """一个"条目"：同 title_path（或同 doc_id）的块组。"""

    key: str
    book: str
    term: str
    chunks: list[Chunk]
    source_type: str

    @property
    def text(self) -> str:
        return "\n".join(c.text for c in self.chunks)

    @property
    def chunk_ids(self) -> list[str]:
        return [c.chunk_id for c in self.chunks]

    def chunk_with_field(self, marker: str) -> list[str]:
        hits = [c.chunk_id for c in self.chunks if marker in c.text]
        return hits if len(hits) == 1 else self.chunk_ids


def group_records(chunks: list[Chunk]) -> "OrderedDict[tuple, Record]":
    """按 (doc_id, title_path) 归组为条目；QA 按 doc_id。"""
    groups: "OrderedDict[tuple, list[Chunk]]" = OrderedDict()
    for c in chunks:
        if c.source_type == "qa":
            key = (c.doc_id,)
        else:
            key = (c.doc_id, tuple(c.title_path))
        groups.setdefault(key, []).append(c)

    records: "OrderedDict[tuple, Record]" = OrderedDict()
    for key, cs in groups.items():
        cs = sorted(cs, key=lambda c: c.seq)
        first = cs[0]
        if first.source_type == "qa":
            book, term = "医案问答", first.meta.get("doc_title", first.doc_id)
        else:
            path = first.title_path
            book = path[0] if path else ""
            term = path[-1] if len(path) >= 2 else book
        records[key] = Record(
            key=str(key), book=book, term=term, chunks=cs, source_type=first.source_type
        )
    return records


class EvalSetBuilder:
    def __init__(
        self,
        chunks: list[Chunk],
        ner: MedicalNER,
        seed: int = 42,
        target_size: int = 700,
        unanswerable_count: int = 30,
    ):
        self.chunks = chunks
        self.ner = ner
        self.rng = random.Random(seed)
        self.target_size = target_size
        self.unanswerable_count = unanswerable_count
        self.records = list(group_records(chunks).values())

    # ------------------------------------------------------------------
    def _records_of_book(self, book_kw: str) -> list[Record]:
        return [r for r in self.records if book_kw in r.book]

    def _pick_template(self, templates: list[tuple[str, str]], rec: Record) -> tuple[str, str]:
        """优先选择条目文本中确实存在对应字段标记的模板。"""
        candidates = [(tpl, fld) for tpl, fld in templates if f"【{fld}】" in rec.text or fld in rec.text]
        if candidates:
            return self.rng.choice(candidates)
        return self.rng.choice(templates)

    # ------------------------------------------------------------------
    def build(self) -> list[EvalItem]:
        items: list[EvalItem] = []
        idx = 0

        def add(question: str, gold: list[str], itype: str, ref: str = "", **meta: Any) -> None:
            nonlocal idx
            items.append(EvalItem(qid=f"q{idx:04d}", question=question, gold_chunk_ids=gold,
                                  item_type=itype, reference_answer=ref, meta=meta))
            idx += 1

        # 1) QA 真实问答
        for r in self.records:
            if r.source_type != "qa":
                continue
            meta0 = r.chunks[0].meta
            question = str(meta0.get("question", "")).strip()
            if not question:
                continue
            add(question, r.chunk_ids, "qa", ref=str(meta0.get("answer", "")),
                label=meta0.get("label", ""), diseases=meta0.get("related_diseases", ""))

        # 2) 中药学
        for r in self._records_of_book("中药学"):
            if not r.term or len(r.term) > 6:
                continue
            tpl, fld = self._pick_template(HERB_TEMPLATES, r)
            add(tpl.format(t=r.term), r.chunk_with_field(fld), "herb", field=fld, term=r.term)

        # 3) 方剂学
        formula_records = [r for r in self._records_of_book("方剂学") if r.term]
        for r in formula_records:
            tpl, fld = self._pick_template(FORMULA_TEMPLATES, r)
            add(tpl.format(t=r.term), r.chunk_with_field(fld), "formula", field=fld, term=r.term)

        # 3b) 多跳：药 → 含该药的方剂
        herb_names = set(self.ner.vocab.get("herb", set()))
        for r in formula_records:
            herbs_in = [h for h in herb_names if h in r.text and len(h) >= 2 and h != r.term]
            if herbs_in:
                h = self.rng.choice(sorted(herbs_in))
                add(f"含有{h}的方剂{r.term}的主治证候和功用是什么？", r.chunk_ids,
                    "multi_hop", term=r.term, bridge_entity=h)

        # 4) 中医内科学
        for r in self._records_of_book("中医内科学"):
            if not r.term or r.term == r.book:
                continue
            tpl, fld = self._pick_template(DISEASE_TEMPLATES, r)
            add(tpl.format(t=r.term), r.chunk_with_field(fld), "disease", field=fld, term=r.term)

        # 5) 中医诊断学
        for r in self._records_of_book("中医诊断学"):
            if not r.term or r.term == r.book:
                continue
            if r.term.endswith("证"):
                tpl, fld = self._pick_template(PATTERN_TEMPLATES, r)
                add(tpl.format(t=r.term), r.chunk_with_field(fld), "pattern", field=fld, term=r.term)
            else:
                add(f"{r.term}的临床意义和诊察要点是什么？", r.chunk_ids, "diagnosis", term=r.term)

        # 6) 古籍原文题
        classic_books: dict[str, list[Record]] = {}
        for r in self.records:
            if r.source_type in ("classic", "ocr_scan", "pdf_scan", "pdf"):
                classic_books.setdefault(r.book, []).append(r)
        for book, recs in classic_books.items():
            for r in recs:
                if len(recs) > 60 and self.rng.random() > 0.4:
                    continue  # 大书抽样控制题量
                if not r.term or r.term == book:
                    continue
                last = r.chunks[0].title_path[-1] if r.chunks[0].title_path else ""
                if re.search(r"第.{1,6}条", last):
                    add(f"《{book}》{last}的原文内容是什么？", r.chunk_ids, "classic",
                        book=book, clause=last)
                    continue
                ents = [e for e in self.ner.recognize(r.text) if e.type in ("formula", "herb", "disease", "pattern")]
                if ents:
                    key_ent = ents[0].text
                    add(f"《{book}》中关于{key_ent}是怎么论述的？", r.chunk_ids,
                        "classic", book=book, entity=key_ent)
                else:
                    body = re.sub(r"^〈[^〉]*〉\s*", "", r.text.strip())
                    snippet = body[:14].strip()
                    if len(snippet) >= 8:
                        add(f"“{snippet}……”这段原文出自哪里？其完整内容是什么？", r.chunk_ids,
                            "classic", book=book)

        # 7) 医案题
        for r in self._records_of_book("教学医案"):
            ents = {e.text for e in self.ner.recognize(r.text)
                    if e.type in ("pattern", "disease") and len(e.text) >= 3}
            if not ents:
                ents = {e.text for e in self.ner.recognize(r.text) if e.type == "formula"}
            if not ents:
                continue
            t = self.rng.choice(sorted(ents))
            tpl, fld = self.rng.choice(CASE_TEMPLATES)
            add(tpl.format(t=t), r.chunk_ids, "case", term=t)

        # 8) 不可回答题：候选池按"与语料 bigram 重叠率"自动校验，
        #    剔除语料实际覆盖的主题（保证 gold 标注诚实），取重叠最低的若干条
        unans = self._select_unanswerable()
        for q, ov in unans:
            add(q, [], "unanswerable", corpus_bigram_overlap=round(ov, 3))

        # 分层采样到 target_size
        items = self._stratified_sample(items)
        for i, it in enumerate(items):
            it.qid = f"q{i:04d}"
        return items

    def _select_unanswerable(self) -> list[tuple[str, float]]:
        """从候选池中选择与语料重叠率最低的问题。"""
        corpus_bigrams: set[str] = set()
        for c in self.chunks:
            text = "".join(c.text.split())
            corpus_bigrams.update(text[i:i + 2] for i in range(len(text) - 1))

        def overlap(q: str) -> float:
            qb = _bigrams_of(q)
            return len(qb & corpus_bigrams) / max(len(qb), 1)

        scored = sorted(((q, overlap(q)) for q in UNANSWERABLE_QUESTIONS), key=lambda x: x[1])
        return scored[: self.unanswerable_count]

    def _stratified_sample(self, items: list[EvalItem]) -> list[EvalItem]:
        if len(items) <= self.target_size:
            return items
        by_type: dict[str, list[EvalItem]] = {}
        for it in items:
            by_type.setdefault(it.item_type, []).append(it)
        kept: list[EvalItem] = []
        unans = by_type.pop("unanswerable", [])
        kept.extend(unans)
        remaining = self.target_size - len(unans)
        pool_total = sum(len(v) for v in by_type.values())
        for t, its in by_type.items():
            self.rng.shuffle(its)
            quota = max(1, round(remaining * len(its) / max(pool_total, 1)))
            kept.extend(its[:quota])
        self.rng.shuffle(kept)
        return kept[: self.target_size]

    # ------------------------------------------------------------------
    @staticmethod
    def save(items: list[EvalItem], path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            for it in items:
                f.write(json.dumps(it.to_dict(), ensure_ascii=False) + "\n")

    @staticmethod
    def load(path: str | Path) -> list[EvalItem]:
        items = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    items.append(EvalItem.from_dict(json.loads(line)))
        return items

    @staticmethod
    def type_counts(items: list[EvalItem]) -> dict[str, int]:
        out: dict[str, int] = {}
        for it in items:
            out[it.item_type] = out.get(it.item_type, 0) + 1
        return out
