"""医疗 NER：Trie 词典最大匹配 + 医疗实体加权重要度。

设计说明：中医文本实体高度术语化（药名/方剂/证型/症状），词典最大匹配
比通用序列标注模型更可控、零推理依赖，且便于与教材词表动态合并。
重要度 importance(text) 由命中实体的类型权重聚合而成（饱和函数保证 [0,1]），
用于短期记忆衰减加权与长期记忆入库阈值。
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .seed_vocab import SEED_VOCAB


@dataclass
class Entity:
    text: str
    type: str          # herb | formula | symptom | pattern | disease | western_disease | meridian | other
    start: int
    end: int


class _TrieNode:
    __slots__ = ("children", "term_type")

    def __init__(self) -> None:
        self.children: dict[str, _TrieNode] = {}
        self.term_type: str | None = None


class MedicalNER:
    """基于词典 Trie 的最大正向匹配 NER。"""

    # 类型权重（config 可覆盖）
    DEFAULT_WEIGHTS = {
        "formula": 1.0,
        "herb": 0.9,
        "disease": 0.85,
        "pattern": 0.85,
        "symptom": 0.7,
        "western_disease": 0.6,
        "meridian": 0.45,
        "other": 0.3,
    }

    def __init__(
        self,
        vocab: dict[str, Iterable[str]] | None = None,
        weights: dict[str, float] | None = None,
        base_importance: float = 0.15,
        max_entity_bonus: float = 0.85,
    ):
        self.vocab: dict[str, set[str]] = {
            t: set(terms) for t, terms in (vocab or SEED_VOCAB).items()
        }
        self.weights = {**self.DEFAULT_WEIGHTS, **(weights or {})}
        self.base_importance = base_importance
        self.max_entity_bonus = max_entity_bonus
        self._root = _TrieNode()
        self._build_trie()
        self._all_terms: list[str] = sorted(
            {term for terms in self.vocab.values() for term in terms}, key=len, reverse=True
        )

    # ------------------------------------------------------------------
    def _build_trie(self) -> None:
        # 同词多类型时优先级：formula > herb > pattern > disease > western_disease > symptom > meridian > other
        priority = ["formula", "herb", "pattern", "disease", "western_disease", "symptom", "meridian", "other"]
        merged: dict[str, str] = {}
        for t in reversed(priority):
            for term in self.vocab.get(t, set()):
                if len(term) >= 2:
                    merged[term] = t
        for term, ttype in merged.items():
            node = self._root
            for ch in term:
                node = node.children.setdefault(ch, _TrieNode())
            node.term_type = ttype
        self.term_types = merged

    def add_terms(self, terms: dict[str, str]) -> None:
        """动态追加词条（term → type），重建 Trie。"""
        for t, terms_set in self.vocab.items():
            extra = [term for term, ty in terms.items() if ty == t]
            terms_set.update(extra)
        self._build_trie()

    # ------------------------------------------------------------------
    def recognize(self, text: str) -> list[Entity]:
        """最大正向匹配识别医疗实体。"""
        entities: list[Entity] = []
        i = 0
        n = len(text)
        while i < n:
            node = self._root
            j = i
            last_match: tuple[int, str] | None = None
            while j < n and text[j] in node.children:
                node = node.children[text[j]]
                j += 1
                if node.term_type is not None:
                    last_match = (j, node.term_type)
            if last_match:
                end, ttype = last_match
                entities.append(Entity(text=text[i:end], type=ttype, start=i, end=end))
                i = end
            else:
                i += 1
        return entities

    def entity_types(self, text: str) -> dict[str, list[str]]:
        """按类型分组返回去重实体文本。"""
        out: dict[str, list[str]] = {}
        for e in self.recognize(text):
            bucket = out.setdefault(e.type, [])
            if e.text not in bucket:
                bucket.append(e.text)
        return out

    # ------------------------------------------------------------------
    def importance(self, text: str) -> float:
        """医疗实体加权重要度 ∈ [base, base+bonus]，饱和聚合。"""
        types = self.entity_types(text)
        raw = sum(self.weights.get(t, 0.3) * len(v) for t, v in types.items())
        bonus = self.max_entity_bonus * (1.0 - math.exp(-raw / 2.0))
        return round(min(1.0, self.base_importance + bonus), 4)

    @property
    def all_terms(self) -> list[str]:
        return self._all_terms

    # ------------------------------------------------------------------
    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        data = {t: sorted(s) for t, s in self.vocab.items()}
        p.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path, **kwargs) -> "MedicalNER":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(vocab=data, **kwargs)


def build_vocab_from_corpus(
    textbook_dir: str | Path,
    huatuo_jsonl: str | Path | None = None,
    extra_terms: dict[str, str] | None = None,
) -> dict[str, list[str]]:
    """种子词表 + 教材条目 + Huatuo 疾病词 合并。

    教材归类：中药学→herb，方剂学→formula，中医内科学→disease，中医诊断学→pattern/other。
    """
    vocab: dict[str, set[str]] = {t: set(s) for t, s in SEED_VOCAB.items()}
    book_type_map = {
        "中药学": "herb",
        "方剂学": "formula",
        "中医内科学": "disease",
        "中医诊断学": "pattern",
    }
    tdir = Path(textbook_dir)
    if tdir.exists():
        for jf in sorted(tdir.glob("*.json")):
            try:
                data = json.loads(jf.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue
            book = data.get("book", jf.stem)
            ttype = book_type_map.get(book, "other")
            for chapter in data.get("chapters", []):
                for record in chapter.get("records", []):
                    term = (record.get("term") or "").strip()
                    if 2 <= len(term) <= 12:
                        vocab.setdefault(ttype, set()).add(term)

    if huatuo_jsonl and Path(huatuo_jsonl).exists():
        diseases: set[str] = set()
        with open(huatuo_jsonl, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                for d in str(row.get("related_diseases", "")).split("|"):
                    d = d.strip()
                    if 2 <= len(d) <= 12:
                        diseases.add(d)
        vocab.setdefault("western_disease", set()).update(diseases)

    for term, ttype in (extra_terms or {}).items():
        vocab.setdefault(ttype, set()).add(term)

    return {t: sorted(s) for t, s in vocab.items()}
