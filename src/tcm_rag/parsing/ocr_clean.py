"""OCR 噪声清洗与噪声度评估。

针对古籍/教材扫描件的常见 OCR 干扰：
1. 形近字混淆（己/已/巳、未/末、日/曰、草/革、热/熟……）
2. CJK 字符间被插入空格、全角/半角混用
3. 乱码残留（U+FFFD、□ 等占位符）、控制字符
4. 标点误识

清洗策略：
- 规范化（去控制字符、全角转半角、去 CJK 间空格、压缩空白）
- 词表驱动的形近字还原：仅当"替换后能命中医药词表、且替换前不是词表词"时才纠正，
  避免误改正常的多音/多义字
- 静态高频医疗 OCR 错误词表兜底

噪声度 noise_score ∈ [0,1]，由乱码字符占比、CJK 间插空格占比、
被纠正的形近字占比加权得到，供二次证据过滤使用。
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Iterable

# 形近混淆对（对称使用）
CONFUSABLE_PAIRS: list[tuple[str, str]] = [
    ("己", "已"), ("巳", "已"), ("未", "末"), ("日", "曰"), ("土", "士"),
    ("干", "千"), ("入", "人"), ("太", "大"), ("玉", "王"), ("自", "白"),
    ("刀", "刃"), ("天", "夫"), ("贝", "见"), ("戊", "戌"), ("汨", "汩"),
    ("草", "革"), ("热", "熟"), ("脾", "牌"), ("风", "凤"), ("灸", "炙"),
    ("祟", "崇"), ("盲", "肓"), ("折", "拆"), ("酒", "洒"), ("治", "冶"),
    ("令", "今"), ("丸", "九"), ("汤", "场"), ("梗", "便"), ("芍", "苍"),
    ("茯", "伏"), ("苓", "荃"), ("芪", "杞"), ("归", "旧"), ("芎", "穹"),
    ("冬", "东"), ("仁", "仕"), ("蜜", "密"), ("桂", "挂"), ("麻", "床"),
    ("黄", "寅"), ("连", "莲"), ("翘", "翅"), ("银", "垠"), ("膏", "高"),
    ("附", "付"), ("乌", "鸟"), ("梅", "悔"), ("姜", "美"), ("枣", "刺"),
]

# 高频医疗领域 OCR 错误词（无词表时兜底）
STATIC_FIXES: dict[str, str] = {
    "甘革": "甘草", "寒熟": "寒热", "补牌": "补脾", "黄茂": "黄芪",
    "栓楼": "栝楼", "括楼": "栝楼", "桂枚": "桂枝", "伤塞": "伤寒",
    "温瘸": "温疟", "咳逆上乞": "咳逆上气", "五赃六腑": "五脏六腑",
    "心下痞便": "心下痞硬", "烦燥": "烦躁", "麻黄汤王": "麻黄汤主之",
    "银翹": "银翘", "伏苓": "茯苓", "灸甘草": "炙甘草", "栓蒌": "栝蒌",
}

_CJK = r"\u3400-\u4dbf\u4e00-\u9fff"
RE_CJK_SPACE = re.compile(rf"(?<=[{_CJK}])[ \t\u3000]+(?=[{_CJK}])")
RE_MULTI_SPACE = re.compile(r"[ \t]{2,}")
RE_MULTI_NL = re.compile(r"\n{3,}")
RE_GARBAGE = re.compile(r"[\ufffd\u25a1\u25a0\u25c6\u2605\ue000-\uf8ff]")
RE_CTRL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


@dataclass
class CleanResult:
    text: str
    noise_before: float
    noise_after: float
    n_fixes: int = 0
    n_space_removed: int = 0
    n_garbage: int = 0
    fixes_applied: list[str] = field(default_factory=list)


class OCRTextCleaner:
    """OCR 文本清洗器。vocab_terms 可选：医药词表（用于形近字安全还原）。"""

    def __init__(self, vocab_terms: Iterable[str] | None = None, enabled: bool = True):
        self.enabled = enabled
        self.vocab: set[str] = {t for t in (vocab_terms or []) if len(t) >= 2}
        self._variant_map: dict[str, str] = {}
        self._build_variant_map()

    def _confusables_of(self, ch: str) -> list[str]:
        out = []
        for a, b in CONFUSABLE_PAIRS:
            if ch == a:
                out.append(b)
            elif ch == b:
                out.append(a)
        return out

    def _build_variant_map(self) -> None:
        """对词表中每个词生成单字形近变体 → 原词 的映射（长词优先）。"""
        for term in sorted(self.vocab, key=len, reverse=True):
            for i, ch in enumerate(term):
                for alt in self._confusables_of(ch):
                    variant = term[:i] + alt + term[i + 1:]
                    if variant == term or variant in self.vocab:
                        continue
                    # 同一变体若映射到多个词，保留更长/首个（按字典序稳定）
                    self._variant_map.setdefault(variant, term)

    # ------------------------------------------------------------------
    def normalize(self, text: str) -> tuple[str, int, int]:
        """基础规范化，返回 (文本, 去除空格数, 乱码字符数)。"""
        n_garbage = len(RE_GARBAGE.findall(text))
        text = RE_CTRL.sub("", text)
        text = RE_GARBAGE.sub("", text)
        # 全角字母/数字 → 半角（保留中文全角标点，，。！？：；等）
        out = []
        for ch in text:
            code = ord(ch)
            if (0xFF10 <= code <= 0xFF19) or (0xFF21 <= code <= 0xFF3A) or (0xFF41 <= code <= 0xFF5A):
                out.append(chr(code - 0xFEE0))
            else:
                out.append(ch)
        text = "".join(out)
        n_space = len(RE_CJK_SPACE.findall(text))
        text = RE_CJK_SPACE.sub("", text)
        text = RE_MULTI_SPACE.sub(" ", text)
        text = RE_MULTI_NL.sub("\n\n", text)
        return text, n_space, n_garbage

    def fix_confusions(self, text: str) -> tuple[str, int, list[str]]:
        """词表驱动 + 静态词表的形近字纠正。返回 (文本, 纠正次数, 明细)。"""
        fixes: list[str] = []
        n = 0
        # 静态高频错误优先
        for bad, good in STATIC_FIXES.items():
            if bad in text and bad != good:
                c = text.count(bad)
                text = text.replace(bad, good)
                n += c
                fixes.append(f"{bad}→{good}×{c}")
        # 词表变体纠正：仅当变体命中且原词不在该位置时替换
        if self._variant_map:
            for variant, term in self._variant_map.items():
                if variant in text:
                    c = text.count(variant)
                    text = text.replace(variant, term)
                    n += c
                    fixes.append(f"{variant}→{term}×{c}")
        return text, n, fixes

    def clean(self, text: str) -> CleanResult:
        noise_before = self.noise_score(text)
        if not self.enabled:
            return CleanResult(text=text, noise_before=noise_before, noise_after=noise_before)
        norm, n_space, n_garbage = self.normalize(text)
        fixed, n_fixes, fixes = self.fix_confusions(norm)
        noise_after = self.noise_score(fixed, n_fixes=0)
        return CleanResult(
            text=fixed,
            noise_before=noise_before,
            noise_after=noise_after,
            n_fixes=n_fixes,
            n_space_removed=n_space,
            n_garbage=n_garbage,
            fixes_applied=fixes[:50],
        )

    # ------------------------------------------------------------------
    def noise_score(self, text: str, n_fixes: int | None = None) -> float:
        """噪声度 ∈ [0,1]：乱码占比 + CJK 插空格占比 + 形近字纠正占比 加权。"""
        if not text:
            return 0.0
        n = len(text)
        garbage = len(RE_GARBAGE.findall(text))
        spaces = len(RE_CJK_SPACE.findall(text))
        if n_fixes is None:
            # 估算潜在形近错误：统计混淆字集中字符在非词表上下文的出现（粗估）
            fixes, cnt, _ = self.fix_confusions(text)  # noqa: F841
            n_fixes = cnt
        score = (garbage * 1.0 + spaces * 0.6 + n_fixes * 1.2) / max(n, 1)
        return round(min(score, 1.0), 4)
