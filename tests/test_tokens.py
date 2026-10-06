"""token 估算与句子切分测试。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tcm_rag.utils.tokens import count_tokens, split_sentences, truncate_to_tokens


def test_count_tokens_cjk():
    assert count_tokens("") == 0
    # 中文 1 字 ≈ 1 token
    n = count_tokens("桂枝汤主之")
    assert 5 <= n <= 6


def test_count_tokens_ascii():
    n = count_tokens("aspirin 100mg")
    assert 2 <= n <= 6


def test_truncate_to_tokens():
    text = "太阳之为病脉浮头项强痛而恶寒" * 20
    truncated = truncate_to_tokens(text, 30)
    assert count_tokens(truncated) <= 30
    assert text.startswith(truncated)
    assert truncate_to_tokens("短文本", 100) == "短文本"


def test_split_sentences():
    text = "太阳之为病，脉浮。头项强痛而恶寒！发热？汗出；"
    sents = split_sentences(text)
    assert len(sents) == 4
    assert sents[0].endswith("。")
    assert all(s.strip() for s in sents)
