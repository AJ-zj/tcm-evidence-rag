"""OCR 清洗器测试：形近字还原、插空格清理、乱码去除、噪声度评估。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tcm_rag.parsing.ocr_clean import OCRTextCleaner

VOCAB = ["甘草", "麻黄", "桂枝汤", "伤寒", "寒热", "脾", "黄芪", "茯苓"]


def make_cleaner():
    return OCRTextCleaner(vocab_terms=VOCAB)


def test_fix_confusable_with_vocab():
    c = make_cleaner()
    res = c.clean("甘革 味甘平，主五赃六府寒熟邪气。")
    assert "甘草" in res.text
    assert "甘革" not in res.text
    assert "五脏六腑" in res.text or "五赃六腑" not in res.text
    assert res.n_fixes >= 2


def test_remove_cjk_spaces():
    c = make_cleaner()
    res = c.clean("太 阳 之 为 病 ，脉 浮 。")
    assert "太 阳" not in res.text
    assert "太阳之为病" in res.text
    assert res.n_space_removed >= 4


def test_remove_garbage_chars():
    c = make_cleaner()
    res = c.clean("麻黄\ufffd汤□主之◆")
    assert "\ufffd" not in res.text and "□" not in res.text and "◆" not in res.text
    assert res.n_garbage >= 3


def test_static_fixes():
    c = OCRTextCleaner(vocab_terms=[])
    res = c.clean("伤塞中风，头痛发热。")
    assert "伤寒中风" in res.text


def test_noise_score_ordering():
    c = make_cleaner()
    clean_text = "桂枝汤主之。太阳病，头痛发热，汗出恶风。"
    noisy_text = "桂 枝 汤 王 之。□太 阳 病\ufffd，头 痛 发 热，汗 出 恶 风。"
    assert c.noise_score(noisy_text) > c.noise_score(clean_text)
    assert c.noise_score(clean_text) < 0.05


def test_clean_reduces_noise():
    c = make_cleaner()
    noisy = "甘 革 味甘平□，主五赃六府寒熟邪气，麻 黄 汤王 之。"
    res = c.clean(noisy)
    assert res.noise_after <= res.noise_before
    assert "甘草" in res.text


def test_fullwidth_normalization():
    c = make_cleaner()
    res = c.clean("剂量３ｇ，每日２次。")
    assert "3g" in res.text and "2" in res.text
