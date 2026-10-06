"""向量化后端：BGE（sentence-transformers）与离线哈希回退。"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence

import numpy as np


class Embedder(ABC):
    """嵌入器接口：文档与查询分开编码（BGE 检索场景查询侧需加指令）。"""

    name: str = "base"
    dim: int = 0

    @abstractmethod
    def encode_documents(self, texts: Sequence[str], batch_size: int = 64) -> np.ndarray:
        ...

    @abstractmethod
    def encode_queries(self, texts: Sequence[str], batch_size: int = 64) -> np.ndarray:
        ...

    def encode_one(self, text: str, is_query: bool = False) -> np.ndarray:
        fn = self.encode_queries if is_query else self.encode_documents
        return fn([text])[0]


class BGEEmbedder(Embedder):
    """BAAI/bge-small-zh-v1.5：512 维，L2 归一化后内积即余弦相似度。"""

    def __init__(
        self,
        model_name: str = "BAAI/bge-small-zh-v1.5",
        device: str = "cpu",
        query_instruction: str = "为这个句子生成表示以用于检索相关文章：",
        hf_endpoint: str | None = None,
    ):
        import os

        if hf_endpoint:
            os.environ.setdefault("HF_ENDPOINT", hf_endpoint)
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name, device=device)
        self.name = model_name
        get_dim = getattr(self._model, "get_embedding_dimension", None) or self._model.get_sentence_embedding_dimension
        self.dim = int(get_dim())
        self.query_instruction = query_instruction

    def encode_documents(self, texts: Sequence[str], batch_size: int = 64) -> np.ndarray:
        vecs = self._model.encode(
            list(texts),
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=len(texts) > 512,
            convert_to_numpy=True,
        )
        return np.asarray(vecs, dtype=np.float32)

    def encode_queries(self, texts: Sequence[str], batch_size: int = 64) -> np.ndarray:
        prefixed = [self.query_instruction + t for t in texts]
        vecs = self._model.encode(
            prefixed,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return np.asarray(vecs, dtype=np.float32)


class HashingEmbedder(Embedder):
    """离线回退：字符 unigram+bigram 哈希向量（次线性 TF，L2 归一化）。

    无网络/无 torch 环境下保证系统可运行；语义能力弱于 BGE，
    但对中文短查询的字面匹配检索仍有效。
    """

    def __init__(self, dim: int = 512):
        self.name = f"hash-{dim}"
        self.dim = dim

    def _vectorize(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        chars = [c for c in text if not c.isspace()]
        grams: list[str] = list(chars)
        grams += [chars[i] + chars[i + 1] for i in range(len(chars) - 1)]
        counts: dict[str, int] = {}
        for g in grams:
            counts[g] = counts.get(g, 0) + 1
        for g, c in counts.items():
            h = hash(g) & 0x7FFFFFFF
            idx = h % self.dim
            sign = 1.0 if (h >> 31) & 1 == 0 else -1.0
            v[idx] += sign * (1.0 + np.log(c))  # 次线性 TF
        norm = float(np.linalg.norm(v))
        if norm > 0:
            v /= norm
        return v

    def encode_documents(self, texts: Sequence[str], batch_size: int = 64) -> np.ndarray:
        return np.stack([self._vectorize(t) for t in texts]) if texts else np.zeros((0, self.dim), np.float32)

    def encode_queries(self, texts: Sequence[str], batch_size: int = 64) -> np.ndarray:
        return self.encode_documents(texts)


def create_embedder(cfg) -> Embedder:
    """按配置创建嵌入器；BGE 加载失败自动降级为哈希回退。"""
    provider = cfg.get("embedding.provider", "bge")
    if provider == "bge":
        kwargs = dict(
            model_name=cfg.get("embedding.model_name", "BAAI/bge-small-zh-v1.5"),
            device=cfg.get("embedding.device", "cpu"),
            query_instruction=cfg.get(
                "embedding.query_instruction", "为这个句子生成表示以用于检索相关文章："
            ),
            hf_endpoint=cfg.get("embedding.hf_endpoint"),
        )
        try:
            return BGEEmbedder(**kwargs)
        except Exception:  # noqa: BLE001
            # 网络检查（huggingface.co 不可达等）偶发挂起失败：
            # 强制走本地缓存离线模式重试一次，仍失败再降级哈希嵌入。
            import os
            import warnings

            os.environ["HF_HUB_OFFLINE"] = "1"
            os.environ["TRANSFORMERS_OFFLINE"] = "1"
            try:
                emb = BGEEmbedder(**kwargs)
                warnings.warn("BGE 通过离线缓存模式加载成功。")
                return emb
            except Exception as e:  # noqa: BLE001
                warnings.warn(f"BGE 加载失败（{e}），回退到哈希嵌入。检索质量会下降。")
    return HashingEmbedder(dim=cfg.get("embedding.dim", 512))
