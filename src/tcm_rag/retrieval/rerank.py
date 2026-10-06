"""重排层：特征重排（默认，零额外依赖）与 CrossEncoder 重排（可选 BGE-reranker）。

特征重排融合六类信号：稠密相似度、BM25 归一分、RRF 融合分、
医疗实体覆盖率、OCR 清洁度、来源先验——对中医古籍场景显式抑制噪声证据。
"""
from __future__ import annotations

from typing import Protocol, Sequence

from ..embedding import Embedder
from ..ner.medical_ner import MedicalNER
from ..schema import Chunk, Evidence

SOURCE_PRIOR = {
    "textbook": 1.0,
    "classic": 0.95,
    "case": 0.9,
    "qa": 0.8,
    "pdf": 0.85,
    "pdf_scan": 0.75,
    "ocr_scan": 0.7,
}


class Reranker(Protocol):
    def rerank(self, query: str, evidences: Sequence[Evidence]) -> list[Evidence]:
        ...


def _bigrams(text: str) -> set[str]:
    chars = "".join(text.split())
    return {chars[i:i + 2] for i in range(len(chars) - 1)}


class FeatureReranker:
    """线性特征加权重排。所有特征归一到 [0,1]，权重和为 1。"""

    name = "feature"

    def __init__(
        self,
        ner: MedicalNER,
        embedder: Embedder | None = None,
        weights: dict[str, float] | None = None,
    ):
        self.ner = ner
        self.embedder = embedder
        self.weights = {
            "dense": 0.30,
            "bm25": 0.15,
            "rrf": 0.10,
            "entity_overlap": 0.25,
            "clean": 0.12,
            "source_prior": 0.08,
        }
        if weights:
            self.weights.update(weights)

    # ------------------------------------------------------------------
    def _entity_overlap(self, query: str, chunk: Chunk) -> float:
        q_ents = {e.text for e in self.ner.recognize(query)}
        if not q_ents:
            # 无词典实体时退化为查询词 bigram 覆盖
            qb = _bigrams(query)
            cb = _bigrams(chunk.text)
            return len(qb & cb) / max(len(qb), 1)
        c_ents = {e.text for e in self.ner.recognize(chunk.text)}
        return len(q_ents & c_ents) / len(q_ents)

    def score(self, query: str, ev: Evidence) -> float:
        cs = ev.channel_scores
        dense = max(0.0, min(1.0, cs.get("dense", 0.0)))
        bm25 = max(0.0, min(1.0, cs.get("bm25_norm", 0.0)))
        rrf = max(0.0, min(1.0, cs.get("rrf_norm", 0.0)))
        ent = self._entity_overlap(query, ev.chunk)
        clean = max(0.0, 1.0 - ev.chunk.ocr_noise * 4.0)  # 噪声放大惩罚
        prior = SOURCE_PRIOR.get(ev.chunk.source_type, 0.8)
        w = self.weights
        total = (
            w["dense"] * dense
            + w["bm25"] * bm25
            + w["rrf"] * rrf
            + w["entity_overlap"] * ent
            + w["clean"] * clean
            + w["source_prior"] * prior
        )
        return round(total, 4)

    def rerank(self, query: str, evidences: Sequence[Evidence]) -> list[Evidence]:
        out = []
        for ev in evidences:
            s = self.score(query, ev)
            ev.channel_scores = {**ev.channel_scores, "feature_rerank": s}
            ev.score = s
            out.append(ev)
        out.sort(key=lambda e: e.score, reverse=True)
        return out


class CrossEncoderReranker:
    """BGE-reranker CrossEncoder 重排（可选）；与特征分加权融合。"""

    name = "cross_encoder"

    def __init__(
        self,
        model_name: str,
        ner: MedicalNER,
        feature_weight: float = 0.3,
        device: str = "cpu",
        hf_endpoint: str | None = None,
    ):
        import os

        if hf_endpoint:
            os.environ.setdefault("HF_ENDPOINT", hf_endpoint)
        from sentence_transformers import CrossEncoder

        self._model = CrossEncoder(model_name, device=device)
        self._feature = FeatureReranker(ner)
        self.feature_weight = feature_weight

    def rerank(self, query: str, evidences: Sequence[Evidence]) -> list[Evidence]:
        if not evidences:
            return []
        pairs = [(query, ev.chunk.text[:1024]) for ev in evidences]
        logits = self._model.predict(pairs, show_progress_bar=False)
        import math

        out = []
        for ev, lg in zip(evidences, logits):
            ce = 1.0 / (1.0 + math.exp(-float(lg)))  # sigmoid → [0,1]
            feat = self._feature.score(query, ev)
            final = (1 - self.feature_weight) * ce + self.feature_weight * feat
            ev.channel_scores = {**ev.channel_scores, "ce_sigmoid": round(ce, 4), "feature_rerank": feat}
            ev.score = round(final, 4)
            out.append(ev)
        out.sort(key=lambda e: e.score, reverse=True)
        return out


class ApiReranker:
    """远程 CrossEncoder 重排（SiliconFlow /v1/rerank 等 Jina 兼容接口）。

    无需本地下载模型；与特征分加权融合，接口异常时自动回退纯特征重排。
    """

    name = "api_cross_encoder"

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        ner: MedicalNER,
        feature_weight: float = 0.3,
        timeout: float = 30.0,
        max_doc_chars: int = 1500,
    ):
        self._feature = FeatureReranker(ner)
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.feature_weight = feature_weight
        self.timeout = timeout
        self.max_doc_chars = max_doc_chars
        self.last_error: str = ""

    def _api_scores(self, query: str, evidences: Sequence[Evidence]) -> list[float] | None:
        import httpx

        payload = {
            "model": self.model,
            "query": query,
            "documents": [ev.chunk.text[: self.max_doc_chars] for ev in evidences],
            "return_documents": False,
        }
        try:
            resp = httpx.post(
                f"{self.base_url}/rerank",
                json=payload,
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=self.timeout,
            )
            resp.raise_for_status()
            results = resp.json()["results"]
            scores = [0.0] * len(evidences)
            for item in results:
                idx, s = int(item["index"]), float(item["relevance_score"])
                if 0 <= idx < len(scores):
                    scores[idx] = max(0.0, min(1.0, s))
            self.last_error = ""
            return scores
        except Exception as e:  # noqa: BLE001
            self.last_error = str(e)
            return None

    def rerank(self, query: str, evidences: Sequence[Evidence]) -> list[Evidence]:
        if not evidences:
            return []
        ce_scores = self._api_scores(query, evidences)
        if ce_scores is None:
            import warnings

            warnings.warn(f"API 重排失败（{self.last_error}），回退特征重排。")
            return self._feature.rerank(query, evidences)
        out = []
        for ev, ce in zip(evidences, ce_scores):
            feat = self._feature.score(query, ev)
            final = (1 - self.feature_weight) * ce + self.feature_weight * feat
            ev.channel_scores = {**ev.channel_scores, "ce_api": round(ce, 4), "feature_rerank": feat}
            ev.score = round(final, 4)
            out.append(ev)
        out.sort(key=lambda e: e.score, reverse=True)
        return out


def create_reranker(cfg, ner: MedicalNER, embedder: Embedder | None = None) -> Reranker:
    kind = cfg.get("retrieval.reranker", "feature")
    if kind == "cross_encoder":
        try:
            return CrossEncoderReranker(
                model_name=cfg.get("retrieval.cross_encoder_model", "BAAI/bge-reranker-base"),
                ner=ner,
                hf_endpoint=cfg.get("embedding.hf_endpoint"),
            )
        except Exception as e:  # noqa: BLE001
            import warnings

            warnings.warn(f"CrossEncoder 加载失败（{e}），回退特征重排。")
    if kind == "api":
        base_url = cfg.get("retrieval.api_rerank_url") or cfg.get("llm.base_url") or ""
        api_key = cfg.get("retrieval.api_rerank_key") or cfg.get("llm.api_key") or ""
        model = cfg.get("retrieval.api_rerank_model", "BAAI/bge-reranker-v2-m3")
        if base_url and api_key:
            return ApiReranker(
                base_url=base_url, api_key=api_key, model=model, ner=ner,
                timeout=float(cfg.get("llm.timeout_seconds", 120)) or 30.0,
            )
        import warnings

        warnings.warn("retrieval.reranker=api 但缺少 base_url/api_key（llm.* 未配置），回退特征重排。")
    return FeatureReranker(ner=ner, embedder=embedder)
