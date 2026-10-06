"""LLM 客户端抽象：OpenAI 兼容 API（可配 DashScope/Qwen 等）+ 离线回退标记。

系统设计为"LLM 可选"：未配置 API 时，Agent 的检索决策由证据统计信号驱动
（覆盖率/一致性/置信度），回答由离线抽取式引擎生成，全链路仍可运行与评估。
"""
from __future__ import annotations

import json
from typing import Any


class LLMError(RuntimeError):
    pass


class OpenAICompatibleLLM:
    """任何 OpenAI 兼容 /chat/completions 接口（DashScope、vLLM、Ollama…）。"""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        temperature: float = 0.2,
        max_tokens: int = 1024,
        timeout: float = 90.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.available = True

    def complete(self, system: str, user: str, **overrides: Any) -> str:
        import httpx

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": overrides.get("temperature", self.temperature),
            "max_tokens": overrides.get("max_tokens", self.max_tokens),
        }
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        try:
            resp = httpx.post(
                f"{self.base_url}/chat/completions",
                json=payload,
                headers=headers,
                timeout=self.timeout,
            )
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"].strip()
        except Exception as e:  # noqa: BLE001
            raise LLMError(f"LLM 调用失败: {e}") from e

    def complete_json(self, system: str, user: str, **overrides: Any) -> dict[str, Any]:
        text = self.complete(system, user, **overrides)
        # 容错解析：剥掉 ```json 包裹
        text = text.strip()
        if text.startswith("```"):
            text = text.strip("`")
            if text.startswith("json"):
                text = text[4:]
        start, end = text.find("{"), text.rfind("}")
        if start >= 0 and end > start:
            text = text[start:end + 1]
        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            raise LLMError(f"LLM 输出非合法 JSON: {e}\n原文: {text[:500]}") from e


def create_llm(cfg) -> OpenAICompatibleLLM | None:
    provider = cfg.get("llm.provider", "offline")
    if provider == "offline":
        return None
    base_url = cfg.get("llm.base_url") or ""
    api_key = cfg.get("llm.api_key") or ""
    model = cfg.get("llm.model") or ""
    if not (base_url and api_key and model):
        import warnings

        warnings.warn("llm.provider=openai_compatible 但缺少 base_url/api_key/model，回退离线模式。")
        return None
    return OpenAICompatibleLLM(
        base_url=base_url,
        api_key=api_key,
        model=model,
        temperature=cfg.get("llm.temperature", 0.2),
        max_tokens=cfg.get("llm.max_tokens", 1024),
        timeout=cfg.get("llm.timeout_seconds", 90),
    )
