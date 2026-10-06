"""配置加载：config/default.yaml + 环境变量覆盖（.env）。"""
from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _load_dotenv() -> None:
    """轻量 .env 加载（存在 python-dotenv 则用之，否则手工解析）。"""
    env_file = PROJECT_ROOT / ".env"
    if not env_file.exists():
        return
    try:
        from dotenv import load_dotenv

        load_dotenv(env_file)
        return
    except ImportError:
        pass
    for line in env_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip())


class Config:
    """点号路径访问的配置对象，如 cfg.get('retrieval.final_top_k')。"""

    def __init__(self, data: dict[str, Any], root: Path):
        self._data = data
        self.root = root

    def get(self, path: str, default: Any = None) -> Any:
        node: Any = self._data
        for part in path.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def section(self, name: str) -> dict[str, Any]:
        value = self._data.get(name, {})
        return copy.deepcopy(value) if isinstance(value, dict) else {}

    def path(self, relative: str) -> Path:
        p = Path(relative)
        return p if p.is_absolute() else (self.root / p)

    @property
    def data(self) -> dict[str, Any]:
        return copy.deepcopy(self._data)


def load_config(config_file: str | Path | None = None) -> Config:
    _load_dotenv()
    cfg_file = Path(config_file) if config_file else PROJECT_ROOT / "config" / "default.yaml"
    with open(cfg_file, encoding="utf-8") as f:
        data = yaml.safe_load(f)

    # 环境变量覆盖（部署时优先级最高）
    if os.environ.get("EMBEDDING_PROVIDER"):
        data["embedding"]["provider"] = os.environ["EMBEDDING_PROVIDER"]
    if os.environ.get("LLM_PROVIDER"):
        data["llm"]["provider"] = os.environ["LLM_PROVIDER"]
    if os.environ.get("LLM_BASE_URL"):
        data["llm"]["base_url"] = os.environ["LLM_BASE_URL"]
    if os.environ.get("LLM_API_KEY"):
        data["llm"]["api_key"] = os.environ["LLM_API_KEY"]
    if os.environ.get("LLM_MODEL"):
        data["llm"]["model"] = os.environ["LLM_MODEL"]
    if os.environ.get("HF_ENDPOINT"):
        data["embedding"]["hf_endpoint"] = os.environ["HF_ENDPOINT"]

    return Config(data, PROJECT_ROOT)
