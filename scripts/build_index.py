"""构建全部索引 artifacts：语料 → 清洗 → 切分 → 向量化 → FAISS-HNSW + BM25。

用法： .venv/Scripts/python scripts/build_index.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tcm_rag.pipeline import RAGSystem  # noqa: E402


def main() -> None:
    system = RAGSystem.build(verbose=True)
    print("\n=== manifest ===")
    print(json.dumps(system.manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
