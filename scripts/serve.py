"""启动 Web 服务： .venv/Scripts/python scripts/serve.py [--port 8000]"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", type=str, default="127.0.0.1")
    args = parser.parse_args()

    import uvicorn

    # 预热：启动时即加载模型与索引，避免首个请求等待
    from tcm_rag.api.server import get_system

    print("预热加载索引与模型…")
    get_system()
    print(f"启动服务： http://{args.host}:{args.port}")
    uvicorn.run("tcm_rag.api.server:app", host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
