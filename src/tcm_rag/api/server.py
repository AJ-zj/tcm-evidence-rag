"""FastAPI 服务：对话 / 检索 / 会话记忆状态 / 健康检查 + 静态 Web 演示界面。

启动： .venv/Scripts/python -m uvicorn tcm_rag.api.server:app --port 8000
或：   .venv/Scripts/python scripts/serve.py
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from tcm_rag.pipeline import RAGSystem  # noqa: E402

app = FastAPI(title="中医循证检索增强问答系统", version="1.0.0")

_SYSTEM: RAGSystem | None = None


def get_system() -> RAGSystem:
    global _SYSTEM
    if _SYSTEM is None:
        _SYSTEM = RAGSystem.load(verbose=True)
    return _SYSTEM


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=2000)
    session_id: str = "web_demo"
    mode: str = Field(default="react", pattern="^(react|static)$")


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=500)
    mode: str = Field(default="full", pattern="^(dense|bm25|hybrid|hybrid_rerank|full)$")
    top_k: int = Field(default=5, ge=1, le=30)


@app.get("/api/health")
def health() -> dict[str, Any]:
    s = get_system()
    return {
        "status": "ok",
        "n_chunks": len(s.store),
        "embedder": s.embedder.name,
        "llm": "openai_compatible" if s.llm else "offline_extractive",
        "manifest": s.manifest,
    }


@app.post("/api/chat")
def chat(req: ChatRequest) -> dict[str, Any]:
    s = get_system()
    result = s.chat(req.session_id, req.message, decision_mode=req.mode)
    s.save_long_term_memory()
    return result.to_dict()


@app.post("/api/search")
def search(req: SearchRequest) -> dict[str, Any]:
    s = get_system()
    rr = s.search(req.query, mode=req.mode, top_k=req.top_k)
    return {
        "query": rr.query,
        "mode": rr.mode,
        "timings_ms": rr.timings_ms,
        "total_ms": rr.total_ms,
        "evidences": [
            {
                "chunk_id": e.chunk.chunk_id,
                "title": e.chunk.title,
                "source_type": e.chunk.source_type,
                "score": round(e.score, 4),
                "ocr_noise": e.chunk.ocr_noise,
                "channel_scores": {k: round(v, 4) for k, v in e.channel_scores.items()},
                "text": e.chunk.text[:600],
            }
            for e in rr.evidences
        ],
        "dropped": [
            {"chunk_id": e.chunk.chunk_id, "reason": e.drop_reason, "score": round(e.score, 4)}
            for e in rr.dropped
        ],
    }


@app.get("/api/session/{session_id}")
def session_state(session_id: str) -> dict[str, Any]:
    s = get_system()
    sess = s.session(session_id)
    return sess.state()


@app.delete("/api/session/{session_id}")
def reset_session(session_id: str) -> dict[str, str]:
    get_system().reset_session(session_id)
    return {"status": "reset", "session_id": session_id}


@app.get("/")
def index() -> FileResponse:
    static = Path(__file__).parent / "static" / "index.html"
    if not static.exists():
        raise HTTPException(500, "static/index.html 缺失")
    return FileResponse(static)


app.mount("/static", StaticFiles(directory=Path(__file__).parent / "static"), name="static")
