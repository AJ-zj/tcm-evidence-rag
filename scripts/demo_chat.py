"""CLI 交互式演示：多轮对话 + 证据引用 + ReAct 决策轨迹 + 记忆状态。

用法： .venv/Scripts/python scripts/demo_chat.py [--mode react|static] [--session demo]
命令： /trace 查看上一轮决策轨迹；/memory 查看记忆状态；/mode react|static 切换；/quit 退出
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tcm_rag.pipeline import RAGSystem  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="react", choices=["react", "static"])
    parser.add_argument("--session", default="cli_demo")
    args = parser.parse_args()

    print("加载系统…")
    system = RAGSystem.load(verbose=True)
    print(f"就绪：{len(system.store)} 个证据块 | LLM: {'在线' if system.llm else '离线抽取式'} | 模式: {args.mode}")
    print("输入问题开始对话（/quit 退出，/trace 轨迹，/memory 记忆，/mode 切换）\n")

    mode = args.mode
    last = None
    while True:
        try:
            question = input("问> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not question:
            continue
        if question == "/quit":
            break
        if question == "/trace" and last is not None:
            for step in last.trace:
                print(f"\n── Round {step.round} ──")
                print(f"Thought: {step.thought}")
                print(f"Action: {step.action} {step.action_input}")
                print(f"Observation: {step.observation[:300]}")
            continue
        if question == "/memory":
            sess = system.session(args.session)
            print(f"短期记忆摘要: {sess.short_term.summary[:200] or '(空)'}")
            print(f"活跃轮次: {len(sess.short_term._active_turns())}")
            if sess.long_term:
                import json as _json

                print("长期记忆: " + _json.dumps(sess.long_term.to_dict(), ensure_ascii=False))
            continue
        if question.startswith("/mode"):
            parts = question.split()
            if len(parts) > 1 and parts[1] in ("react", "static"):
                mode = parts[1]
                print(f"已切换到 {mode} 模式")
            continue

        result = system.chat(args.session, question, decision_mode=mode)
        last = result
        print(f"\n答> {result.answer}")
        if result.citations:
            print("\n引用证据:")
            for c in result.citations[:5]:
                print(f"  {c}")
        print(
            f"\n[confidence={result.confidence:.2f} consistency={result.consistency:.2f} "
            f"coverage={result.coverage:.2f} rounds={result.rounds_used} "
            f"latency={result.latency_ms:.0f}ms"
            + (" REFUSED" if result.refused else "")
            + "]\n"
        )
    system.save_long_term_memory()
    print("再见。")


if __name__ == "__main__":
    main()
