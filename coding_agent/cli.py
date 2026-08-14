"""Command-line interface for the coding agent.

Usage examples::

    python -m coding_agent --root C:\\path\\to\\project analyze --output report.md
    python -m coding_agent --root . run "add a unit test for utils.py"
    python -m coding_agent --root . --read-only chat
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.errors import GraphRecursionError
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from . import __version__
from .config import Config
from .graph import build_graph
from .llm import build_llm
from .prompts import ANALYZE_TASK, build_system_prompt

console = Console()


# ------------------------------------------------------------------ helpers --
def _compact_args(args: dict, limit: int = 200) -> str:
    parts = []
    for k, v in args.items():
        s = str(v)
        if len(s) > limit:
            s = s[:limit] + "..."
        s = s.replace("\n", "\\n")
        parts.append(f"{k}={s}")
    return ", ".join(parts)


def _print_message(m) -> None:
    if isinstance(m, SystemMessage):
        return
    if isinstance(m, HumanMessage):
        return  # the user already saw their own input
    if isinstance(m, AIMessage):
        if m.content:
            console.print(Markdown(str(m.content)))
        for tc in m.tool_calls or []:
            console.print(
                f"[bold cyan]⚙ {tc.get('name')}([/bold cyan]"
                f"[cyan]{_compact_args(tc.get('args') or {})}[/cyan]"
                f"[bold cyan])[/bold cyan]"
            )
    elif isinstance(m, ToolMessage):
        content = str(m.content)
        console.print(
            Panel(
                content[:4000],
                title=f"tool: {getattr(m, 'name', '?')}",
                border_style="dim",
                expand=False,
            )
        )


def _invoke_config(cfg: Config) -> dict:
    return {
        "recursion_limit": cfg.max_iterations + 10,
        "configurable": {
            "project_root": str(cfg.project_root),
            "read_only": cfg.read_only,
            "allow_shell": cfg.allow_shell,
            "shell_timeout": cfg.shell_timeout,
            "tool_output_limit": cfg.tool_output_limit,
            "file_read_limit": cfg.file_read_limit,
        },
    }


def run_agent(cfg: Config, task: str, mode: str, initial_messages: list | None = None) -> dict:
    """Run the agent loop and stream progress. Returns the final state."""
    graph = build_graph(cfg, build_llm(cfg))
    msgs = list(initial_messages or [])
    if not msgs or not isinstance(msgs[0], SystemMessage):
        msgs.insert(0, SystemMessage(build_system_prompt(cfg, mode)))
    msgs.append(HumanMessage(task))

    state: dict = {
        "messages": msgs,
        "project_root": str(cfg.project_root),
        "mode": mode,
        "task": task,
        "finished": False,
        "final_summary": "",
    }
    seen = len(msgs)  # don't re-print history we already know about
    final = state
    try:
        for snapshot in graph.stream(state, config=_invoke_config(cfg), stream_mode="values"):
            final = snapshot
            for m in snapshot["messages"][seen:]:
                _print_message(m)
            seen = len(snapshot["messages"])
    except GraphRecursionError:
        console.print(
            "[yellow]⚠ Agent stopped: it hit the maximum number of steps without "
            "finishing. Partial progress is kept.[/yellow]"
        )
    return final


def _extract_analysis(final: dict) -> str:
    """Return the full analysis report.

    Priority: the <final_analysis> block from the last assistant message first
    (that's the actual report); the `finish` summary second (short fallback);
    then any remaining assistant text.
    """
    last_text = ""
    for m in final["messages"]:
        if isinstance(m, AIMessage) and m.content:
            last_text = str(m.content)
    m = re.search(r"<final_analysis>(.*?)</final_analysis>", last_text, re.S)
    if m:
        return m.group(1).strip()
    if final.get("final_summary"):
        return str(final["final_summary"])
    return last_text.strip()


# ------------------------------------------------------------------- modes --
def cmd_analyze(cfg: Config, args) -> int:
    console.print(f"[bold]Analyzing project:[/bold] {cfg.project_root}")
    final = run_agent(cfg, ANALYZE_TASK, "analyze")
    report = _extract_analysis(final)

    if args.output:
        out = Path(args.output)
        out.write_text(report, encoding="utf-8")
        console.print(f"\n[green]✔ Report written to {out}[/green]")
    else:
        console.print("\n" + Panel(report, title="Architecture Analysis", border_style="green"))
    return 0


def cmd_run(cfg: Config, task: str) -> int:
    console.print(f"[bold]Running task:[/bold] {task}")
    run_agent(cfg, task, "run")
    return 0


def cmd_chat(cfg: Config) -> int:
    console.print(
        Panel(
            f"[bold]coding-agent[/bold] · project: [cyan]{cfg.project_root}[/cyan]\n"
            "Type your question. Commands: [cyan]exit[/cyan] / [cyan]quit[/cyan] to leave.",
            border_style="blue",
        )
    )
    history: list = []
    while True:
        try:
            user_input = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\nbye 👋")
            break
        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit", "/exit", "/quit"):
            break
        final = run_agent(cfg, user_input, "chat", initial_messages=history)
        history = final["messages"]


# ---------------------------------------------------------------------- cli --
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="coding-agent",
        description="A LangGraph + DeepSeek CLI coding agent for architecture analysis and code modification.",
    )
    parser.add_argument("--version", action="version", version=f"coding-agent {__version__}")
    parser.add_argument("--root", default=None, help="project root to operate on (default: cwd)")
    parser.add_argument("--model", default=None, help="model id (default: deepseek-chat)")
    parser.add_argument("--api-key", default=None, help="DeepSeek API key (overrides env)")
    parser.add_argument("--read-only", action="store_true", help="read-only: no edits, no shell")
    parser.add_argument("--iterations", type=int, default=None, help="max agent loop steps")

    sub = parser.add_subparsers(dest="command", required=True)

    p_analyze = sub.add_parser("analyze", help="produce an architecture analysis of the project")
    p_analyze.add_argument("--output", default=None, help="write the report to this file")

    p_run = sub.add_parser("run", help="execute a one-shot task (may modify code)")
    p_run.add_argument("task", nargs="+", help="the task description")

    sub.add_parser("chat", help="start an interactive session")

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    cfg = Config.from_env(
        project_root=args.root,
        model=args.model,
        api_key=args.api_key,
        read_only=args.read_only if args.read_only else None,
        max_iterations=args.iterations,
    )

    if not cfg.api_key:
        console.print(
            "[red]No API key configured.[/red]\n"
            "Set DEEPSEEK_API_KEY in the environment, or create a .env file:\n"
            "  DEEPSEEK_API_KEY=sk-...\n"
            f"Model: {cfg.model} @ {cfg.base_url}"
        )
        return 2

    if not cfg.project_root.is_dir():
        console.print(f"[red]Project root is not a directory:[/red] {cfg.project_root}")
        return 2

    try:
        if args.command == "analyze":
            return cmd_analyze(cfg, args)
        if args.command == "run":
            return cmd_run(cfg, " ".join(args.task))
        if args.command == "chat":
            return cmd_chat(cfg)
    except KeyboardInterrupt:
        console.print("\nInterrupted.")
        return 130
    return 0


if __name__ == "__main__":
    sys.exit(main())
