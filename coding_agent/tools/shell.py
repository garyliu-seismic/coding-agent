"""Shell execution tool."""
from __future__ import annotations

import os
import subprocess
from typing import Optional

from langchain_core.runnables.config import RunnableConfig
from langchain_core.tools import tool

from .filesystem import _cfg, _root


@tool
def run_shell(
    command: str,
    workdir: str = ".",
    timeout: Optional[int] = None,
    config: RunnableConfig = None,
) -> str:
    """Run a shell command in `workdir` (relative to project root). Uses PowerShell on Windows, bash otherwise. Returns stdout, stderr and the exit code. Prefer this over guessing at build/test outputs."""
    if _cfg(config).get("read_only"):
        return "Error: read-only mode; run_shell is disabled."
    root = _root(config)
    cwd = (root / workdir).resolve()
    if not cwd.exists():
        return f"Error: workdir does not exist: {workdir}"
    t = timeout or int(_cfg(config).get("shell_timeout", 120))
    if os.name == "nt":
        cmd = ["powershell", "-NoProfile", "-NonInteractive", "-Command", command]
    else:
        cmd = ["bash", "-lc", command]
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(cwd),
            capture_output=True,
            text=True,
            encoding="utf-8",  # Windows consoles often use gbk; force UTF-8
            errors="replace",
            timeout=t,
        )
    except subprocess.TimeoutExpired:
        return f"Error: command timed out after {t}s:\n$ {command}"
    parts = []
    if proc.stdout:
        parts.append(proc.stdout.strip())
    if proc.stderr:
        parts.append("[stderr]\n" + proc.stderr.strip())
    out = "\n".join(parts)
    limit = int(_cfg(config).get("tool_output_limit", 20_000))
    if len(out) > limit:
        out = out[:limit] + f"\n...[truncated, {len(out) - limit:,} chars omitted]"
    if not out:
        return f"$ {command}\n(exit code {proc.returncode}, no output)"
    return f"$ {command}\n(exit code {proc.returncode})\n{out}"
