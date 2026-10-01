"""Shell execution tool."""
from __future__ import annotations

import os
import re
import subprocess
from typing import Optional

from langchain_core.runnables.config import RunnableConfig
from langchain_core.tools import tool

from .filesystem import _cfg, _root


# ------------------------------------------------------------------ guard ---
# Disaster-level command patterns. Refused by default because they can wipe
# whole disks/system state, take the machine down, or escalate privileges.
# Normal project-scoped commands (e.g. `rm -rf ./build`) are NOT blocked.
# Disable this guard with CODING_AGENT_ALLOW_DANGEROUS=1.
_DANGEROUS_PATTERNS: list[tuple[str, str]] = [
    # POSIX: recursive delete of filesystem roots / home / system dirs
    (r"\brm\s+(?:-[a-z]*[rf][a-z]*\s+)+(?:--no-preserve-root\s+)?/(?:\s|$)", "recursive delete of '/'"),
    (r"\brm\s+(?:-[a-z]*[rf][a-z]*\s+)+(?:--no-preserve-root\s+)?~(?:\s|/|$)", "recursive delete of home '~'"),
    (r"\brm\s+(?:-[a-z]*[rf][a-z]*\s+)+(?:--no-preserve-root\s+)?\$(?:HOME|PWD)(?:\s|/|$)", "recursive delete of $HOME/$PWD"),
    (r"\brm\s+(?:-[a-z]*[rf][a-z]*\s+)+/(?:home|usr|etc|var|bin|sbin|root|boot|lib)(?:\s|$)", "recursive delete of a system directory"),
    # raw device writes / formatting
    (r"\bdd\s+.*\bof=/dev/", "raw write to a /dev/ device"),
    (r"\bmkfs(?:\.\w+)?\b", "filesystem formatting (mkfs)"),
    # Windows destructive commands
    (r"\bformat\s+[A-Za-z]:", "formatting a Windows drive"),
    (r"\b(?:del|erase|rd)\b[^\n]*/[sS][^\n]*[A-Za-z]:\\(?:\s|$|\*)", "recursive delete of a Windows drive root"),
    (r"\bRemove-Item\b[^\n]*-Recurse[^\n]*[A-Za-z]:\\(?:\s|$|\*)", "recursive Remove-Item of a Windows drive root"),
    (r"\b(?:Clear-Disk|Format-Volume)\b", "disk wipe / format (PowerShell)"),
    # shutdown / reboot
    (r"\b(?:shutdown|reboot|halt|poweroff)\b", "shutdown / reboot"),
    (r"\b(?:Stop-Computer|Restart-Computer|shutdown\.exe)\b", "shutdown / reboot (Windows)"),
    # privilege escalation
    (r"\bsudo\b", "privilege escalation (sudo)"),
    # fork bomb
    (r":\s*\(\)\s*\{\s*:\s*\|:\s*&\s*\}\s*;?\s*:", "fork bomb"),
    # remote script piped into a shell
    (r"\b(?:curl|wget)\b[^|;\n]*\|\s*(?:sudo\s+)?(?:ba|z|da)?sh\b", "piping a remote script into a shell"),
]

_DANGEROUS_RE = [(re.compile(p, re.IGNORECASE), label) for p, label in _DANGEROUS_PATTERNS]


def _dangerous_reason(command: str) -> str | None:
    """Return a human-readable reason if ``command`` looks dangerous, else None."""
    for pattern, label in _DANGEROUS_RE:
        if pattern.search(command):
            return label
    return None


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
    if not _cfg(config).get("allow_dangerous_commands", False):
        reason = _dangerous_reason(command)
        if reason:
            return (
                f"Error: refusing to run this command — {reason}.\n"
                "This is a destructive / high-risk operation. If you are certain, "
                "the user must set CODING_AGENT_ALLOW_DANGEROUS=1 and re-run."
            )
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
        head = int(limit * 0.7)
        tail = limit - head
        omitted = len(out) - limit
        out = (
            out[:head]
            + f"\n...[truncated, {omitted:,} chars omitted]...\n"
            + out[-tail:]
        )
    if not out:
        return f"$ {command}\n(exit code {proc.returncode}, no output)"
    return f"$ {command}\n(exit code {proc.returncode})\n{out}"
