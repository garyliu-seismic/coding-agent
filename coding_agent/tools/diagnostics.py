"""Static diagnostics tool: syntax / lint / type-check + conflict-marker scan.

This is a pragmatic, dependency-light alternative to a full LSP client: it
shells out to the project's linters / type-checkers (when installed) and
reports their output, plus always scans for unresolved merge-conflict markers.
It is read-only and remains available in ``--read-only`` mode.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional

from langchain_core.runnables.config import RunnableConfig
from langchain_core.tools import tool

from .filesystem import _cfg, _root

# Unresolved merge-conflict markers.
_CONFLICT_MARKERS = ("<<<<<<<", ">>>>>>>")

# Directories / files we never scan (VCS, venvs, build artifacts, binaries).
_SKIP_DIRS = {
    ".git", ".hg", ".svn", ".venv", "venv", "__pycache__", "node_modules",
    "dist", "build", ".tox", ".mypy_cache", ".pytest_cache", ".idea", ".vscode",
}
_SKIP_SUFFIXES = {
    ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp", ".ico", ".zip", ".pyc",
    ".pyo", ".lock", ".ipynb", ".min.js", ".map", ".woff", ".woff2", ".ttf",
}


def _which(cmd: str) -> Optional[str]:
    return shutil.which(cmd)


def _run(cmd: list[str], cwd: Path, timeout: int) -> tuple[int, str]:
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(cwd),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return -1, f"(timed out after {timeout}s)"
    out = (proc.stdout or "").strip()
    err = (proc.stderr or "").strip()
    return proc.returncode, "\n".join(x for x in (out, err) if x)


def _detect_language(root: Path) -> str:
    """Best-effort language detection for the project root."""
    if (root / "pyproject.toml").exists() or (root / "setup.py").exists() or (root / "requirements.txt").exists():
        return "python"
    if (root / "package.json").exists():
        return "javascript"
    if (root / "Cargo.toml").exists():
        return "rust"
    if (root / "go.mod").exists():
        return "go"
    return "unknown"


def _scan_conflict_markers(root: Path, target: Path) -> list[str]:
    """Return a list of 'path: conflict marker' strings under ``target``."""
    hits: list[str] = []
    base = target if target != root else root
    for p in base.rglob("*"):
        if not p.is_file():
            continue
        if any(part in _SKIP_DIRS for part in p.parts):
            continue
        if p.suffix.lower() in _SKIP_SUFFIXES:
            continue
        try:
            text = p.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for marker in _CONFLICT_MARKERS:
            if marker in text:
                hits.append(f"{p.relative_to(root)}: conflict marker {marker!r}")
                break
    return hits


@tool
def run_diagnostics(
    path: str = ".",
    timeout: Optional[int] = None,
    config: RunnableConfig = None,
) -> str:
    """Run static diagnostics on the project and report issues (Python syntax errors, ruff/mypy lint & type errors, unresolved merge-conflict markers, git whitespace errors). Use this to locate compile/lint errors instead of guessing. `path` is relative to the project root (default '.' = whole project). Read-only."""
    root = _root(config).resolve()
    target = (root / path).resolve()
    if not target.exists():
        return f"Error: path does not exist: {path}"
    t = timeout or int(_cfg(config).get("diagnostics_timeout", 120))

    lang = _detect_language(root)
    sections: list[str] = [f"Language detected: {lang}"]

    if lang == "python":
        # 1) byte-compile everything (always available, no extra deps)
        rc, out = _run([sys.executable, "-m", "compileall", "-q", str(target)], root, t)
        if rc == 0:
            sections.append("## python -m compileall (syntax)\n(clean)" if not out else f"## python -m compileall (syntax)\n{out}")
        else:
            sections.append(f"## python -m compileall (syntax) [exit {rc}]\n{out or '(no output)'}")

        # 2) ruff — fast linter (if installed)
        if _which("ruff"):
            rc, out = _run(["ruff", "check", str(target)], root, t)
            sections.append(f"## ruff check [exit {rc}]\n{out or '(clean)'}")
        else:
            sections.append("## ruff check\n(not installed — skipped)")

        # 3) mypy — type checker (if installed)
        if _which("mypy"):
            rc, out = _run(["mypy", str(target)], root, t)
            sections.append(f"## mypy [exit {rc}]\n{out or '(clean)'}")
        else:
            sections.append("## mypy\n(not installed — skipped)")
    else:
        sections.append(
            "## language-specific linters\n"
            f"(no linter configured for '{lang}'; install ruff/mypy for Python, "
            "or run your project's build command via run_shell)"
        )

    # 4) merge-conflict markers (language-agnostic)
    hits = _scan_conflict_markers(root, target)
    sections.append("## merge-conflict scan\n" + ("\n".join(hits) if hits else "(no conflict markers)"))

    # 5) git whitespace / conflict errors (language-agnostic)
    if (root / ".git").exists():
        rc, out = _run(["git", "diff", "--check"], root, t)
        sections.append(f"## git diff --check [exit {rc}]\n{out or '(clean)'}")

    return "\n\n".join(sections)
