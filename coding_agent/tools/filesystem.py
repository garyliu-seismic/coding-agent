"""Filesystem / search tools exposed to the agent.

Tools read their runtime config (project root, limits, read-only flag) from the
injected ``RunnableConfig`` (the ``configurable`` section passed at invoke time).
"""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Optional

from langchain_core.runnables.config import RunnableConfig
from langchain_core.tools import tool

from ..config import IGNORED_DIRS


# ---------------------------------------------------------------- helpers ---
def _cfg(config: RunnableConfig) -> dict:
    return (config or {}).get("configurable", None) or {}


def _root(config: RunnableConfig) -> Path:
    return Path(_cfg(config).get("project_root", "."))


def _cap(text: str, limit_key: str, config: RunnableConfig) -> str:
    limit = int(_cfg(config).get(limit_key, 60_000))
    if len(text) > limit:
        return (
            text[:limit]
            + f"\n...[truncated {len(text) - limit:,} chars; read in line ranges]"
        )
    return text


def _within(root: Path, p: Path) -> bool:
    try:
        p.relative_to(root)
        return True
    except ValueError:
        return False


def _iter_files(base: Path):
    for dirpath, dirnames, filenames in os.walk(base):
        dirnames[:] = [d for d in dirnames if d not in IGNORED_DIRS]
        for name in filenames:
            yield Path(dirpath) / name


# ------------------------------------------------------------------ tools ----
@tool
def list_directory(
    path: str = ".",
    max_depth: int = 2,
    config: RunnableConfig = None,
) -> str:
    """List the files and directories under `path` (relative to project root) up to `max_depth` levels. Use this first to understand the project structure."""
    root = _root(config)
    target = (root / path).resolve()
    if not target.exists():
        return f"Error: path does not exist: {path}"
    if not _within(root, target):
        return f"Error: path escapes project root: {path}"
    if target.is_file():
        return target.relative_to(root).as_posix()

    lines: list[str] = []

    def walk(d: Path, depth: int) -> None:
        if depth > max_depth:
            return
        entries = sorted(d.iterdir(), key=lambda p: (p.is_file(), p.name.lower()))
        for e in entries:
            if e.name in IGNORED_DIRS:
                continue
            rel = e.relative_to(root).as_posix()
            if e.is_dir():
                lines.append(f"{'  ' * depth}[dir ] {rel}/")
                walk(e, depth + 1)
            else:
                try:
                    size = e.stat().st_size
                except OSError:
                    size = 0
                lines.append(f"{'  ' * depth}[file] {rel}  ({size:,} B)")

    walk(target, 0)
    return "\n".join(lines) if lines else "(empty directory)"


@tool
def read_file(
    path: str,
    start_line: Optional[int] = None,
    end_line: Optional[int] = None,
    config: RunnableConfig = None,
) -> str:
    """Read a text file. Use `start_line`/`end_line` (1-based, inclusive) to read only a range of a large file. Output is line-numbered so you can reference lines later."""
    root = _root(config)
    p = (root / path).resolve()
    if not _within(root, p):
        return f"Error: path escapes project root: {path}"
    if not p.is_file():
        return f"Error: not a file or does not exist: {path}"
    try:
        raw = p.read_text(encoding="utf-8", errors="replace")
    except OSError as e:
        return f"Error reading {path}: {e}"
    lines = raw.splitlines()
    total = len(lines)
    start = start_line or 1
    end = min(end_line or total, total)
    if start < 1 or start > total:
        return f"Error: start_line {start} out of range (file has {total} lines)."
    numbered = "\n".join(f"{i}: {lines[i - 1]}" for i in range(start, end + 1))
    header = f"--- {path} (lines {start}-{end} of {total}) ---\n"
    return _cap(header + numbered, "file_read_limit", config)


@tool
def grep_search(
    pattern: str,
    path: str = ".",
    is_regexp: bool = True,
    max_results: int = 100,
    config: RunnableConfig = None,
) -> str:
    """Search file contents under `path` for `pattern` (regex if `is_regexp`, else plain substring). Returns 'file:line: content' matches. Use this to find where symbols are used."""
    root = _root(config)
    target = (root / path).resolve()
    if not target.exists():
        return f"Error: path does not exist: {path}"
    if not _within(root, target):
        return f"Error: path escapes project root: {path}"
    try:
        rx = re.compile(pattern) if is_regexp else re.compile(re.escape(pattern))
    except re.error as e:
        return f"Error: invalid regex: {e}"

    matches: list[str] = []
    files = [target] if target.is_file() else list(_iter_files(target))
    for p in files:
        try:
            text = p.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        rel = p.relative_to(root).as_posix()
        for i, line in enumerate(text.splitlines(), 1):
            if rx.search(line):
                matches.append(f"{rel}:{i}: {line.strip()[:200]}")
                if len(matches) >= max_results:
                    return "\n".join(matches) + f"\n...[stopped at {max_results} matches]"
    return "\n".join(matches) if matches else "(no matches)"


def _glob_to_regex(glob_pattern: str) -> re.Pattern:
    """Convert a glob pattern (with `**`, `*`, `?`, and `{a,b}` braces) to a regex.

    fnmatch does not support `**` (match across `/`) or `{a,b}` alternation,
    both of which the model commonly uses. We translate them here instead.
    """
    out: list[str] = []
    i = 0
    n = len(glob_pattern)
    while i < n:
        ch = glob_pattern[i]
        if ch == "*":
            if i + 1 < n and glob_pattern[i + 1] == "*":
                out.append(".*")  # ** crosses directory separators
                i += 2
                # swallow a following '/' so '**/' and '**' both work
                if i < n and glob_pattern[i] == "/":
                    i += 1
            else:
                out.append("[^/]*")
                i += 1
        elif ch == "?":
            out.append("[^/]")
            i += 1
        elif ch == "{":
            end = glob_pattern.find("}", i)
            if end != -1:
                parts = [p for p in glob_pattern[i + 1 : end].split(",") if p]
                out.append("(?:" + "|".join(re.escape(p) for p in parts) + ")")
                i = end + 1
            else:
                out.append(re.escape(ch))
                i += 1
        else:
            out.append(re.escape(ch))
            i += 1
    return re.compile("^" + "".join(out) + "$")


@tool
def file_search(
    glob_pattern: str,
    path: str = ".",
    config: RunnableConfig = None,
) -> str:
    """Find files by filename glob pattern. Supports `**` (any depth), `*`, `?`, and `{a,b}` alternatives — e.g. '**/*.ts', '*service*.py', 'manifest*.xml', '*.{json,yml}'. Returns matching relative paths."""
    root = _root(config)
    target = (root / path).resolve()
    if not target.exists():
        return f"Error: path does not exist: {path}"
    if not _within(root, target):
        return f"Error: path escapes project root: {path}"
    rx = _glob_to_regex(glob_pattern)
    results = []
    files = [target] if target.is_file() else list(_iter_files(target))
    for p in files:
        rel = p.relative_to(root).as_posix()
        if rx.match(rel) or rx.match(p.name):
            results.append(rel)
    return "\n".join(sorted(results)) if results else "(no matches)"


@tool
def write_file(
    path: str,
    content: str,
    config: RunnableConfig = None,
) -> str:
    """Write `content` to `path` (relative to project root), creating directories as needed. This OVERWRITES the file — only use it to create or fully rewrite a file; prefer `edit_file` for targeted changes."""
    if _cfg(config).get("read_only"):
        return "Error: read-only mode; write_file is disabled."
    root = _root(config)
    p = (root / path).resolve()
    if not _within(root, p):
        return f"Error: path escapes project root: {path}"
    p.parent.mkdir(parents=True, exist_ok=True)
    existed = p.exists()
    old = p.read_text(encoding="utf-8", errors="replace") if existed else ""
    p.write_text(content, encoding="utf-8")
    if existed and old == content:
        return f"OK: {path} unchanged (content identical)."
    added = len(set(content.splitlines()) - set(old.splitlines()))
    removed = len(set(old.splitlines()) - set(content.splitlines()))
    return (
        f"OK: wrote {path} ({'overwrote' if existed else 'created'}, "
        f"~{added} lines added / ~{removed} removed)."
    )


@tool
def edit_file(
    path: str,
    old_string: str,
    new_string: str,
    config: RunnableConfig = None,
) -> str:
    """Apply a single text replacement in `path`. `old_string` must appear EXACTLY once in the file — include surrounding context to make it unique. Use this for surgical edits."""
    if _cfg(config).get("read_only"):
        return "Error: read-only mode; edit_file is disabled."
    root = _root(config)
    p = (root / path).resolve()
    if not _within(root, p):
        return f"Error: path escapes project root: {path}"
    if not p.is_file():
        return f"Error: not a file: {path}"
    text = p.read_text(encoding="utf-8")
    count = text.count(old_string)
    if count == 0:
        return (
            f"Error: old_string not found in {path}. "
            "Use read_file to check the exact content and whitespace."
        )
    if count > 1:
        return (
            f"Error: old_string found {count} times in {path}; "
            "include more surrounding context to make it unique."
        )
    p.write_text(text.replace(old_string, new_string, 1), encoding="utf-8")
    return f"OK: edited {path} (1 replacement applied)."


@tool
def delete_file(
    path: str,
    config: RunnableConfig = None,
) -> str:
    """Delete the file at `path` (relative to project root). Use sparingly."""
    if _cfg(config).get("read_only"):
        return "Error: read-only mode; delete_file is disabled."
    root = _root(config)
    p = (root / path).resolve()
    if not _within(root, p):
        return f"Error: path escapes project root: {path}"
    if p.is_file():
        p.unlink()
        return f"OK: deleted {path}"
    return f"Error: not a file: {path}"
