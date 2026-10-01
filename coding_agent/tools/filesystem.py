"""Filesystem / search tools exposed to the agent.

Tools read their runtime config (project root, limits, read-only flag) from the
injected ``RunnableConfig`` (the ``configurable`` section passed at invoke time).

edit_file design (mirrors pi / Claude approach)
------------------------------------------------
Key improvements over a naive str.replace():

1. **Multi-edit in one call** — `edits` is a list of {old_string, new_string}
   pairs applied atomically.  All matches are found against the *original* file;
   edits are then applied in reverse position order so earlier offsets stay valid.

2. **Fuzzy matching fallback** — when exact match fails we try again after:
   - Stripping trailing whitespace per line
   - Normalising smart quotes / em-dashes / non-breaking spaces to ASCII
   This catches the most common LLM failure mode: generating old_string with
   subtly wrong Unicode punctuation copied from rendered Markdown.

3. **CRLF preservation** — line endings are detected, normalised to LF for
   matching/replacement, then restored.  Windows files stay Windows files.

4. **Overlap detection** — edits that target the same region are rejected
   with a clear error rather than silently producing garbage.
"""
from __future__ import annotations

import os
import re
import shutil
import unicodedata
from pathlib import Path
from typing import List, Optional

from langchain_core.runnables.config import RunnableConfig
from langchain_core.tools import tool
from pydantic import BaseModel

from ..config import IGNORED_DIRS


# ---------------------------------------------------------------- helpers ---
def _cfg(config: RunnableConfig) -> dict:
    return (config or {}).get("configurable", None) or {}


def _root(config: RunnableConfig) -> Path:
    return Path(_cfg(config).get("project_root", "."))


def _cap(text: str, limit_key: str, config: RunnableConfig) -> str:
    limit = int(_cfg(config).get(limit_key, 60_000))
    if len(text) <= limit:
        return text
    head = int(limit * 0.7)
    tail = limit - head
    omitted = len(text) - limit
    return (
        text[:head]
        + f"\n...[truncated {omitted:,} chars; read middle in line ranges]...\n"
        + text[-tail:]
    )


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


# -------------------------------------------------------- backup / restore ---

_BAK_DIR = ".coding-agent-bak"


def _backup(root: Path, rel_path: str) -> None:
    """Save a copy of rel_path into .coding-agent-bak/<rel_path>.bak.

    Keeps exactly ONE backup per file: the state just before the most recent
    write.  Enough for the agent to undo one bad edit without needing git.
    """
    src = root / rel_path
    if not src.is_file():
        return
    bak = root / _BAK_DIR / (rel_path + ".bak")
    bak.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(str(src), str(bak))


# -------------------------------------------------------- edit_file helpers --

def _detect_line_ending(content: str) -> str:
    """Return '\\r\\n' if CRLF dominates, else '\\n'."""
    crlf = content.count("\r\n")
    lf = content.count("\n") - crlf
    return "\r\n" if crlf > lf else "\n"


def _normalize_to_lf(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


def _restore_line_endings(text: str, ending: str) -> str:
    if ending == "\r\n":
        return text.replace("\n", "\r\n")
    return text


def _normalize_for_fuzzy(text: str) -> str:
    """Normalise text for fuzzy match (mirrors pi's normalizeForFuzzyMatch).

    Strips trailing whitespace per line; normalises Unicode punctuation
    variants (smart quotes, em-dashes, non-breaking spaces) to ASCII.
    """
    text = unicodedata.normalize("NFKC", text)
    text = "\n".join(line.rstrip() for line in text.split("\n"))
    text = re.sub(r"[\u2018\u2019\u201A\u201B]", "'", text)       # smart single quotes
    text = re.sub(r'[\u201C\u201D\u201E\u201F]', '"', text)        # smart double quotes
    text = re.sub(r"[\u2010\u2011\u2012\u2013\u2014\u2015\u2212]", "-", text)  # dashes
    text = re.sub(r"[\u00A0\u2002-\u200A\u202F\u205F\u3000]", " ", text)       # spaces
    return text


def _count_occurrences_fuzzy(content_lf: str, old_lf: str) -> int:
    """Count occurrences using fuzzy-normalised comparison."""
    fc = _normalize_for_fuzzy(content_lf)
    fo = _normalize_for_fuzzy(old_lf)
    if not fo:
        return 0
    return fc.count(fo)


class _Match:
    __slots__ = ("index", "length", "fuzzy", "base")

    def __init__(self, index: int, length: int, fuzzy: bool, base: str):
        self.index = index
        self.length = length
        self.fuzzy = fuzzy
        self.base = base  # content string in which index/length are valid


def _fuzzy_find(content_lf: str, old_lf: str) -> _Match | None:
    """Exact match first; then fuzzy (trailing-ws + unicode normalisation)."""
    idx = content_lf.find(old_lf)
    if idx != -1:
        return _Match(idx, len(old_lf), False, content_lf)

    fc = _normalize_for_fuzzy(content_lf)
    fo = _normalize_for_fuzzy(old_lf)
    idx = fc.find(fo)
    if idx != -1:
        return _Match(idx, len(fo), True, fc)

    return None


def _apply_single_edit_in_lf(content: str, old: str, new: str, path: str) -> str:
    """Apply one replacement.  When fuzzy matching is used, only the touched
    lines are taken from the normalised content; unchanged lines come from the
    original (preserving original spacing/encoding for untouched regions)."""
    m = _fuzzy_find(content, old)
    if m is None:
        raise ValueError(
            f"old_string not found in {path}.\n"
            "Tip: use read_file to verify exact content including whitespace."
        )

    if not m.fuzzy:
        # Exact — simple splice
        return content[: m.index] + new + content[m.index + m.length :]

    # Fuzzy — splice in normalised space, then overlay touched lines onto original
    lines_orig = content.split("\n")
    lines_fuzzy = m.base.split("\n")

    # Locate start/end line of the matched region in fuzzy space
    offset = 0
    start_line = end_line = 0
    for i, line in enumerate(lines_fuzzy):
        next_offset = offset + len(line) + 1  # +1 for \n
        if offset <= m.index < next_offset:
            start_line = i
        if offset < m.index + m.length <= next_offset:
            end_line = i
            break
        elif m.index + m.length == len(m.base) and i == len(lines_fuzzy) - 1:
            end_line = i
            break
        offset = next_offset

    # Apply replacement in fuzzy space
    replaced_base = m.base[: m.index] + new + m.base[m.index + m.length :]
    replaced_lines = replaced_base.split("\n")

    # How many lines did the old matched region span?
    old_span = end_line - start_line + 1
    # How many lines does the replacement span?
    new_span = new.count("\n") + 1

    # Reconstruct: original lines before, replaced slice, original lines after
    result_lines = (
        lines_orig[:start_line]
        + replaced_lines[start_line : start_line + new_span]
        + lines_orig[end_line + 1 :]
    )
    return "\n".join(result_lines)


def _diff_summary(old_lf: str, new_lf: str, context: int = 1) -> str:
    """Return a compact human-readable diff with 1-based line numbers.

    Format (shown to the agent in the edit_file return value)::

        Changes:
          L5:  -  x = 1
               +  x = 99
          L12-L14: - def old_name(a, b):
                   -     return a + b
                   + def new_name(a, b, c):
                   +     return a + b + c

    Keeps `context` unchanged lines before/after each hunk so the agent can
    confirm the right location was edited.
    """
    old_lines = old_lf.splitlines()
    new_lines = new_lf.splitlines()

    # Collect changed line ranges (0-based) in old content
    # Simple LCS-free approach: mark lines that differ after alignment via
    # difflib.SequenceMatcher.
    import difflib
    matcher = difflib.SequenceMatcher(None, old_lines, new_lines, autojunk=False)
    hunks: list[str] = []

    for group in matcher.get_grouped_opcodes(context):
        # group is a list of (tag, i1, i2, j1, j2) opcodes
        # Determine 1-based line range in old file for the hunk header
        first_old = group[0][1] + 1
        last_old  = group[-1][2]      # exclusive → last 1-based
        if first_old == last_old:
            hunk_header = f"  L{first_old}:"
        else:
            hunk_header = f"  L{first_old}-L{last_old}:"

        hunk_lines: list[str] = []
        for tag, i1, i2, j1, j2 in group:
            if tag == "equal":
                for line in old_lines[i1:i2]:
                    hunk_lines.append(f"      {line}")
            elif tag in ("replace", "delete"):
                for line in old_lines[i1:i2]:
                    hunk_lines.append(f"    - {line}")
                if tag == "replace":
                    for line in new_lines[j1:j2]:
                        hunk_lines.append(f"    + {line}")
            elif tag == "insert":
                for line in new_lines[j1:j2]:
                    hunk_lines.append(f"    + {line}")

        hunks.append(hunk_header + "\n" + "\n".join(hunk_lines))

    if not hunks:
        return ""
    return "Changes:\n" + "\n".join(hunks)


# -------------------------------------------------------- edit schema --------

class _EditEntry(BaseModel):
    old_string: str
    new_string: str


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
        raw = p.read_bytes().decode("utf-8", errors="replace").replace("\r\n", "\n")
    except OSError as e:
        return f"Error reading {path}: {e}"
    lines = raw.split("\n")
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
            text = p.read_bytes().decode("utf-8", errors="replace").replace("\r\n", "\n")
        except OSError:
            continue
        rel = p.relative_to(root).as_posix()
        for i, line in enumerate(text.split("\n"), 1):
            if rx.search(line):
                s = line.strip()
                if len(s) > 800:
                    s = s[:800] + f"...[line truncated, {len(s)} chars total; use read_file {rel} {i}-{i}]"
                matches.append(f"{rel}:{i}: {s}")
                if len(matches) >= max_results:
                    return "\n".join(matches) + f"\n...[stopped at {max_results} matches]"
    return "\n".join(matches) if matches else "(no matches)"


def _glob_to_regex(glob_pattern: str) -> re.Pattern:
    """Convert a glob pattern (with `**`, `*`, `?`, and `{a,b}` braces) to a regex."""
    out: list[str] = []
    i = 0
    n = len(glob_pattern)
    while i < n:
        ch = glob_pattern[i]
        if ch == "*":
            if i + 1 < n and glob_pattern[i + 1] == "*":
                out.append(".*")
                i += 2
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
    if existed:
        _backup(root, path)
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
    edits: List[_EditEntry],
    config: RunnableConfig = None,
) -> str:
    """Apply one or more precise text replacements to a file in a single call.

    `edits` is a list of {old_string, new_string} pairs.

    Rules:
    - Each old_string must appear EXACTLY ONCE in the file. Include enough
      surrounding lines to make it unique.
    - All old_strings are matched against the ORIGINAL file — not against the
      result of earlier edits in the same call.
    - Edits must not overlap; merge overlapping changes into one edit.
    - Fuzzy matching is tried automatically when exact match fails: trailing
      whitespace differences and Unicode punctuation variants (smart quotes,
      em-dashes, non-breaking spaces) are tolerated.
    - Line endings (CRLF / LF) are detected and preserved.

    Prefer one call with multiple edits over multiple single-edit calls when
    changing several locations in the same file.
    """
    if _cfg(config).get("read_only"):
        return "Error: read-only mode; edit_file is disabled."
    root = _root(config)
    p = (root / path).resolve()
    if not _within(root, p):
        return f"Error: path escapes project root: {path}"
    if not p.is_file():
        return f"Error: not a file: {path}"
    if not edits:
        return "Error: edits list is empty — nothing to do."

    raw = p.read_text(encoding="utf-8", errors="replace")
    ending = _detect_line_ending(raw)
    content = _normalize_to_lf(raw)

    # Backup before writing so restore_file can undo this edit.
    _backup(root, path)

    # --- Validate all edits against original content before applying any ---
    errors: list[str] = []
    for i, e in enumerate(edits):
        old_lf = _normalize_to_lf(e.old_string)
        if not old_lf:
            errors.append(f"edits[{i}].old_string is empty.")
            continue
        occ = _count_occurrences_fuzzy(content, old_lf)
        if occ == 0:
            errors.append(
                f"edits[{i}].old_string not found in {path}. "
                "Use read_file to check exact content and whitespace."
            )
        elif occ > 1:
            errors.append(
                f"edits[{i}].old_string found {occ} times in {path}. "
                "Add more surrounding context to make it unique."
            )
    if errors:
        return "Error — no changes made:\n" + "\n".join(f"  • {e}" for e in errors)

    # --- Find match positions (against original) ---
    matched: list[tuple[int, int, str, bool]] = []  # (start, length, new_lf, is_fuzzy)
    for i, e in enumerate(edits):
        old_lf = _normalize_to_lf(e.old_string)
        new_lf = _normalize_to_lf(e.new_string)
        m = _fuzzy_find(content, old_lf)
        if m is None:
            return f"Error: edits[{i}].old_string disappeared during validation (race condition?)."
        matched.append((m.index, m.length, new_lf, m.fuzzy))

    # --- Check for overlaps ---
    sorted_m = sorted(enumerate(matched), key=lambda x: x[1][0])
    for j in range(1, len(sorted_m)):
        prev_i, (prev_start, prev_len, _, _) = sorted_m[j - 1]
        cur_i, (cur_start, _, _, _) = sorted_m[j]
        if prev_start + prev_len > cur_start:
            return (
                f"Error: edits[{prev_i}] and edits[{cur_i}] overlap in {path}. "
                "Merge them into one edit or target disjoint regions."
            )

    # --- Apply edits ---
    any_fuzzy = any(m[3] for m in matched)

    if any_fuzzy:
        # Fuzzy path: apply one at a time (left-to-right on the evolving content)
        # Note: positions shift after each edit, so we re-find each time.
        result = content
        for i, e in enumerate(edits):
            old_lf = _normalize_to_lf(e.old_string)
            new_lf = _normalize_to_lf(e.new_string)
            try:
                result = _apply_single_edit_in_lf(result, old_lf, new_lf, path)
            except ValueError as exc:
                return f"Error applying edits[{i}]: {exc}"
    else:
        # All exact: apply right-to-left for offset stability
        result = content
        for _start, _length, new_lf, _fuzzy in sorted(matched, key=lambda x: -x[0]):
            result = result[:_start] + new_lf + result[_start + _length:]

    if result == content:
        return f"Error: replacements produced identical content in {path} — no changes made."

    final = _restore_line_endings(result, ending)
    p.write_text(final, encoding="utf-8")

    diff = _diff_summary(content, result, context=1)
    n = len(edits)
    header = f"OK: edited {path} ({n} replacement{'s' if n > 1 else ''} applied)."
    return header + ("\n" + diff if diff else "")


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


@tool
def restore_file(
    path: str,
    config: RunnableConfig = None,
) -> str:
    """Restore a file to its state before the last write_file or edit_file call.

    Every write_file / edit_file automatically saves a backup to
    `.coding-agent-bak/<path>.bak` before overwriting.  Call this tool to
    undo a bad edit.  Only ONE level of undo is available per file.

    Use this when:
    - An edit produced incorrect code and you want to start over.
    - write_file overwrote the wrong content.
    """
    if _cfg(config).get("read_only"):
        return "Error: read-only mode; restore_file is disabled."
    root = _root(config)
    p = (root / path).resolve()
    if not _within(root, p):
        return f"Error: path escapes project root: {path}"
    bak = root / _BAK_DIR / (path + ".bak")
    if not bak.is_file():
        return (
            f"Error: no backup found for {path}. "
            "restore_file only works after a write_file or edit_file call."
        )
    shutil.copy2(str(bak), str(p))
    bak.unlink()  # consume the backup so a second restore doesn't double-undo
    return f"OK: restored {path} from backup (backup consumed — further restore not possible)."


@tool
def move_file(
    src: str,
    dst: str,
    config: RunnableConfig = None,
) -> str:
    """Move or rename a file within the project root.

    Both `src` and `dst` are relative to the project root.
    Parent directories of `dst` are created automatically.
    Use this when refactoring requires reorganising files or renaming modules.
    Remember to update import paths in other files with edit_file after moving.
    """
    if _cfg(config).get("read_only"):
        return "Error: read-only mode; move_file is disabled."
    root = _root(config)
    s = (root / src).resolve()
    d = (root / dst).resolve()
    if not _within(root, s):
        return f"Error: src escapes project root: {src}"
    if not _within(root, d):
        return f"Error: dst escapes project root: {dst}"
    if not s.exists():
        return f"Error: src does not exist: {src}"
    if d.exists():
        return f"Error: dst already exists: {dst}. Delete it first to overwrite."
    d.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(s), str(d))
    return f"OK: moved {src} → {dst}"
