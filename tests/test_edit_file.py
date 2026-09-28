"""Tests for the improved edit_file tool.

Covers:
- Exact single edit
- Multi-edit in one call (both applied, untouched lines preserved)
- Fuzzy match: trailing whitespace in old_string
- Fuzzy match: smart-quote in old_string
- CRLF file preservation
- Overlap detection
- Empty old_string rejection
- Duplicate old_string rejection
- Not-found error
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from coding_agent.config import Config
from coding_agent.tools.filesystem import edit_file

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _cfg(root: Path) -> dict:
    return {
        "configurable": {
            "project_root": str(root),
            "read_only": False,
            "file_read_limit": 60_000,
        }
    }


def _write(root: Path, rel: str, content: str) -> Path:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    return p


def _invoke(root: Path, path: str, edits: list[dict]) -> str:
    return edit_file.invoke({"path": path, "edits": edits}, config=_cfg(root))


# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------

def test_single_exact_edit(tmp_path):
    _write(tmp_path, "a.py", "x = 1\ny = 2\n")
    out = _invoke(tmp_path, "a.py", [{"old_string": "x = 1", "new_string": "x = 99"}])
    assert "OK" in out
    assert (tmp_path / "a.py").read_text() == "x = 99\ny = 2\n"


def test_multi_edit_both_applied(tmp_path):
    _write(tmp_path, "b.py", "a = 1\nb = 2\nc = 3\n")
    out = _invoke(tmp_path, "b.py", [
        {"old_string": "a = 1", "new_string": "a = 10"},
        {"old_string": "c = 3", "new_string": "c = 30"},
    ])
    assert "OK" in out and "2 replacements" in out
    content = (tmp_path / "b.py").read_text()
    assert "a = 10" in content
    assert "b = 2" in content   # untouched
    assert "c = 30" in content


def test_multi_edit_middle_line_preserved(tmp_path):
    _write(tmp_path, "c.py", "def foo():\n    pass\n\ndef bar():\n    pass\n")
    out = _invoke(tmp_path, "c.py", [
        {"old_string": "def foo():\n    pass", "new_string": "def foo():\n    return 1"},
        {"old_string": "def bar():\n    pass", "new_string": "def bar():\n    return 2"},
    ])
    assert "OK" in out
    content = (tmp_path / "c.py").read_text()
    assert "return 1" in content
    assert "return 2" in content


def test_fuzzy_trailing_whitespace(tmp_path):
    """old_string with trailing spaces should still match via fuzzy."""
    _write(tmp_path, "d.py", "x = 1\ny = 2\n")
    # old_string has trailing space — exact match fails, fuzzy should save it
    out = _invoke(tmp_path, "d.py", [{"old_string": "x = 1   ", "new_string": "x = 99"}])
    assert "OK" in out, f"Expected OK, got: {out}"
    assert "x = 99" in (tmp_path / "d.py").read_text()


def test_fuzzy_smart_quote(tmp_path):
    """Smart quote in old_string normalises to ASCII apostrophe."""
    _write(tmp_path, "e.py", "msg = 'hello'\n")
    # Use smart single quote in old_string
    out = _invoke(tmp_path, "e.py", [{"old_string": "msg = \u2018hello\u2019", "new_string": "msg = 'world'"}])
    assert "OK" in out, f"Expected OK, got: {out}"
    assert "world" in (tmp_path / "e.py").read_text()


def test_crlf_preserved(tmp_path):
    """CRLF line endings in the file must be preserved after edit."""
    p = tmp_path / "f.py"
    p.write_bytes(b"x = 1\r\ny = 2\r\n")
    out = _invoke(tmp_path, "f.py", [{"old_string": "x = 1", "new_string": "x = 99"}])
    assert "OK" in out
    raw = p.read_bytes()
    assert b"\r\n" in raw, "CRLF should be preserved"
    assert b"x = 99" in raw


def test_not_found_error(tmp_path):
    _write(tmp_path, "g.py", "x = 1\n")
    out = _invoke(tmp_path, "g.py", [{"old_string": "z = 999", "new_string": "z = 0"}])
    assert "not found" in out.lower()
    assert "Error" in out


def test_duplicate_old_string_rejected(tmp_path):
    _write(tmp_path, "h.py", "x = 1\nx = 1\n")
    out = _invoke(tmp_path, "h.py", [{"old_string": "x = 1", "new_string": "x = 2"}])
    assert "times" in out or "2 times" in out
    assert "Error" in out


def test_empty_old_string_rejected(tmp_path):
    _write(tmp_path, "i.py", "x = 1\n")
    out = _invoke(tmp_path, "i.py", [{"old_string": "", "new_string": "x = 2"}])
    assert "empty" in out.lower()
    assert "Error" in out


def test_overlap_rejected(tmp_path):
    _write(tmp_path, "j.py", "abcdef\n")
    # Two edits that overlap: "abcde" and "bcdef"
    out = _invoke(tmp_path, "j.py", [
        {"old_string": "abcde", "new_string": "ABCDE"},
        {"old_string": "bcdef", "new_string": "BCDEF"},
    ])
    assert "overlap" in out.lower()
    assert "Error" in out


def test_read_only_blocked(tmp_path):
    _write(tmp_path, "k.py", "x = 1\n")
    cfg = {"configurable": {"project_root": str(tmp_path), "read_only": True}}
    out = edit_file.invoke(
        {"path": "k.py", "edits": [{"old_string": "x = 1", "new_string": "x = 2"}]},
        config=cfg,
    )
    assert "read-only" in out.lower()
    assert (tmp_path / "k.py").read_text() == "x = 1\n"  # unchanged


def test_identical_content_error(tmp_path):
    _write(tmp_path, "l.py", "x = 1\n")
    out = _invoke(tmp_path, "l.py", [{"old_string": "x = 1", "new_string": "x = 1"}])
    assert "identical" in out.lower() or "no changes" in out.lower()
