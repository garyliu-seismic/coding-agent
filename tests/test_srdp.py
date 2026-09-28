"""Tests for srdp_list and srdp_read tools — no real SRDP needed.

Creates synthetic zip structures in tmp_path that mirror the SRDP layout.
"""
from __future__ import annotations

import io
import zipfile
from pathlib import Path

from coding_agent.tools.srdp import srdp_list, srdp_read


def _cfg(root: Path) -> dict:
    return {"configurable": {"project_root": str(root), "file_read_limit": 60_000}}


def _make_srdp(path: Path, entries: dict[str, bytes]) -> None:
    """Write a synthetic SRDP zip."""
    with zipfile.ZipFile(str(path), "w") as zf:
        for name, data in entries.items():
            zf.writestr(name, data)


def _make_nested_zip(entries: dict[str, bytes]) -> bytes:
    """Create an in-memory zip and return its bytes."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, data in entries.items():
            zf.writestr(name, data)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# srdp_list
# ---------------------------------------------------------------------------

def test_srdp_list_basic(tmp_path):
    srdp = tmp_path / "test.zip"
    _make_srdp(srdp, {
        "index.xml": b"<SRDP><Engine>V2</Engine></SRDP>",
        "MainRequestDocumentInput.xml": b"<RequestDocument/>",
        "UserInfo.xml": b"<UserInfo/>",
    })
    out = srdp_list.invoke({"srdp_path": "test.zip"}, config=_cfg(tmp_path))
    assert "index.xml" in out
    assert "UserInfo.xml" in out
    assert "[text]" in out


def test_srdp_list_shows_nested_bin(tmp_path):
    nested = _make_nested_zip({
        "ppt/slides/slide1.xml": b"<slide/>",
        "ppt/presentation.xml": b"<prs/>",
    })
    srdp = tmp_path / "test.zip"
    _make_srdp(srdp, {
        "index.xml": b"<SRDP/>",
        "ExtContent/abc123.bin": nested,
    })
    out = srdp_list.invoke({"srdp_path": "test.zip"}, config=_cfg(tmp_path))
    assert "ExtContent/abc123.bin" in out
    assert "nested zip" in out
    assert "ppt/slides/slide1.xml" in out


def test_srdp_list_not_found(tmp_path):
    out = srdp_list.invoke({"srdp_path": "missing.zip"}, config=_cfg(tmp_path))
    assert "Error" in out


def test_srdp_list_bad_zip(tmp_path):
    bad = tmp_path / "bad.zip"
    bad.write_bytes(b"not a zip")
    out = srdp_list.invoke({"srdp_path": "bad.zip"}, config=_cfg(tmp_path))
    assert "Error" in out


# ---------------------------------------------------------------------------
# srdp_read — top-level text
# ---------------------------------------------------------------------------

def test_srdp_read_text_entry(tmp_path):
    srdp = tmp_path / "test.zip"
    _make_srdp(srdp, {"index.xml": b"<SRDP><Engine>V2</Engine></SRDP>"})
    out = srdp_read.invoke(
        {"srdp_path": "test.zip", "entry": "index.xml"}, config=_cfg(tmp_path)
    )
    assert "<Engine>V2</Engine>" in out


def test_srdp_read_entry_not_found(tmp_path):
    srdp = tmp_path / "test.zip"
    _make_srdp(srdp, {"index.xml": b"<SRDP/>"})
    out = srdp_read.invoke(
        {"srdp_path": "test.zip", "entry": "missing.xml"}, config=_cfg(tmp_path)
    )
    assert "Error" in out
    assert "not found" in out


# ---------------------------------------------------------------------------
# srdp_read — nested bin (ExtContent)
# ---------------------------------------------------------------------------

def test_srdp_read_bin_lists_contents_without_inner_entry(tmp_path):
    nested = _make_nested_zip({"ppt/slides/slide1.xml": b"<slide/>"})
    srdp = tmp_path / "test.zip"
    _make_srdp(srdp, {"ExtContent/abc.bin": nested})
    out = srdp_read.invoke(
        {"srdp_path": "test.zip", "entry": "ExtContent/abc.bin"},
        config=_cfg(tmp_path),
    )
    assert "ppt/slides/slide1.xml" in out
    assert "nested zip" in out or "Entries" in out


def test_srdp_read_bin_inner_entry(tmp_path):
    nested = _make_nested_zip({"ppt/slides/slide1.xml": b"<slide><title>Hello</title></slide>"})
    srdp = tmp_path / "test.zip"
    _make_srdp(srdp, {"ExtContent/abc.bin": nested})
    out = srdp_read.invoke(
        {
            "srdp_path": "test.zip",
            "entry": "ExtContent/abc.bin",
            "inner_entry": "ppt/slides/slide1.xml",
        },
        config=_cfg(tmp_path),
    )
    assert "Hello" in out


def test_srdp_read_bin_inner_entry_not_found(tmp_path):
    nested = _make_nested_zip({"ppt/slides/slide1.xml": b"<slide/>"})
    srdp = tmp_path / "test.zip"
    _make_srdp(srdp, {"ExtContent/abc.bin": nested})
    out = srdp_read.invoke(
        {
            "srdp_path": "test.zip",
            "entry": "ExtContent/abc.bin",
            "inner_entry": "does/not/exist.xml",
        },
        config=_cfg(tmp_path),
    )
    assert "Error" in out


def test_srdp_read_bin_not_a_zip(tmp_path):
    srdp = tmp_path / "test.zip"
    _make_srdp(srdp, {"DataSourceBytes/data.bin": b"\x00\x01\x02binary"})
    out = srdp_read.invoke(
        {"srdp_path": "test.zip", "entry": "DataSourceBytes/data.bin"},
        config=_cfg(tmp_path),
    )
    assert "not a zip" in out or "binary" in out.lower()


# ---------------------------------------------------------------------------
# srdp_read — Office zip (pptx inside SRDP)
# ---------------------------------------------------------------------------

def test_srdp_read_pptx_lists_contents(tmp_path):
    pptx_bytes = _make_nested_zip({
        "ppt/presentation.xml": b"<prs/>",
        "ppt/slides/slide1.xml": b"<slide/>",
    })
    srdp = tmp_path / "test.zip"
    _make_srdp(srdp, {"InstanceLiveDoc.pptx": pptx_bytes})
    out = srdp_read.invoke(
        {"srdp_path": "test.zip", "entry": "InstanceLiveDoc.pptx"},
        config=_cfg(tmp_path),
    )
    assert "ppt/presentation.xml" in out
    assert "ppt/slides/slide1.xml" in out


def test_srdp_read_pptx_inner_xml(tmp_path):
    pptx_bytes = _make_nested_zip({
        "ppt/slides/slide1.xml": b"<slide><sp>content</sp></slide>",
    })
    srdp = tmp_path / "test.zip"
    _make_srdp(srdp, {"InstanceLiveDoc.pptx": pptx_bytes})
    out = srdp_read.invoke(
        {
            "srdp_path": "test.zip",
            "entry": "InstanceLiveDoc.pptx",
            "inner_entry": "ppt/slides/slide1.xml",
        },
        config=_cfg(tmp_path),
    )
    assert "content" in out


# ---------------------------------------------------------------------------
# path escape guard
# ---------------------------------------------------------------------------

def test_srdp_path_escape_blocked(tmp_path):
    out = srdp_list.invoke({"srdp_path": "../../etc/passwd"}, config=_cfg(tmp_path))
    assert "Error" in out
    assert "escapes" in out
