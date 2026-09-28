"""Tests for read_image_meta — no real SRDP / no network needed.

Synthesises minimal valid image bytes in-memory for each format.
"""
from __future__ import annotations

import io
import struct
import zipfile
import zlib
from pathlib import Path

import pytest

from coding_agent.tools.image_meta import read_image_meta


def _cfg(root: Path) -> dict:
    return {"configurable": {"project_root": str(root), "file_read_limit": 60_000}}


# ---------------------------------------------------------------------------
# minimal image builders
# ---------------------------------------------------------------------------

def _make_png(w: int = 64, h: int = 32, mode: str = "RGB", dpi: int | None = None) -> bytes:
    color_type = {"RGB": 2, "RGBA": 6, "L": 0}[mode]
    channels   = {"RGB": 3, "RGBA": 4, "L": 1}[mode]

    def chunk(name: bytes, data: bytes) -> bytes:
        raw = name + data
        return struct.pack(">I", len(data)) + raw + struct.pack(">I", zlib.crc32(raw) & 0xFFFFFFFF)

    ihdr = chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, color_type, 0, 0, 0))

    chunks = [ihdr]
    if dpi:
        ppm = round(dpi / 0.0254)
        phys = chunk(b"pHYs", struct.pack(">IIB", ppm, ppm, 1))
        chunks.append(phys)

    row = b"\x00" + b"\xff" * (w * channels)
    idat = chunk(b"IDAT", zlib.compress(row * h))
    chunks.append(idat)
    chunks.append(chunk(b"IEND", b""))

    return b"\x89PNG\r\n\x1a\n" + b"".join(chunks)


def _make_jpeg(w: int = 64, h: int = 32) -> bytes:
    """Minimal JPEG: JFIF APP0 + SOF0 + EOI (not valid JPEG data but parseable headers)."""
    soi  = b"\xff\xd8"
    # APP0 JFIF with DPI=72
    app0 = b"\xff\xe0\x00\x10JFIF\x00\x01\x01\x01\x00\x48\x00\x48\x00\x00"
    # SOF0
    sof_data = struct.pack(">BHHB", 8, h, w, 3) + b"\x01\x11\x00\x02\x11\x01\x03\x11\x01"
    sof0 = b"\xff\xc0" + struct.pack(">H", 2 + len(sof_data)) + sof_data
    eoi  = b"\xff\xd9"
    return soi + app0 + sof0 + eoi


def _make_gif(w: int = 32, h: int = 16) -> bytes:
    header = b"GIF89a"
    lsd = struct.pack("<HHBBB", w, h, 0x00, 0, 0)
    trailer = b"\x3b"
    return header + lsd + trailer


def _make_bmp(w: int = 32, h: int = 16) -> bytes:
    row_size = (w * 3 + 3) & ~3
    pixel_data = b"\xff\x00\x00" * w + b"\x00" * (row_size - w * 3)
    pixel_data *= h
    file_size = 54 + len(pixel_data)
    file_header = b"BM" + struct.pack("<IHHI", file_size, 0, 0, 54)
    dib = struct.pack("<IiiHHIIiiII", 40, w, h, 1, 24, 0, len(pixel_data), 0, 0, 0, 0)
    return file_header + dib + pixel_data


def _make_svg(w: float = 200.0, h: float = 100.0) -> bytes:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{w}" height="{h}" viewBox="0 0 {w} {h}">'
        f'<rect width="{w}" height="{h}" fill="red"/>'
        f"</svg>"
    ).encode()


def _make_zip(entries: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, data in entries.items():
            zf.writestr(name, data)
    return buf.getvalue()


def _write_file(root: Path, name: str, data: bytes) -> None:
    (root / name).write_bytes(data)


def _write_zip(root: Path, name: str, entries: dict[str, bytes]) -> None:
    (root / name).write_bytes(_make_zip(entries))


# ---------------------------------------------------------------------------
# PNG
# ---------------------------------------------------------------------------

def test_png_rgb_no_dpi(tmp_path):
    _write_file(tmp_path, "img.png", _make_png(64, 32, "RGB"))
    out = read_image_meta.invoke({"path": "img.png"}, config=_cfg(tmp_path))
    assert "PNG" in out
    assert "64 x 32" in out
    assert "RGB" in out
    assert "dpi" not in out


def test_png_rgba_with_dpi(tmp_path):
    _write_file(tmp_path, "img.png", _make_png(128, 64, "RGBA", dpi=300))
    out = read_image_meta.invoke({"path": "img.png"}, config=_cfg(tmp_path))
    assert "RGBA" in out
    assert "300" in out
    assert "128 x 64" in out


# ---------------------------------------------------------------------------
# JPEG
# ---------------------------------------------------------------------------

def test_jpeg_basic(tmp_path):
    _write_file(tmp_path, "photo.jpg", _make_jpeg(640, 480))
    out = read_image_meta.invoke({"path": "photo.jpg"}, config=_cfg(tmp_path))
    assert "JPEG" in out
    assert "640 x 480" in out
    assert "72" in out   # DPI from JFIF


# ---------------------------------------------------------------------------
# GIF
# ---------------------------------------------------------------------------

def test_gif_basic(tmp_path):
    _write_file(tmp_path, "anim.gif", _make_gif(32, 16))
    out = read_image_meta.invoke({"path": "anim.gif"}, config=_cfg(tmp_path))
    assert "GIF" in out
    assert "32 x 16" in out


# ---------------------------------------------------------------------------
# BMP
# ---------------------------------------------------------------------------

def test_bmp_basic(tmp_path):
    _write_file(tmp_path, "icon.bmp", _make_bmp(32, 16))
    out = read_image_meta.invoke({"path": "icon.bmp"}, config=_cfg(tmp_path))
    assert "BMP" in out
    assert "32 x 16" in out
    assert "24" in out   # bit depth


# ---------------------------------------------------------------------------
# SVG
# ---------------------------------------------------------------------------

def test_svg_basic(tmp_path):
    _write_file(tmp_path, "logo.svg", _make_svg(200, 100))
    out = read_image_meta.invoke({"path": "logo.svg"}, config=_cfg(tmp_path))
    assert "SVG" in out
    assert "200" in out
    assert "100" in out
    assert "viewBox" in out


# ---------------------------------------------------------------------------
# image inside a zip (Pattern 2: zip → image)
# ---------------------------------------------------------------------------

def test_image_inside_zip(tmp_path):
    png = _make_png(50, 50, "RGBA", dpi=96)
    _write_zip(tmp_path, "package.zip", {"media/logo.png": png})
    out = read_image_meta.invoke(
        {"path": "package.zip", "zip_entry": "media/logo.png"},
        config=_cfg(tmp_path),
    )
    assert "PNG" in out
    assert "50 x 50" in out
    assert "RGBA" in out


# ---------------------------------------------------------------------------
# image inside nested zip (Pattern 3: zip → inner-zip → image)
# ---------------------------------------------------------------------------

def test_image_inside_nested_zip(tmp_path):
    png = _make_png(100, 80, "RGB", dpi=150)
    # inner zip (simulates a .bin or .pptx)
    inner_zip = _make_zip({"ppt/media/slide1_img.png": png})
    # outer SRDP zip
    _write_zip(tmp_path, "srdp.zip", {"ExtContent/abc.bin": inner_zip})

    out = read_image_meta.invoke(
        {
            "path": "srdp.zip",
            "zip_entry": "ExtContent/abc.bin",
            "inner_entry": "ppt/media/slide1_img.png",
        },
        config=_cfg(tmp_path),
    )
    assert "PNG" in out
    assert "100 x 80" in out
    assert "150" in out


def test_svg_inside_nested_zip(tmp_path):
    svg = _make_svg(300.5, 200.0)
    inner_zip = _make_zip({"ppt/media/icon.svg": svg})
    _write_zip(tmp_path, "srdp.zip", {"ExtContent/x.bin": inner_zip})

    out = read_image_meta.invoke(
        {"path": "srdp.zip", "zip_entry": "ExtContent/x.bin", "inner_entry": "ppt/media/icon.svg"},
        config=_cfg(tmp_path),
    )
    assert "SVG" in out
    assert "300" in out


# ---------------------------------------------------------------------------
# error cases
# ---------------------------------------------------------------------------

def test_file_not_found(tmp_path):
    out = read_image_meta.invoke({"path": "nope.png"}, config=_cfg(tmp_path))
    assert "Error" in out


def test_non_image_extension_rejected(tmp_path):
    _write_file(tmp_path, "data.xml", b"<xml/>")
    out = read_image_meta.invoke({"path": "data.xml"}, config=_cfg(tmp_path))
    assert "Error" in out


def test_zip_entry_not_found(tmp_path):
    _write_zip(tmp_path, "pkg.zip", {"a.png": _make_png()})
    out = read_image_meta.invoke(
        {"path": "pkg.zip", "zip_entry": "missing.png"},
        config=_cfg(tmp_path),
    )
    assert "Error" in out
    assert "not found" in out


def test_inner_entry_not_found(tmp_path):
    inner = _make_zip({"img.png": _make_png()})
    _write_zip(tmp_path, "outer.zip", {"inner.bin": inner})
    out = read_image_meta.invoke(
        {"path": "outer.zip", "zip_entry": "inner.bin", "inner_entry": "missing.png"},
        config=_cfg(tmp_path),
    )
    assert "Error" in out


def test_path_escape_blocked(tmp_path):
    out = read_image_meta.invoke({"path": "../../etc/shadow.png"}, config=_cfg(tmp_path))
    assert "Error" in out
    assert "escapes" in out
