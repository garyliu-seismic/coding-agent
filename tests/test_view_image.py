from pathlib import Path
from langchain_core.messages import ToolMessage

from coding_agent.tools.image_view import view_image
from coding_agent.tools.image_meta import _extract_metadata


def test_view_image_small(tmp_path: Path):
    # Create a small red PNG
    from PIL import Image
    p = tmp_path / "red.png"
    img = Image.new("RGB", (100, 50), (255, 0, 0))
    img.save(p)

    res = view_image(str(p), config={"configurable": {"project_root": str(tmp_path)}})
    assert "image_url" in res
    assert "data:image/png;base64," in res


def test_view_image_downscale(tmp_path: Path):
    from PIL import Image
    p = tmp_path / "big.png"
    img = Image.new("RGB", (4000, 2000), (0, 255, 0))
    img.save(p)

    res = view_image(str(p), max_side=800, config={"configurable": {"project_root": str(tmp_path)}})
    assert "image_url" in res
    assert "data:image/png;base64," in res


def test_view_image_zip(tmp_path: Path):
    import zipfile
    from PIL import Image
    inner = tmp_path / "inner.png"
    Image.new("RGB", (64, 64), (0, 0, 255)).save(inner)
    z = tmp_path / "a.zip"
    with zipfile.ZipFile(z, "w") as zf:
        zf.write(inner, "img.png")

    res = view_image(str(z.name), zip_entry="img.png", config={"configurable": {"project_root": str(tmp_path)}})
    assert "image_url" in res


def test_extract_metadata_compat():
    # Ensure image_meta._extract_metadata still works for small PNG
    from PIL import Image
    import io
    img = Image.new("RGB", (10, 20), (1, 2, 3))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    meta = _extract_metadata(buf.getvalue(), "t.png")
    assert meta["width"] == 10
    assert meta["height"] == 20
