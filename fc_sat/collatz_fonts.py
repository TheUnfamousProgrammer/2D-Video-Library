"""Read OFL fonts and decide whether the multiply and divide signs can be drawn."""

from __future__ import annotations

from pathlib import Path

from PIL import ImageFont

ROOT = Path(__file__).resolve().parents[1]
FONT_DIR = ROOT / "assets" / "fonts"
MONO_PATH = FONT_DIR / "JetBrainsMono-ExtraBold.ttf"
WORD_PATH = FONT_DIR / "Montserrat-ExtraBold.ttf"
MULTIPLY = "\u00d7"
DIVIDE = "\u00f7"


def font_file(kind: str) -> Path:
    path = MONO_PATH if kind == "mono" else WORD_PATH
    if not path.exists():
        raise SystemExit(
            f"Missing {path.name}. Place the OFL file in assets/fonts/ "
            "(JetBrains Mono ExtraBold and Montserrat ExtraBold)."
        )
    return path


def load_font(kind: str, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(font_file(kind)), size=max(1, int(size)))


def glyph_present(font: ImageFont.FreeTypeFont, char: str) -> bool:
    box = font.getbbox(char)
    if box is None:
        return False
    width = box[2] - box[0]
    height = box[3] - box[1]
    if width <= 0 or height <= 0:
        return False
    missing = font.getbbox("\ufffd")
    if missing is not None and box == missing and char != "\ufffd":
        return False
    return True


def operation_glyphs(kind: str = "word") -> tuple[str, str]:
    """Return the multiply and divide characters the font can draw."""
    font = load_font(kind, 64)
    multiply = MULTIPLY if glyph_present(font, MULTIPLY) else "x"
    divide = DIVIDE if glyph_present(font, DIVIDE) else "/"
    return multiply, divide
