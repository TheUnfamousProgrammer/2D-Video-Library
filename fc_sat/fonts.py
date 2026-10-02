"""OFL font files shared by generators that draw type.

JetBrains Mono ExtraBold is the number face. Montserrat ExtraBold is the word face.
Both live in ``assets/fonts``. A missing file is a hard error with an install hint.
"""

from __future__ import annotations

from pathlib import Path

from PIL import ImageFont

ROOT = Path(__file__).resolve().parents[1]
FONT_DIR = ROOT / "assets" / "fonts"
PATHS = {
    "mono": FONT_DIR / "JetBrainsMono-ExtraBold.ttf",
    "word": FONT_DIR / "Montserrat-ExtraBold.ttf",
}


def font_file(kind: str) -> Path:
    try:
        path = PATHS[kind]
    except KeyError as exc:
        raise SystemExit(f"Unknown font kind {kind!r}. Use 'mono' or 'word'.") from exc
    if not path.exists():
        raise SystemExit(
            f"Missing {path.name}. Place the OFL file in assets/fonts/ "
            "(JetBrains Mono ExtraBold and Montserrat ExtraBold)."
        )
    return path


def load_font(kind: str, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(font_file(kind)), size=max(1, int(size)))


def glyph_present(font: ImageFont.FreeTypeFont, char: str) -> bool:
    """True when the font draws ``char`` as something other than the missing-glyph box."""
    box = font.getbbox(char)
    if box is None:
        return False
    if box[2] - box[0] <= 0 or box[3] - box[1] <= 0:
        return False
    missing = font.getbbox("\ufffd")
    if missing is not None and box == missing and char != "\ufffd":
        return False
    return True
