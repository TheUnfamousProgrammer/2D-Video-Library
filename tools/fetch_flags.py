"""Download pinned flag-icons SVGs and rasterize cast flags.

The pin is confirmed against GitHub before any download. A missing tag exits
with the latest release tag that GitHub actually returned.
"""

from __future__ import annotations

import io
import json
import shutil
import subprocess
import sys
import tarfile
import urllib.error
import urllib.request
from pathlib import Path

import yaml
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
PIN = "v7.5.0"
REPO = "lipis/flag-icons"
SVG_DIR = ROOT / "assets" / "flags" / "svg"
PNG_DIR = ROOT / "assets" / "flags" / "png"
CAMEO_DIR = ROOT / "assets" / "cameos"
RASTER = 512
UA = "fc-sat-fetch-flags"


def _request(url: str, timeout: float = 120) -> tuple[int, bytes]:
    req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "application/vnd.github+json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read()


def confirm_pin(pin: str = PIN, fetch=_request) -> str:
    status, body = fetch(f"https://api.github.com/repos/{REPO}/git/refs/tags/{pin}")
    if status == 200:
        return pin
    latest_status, latest_body = fetch(f"https://api.github.com/repos/{REPO}/releases/latest")
    latest = "unknown"
    if latest_status == 200:
        try:
            latest = str(json.loads(latest_body.decode("utf-8")).get("tag_name") or "unknown")
        except json.JSONDecodeError:
            latest = "unknown"
    raise SystemExit(
        f"pinned flag-icons tag {pin} does not exist (GitHub status {status}); "
        f"latest release is {latest}. The pin was not changed."
    )


def cast_codes() -> list[str]:
    codes: list[str] = []
    seen: set[str] = set()
    for path in sorted((ROOT / "configs" / "casts").glob("*.yaml")):
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise SystemExit(f"{path} is not a cast list")
        for item in data:
            code = str(item["iso2"]).lower()
            if code not in seen:
                seen.add(code)
                codes.append(code)
    if not codes:
        raise SystemExit("no cast codes found")
    return codes


def _hint() -> str:
    return (
        "No SVG rasterizer is available. Install one of:\n"
        "  pip install cairosvg    (also needs the cairo library: brew install cairo)\n"
        "  brew install resvg\n"
        "  brew install librsvg    (provides rsvg-convert)"
    )


def rasterize_svg(svg: Path, png: Path) -> str:
    """Rasterize to 512x512. Returns the tool name that succeeded."""
    png.parent.mkdir(parents=True, exist_ok=True)
    errors: list[str] = []
    try:
        import cairosvg

        cairosvg.svg2png(url=str(svg), write_to=str(png), output_width=RASTER, output_height=RASTER)
        return "cairosvg"
    except Exception as exc:
        errors.append(f"cairosvg: {exc}")
    resvg = shutil.which("resvg")
    if resvg:
        result = subprocess.run(
            [resvg, "-w", str(RASTER), "-h", str(RASTER), str(svg), str(png)],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0 and png.is_file():
            return "resvg"
        errors.append(f"resvg: {result.stderr.strip() or result.returncode}")
    rsvg = shutil.which("rsvg-convert")
    if rsvg:
        result = subprocess.run(
            [rsvg, "-w", str(RASTER), "-h", str(RASTER), "-o", str(png), str(svg)],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0 and png.is_file():
            return "rsvg-convert"
        errors.append(f"rsvg-convert: {result.stderr.strip() or result.returncode}")
    detail = "\n".join(errors) if errors else "none of the tools imported or were on PATH"
    raise SystemExit(_hint() + f"\nTried:\n{detail}")


def download_svgs(pin: str, codes: list[str]) -> None:
    SVG_DIR.mkdir(parents=True, exist_ok=True)
    needed = [code for code in codes if not (SVG_DIR / f"{code}.svg").is_file()]
    if not needed:
        return
    url = f"https://codeload.github.com/{REPO}/tar.gz/refs/tags/{pin}"
    status, body = _request(url)
    if status != 200 or not body.startswith(b"\x1f\x8b"):
        raise SystemExit(f"could not download flag-icons {pin} (HTTP {status})")
    wanted = {f"flags/1x1/{code}.svg": code for code in needed}
    found: set[str] = set()
    with tarfile.open(fileobj=io.BytesIO(body), mode="r:gz") as archive:
        for member in archive.getmembers():
            if not member.isfile():
                continue
            for suffix, code in wanted.items():
                if member.name.endswith(suffix):
                    extracted = archive.extractfile(member)
                    if extracted is None:
                        continue
                    (SVG_DIR / f"{code}.svg").write_bytes(extracted.read())
                    found.add(code)
                    break
    missing = [code.upper() for code in needed if code not in found]
    if missing:
        raise SystemExit("cast flag missing from flag-icons " + pin + ": " + ", ".join(missing))


def available_rasterizer() -> str:
    try:
        import cairosvg  # noqa: F401

        return "cairosvg"
    except Exception:
        pass
    if shutil.which("resvg"):
        return "resvg"
    if shutil.which("rsvg-convert"):
        return "rsvg-convert"
    return "unknown"


def rasterize_cached(codes: list[str]) -> str:
    tool = ""
    for code in codes:
        svg = SVG_DIR / f"{code}.svg"
        png = PNG_DIR / f"{code}.png"
        if not svg.is_file():
            raise SystemExit(f"cast flag missing: {code.upper()} ({svg})")
        if png.is_file() and png.stat().st_mtime >= svg.stat().st_mtime:
            continue
        tool = rasterize_svg(svg, png)
    return tool or "cache"


def write_credits(pin: str, tool: str) -> None:
    text = (
        "# Flag credits\n\n"
        f"Country flags are the 1x1 SVGs from [flag-icons](https://github.com/lipis/flag-icons) "
        f"tag `{pin}`.\n\n"
        "License: MIT. Copyright (c) Panayiotis Lipiridis.\n\n"
        f"Rasterized to {RASTER}x{RASTER} PNG with `{tool}`.\n"
    )
    (SVG_DIR.parent / "CREDITS.md").write_text(text, encoding="utf-8")


def _public_domain(metadata: dict) -> bool:
    blob = " ".join(str(item.get("value", "")) for item in metadata.values() if isinstance(item, dict))
    lowered = blob.lower()
    return "public domain" in lowered


def _draw_burgee(path: Path) -> None:
    image = Image.new("RGBA", (RASTER, RASTER), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    red = (196, 30, 58, 255)
    white = (255, 255, 255, 255)
    blue = (0, 40, 104, 255)
    draw.polygon([(40, 80), (470, 80), (360, 256), (470, 432), (40, 432)], fill=red)
    for top in (140, 250, 360):
        draw.polygon([(70, top), (430, top), (400, top + 36), (70, top + 36)], fill=white)
    draw.polygon([(40, 80), (250, 256), (40, 432)], fill=blue)
    draw.ellipse((108, 176, 228, 336), fill=white)
    draw.ellipse((138, 216, 198, 296), fill=red)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def fetch_cameo() -> str:
    """Return 'commons' or 'stylized'."""
    CAMEO_DIR.mkdir(parents=True, exist_ok=True)
    api = (
        "https://commons.wikimedia.org/w/api.php?action=query&format=json"
        "&titles=File:Flag_of_Ohio.svg&prop=imageinfo&iiprop=url|extmetadata"
    )
    note = "Stylized swallowtail burgee drawn with Pillow. Not the official artwork."
    kind = "stylized"
    try:
        status, body = _request(api)
    except (urllib.error.URLError, TimeoutError, OSError):
        status, body = 0, b""
    if status == 200:
        try:
            payload = json.loads(body.decode("utf-8"))
            pages = payload["query"]["pages"]
            page = next(iter(pages.values()))
            info = page["imageinfo"][0]
            if _public_domain(info.get("extmetadata", {})):
                svg_status, svg = _request(info["url"])
                if svg_status == 200 and svg.startswith(b"<"):
                    svg_path = CAMEO_DIR / "ohio.svg"
                    svg_path.write_bytes(svg)
                    rasterize_svg(svg_path, CAMEO_DIR / "ohio.png")
                    note = (
                        "Flag of Ohio from Wikimedia Commons "
                        f"({info['url']}). License metadata says public domain."
                    )
                    kind = "commons"
        except (KeyError, StopIteration, json.JSONDecodeError, TypeError):
            kind = "stylized"
    if kind != "commons":
        _draw_burgee(CAMEO_DIR / "ohio.png")
    (CAMEO_DIR / "CREDITS.md").write_text("# Cameo credits\n\n" + note + "\n", encoding="utf-8")
    return kind


def main() -> None:
    pin = confirm_pin()
    codes = cast_codes()
    download_svgs(pin, codes)
    tool = rasterize_cached(codes)
    if tool == "cache":
        if not any(PNG_DIR.glob("*.png")):
            raise SystemExit("no flag PNGs were written")
        tool = available_rasterizer()
    write_credits(pin, tool)
    kind = fetch_cameo()
    print(f"flags: {len(codes)} from flag-icons {pin} via {tool}; cameo: {kind}")


if __name__ == "__main__":
    main()
