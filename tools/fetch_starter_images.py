#!/usr/bin/env python3
"""Download a few public-domain starter pictures from Wikimedia Commons.

A file is kept only when LicenseShortName says public domain or CC0.
Anything else is deleted (or never saved) and the reason is printed.
Missing files are skipped. Credits land in assets/images/credits.json.
"""

from __future__ import annotations

import hashlib
import html
import json
import re
import sys
import urllib.parse
import urllib.request
from pathlib import Path

UA = "fc-sat-pixel-morph/1.0 (local educational renderer; contact: local)"
# commons.wikimedia.org resets on some networks. en.wikipedia.org serves the
# same Commons file record, including LicenseShortName. Special:FilePath is
# tried first; the canonical upload.wikimedia.org URL is the fallback.
APIS = (
    "https://en.wikipedia.org/w/api.php",
    "https://commons.wikimedia.org/w/api.php",
)
THUMB_WIDTH = 2000
ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "assets" / "images"

CANDIDATES = (
    {
        "slug": "mona_lisa",
        "title": "Mona Lisa",
        "files": ("Mona Lisa, by Leonardo da Vinci, from C2RMF retouched.jpg",),
        "search": 'Mona Lisa Leonardo da Vinci',
    },
    {
        "slug": "starry_night",
        "title": "The Starry Night",
        "files": ("Van Gogh - Starry Night - Google Art Project.jpg",),
        "search": "Starry Night Google Art Project Van Gogh",
    },
    {
        "slug": "great_wave",
        "title": "The Great Wave off Kanagawa",
        "files": (
            "The Great Wave off Kanagawa.jpg",
            "Tsunami by hokusai 19th century.jpg",
        ),
        "search": "The Great Wave off Kanagawa Hokusai",
    },
    {
        "slug": "pearl_earring",
        "title": "Girl with a Pearl Earring",
        "files": (
            "Johannes Vermeer - Girl with a Pearl Earring - Google Art Project.jpg",
            "Meisje met de parel.jpg",
        ),
        "search": "Girl with a Pearl Earring Vermeer",
    },
)


def _get(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(request, timeout=180) as response:
        return response.read()


def _api(base: str, params: dict) -> dict:
    params = dict(params)
    params["format"] = "json"
    url = base + "?" + urllib.parse.urlencode(params)
    return json.loads(_get(url).decode("utf-8"))


def _page(title: str) -> dict | None:
    params = {
        "action": "query",
        "titles": title if title.startswith("File:") else f"File:{title}",
        "prop": "imageinfo",
        "iiprop": "url|mime|extmetadata",
        "iiurlwidth": str(THUMB_WIDTH),
    }
    last_error = ""
    for base in APIS:
        try:
            data = _api(base, params)
        except Exception as exc:
            last_error = str(exc)
            continue
        pages = data.get("query", {}).get("pages", {})
        if not pages:
            continue
        page = next(iter(pages.values()))
        if page.get("missing") is not None or "imageinfo" not in page:
            return None
        return page
    if last_error:
        print(f"  api failed for {title}: {last_error}")
    return None


def _plain(value: str) -> str:
    text = re.sub(r"<[^>]+>", "", value or "")
    return html.unescape(re.sub(r"\s+", " ", text)).strip()


def _license_ok(short_name: str) -> tuple[bool, str]:
    text = (short_name or "").strip().lower()
    if not text:
        return False, "LicenseShortName is missing"
    if "cc by" in text or "cc-by" in text or "attribution" in text:
        return False, f"license is {short_name}, not public domain or CC0"
    if "public domain" in text or "cc0" in text or text in {"pd", "pd-old"} or text.startswith("pd-"):
        return True, short_name
    return False, f"license is {short_name}, not public domain or CC0"


def _meta(page: dict) -> tuple[bool, str, str, str]:
    info = page["imageinfo"][0]
    extra = info.get("extmetadata") or {}
    short = _plain((extra.get("LicenseShortName") or {}).get("value", ""))
    artist = _plain((extra.get("Artist") or {}).get("value", "")) or "unknown"
    ok, reason = _license_ok(short)
    title = page.get("title", "")
    return ok, reason, artist, title


def _search(query: str) -> list[str]:
    params = {
        "action": "query",
        "list": "search",
        "srsearch": query,
        "srnamespace": "6",
        "srlimit": "8",
    }
    for base in APIS:
        try:
            data = _api(base, params)
        except Exception:
            continue
        hits = data.get("query", {}).get("search", [])
        return [hit.get("title", "") for hit in hits if hit.get("title")]
    return []


def _download(filename: str, info: dict, dest: Path) -> None:
    quoted = urllib.parse.quote(filename, safe="")
    special = f"https://commons.wikimedia.org/wiki/Special:FilePath/{quoted}?width={THUMB_WIDTH}"
    thumb = info.get("thumburl") or info.get("url")
    data = b""
    errors = []
    for url in (special, thumb):
        if not url:
            continue
        try:
            data = _get(url)
            break
        except Exception as exc:
            errors.append(str(exc))
    if not data:
        raise RuntimeError("; ".join(errors) or "no download url")
    if data[:20].lstrip().lower().startswith((b"<!doctype", b"<html")):
        raise RuntimeError("download returned HTML")
    dest.write_bytes(data)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _try_titles(titles: list[str]) -> tuple[dict, str] | None:
    for title in titles:
        if not title:
            continue
        page = _page(title)
        if page is None:
            print(f"  skip {title}: not found (404)")
            continue
        ok, reason, artist, found = _meta(page)
        if not ok:
            print(f"  delete/skip {found}: {reason}")
            continue
        return page, artist
    return None


def main() -> int:
    DEST.mkdir(parents=True, exist_ok=True)
    credits = []
    for candidate in CANDIDATES:
        print(f"{candidate['title']}:")
        titles = list(candidate["files"])
        found = _try_titles(titles)
        if found is None:
            print(f"  searching: {candidate['search']}")
            found = _try_titles(_search(candidate["search"]))
        if found is None:
            print(f"  no public-domain file kept for {candidate['title']}")
            continue
        page, artist = found
        info = page["imageinfo"][0]
        extra = info.get("extmetadata") or {}
        license_name = _plain((extra.get("LicenseShortName") or {}).get("value", ""))
        filename = page.get("title", "File:image.jpg").split("File:", 1)[-1]
        suffix = Path(filename).suffix.lower() or ".jpg"
        if suffix not in {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff"}:
            suffix = ".jpg"
        dest = DEST / f"{candidate['slug']}{suffix}"
        try:
            _download(filename, info, dest)
        except Exception as exc:
            if dest.exists():
                dest.unlink()
            print(f"  download failed for {filename}: {exc}")
            continue
        source = "https://commons.wikimedia.org/wiki/" + urllib.parse.quote(page.get("title", ""))
        credits.append(
            {
                "file": dest.name,
                "title": candidate["title"],
                "artist": artist,
                "license": license_name,
                "source": source,
                "sha256": _sha256(dest),
            }
        )
        print(f"  kept {dest.name} ({license_name}, {artist})")
    payload = {"images": credits}
    (DEST / "credits.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {len(credits)} credit entries")
    return 0 if credits else 1


if __name__ == "__main__":
    sys.exit(main())
