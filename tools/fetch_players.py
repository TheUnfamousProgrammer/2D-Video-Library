"""Download openly licensed portraits and crop them to square face circles.

Accepts public domain, CC0, CC BY, and CC BY-SA. Skips non-commercial and
no-derivatives licenses. Writes assets/players/{code}.png and CREDITS.md.
"""

from __future__ import annotations

import json
import re
import urllib.parse
import urllib.request
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "assets" / "players"
UA = "fc-sat/1.0 (local portrait crop; contact: local)"

# code, wikipedia title
ROSTER = [
    ("mb", "Kylian Mbappé"),
    ("ly", "Lamine Yamal"),
    ("lm", "Lionel Messi"),
    ("cr", "Cristiano Ronaldo"),
    ("ha", "Erling Haaland"),
    ("mo", "Luka Modrić"),
    ("kd", "Kevin De Bruyne"),
    ("ms", "Mohamed Salah"),
    ("hk", "Harry Kane"),
    ("vj", "Vinícius Júnior"),
    ("jb", "Jude Bellingham"),
    ("rd", "Rodrigo Hernández Cascante"),
    ("rl", "Robert Lewandowski"),
    ("nj", "Neymar"),
    ("kb", "Karim Benzema"),
    ("sh", "Son Heung-min"),
    ("bs", "Bukayo Saka"),
    ("pf", "Phil Foden"),
    ("pd", "Pedri"),
    ("fv", "Federico Valverde"),
    ("ag", "Antoine Griezmann"),
    ("bf", "Bruno Fernandes"),
    ("vd", "Virgil van Dijk"),
    ("tc", "Thibaut Courtois"),
    ("tk", "Toni Kroos"),
    ("jm", "Jamal Musiala"),
    ("fw", "Florian Wirtz"),
    ("vo", "Victor Osimhen"),
    ("kv", "Khvicha Kvaratskhelia"),
    ("ta", "Trent Alexander-Arnold"),
    ("ah", "Achraf Hakimi"),
    ("od", "Ousmane Dembélé"),
]


def _get(url: str) -> bytes:
    import time

    last: Exception | None = None
    for attempt in range(6):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(request, timeout=40) as response:
                return response.read()
        except Exception as exc:
            last = exc
            time.sleep(2.5 * (attempt + 1))
    assert last is not None
    raise last


def _api(base: str, params: dict) -> dict:
    query = urllib.parse.urlencode(params)
    return json.loads(_get(f"{base}?{query}").decode("utf-8"))


def _ok_license(name: str) -> bool:
    text = name.lower().replace("‑", "-")
    if "non-free" in text or "fair use" in text or "copyrighted" in text:
        return False
    if any(token in text for token in ("noncommercial", "non-commercial", "-nc", " nc")):
        return False
    if "noderiv" in text or "-nd" in text or " no derivatives" in text:
        return False
    if "public domain" in text or text.startswith("cc0") or "cc0" in text:
        return True
    return "cc by" in text or "cc-by" in text


def _lead(title: str) -> str | None:
    data = _api(
        "https://en.wikipedia.org/w/api.php",
        {
            "action": "query",
            "titles": title,
            "prop": "pageimages",
            "piprop": "name",
            "redirects": "1",
            "format": "json",
        },
    )
    page = next(iter(data["query"]["pages"].values()))
    name = page.get("pageimage")
    if not name:
        return None
    return "File:" + name


def _info(filename: str) -> tuple[str, str, str] | None:
    data = _api(
        "https://en.wikipedia.org/w/api.php",
        {
            "action": "query",
            "titles": filename,
            "prop": "imageinfo",
            "iiprop": "url|extmetadata",
            "iiurlwidth": "800",
            "format": "json",
        },
    )
    pages = data["query"]["pages"]
    page = next(iter(pages.values()))
    info = (page.get("imageinfo") or [None])[0]
    if not info:
        return None
    meta = info.get("extmetadata") or {}
    license_name = (meta.get("LicenseShortName") or {}).get("value", "")
    artist = (meta.get("Artist") or {}).get("value", "")
    credit = (meta.get("Credit") or {}).get("value", "")
    if not _ok_license(license_name):
        return None
    url = info.get("thumburl") or info.get("url")
    if not url:
        return None
    return url, license_name, artist or credit or "Wikimedia Commons"


def _crop(image: np.ndarray, face: tuple[int, int, int, int]) -> np.ndarray:
    x, y, w, h = face
    cx = x + w / 2
    cy = y + h * 0.42
    side = max(w, h) * 2.15
    half = side / 2
    x0 = int(round(cx - half))
    y0 = int(round(cy - half))
    x1 = int(round(cx + half))
    y1 = int(round(cy + half))
    pad_x0 = max(0, -x0)
    pad_y0 = max(0, -y0)
    pad_x1 = max(0, x1 - image.shape[1])
    pad_y1 = max(0, y1 - image.shape[0])
    if pad_x0 or pad_y0 or pad_x1 or pad_y1:
        image = cv2.copyMakeBorder(image, pad_y0, pad_y1, pad_x0, pad_x1, cv2.BORDER_REPLICATE)
        x0 += pad_x0
        y0 += pad_y0
        x1 += pad_x0
        y1 += pad_y0
    crop = image[y0:y1, x0:x1]
    return cv2.resize(crop, (512, 512), interpolation=cv2.INTER_AREA)


def _best_face(path_bytes: bytes) -> np.ndarray | None:
    raw = np.frombuffer(path_bytes, dtype=np.uint8)
    image = cv2.imdecode(raw, cv2.IMREAD_COLOR)
    if image is None:
        return None
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = cascade.detectMultiScale(gray, scaleFactor=1.08, minNeighbors=5, minSize=(60, 60))
    if len(faces) == 0:
        return None
    face = max(faces, key=lambda box: box[2] * box[3])
    return _crop(image, tuple(int(v) for v in face))


def _square(image: np.ndarray) -> np.ndarray:
    h, w = image.shape[:2]
    side = min(w, int(h * 0.92))
    x0 = max(0, (w - side) // 2)
    y0 = max(0, int(h * 0.02))
    y1 = min(h, y0 + side)
    y0 = y1 - side
    crop = image[y0:y1, x0 : x0 + side]
    return cv2.resize(crop, (512, 512), interpolation=cv2.INTER_AREA)


def main() -> None:
    import time

    OUT.mkdir(parents=True, exist_ok=True)
    credit_path = OUT / "CREDITS.md"
    header = ["# Player portraits", "", "Cropped from Wikimedia Commons. Each file keeps the source license.", ""]
    kept = []
    if credit_path.is_file():
        kept = [line for line in credit_path.read_text(encoding="utf-8").splitlines() if line.startswith("- `")]
    credits = header + kept
    done = {line.split("`")[1].split(".")[0] for line in kept}
    for code, title in ROSTER:
        dest = OUT / f"{code}.png"
        if dest.is_file() and code in done:
            print(f"keep {code}")
            continue
        time.sleep(1.2)
        try:
            filename = _lead(title)
        except Exception as exc:
            print(f"MISS {code} {title} {exc}")
            continue
        if not filename:
            print(f"MISS {code} {title} no page image")
            continue
        time.sleep(0.35)
        try:
            info = _info(filename)
        except Exception as exc:
            print(f"MISS {code} {title} {exc}")
            continue
        if info is None:
            print(f"MISS {code} {title} license {filename}")
            continue
        url, license_name, artist = info
        raw = _get(url)
        face = _best_face(raw)
        if face is None:
            decoded = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
            if decoded is None:
                print(f"MISS {code} {title} decode")
                continue
            face = _square(decoded)
        cv2.imwrite(str(dest), face)
        page = "https://commons.wikimedia.org/wiki/" + urllib.parse.quote(filename.replace(" ", "_"))
        artist = re.sub(r"<[^>]+>", "", artist)
        artist = re.sub(r"\s+", " ", artist).strip()
        credits.append(f"- `{code}.png` — {title}. {license_name}. {artist}. {page}")
        credit_path.write_text("\n".join(credits) + "\n", encoding="utf-8")
        print(f"ok {code} {license_name}")
    credit_path.write_text("\n".join(credits) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
