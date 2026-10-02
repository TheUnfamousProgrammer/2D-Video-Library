"""Parse the Barina project page for the verified convergence bound."""

from __future__ import annotations

import re
import urllib.request

PAGE_URL = "https://pcbarina.fit.vutbr.cz/"

_BOUND = re.compile(
    r"(?:(?P<coef>\d+(?:\.\d+)?)\s*[×x]\s*)?2\s*\^\s*(?P<exp>\d+)",
    re.IGNORECASE,
)


def bound_integer(coef: str | None, exp: int) -> int:
    power = 2**exp
    if coef is None:
        return power
    if "." in coef:
        whole, frac = coef.split(".", 1)
        scale = 10 ** len(frac)
        numer = int(whole) * scale + int(frac)
        return numer * power // scale
    return int(coef) * power


def _plain(html: str) -> str:
    text = re.sub(r"<\s*sup\s*>", "^", html, flags=re.IGNORECASE)
    text = re.sub(r"<\s*/\s*sup\s*>", "", text, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", text)
    text = text.replace("&times;", "×").replace("&asymp;", "≈").replace("\n", " ")
    return re.sub(r"\s+", " ", text)


def bounds_from_html(html: str) -> list[int]:
    """Integers the page states as already verified. Progress targets are ignored."""
    text = re.sub(r"(\d)\.(\d)", r"\1DOT\2", _plain(html))
    found: list[int] = []
    for sentence in re.split(r"[.]", text):
        sentence = sentence.replace("DOT", ".")
        lowered = sentence.lower()
        if "verified" not in lowered or "towards" in lowered:
            continue
        if "numbers below" not in lowered:
            continue
        for bound in _BOUND.finditer(sentence):
            found.append(bound_integer(bound.group("coef"), int(bound.group("exp"))))
    return found


def latest_bound(html: str) -> int | None:
    found = bounds_from_html(html)
    if not found:
        return None
    return max(found)


def fetch_page(timeout: float = 30.0) -> str:
    request = urllib.request.Request(PAGE_URL, headers={"User-Agent": "fc-sat-collatz"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read().decode("utf-8", errors="replace")
