"""Titles, description, and the retention map. Numbers in the post come from the fit."""

from __future__ import annotations

import re

_COMMA = re.compile(r"\d{1,3}(?:,\d{3})+")
_DECIMAL = re.compile(r"\d+\.\d+")
_DIGITS = re.compile(r"\d+")

TITLES = (
    "I threw millions of random lines. Wait for it.",
    "Random lines. Guess what they become.",
    "No curves, no brush, just random lines",
)


def _numbers(text: str) -> list[str]:
    occupied = [False] * (len(text) + 1)
    found: list[tuple[int, str]] = []

    def take(pattern: re.Pattern[str]) -> None:
        for match in pattern.finditer(text):
            if any(occupied[match.start() : match.end()]):
                continue
            found.append((match.start(), match.group()))
            for index in range(match.start(), match.end()):
                occupied[index] = True

    take(_COMMA)
    take(_DECIMAL)
    take(_DIGITS)
    found.sort()
    return [token for _, token in found]


def render_post(meta: dict) -> str:
    kept = int(meta["kept"])
    thrown = int(meta["thrown"])
    ratio = max(1, int(round(thrown / max(kept, 1))))
    credit = str(meta.get("credit") or "").strip()
    credit_line = f"{credit} " if credit else ""
    rights = str(meta.get("rights") or "unknown")
    warning = ""
    if rights == "unknown":
        warning = (
            "\nRights warning: this image's rights are unknown. It may be copyrighted and it may show a real person. "
            "Do not post it until you have permission, and answer the copyright and altered-content questions honestly.\n"
        )
    titles = "\n".join(f"- {title}" for title in TITLES)
    return f"""# Post kit

## Titles
{titles}

## Description
GUESS WHAT THIS BECOMES.
Random lines are thrown and only the ones that bring the drawing closer are kept; likeness is a similarity score.
{credit_line}Music: original, made with code.
#shorts #art #satisfying #generativeart #asmr

## Pinned comments
- What did you think it was before the drop?
- {kept:,} lines kept out of {thrown:,} thrown.

## Thumbnail candidates
- out/stills/frame_0600.png at the build, before the picture snaps in
- out/stills/frame_0768.png at the drop
- out/stills/frame_0940.png at the zoom, where the mouth is only straight lines

The kept-to-thrown ratio on screen is 1 in {ratio}.
{warning}
## Posting checklist
- Set the audience. This picture may show a real person when rights are unknown, so don't treat it as made for kids.
- Answer YouTube's copyright question and the altered-content question. The picture was redrawn with code; the music is original.
- Pin the comment "{kept:,} lines kept out of {thrown:,} thrown."
- After it posts, compare Viewed vs Swiped away and the retention graph with out/retention_map.md.
"""


def lint_post(text: str, meta: dict) -> list[str]:
    errors = []
    kept = int(meta["kept"])
    thrown = int(meta["thrown"])
    ratio = max(1, int(round(thrown / max(kept, 1))))
    allowed = {
        f"{kept:,}",
        str(kept),
        f"{thrown:,}",
        str(thrown),
        f"{ratio:,}",
        str(ratio),
        "1",
    }
    for title in TITLES:
        if len(title) > 60:
            errors.append(f"title longer than 60: {title}")
        if any(word in title.lower() for word in ("drop", "mouth", "face", "it's")):
            errors.append(f"title spoils the reveal: {title}")
    if not text.startswith("# Post kit"):
        errors.append("missing heading")
    if "#shorts #art #satisfying #generativeart #asmr" not in text:
        errors.append("hashtags must end the description and start with #shorts")
    if "Random lines are thrown and only the ones that bring the drawing closer are kept" not in text:
        errors.append("missing the how-it-works sentence")
    if "Music: original, made with code." not in text:
        errors.append("missing music credit")
    if meta.get("credit") and str(meta["credit"]) not in text:
        errors.append("missing the supplied credit")
    if meta.get("rights") == "unknown" and "may be copyrighted" not in text:
        errors.append("missing the rights warning")
    if "Viewed vs Swiped away" not in text:
        errors.append("missing the posting checklist")
    description = text.split("## Pinned comments")[0]
    for token in _numbers(description):
        if token not in allowed:
            errors.append(f"description number {token} has no claim")
    return errors


def retention_map(meta: dict) -> str:
    kept = int(meta["kept"])
    thrown = int(meta["thrown"])
    return f"""# Retention map

The film is 30.4 seconds at 150 bpm. Times are exact frame numbers divided by 60.

| time | frame | what the viewer should feel |
| --- | --- | --- |
| 0.0s | 0 | A line is already in the air. The only question is what it becomes. |
| 0.0–3.2s | 0–191 | Sixteen throws, one every eighth note. It looks like chaos, and the counters are still small. |
| 3.2–12.4s | 192–743 | The throws speed up. The page gets busier, but the subject is not obvious yet. |
| 12.4s | 744 | Everything stops for one silent beat. Lean in. |
| 12.8s | 768 | The drop. Thousands of lines land at once and the picture snaps into view. |
| 14.6–15.7s | 876–940 | The zoom. The mouth, or the main feature, is nothing but crossing straight lines. |
| 16.0–22.0s | 960–1320 | Back out. The counters race toward {kept:,} kept lines. |
| 24.0–27.2s | 1440–1631 | Then and now, or the original beside the lines when the picture's rights allow it. |
| 30.0s | 1800 | The lines fly off the page. |
| 30.4s | 1823 | Blank paper, the first line back in flight, the hook returned. It loops. |

{thrown:,} candidates were thrown to keep {kept:,} lines. The silent beat and the drop are the two moments most likely to decide whether someone stays.
"""
