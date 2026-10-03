"""Titles, description, pinned comments, and the posting checklist. Digits come from claims."""

from __future__ import annotations

from pathlib import Path

from fc_sat.paperfold_claims import ClaimBook, lint_text, load_claims
from fc_sat.paperfold_copy import about_sun, credit, description, pinned_comment
from fc_sat.paperfold_timeline import build_timeline, retention_markdown

ROOT = Path(__file__).resolve().parents[1]

TITLES = (
    "How many folds to reach the Moon?",
    "Fold paper 42 times. Where does it end?",
    "42 folds to the Moon? Here's the catch",
    "Why you can't fold paper to the Moon",
    "The paper you'd need is as wide as the Milky Way",
)


def pinned_sun(book: ClaimBook | None = None) -> str:
    book = book or load_claims()
    return "How many folds to reach the Sun? Answer below."


def sun_folds(book: ClaimBook | None = None) -> int:
    book = book or load_claims()
    return int(book.get("sun_folds").value)


def postkit_markdown(book: ClaimBook | None = None) -> str:
    book = book or load_claims()
    titles = "\n".join(f"- {title}" for title in TITLES)
    return "\n".join(
        [
            "# Post kit",
            "",
            "Titles. Each stays inside the character limit, and the hook is in the opening words.",
            "",
            titles,
            "",
            "## Description",
            "",
            description(book),
            "",
            "## Pinned comment",
            "",
            "A) " + pinned_comment(book),
            "",
            f"B) {pinned_sun(book)} ({sun_folds(book)}, computed)",
            "",
            "## Thumbnail candidates",
            "",
            "- the drop, `out/stills/fold_30.png`",
            "- the tower beside the Moon, `out/stills/fold_42.png`",
            "- the light-year frame, the climax still",
            "",
            "## Assumptions",
            "",
            f"0.1 mm paper, single-direction folds, Gallivan's formula. At 30 folds the paper needed is {about_sun(book)} the Earth to Sun distance. The stack's width is not to scale. Credit on screen: {credit(book)}.",
            "",
            "## Posting checklist",
            "",
            "- Upload with #Shorts in the description (#shorts is already first).",
            "- Audience: not made for kids.",
            "- Copyright: the music is original. Altered content: the art is AI-generated and the music is made with code. Say yes to the altered-content question.",
            "- Pin comment A.",
            "- After it has views, read Viewed vs Swiped away and the retention graph against out/retention_map.md.",
            "",
            "Reply to a place or a distance with `python tools/fold_reply.py`.",
            "",
        ]
    )


def lint_post(book: ClaimBook | None = None) -> list[str]:
    book = book or load_claims()
    errors = []
    for title in TITLES:
        if len(title) > 60:
            errors.append(f"title too long: {title}")
        if len(title) >= 40 and title[:40].count(" ") == 0:
            errors.append(f"hook is not in the first 40 characters: {title}")
    text = postkit_markdown(book)
    errors.extend(
        lint_text(
            text,
            ["pass_moon", "sun_folds", "record_folds", "record_year", "thickness_mm", "au_times_30", "pass_space"],
            book,
            "postkit",
        )
    )
    first = description(book).splitlines()[0]
    if len(first) > 100:
        errors.append(f"description line 1 is {len(first)} characters")
    if "Art: AI-generated." not in text:
        errors.append("description is missing the art credit")
    if "#shorts" not in text.split("#")[0] and not text.strip().endswith("#science"):
        pass
    if "#shorts #math #satisfying #space #science" not in text:
        errors.append("hashtags missing")
    return errors


def write_post(out_dir: Path | None = None) -> Path:
    out_dir = out_dir or (ROOT / "out")
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "postkit.md"
    path.write_text(postkit_markdown())
    (out_dir / "retention_map.md").write_text(retention_markdown(build_timeline()))
    return path
