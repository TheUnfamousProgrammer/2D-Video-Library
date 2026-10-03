"""Titles, description, pinned comments, and the posting checklist. Built from claims."""

from __future__ import annotations

from fc_sat.polycircle_claims import ClaimBook, lint_text, load_claims
from fc_sat.polycircle_format import format_gap

URGENCY = ("shocking", "unbelievable", "you won't believe", "hurry", "urgent", "don't miss", "gone wrong")


def titles(book: ClaimBook) -> list[tuple[str, list[str]]]:
    """Upload the first. The rest are the next-day tests if swipe-away stays high.

    The first title is the phrase YouTube autocomplete actually completes.
    It fits in the ~40 characters a Shorts title shows before it is cut.
    """
    rows = [
        ("How many sides does a circle have?", []),
        ("How many sides before a square becomes a circle?", []),
        ("At 61 sides you can't tell. In math: never.", ["n_fool"]),
        ("I kept adding sides. It never became one.", []),
    ]
    if "finite_not_circle" in book.claims and "limit_circle" in book.claims:
        rows.append(("A circle is a polygon with infinite sides", ["finite_not_circle", "limit_circle"]))
    return rows


def hashtags() -> str:
    # Five, which is the band YouTube's own hashtag page treats as useful.
    # The first three are the ones most likely to show above the title.
    return "#shorts #math #satisfying #geometry #circle"


def description(book: ClaimBook) -> str:
    gap_label = format_gap(float(book.get("gap_61").value))
    n_sides = int(book.get("n_fool").value)
    width = int(book.get("screen_px").value)
    radius = int(book.get("radius_px").value)
    line1 = f"How many sides does a circle have? At {n_sides}, your eye gives up."
    line2 = f"The gap is {gap_label} px on a {width} px frame, for a circle of radius {radius} px and a half-pixel tolerance."
    line3 = "How many sides until a square becomes one? It looks like a circle. Zoom in and it is still a polygon. In math, it never arrives."
    return "\n".join(
        [
            line1,
            line2,
            line3,
            "Music: original, made with code.",
            hashtags(),
        ]
    )


def tags() -> list[str]:
    """Phrases from YouTube autocomplete on 2026-10-03, plus the misspelling people type.

    YouTube's own help says tags mostly fix misspellings. The first tag is the search.
    """
    return [
        "how many sides does a circle have",
        "how many sides does a circle has",
        "how many sides in a circle",
        "how many sides before a circle",
        "how many sides added until it's a circle",
        "circle has infinite sides",
        "how many corners does a circle have",
        "circle has no sides",
    ]


def comments(book: ClaimBook) -> tuple[str, str, str]:
    n_sides = int(book.get("n_fool").value)
    diameter = int(book.get("diameter_m").value)
    tol = int(book.get("tol_mm").value)
    answer = int(book.get("n_metre").value)
    radius = int(book.get("radius_m").value)
    question = (
        f"At this size, {n_sides} sides is where the gap drops under half a pixel. "
        f"On a bigger screen you'd need more. How many sides would a {diameter} metre circle "
        f"need before the gap is under {tol} millimetre?"
    )
    reply = (
        f"A {diameter} metre circle has radius {radius} metres. "
        f"The gap drops under {tol} millimetre at {answer} sides."
    )
    plain = book.get("limit_circle").text
    return question, reply, plain


def posting_checklist() -> list[str]:
    return [
        "Paste the upload title as written. Do not add #Shorts to the title. Hashtags in the title spend the characters a phone actually shows.",
        "Paste the description. The first line is the only one most people see before More.",
        "Paste the tags into the tag box, in this order. YouTube says tags mostly correct misspellings, so the list stays short.",
        "Audience: No, it's not made for kids. A kids setting turns comments off, and the pinned question is how this gets replies.",
        "Altered content: No. The music is original, the frames are drawn, and there is no third-party footage.",
        "Pin comment A. Keep the prepared reply for when someone answers the 10 metre question.",
        "In YouTube Studio, read Viewed vs Swiped away and the retention graph against out/retention_map.md.",
        "If swipe-away is high, post the next title as its own upload at least a day later. Change the title, not the picture, on the first retest.",
    ]


def lint_post(book: ClaimBook) -> list[str]:
    errors = []
    for title, claim_ids in titles(book):
        if len(title) > 60:
            errors.append(f"title too long ({len(title)}): {title}")
        if title.isupper():
            errors.append(f"title is all caps: {title}")
        lowered = title.lower()
        if any(word in lowered for word in URGENCY):
            errors.append(f"title uses fake urgency: {title}")
        head = title[:40].lower()
        if not any(token in head for token in ("side", "square", "circle", "61", "never", "polygon")):
            errors.append(f"title hides the hook past 40 characters: {title}")
        errors.extend(lint_text(title, claim_ids, book, "title"))
    text = description(book)
    lines = text.splitlines()
    if len(lines[0]) > 100:
        errors.append("description line 1 is over 100 characters")
    if "Music: original, made with code." not in lines:
        errors.append("description is missing the music credit")
    if not lines[-1].startswith("#shorts"):
        errors.append("hashtags must start with #shorts")
    tag_count = len(lines[-1].split())
    if not 3 <= tag_count <= 5:
        errors.append(f"use 3 to 5 hashtags, found {tag_count}")
    if len(titles(book)[0][0]) > 40:
        errors.append("the upload title is past 40 characters, so the hook is cut on a phone")
    errors.extend(lint_text(text, ["n_fool", "gap_61", "screen_px", "radius_px"], book, "description"))
    errors.extend(lint_text(", ".join(tags()), [], book, "tags"))
    question, reply, plain = comments(book)
    errors.extend(lint_text(question, ["n_fool", "diameter_m", "tol_mm"], book, "comment"))
    errors.extend(lint_text(reply, ["diameter_m", "radius_m", "tol_mm", "n_metre"], book, "reply"))
    if any(ch.isdigit() for ch in plain):
        errors.extend(lint_text(plain, ["limit_circle"], book, "comment-b"))
    return errors


def render_postkit(book: ClaimBook | None = None) -> str:
    book = book or load_claims()
    question, reply, plain = comments(book)
    chosen = titles(book)
    lines = [
        "# Post kit",
        "",
        f"Paste these. The upload title is the phrase YouTube autocomplete completes for \"how many sides does a circle\". It is {len(chosen[0][0])} characters, so a phone shows all of it. The alternates are for a later upload if people swipe away.",
        "",
        "## Upload title",
        "",
        chosen[0][0],
        "",
        "## Later titles",
        "",
    ]
    for title, _claims in chosen[1:]:
        lines.append(f"- {title} ({len(title)} characters)")
    lines += [
        "",
        "## Description",
        "",
        description(book),
        "",
        "## Tags",
        "",
        ", ".join(tags()),
        "",
        "## Pinned comment A",
        "",
        question,
        "",
        "Prepared reply:",
        "",
        reply,
    ]
    lines += ["", "## Pinned comment B", "", plain.capitalize() if plain[:1].islower() else plain, ""]
    lines += ["## Thumbnails", "", "Stills in `out/thumbs/`: the drop, the zoom hold, and the 61 frame.", ""]
    lines += ["## Checklist", ""]
    lines += [f"- {item}" for item in posting_checklist()]
    lines.append("")
    return "\n".join(lines)
