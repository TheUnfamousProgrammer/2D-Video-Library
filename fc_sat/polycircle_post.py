"""Titles, description, pinned comments, and the posting checklist. Built from claims."""

from __future__ import annotations

from fc_sat.polycircle_claims import ClaimBook, lint_text, load_claims
from fc_sat.polycircle_format import format_gap

URGENCY = ("shocking", "unbelievable", "you won't believe", "hurry", "urgent", "don't miss", "gone wrong")


def titles(book: ClaimBook) -> list[tuple[str, list[str]]]:
    rows = [
        ("How many sides until a square becomes a circle?", []),
        ("At 61 sides you can't tell. In math: never.", ["n_fool"]),
        ("Square to circle: how many sides?", []),
        ("I kept adding sides until it became a circle", []),
    ]
    if "finite_not_circle" in book.claims and "limit_circle" in book.claims:
        rows.append(("A circle is a polygon with infinite sides", ["finite_not_circle", "limit_circle"]))
    return rows


def description(book: ClaimBook) -> str:
    gap_label = format_gap(float(book.get("gap_61").value))
    n_sides = int(book.get("n_fool").value)
    width = int(book.get("screen_px").value)
    radius = int(book.get("radius_px").value)
    line1 = "How many sides before a square becomes a circle?"
    line2 = f"At {n_sides} sides the gap is {gap_label} px on a {width} px frame,"
    line3 = f"for a circle of radius {radius} px, using a half-pixel tolerance."
    return "\n".join(
        [
            line1,
            line2,
            line3,
            "Music: original, made with code.",
            "#shorts #math #geometry #satisfying #circle",
        ]
    )


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
        "Upload with #Shorts.",
        "Set the audience correctly.",
        "On the copyright and altered-content questions: the music is original and there is no third-party footage.",
        "Pin comment A, and keep the prepared reply for when someone answers.",
        "In YouTube Studio, read Viewed vs Swiped away and the retention graph against out/retention_map.md.",
        "If swipe-away is high, post the next hook variant as its own upload at least a day later.",
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
    errors.extend(lint_text(text, ["n_fool", "gap_61", "screen_px", "radius_px"], book, "description"))
    question, reply, plain = comments(book)
    errors.extend(lint_text(question, ["n_fool", "diameter_m", "tol_mm"], book, "comment"))
    errors.extend(lint_text(reply, ["diameter_m", "radius_m", "tol_mm", "n_metre"], book, "reply"))
    if any(ch.isdigit() for ch in plain):
        errors.extend(lint_text(plain, ["limit_circle"], book, "comment-b"))
    return errors


def render_postkit(book: ClaimBook | None = None) -> str:
    book = book or load_claims()
    question, reply, plain = comments(book)
    lines = ["# Post kit", "", "## Titles", ""]
    for title, _claims in titles(book):
        lines.append(f"- {title} ({len(title)} characters)")
    lines += ["", "## Description", "", description(book), "", "## Pinned comment A", "", question, "", "Prepared reply:", "", reply]
    lines += ["", "## Pinned comment B", "", plain.capitalize() if plain[:1].islower() else plain, ""]
    lines += ["## Thumbnails", "", "Stills in `out/thumbs/`: the drop, the zoom hold, and the 61 frame.", ""]
    lines += ["## Checklist", ""]
    lines += [f"- {item}" for item in posting_checklist()]
    lines.append("")
    return "\n".join(lines)
