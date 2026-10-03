"""Titles, description, pinned comments, and the posting checklist."""

from __future__ import annotations

from fc_sat.circlesquare_claims import ClaimBook, lint_text, load_claims
from fc_sat.polycircle_format import format_count, format_gap

URGENCY = ("shocking", "unbelievable", "you won't believe", "hurry", "urgent", "don't miss", "gone wrong")


def titles(book: ClaimBook) -> list[tuple[str, list[str]]]:
    return [
        ("How many circles does it take to draw a square?", []),
        ("At 138 circles you can't tell. In math: never.", ["k_fool"]),
        ("Can spinning circles draw a square?", []),
        ("I kept adding circles until it became a square", []),
        ("Circles can't draw a perfect square. Here's why.", []),
    ]


def hashtags() -> str:
    return "#shorts #math #geometry #satisfying #fourier"


def description(book: ClaimBook) -> str:
    circles = int(book.get("k_fool").value)
    width = int(book.get("screen_w").value)
    side = int(book.get("side_px").value)
    line1 = "How many circles does it take to draw a square?"
    line2 = (
        f"At {circles} circles the gap is under half a pixel on a {width} px frame, "
        f"for a {side} px square."
    )
    line3 = book.get("corners").text
    line4 = "Part 1: How many sides before a square becomes a circle?"
    return "\n".join([line1, line2, line3, line4, "Music: original, made with code.", hashtags()])


def comments(book: ClaimBook) -> tuple[str, str, str]:
    answer = format_count(int(book.get("metre_k").value))
    question = "How many circles would a 10 metre square need to be within 1 millimetre? Answer below."
    reply = f"About {answer}. The gap scales with the square, and a 10 metre square has a 5 metre half-side."
    plain = book.get("smooth_curve").text
    return question, reply, plain


def posting_checklist() -> list[str]:
    return [
        "Upload with #Shorts in the description (it is already the first hashtag).",
        "Audience: No, it's not made for kids.",
        "Altered content / copyright: the music is original and there is no third-party footage.",
        "Pin comment A, the 10 metre question. Keep the prepared reply for when someone answers.",
        "In YouTube Studio, read Viewed vs Swiped away and the retention graph against out/retention_map.md.",
        "If swipe-away is high, post the next hook variant as a separate upload at least a day later.",
    ]


def thumbnail_notes() -> list[str]:
    return [
        "Drop, frame 768: the curve has just become a square.",
        "Corner, frame 954: 36x zoom, the corner is still round.",
        "Answer, frame 1488: the counter sits on 138.",
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
        if not any(token in head for token in ("circle", "square", "138")):
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
    tags = [tag for tag in lines[-1].split() if tag.startswith("#")]
    if not 3 <= len(tags) <= 5:
        errors.append(f"expected 3 to 5 hashtags, got {len(tags)}")
    claim_ids = ["k_fool", "screen_w", "side_px", "part", "corners"]
    errors.extend(lint_text(text, claim_ids, book, "description"))
    question, reply, plain = comments(book)
    errors.extend(lint_text(question, ["square_m", "tol_mm"], book, "comment"))
    errors.extend(lint_text(reply, ["metre_k", "half_m", "square_m"], book, "reply"))
    if plain != book.get("smooth_curve").text:
        errors.append("pinned comment B is not the definitional sentence")
    return errors


def render_postkit(book: ClaimBook | None = None) -> str:
    book = book or load_claims()
    question, reply, plain = comments(book)
    lines = [
        "# Post kit",
        "",
        "Upload the first title. The others are the next-day tests if swipe-away stays high.",
        "",
        "## Titles",
        "",
    ]
    for index, (title, _) in enumerate(titles(book), start=1):
        lines.append(f"{index}. {title}")
    lines += [
        "",
        "## Description",
        "",
        description(book),
        "",
        "## Pinned comments",
        "",
        f"A. {question}",
        f"   Reply: {reply}",
        f"B. {plain}",
        "",
        "## Thumbnail candidates",
        "",
    ]
    for note in thumbnail_notes():
        lines.append(f"- {note}")
    lines += ["", "## Posting checklist", ""]
    for item in posting_checklist():
        lines.append(f"- {item}")
    lines.append("")
    return "\n".join(lines)
