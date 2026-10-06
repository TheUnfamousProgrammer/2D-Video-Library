"""Titles, description, pinned comments, and the posting checklist for the scale short."""

from __future__ import annotations

from fc_sat.scale_claims import ClaimBook, lint_text, load_claims

URGENCY = ("shocking", "unbelievable", "you won't believe", "hurry", "urgent", "don't miss", "gone wrong")
HOOK_WORDS = ("smallest", "biggest", "proton", "universe", "you are here", "scale")


def titles(book: ClaimBook) -> list[tuple[str, list[str]]]:
    return [
        ("The smallest thing to the biggest thing in the universe", []),
        ("From a proton to the whole universe, to scale", []),
        ("You are here: the scale of everything", []),
        ("How big is the universe compared to you?", []),
    ]


def hashtags() -> str:
    return "#shorts #space #science #universe #scale"


def description(book: ClaimBook) -> str:
    proton = book.get("s01_proton")
    universe = book.get("s28_observable_universe")
    lines = [
        "From a proton to the observable universe, every object to scale, side by side.",
        f"The counter is each size in plain metres, from {_display(book, proton.id)} to {_display(book, universe.id)} across.",
        "Sizes are typical published values. Sources are in the pinned comment.",
        "Music: original, made with code.",
        hashtags(),
    ]
    return "\n".join(lines)


def _display(book: ClaimBook, claim_id: str) -> str:
    import yaml

    return str(yaml.safe_load(book.path.read_text())["claims"][claim_id]["display"]).lower()


def comments(book: ClaimBook) -> tuple[str, str]:
    question = "Which jump surprised you most? For me it's the Sun next to UY Scuti."
    sources = "Sources for every size: " + "; ".join(
        f"{claim_id.split('_', 1)[1].replace('_', ' ')}: {str(claim.source).split(';')[0]}"
        for claim_id, claim in book.claims.items()
    )
    return question, sources


def posting_checklist() -> list[str]:
    return [
        "Upload with #Shorts in the description (it is already the first hashtag).",
        "Audience: No, it's not made for kids, so comments stay open.",
        "Altered content / copyright: the music is original and made with code; the object art is our own generated paper-cut set.",
        "Pin comment A. Keep the sources comment ready for the first 'that's wrong' reply.",
        "In YouTube Studio, read Viewed vs Swiped away and the retention graph against out/scale_retention_map.md.",
        "If swipe-away is high, post hook B (HOW SMALL AND HOW BIG DOES IT GET?) as its own upload at least a day later.",
    ]


def thumbnail_notes() -> list[str]:
    return [
        "Hook, frame 0: the proton and SMALLEST THING TO THE BIGGEST IN THE UNIVERSE.",
        "You, frame 790: YOU ARE HERE with the cat and the phone at your feet.",
        "Earth, frame 1220: Earth beside the Moon.",
        "Universe, frame 1650: the observable universe and its two-line counter.",
    ]


def lint_post(book: ClaimBook) -> list[str]:
    errors = []
    every = list(book.claims)
    for title, claim_ids in titles(book):
        if len(title) > 60:
            errors.append(f"title too long ({len(title)}): {title}")
        if title.isupper():
            errors.append(f"title is all caps: {title}")
        lowered = title.lower()
        if any(word in lowered for word in URGENCY):
            errors.append(f"title uses fake urgency: {title}")
        if not any(word in title[:40].lower() for word in HOOK_WORDS):
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
    errors.extend(lint_text(text, every, book, "description"))
    question, sources = comments(book)
    errors.extend(lint_text(question, [], book, "comment A"))
    return errors


def render_postkit(book: ClaimBook | None = None) -> str:
    book = book or load_claims()
    question, sources = comments(book)
    lines = ["# Post kit", "", "Upload the first title. The others are next-day tests if swipe-away stays high.", "", "## Titles", ""]
    for index, (title, _) in enumerate(titles(book), start=1):
        lines.append(f"{index}. {title}")
    lines += ["", "## Description", "", description(book), "", "## Pinned comments", "", f"A. {question}", f"B. {sources}", "", "## Thumbnail candidates", ""]
    lines += [f"- {note}" for note in thumbnail_notes()]
    lines += ["", "## Posting checklist", ""]
    lines += [f"- {item}" for item in posting_checklist()]
    lines.append("")
    return "\n".join(lines)
