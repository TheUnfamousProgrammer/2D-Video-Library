"""Titles, description, pinned comments, and the posting checklist."""

from __future__ import annotations

from fc_sat.coinspin_claims import ClaimBook, lint_text, load_claims

URGENCY = ("shocking", "unbelievable", "you won't believe", "hurry", "urgent", "don't miss", "gone wrong")
HOOK_WORDS = ("coin", "spin", "sat")


def titles(book: ClaimBook) -> list[tuple[str, list[str]]]:
    return [
        ("Roll a coin around a coin: how many spins?", []),
        ("Same size coins. One trip around. 2 spins?", ["equal_spins"]),
        ("The coin question the 1982 SAT got wrong", ["sat_year"]),
        ("Why this coin spins twice, not once", []),
    ]


def hashtags() -> str:
    return "#shorts #math #puzzle #geometry #satisfying"


def description(book: ClaimBook) -> str:
    equal = int(book.get("equal_spins").value)
    rolled = int(book.get("road_spins").value)
    trip = int(book.get("trip_spins").value)
    year = int(book.get("sat_year").value)
    answer = int(book.get("sat_spins").value)
    intended = int(book.get("sat_intended").value)
    lines = [
        "Roll a coin once around a coin of the same size. How many times does it spin?",
        f"{equal}. Rolling along the other coin's edge gives {rolled}, and the trip around gives {trip} more, even with no rolling at all.",
        f"The {year} SAT asked this with a coin 3x wider. The test's answer was {intended}; the real answer is {answer}.",
        "1 spin = the face coming back upright.",
        "Music: original, made with code.",
        hashtags(),
    ]
    return "\n".join(lines)


def comments(book: ClaimBook) -> tuple[str, str, str, str]:
    ratio = int(book.get("ten_ratio").value)
    spins = int(book.get("ten_spins").value)
    inside = int(book.get("inside_spins").value)
    question = f"A coin rolls once around a coin {ratio}x wider. How many spins? Answer below."
    reply = f"{spins}. Rolling along the rim gives {ratio}, and the trip around gives 1 more."
    bonus = f"Roll it around the inside of a ring 3x wider and you get {inside} spins. Inside, the trip takes one away."
    earth = (
        "Earth does this too: about 365.24 days a year, but about 366.24 turns against the stars. "
        "The trip around the Sun adds 1. Want that one next?"
    )
    return question, reply, bonus, earth


def posting_checklist() -> list[str]:
    return [
        "Upload with #Shorts in the description (it is already the first hashtag).",
        "Audience: No, it's not made for kids, so the pinned comment can collect answers.",
        "Altered content / copyright: the music is original, the drawing is made with code, there is no third-party footage.",
        "Pin comment A, the 10x question. Keep the prepared reply. Post comment B under the first wrong answer and comment C when someone asks for more.",
        "In YouTube Studio, read Viewed vs Swiped away and the retention graph against out/coinspin_retention_map.md.",
        "If swipe-away is high, post hook D (EVEN THE SAT GOT THIS WRONG.) as its own upload at least a day later.",
    ]


def thumbnail_notes() -> list[str]:
    return [
        "Hook, frame 0: the face coin on top, the question, SPINS ?.",
        "Halfway, frame 300: the coin at the bottom, upright, ALREADY 1 SPIN!",
        "Drop, frame 790: 2 SPINS! with two upright faces.",
        "Proof, frame 1260: 1 + 1 = 2 with the bolted rod.",
        "SAT, frame 1640: 4 upright faces in a cross and the struck-out 3.",
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
    claim_ids = ["unit", "equal_spins", "road_spins", "trip_spins", "sat_year", "sat_ratio", "sat_intended", "sat_spins"]
    errors.extend(lint_text(text, claim_ids, book, "description"))
    question, reply, bonus, earth = comments(book)
    errors.extend(lint_text(question, ["ten_ratio"], book, "comment A"))
    errors.extend(lint_text(reply, ["ten_ratio", "ten_spins", "trip_spins"], book, "reply"))
    errors.extend(lint_text(bonus, ["inside_spins", "sat_ratio"], book, "comment B"))
    errors.extend(lint_text(earth, ["tropical_year", "earth_turns", "trip_spins"], book, "comment C"))
    return errors


def render_postkit(book: ClaimBook | None = None) -> str:
    book = book or load_claims()
    question, reply, bonus, earth = comments(book)
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
        f"B. {bonus}",
        f"C. {earth}",
        f"D. {book.get('trip_rule').text}",
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
