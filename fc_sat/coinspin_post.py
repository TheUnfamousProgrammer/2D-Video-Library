"""Titles, description, pinned comments, and the posting checklist."""

from __future__ import annotations

from fc_sat.coinspin_claims import ClaimBook, lint_text, load_claims
from fc_sat.coinspin_math import format_hms

URGENCY = ("shocking", "unbelievable", "you won't believe", "hurry", "urgent", "don't miss", "gone wrong")
HOOK_WORDS = ("coin", "spin", "sat", "earth", "366")


def titles(book: ClaimBook) -> list[tuple[str, list[str]]]:
    return [
        ("How many times does the coin spin?", []),
        ("The coin question the 1982 SAT got wrong", ["sat_year"]),
        ("Why does Earth spin 366 times a year?", ["sidereal_whole"]),
        ("Roll a coin around a coin. Count the spins.", []),
        ("Same size coins. One lap. 2 spins?", ["equal_spins"]),
    ]


def hashtags() -> str:
    return "#shorts #math #puzzle #geometry #space"


def description(book: ClaimBook) -> str:
    equal = int(book.get("equal_spins").value)
    year = int(book.get("sat_year").value)
    answer = int(book.get("sat_spins").value)
    days = float(book.get("tropical_year").value)
    spins = float(book.get("sidereal_year").value)
    hours, minutes, seconds = (int(part[:-1]) for part in format_hms(float(book.get("sidereal_day_s").value)).split())
    day = f"{hours} h {minutes} m {seconds} s"
    lines = [
        "How many times does a coin spin when it rolls once around another coin?",
        f"Same size coins: {equal} spins. 1 from rolling along the rim, and 1 more from the trip around.",
        f"The {year} SAT asked this with a coin one third the size. The answer is {answer}, and it was not one of the choices.",
        f"Earth does it too: about {days:.2f} days a year, but about {spins:.2f} spins against the stars, one every {day}.",
        book.get("not_to_scale").text,
        "Music: original, made with code.",
        hashtags(),
    ]
    return "\n".join(lines)


def comments(book: ClaimBook) -> tuple[str, str, str]:
    ratio = int(book.get("ten_ratio").value)
    spins = int(book.get("ten_spins").value)
    inside = int(book.get("inside_spins").value)
    question = f"A coin rolls once around a coin {ratio}x wider. How many spins? Answer below."
    reply = f"{spins}. Rolling along the rim gives {ratio}, and the trip around gives 1 more."
    bonus = f"Roll it around the inside of a ring 3x wider and you get {inside} spins. Inside, the trip takes one away."
    return question, reply, bonus


def posting_checklist() -> list[str]:
    return [
        "Upload with #Shorts in the description (it is already the first hashtag).",
        "Audience: No, it's not made for kids, so the pinned comment can collect answers.",
        "Altered content / copyright: the music is original, the drawing is made with code, there is no third-party footage.",
        "Pin comment A, the 10x question. Keep the prepared reply for when someone answers. Post comment B as a reply to the first wrong answer.",
        "In YouTube Studio, read Viewed vs Swiped away and the retention graph against out/coinspin_retention_map.md.",
        "If swipe-away is high, post hook D (THE SAT GOT THIS WRONG.) as its own upload at least a day later.",
    ]


def thumbnail_notes() -> list[str]:
    return [
        "Hook, frame 0: two coins, the question, SPINS 0.",
        "Tension, frame 700: the counter says 3, the test's answer is highlighted, and the coin is not home yet.",
        "Drop, frame 780: 4, with every choice struck out.",
        "Earth, frame 1500: DAYS 365, SPINS 366.",
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
    claim_ids = ["equal_spins", "plus_one", "sat_year", "sat_spins", "sat_ratio", "tropical_year", "sidereal_year", "sidereal_day_s"]
    errors.extend(lint_text(text, claim_ids, book, "description"))
    question, reply, bonus = comments(book)
    errors.extend(lint_text(question, ["ten_ratio"], book, "comment"))
    errors.extend(lint_text(reply, ["ten_ratio", "ten_spins", "plus_one"], book, "reply"))
    errors.extend(lint_text(bonus, ["inside_spins", "sat_ratio"], book, "comment B"))
    return errors


def render_postkit(book: ClaimBook | None = None) -> str:
    book = book or load_claims()
    question, reply, bonus = comments(book)
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
        f"C. {book.get('trip_rule').text}",
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
