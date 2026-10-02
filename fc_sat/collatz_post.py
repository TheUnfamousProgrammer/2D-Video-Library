"""YouTube copy built from the claims book, then linted."""

from __future__ import annotations

from fc_sat.collatz_claims import ClaimBook, lint_text
from fc_sat.collatz_format import format_int

URGENCY = ("urgent", "shocking", "hurry", "you won't believe", "don't miss", "act now")


TITLES = [
    ("Pick any number. It always ends at 1.", ["ends_at_one"]),
    ("Why does every number end at 1?", ["ends_at_one"]),
    ("27 takes 111 steps to reach 1", ["start_27", "steps_27", "ends_at_one"]),
    ("Can you find a number that breaks this rule?", []),
    ("The simplest math problem nobody can solve", []),
    ("A 1-rule game mathematicians can't solve", ["ends_at_one"]),
    ("Every number ends at 1... or does it?", ["ends_at_one"]),
    ("The Collatz conjecture in 30 seconds", []),
]


def _title_errors(title: str, claims: list[str], book: ClaimBook) -> list[str]:
    errors = []
    if len(title) > 60:
        errors.append(f"title longer than 60: {title}")
    if title.startswith("#"):
        errors.append(f"title starts with a hashtag: {title}")
    letters = [ch for ch in title if ch.isalpha()]
    if letters and all(ch.isupper() for ch in letters):
        errors.append(f"title is all caps: {title}")
    lowered = title.lower()
    if any(word in lowered for word in URGENCY):
        errors.append(f"title uses fake urgency: {title}")
    errors.extend(lint_text(title, claims, book, f"title {title}"))
    return errors


def description(book: ClaimBook, reply: bool) -> str:
    verified = book.get("c_verified")
    steps = book.get("steps_27").value
    top = format_int(book.get("peak_27").value)
    bound = format_int(int(verified.value))
    line1 = "Pick any number. The Collatz conjecture says it always ends at 1."
    body = (
        f"Even numbers are halved. Odd numbers use 3n+1. "
        f"27 takes {steps} steps and peaks at {top}. "
        f"Computers have checked every start below 2075×2^60 ({bound}). "
        f"It is still unsolved. {verified.source}"
    )
    lines = [line1, "", body, ""]
    if reply:
        lines.append("Comment a number and I'll reply with its steps.")
        lines.append("")
    lines.append("Narration: AI voice. Verification record: D. Barina, J. Supercomput. 81, 810 (2025).")
    lines.append("")
    lines.append("#shorts #math #collatzconjecture #unsolved #mathshorts")
    return "\n".join(lines)


def description_errors(text: str, book: ClaimBook) -> list[str]:
    errors = []
    first = text.splitlines()[0]
    if len(first) > 100:
        errors.append(f"description line 1 is {len(first)} characters")
    for needle in ("Collatz conjecture", "3n+1", "unsolved"):
        if needle not in text:
            errors.append(f"description missing {needle}")
    tags = text.strip().splitlines()[-1].split()
    if not tags or tags[0] != "#shorts" or not 3 <= len(tags) <= 5:
        errors.append(f"hashtags must be 3 to 5 and start with #shorts, got {tags}")
    errors.extend(
        lint_text(
            text,
            ["ends_at_one", "rule_mul", "rule_addend", "start_27", "steps_27", "peak_27", "c_verified", "c_unproven"],
            book,
            "description",
        )
    )
    return errors


def comments(book: ClaimBook) -> str:
    steps_97 = book.get("steps_97").value
    return "\n".join(
        [
            "A) Comment a number. I'll reply with how many steps it takes.",
            (
                f"B) Fun fact: under 100 the longest is 97 with {steps_97} steps. "
                f"27 only needs {book.get('steps_27').value}. Can you find a number that beats {steps_97}?"
            ),
            f"C) {book.get('c_erdos').text} — {book.get('c_erdos').source}",
            "",
            "Pin A. Post B as a follow-up.",
        ]
    )


def checklist() -> str:
    return "\n".join(
        [
            "## First hour",
            "",
            "- Upload with #Shorts.",
            "- Audience setting is not made for kids.",
            "- Answer the copyright and altered-content questions. A plainly animated explainer with an AI narrator is probably not the realistic-altered-content toggle, but check the current Studio wording.",
            "- Pin comment A immediately.",
            "- Reply to number comments with tools/collatz_reply.py.",
            "- In Studio, read Viewed vs Swiped away against out/retention_map.md.",
            "- If swipe-away is above about 70% in the first 2 seconds, switch the hook variant.",
            "- Compare with this channel's own baseline, not a universal number.",
            "- Post the next hook variant as its own upload, at least a day later.",
        ]
    )


def render_postkit(book: ClaimBook, reply: bool = True) -> tuple[str, list[str]]:
    errors: list[str] = []
    rows = ["# Post kit", "", "## Titles", ""]
    for index, (title, claims) in enumerate(TITLES, start=1):
        rows.append(f"{index}. {title} ({len(title)} characters)")
        errors.extend(_title_errors(title, claims, book))
    text = description(book, reply)
    errors.extend(description_errors(text, book))
    rows.extend(["", "## Description", "", text, "", "## Pinned comments", "", comments(book), "", checklist(), ""])
    comment_errors = lint_text(
        comments(book),
        ["steps_97", "start_97", "steps_27", "start_27", "c_erdos"],
        book,
        "comments",
    )
    errors.extend(comment_errors)
    return "\n".join(rows), errors
