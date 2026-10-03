"""Sentences that leave the timeline. Digits come from claims.

On screen the record credit is only "Britney Gallivan, 2002".
The description and the pinned comment say "about 400x", from L(30) / 1 AU
rounded to two significant figures.
"""

from __future__ import annotations

from fc_sat.paperfold_claims import ClaimBook, lint_text, load_claims


def credit(book: ClaimBook | None = None) -> str:
    book = book or load_claims()
    return f"Britney Gallivan, {int(book.get('record_year').value)}"


def about_sun(book: ClaimBook | None = None) -> str:
    book = book or load_claims()
    return f"about {int(book.get('au_times_30').value)}x"


def description(book: ClaimBook | None = None) -> str:
    book = book or load_claims()
    folds = int(book.get("pass_moon").value)
    thick = book.get("thickness_mm").value
    record = int(book.get("record_folds").value)
    line1 = "How many folds does it take to reach the Moon?"
    line2 = (
        f"In theory, {folds} folds of {thick} mm paper, each in a single direction, "
        f"using Gallivan's formula. At 30 folds the paper needed is {about_sun(book)} "
        "the distance from the Earth to the Sun. The real record is "
        f"{record} folds."
    )
    return "\n".join(
        [
            line1,
            line2,
            "Music: original, made with code.",
            "#shorts #math #satisfying #space #science",
        ]
    )


def pinned_comment(book: ClaimBook | None = None) -> str:
    book = book or load_claims()
    folds = int(book.get("pass_moon").value)
    record = int(book.get("record_folds").value)
    return (
        f"In theory, {folds} folds. In real life, the record is {record}. "
        f"The paper for 30 folds is {about_sun(book)} an Earth-to-Sun trip. "
        "How many folds do you think a normal sheet can take?"
    )


def lint_copy(book: ClaimBook | None = None) -> list[str]:
    book = book or load_claims()
    errors = []
    errors.extend(lint_text(description(book), ["pass_moon", "thickness_mm", "au_times_30", "record_folds", "pass_space"], book, "description"))
    errors.extend(lint_text(pinned_comment(book), ["pass_moon", "record_folds", "au_times_30", "pass_space"], book, "pinned"))
    errors.extend(lint_text(credit(book), ["record_year"], book, "credit"))
    if "about 400x" not in description(book) or "about 400x" not in pinned_comment(book):
        errors.append('description and pinned comment must say "about 400x"')
    if credit(book) != "Britney Gallivan, 2002":
        errors.append(f"on-screen credit must be 'Britney Gallivan, 2002', got {credit(book)!r}")
    return errors
