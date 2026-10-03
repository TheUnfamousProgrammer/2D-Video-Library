"""Official polygon names. Wikipedia's list, with the regular 4-gon called a square.

Names longer than 22 letters are left out. The picture only shows a name when that
count stays on screen for at least 12 frames, so the fast roll does not strobe.
"""

from __future__ import annotations

from fc_sat.polycircle_timeline import Timeline, counter_number

# Hold of one eighth note at 60 fps. Shorter than this, the word cannot be read.
MIN_HOLD = 12
MAX_LETTERS = 22

# https://en.wikipedia.org/wiki/List_of_polygons
# Teens use the usual English names. From 21 up, the kai form is the systematic one.
_NAMES: dict[int, str] = {
    4: "SQUARE",
    5: "PENTAGON",
    6: "HEXAGON",
    7: "HEPTAGON",
    8: "OCTAGON",
    9: "NONAGON",
    10: "DECAGON",
    11: "HENDECAGON",
    12: "DODECAGON",
    13: "TRIDECAGON",
    14: "TETRADECAGON",
    15: "PENTADECAGON",
    16: "HEXADECAGON",
    17: "HEPTADECAGON",
    18: "OCTADECAGON",
    19: "ENNEADECAGON",
    20: "ICOSAGON",
    21: "ICOSIKAIHENAGON",
    22: "ICOSIKAIDIGON",
    23: "ICOSIKAITRIGON",
    24: "ICOSIKAITETRAGON",
    25: "ICOSIKAIPENTAGON",
    26: "ICOSIKAIHEXAGON",
    27: "ICOSIKAIHEPTAGON",
    28: "ICOSIKAIOCTAGON",
    29: "ICOSIKAIENNEAGON",
    30: "TRIACONTAGON",
    40: "TETRACONTAGON",
    50: "PENTACONTAGON",
    60: "HEXACONTAGON",
    61: "HEXACONTAKAIHENAGON",
    70: "HEPTACONTAGON",
    80: "OCTACONTAGON",
    90: "ENNEACONTAGON",
    96: "ENNEACONTAKAIHEXAGON",
    100: "HECTOGON",
    1000: "CHILIAGON",
    10000: "MYRIAGON",
    1000000: "MEGAGON",
}

# A countably infinite polygon. Shown while the counter is the infinity sign.
APEIROGON = "APEIROGON"


def polygon_name(n_sides: int) -> str:
    name = _NAMES.get(int(n_sides), "")
    if len(name) > MAX_LETTERS:
        return ""
    return name


def shape_labels(timeline: Timeline) -> list[str]:
    """One label per frame. Empty when the count has no short name, or it changes too fast."""
    shown = [counter_number(timeline, frame) for frame in range(timeline.n_frames)]
    labels = [""] * timeline.n_frames
    frame = 0
    while frame < timeline.n_frames:
        end = frame + 1
        while end < timeline.n_frames and shown[end] == shown[frame]:
            end += 1
        if shown[frame] is None and frame >= 1632:
            name = APEIROGON
        elif shown[frame] is None:
            name = ""
        else:
            name = polygon_name(shown[frame])
        if name and end - frame >= MIN_HOLD:
            for index in range(frame, end):
                labels[index] = name
        frame = end
    return labels
