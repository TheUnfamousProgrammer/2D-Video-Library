"""Tower height and Gallivan's single-direction paper length.

h(n) = t * 2^n. Each fold doubles the stack.
L(n) = (pi * t / 6) * (2^n + 4) * (2^n - 1), Britney Gallivan, 2001,
for folds all in one direction. t and L are in the same units.
"""

from __future__ import annotations

import math

# 0.1 mm printer paper. An assumption, shown on screen. Not a measured sheet.
THICKNESS_M = 1.0e-4

# NASA Science, "The Moon is an average of 238,855 miles (384,400 kilometers) away."
MOON_M = 384_400e3

# 8,848.86 m, China-Nepal joint announcement, 8 December 2020, rounded to the nearest metre.
EVEREST_M = 8849.0

# CTBUH architectural height, Burj Khalifa.
BURJ_M = 828.0

# FAI Karman line.
KARMAN_M = 100_000.0

# WGS84 equatorial circumference, 2 * pi * 6,378.137 km, rounded to the kilometre.
EARTH_CIRC_M = 40_075e3

# IAU 2012 Resolution B2. Exactly 149,597,870,700 m.
AU_M = 149_597_870_700.0

# IAU light-year 9.4607304725808e15 m, rounded to 5 significant figures as specified.
LIGHT_YEAR_M = 9.4607e15

# NASA: the Milky Way is about 100,000 light-years across.
MILKY_WAY_LY = 100_000.0

# Assumption, shown as a comparison, not a measurement of a particular person.
PERSON_M = 1.8

# A desk mug and a phone, assumptions used as scale references.
MUG_M = 0.095
PHONE_M = 0.15

# NASA: the ISS orbits at about 408 km.
ISS_M = 408_000.0

# NASA planetary fact sheet, Earth volumetric mean radius.
EARTH_RADIUS_M = 6_371e3

# Guinness World Records: Britney Gallivan, 12 folds, 27 January 2002, 4,000 ft (1,219 m).
RECORD_FOLDS = 12
RECORD_LENGTH_M = 1219.0

# Popular comoving diameter, about 93 billion light-years. The reply tool stops here.
OBSERVABLE_LY = 93e9

TOWER_PX = 700.0
AXIS_X0 = 130.0
AXIS_WIDTH = 820.0
AXIS_LO_M = 0.1
AXIS_HI_M = 1.0e21

MILESTONES = (
    ("mug", MUG_M, "A MUG"),
    ("phone", PHONE_M, "A PHONE"),
    ("person", PERSON_M, "A PERSON"),
    ("burj", BURJ_M, "BURJ KHALIFA"),
    ("everest", EVEREST_M, "EVEREST"),
    ("karman", KARMAN_M, "SPACE"),
    ("iss", ISS_M, "THE ISS"),
    ("moon", MOON_M, "THE MOON"),
)


def height_m(n: int) -> float:
    """Stack thickness after n folds, in metres. h(0) is one sheet."""
    if n < 0:
        raise ValueError(f"fold count is negative: {n}")
    return THICKNESS_M * float(2 ** int(n))


def length_m(n: int) -> float:
    """Minimum strip length for n single-direction folds, in metres."""
    if n < 0:
        raise ValueError(f"fold count is negative: {n}")
    layers = float(2 ** int(n))
    return (math.pi * THICKNESS_M / 6.0) * (layers + 4.0) * (layers - 1.0)


def first_fold_past(distance_m: float) -> int:
    """Smallest n with h(n) >= distance. n = 0 is the unfolded sheet."""
    if distance_m <= 0:
        raise ValueError(f"distance must be positive, got {distance_m}")
    n = 0
    limit = OBSERVABLE_LY * LIGHT_YEAR_M
    while height_m(n) < distance_m:
        n += 1
        if height_m(n) > limit and distance_m > limit:
            break
        if n > 120:
            break
    return n


def passes_observable(distance_m: float) -> bool:
    return distance_m > OBSERVABLE_LY * LIGHT_YEAR_M


def milestone_fold(name: str) -> int:
    for key, distance, _label in MILESTONES:
        if key == name:
            return first_fold_past(distance)
    raise KeyError(name)


def milestone_folds() -> dict[str, int]:
    return {key: first_fold_past(distance) for key, distance, _label in MILESTONES}


def screen_scale(height: float) -> float:
    """Pixels per metre so the stack stays TOWER_PX tall."""
    return TOWER_PX / float(height)


def tower_px(height: float) -> float:
    return screen_scale(height) * float(height)


def milestone_visible(item_m: float, tower_m: float) -> bool:
    """Draw a marker only when it sits between 0.002 and 4 times the tower."""
    if tower_m <= 0 or item_m <= 0:
        return False
    ratio = item_m / tower_m
    return 0.002 <= ratio <= 4.0


def log_axis_x(meters: float, x0: float = AXIS_X0, width: float = AXIS_WIDTH) -> float:
    """Horizontal position of a length on the twist axis. 0.1 m at the left, 1e21 m at the right."""
    if meters <= 0:
        raise ValueError(f"axis length must be positive, got {meters}")
    lo = math.log10(AXIS_LO_M)
    hi = math.log10(AXIS_HI_M)
    u = (math.log10(meters) - lo) / (hi - lo)
    return x0 + u * width


def sigfigs(value: float, digits: int) -> float:
    """Round to a number of significant digits. 403.5 with 2 digits is 400."""
    if value == 0:
        return 0.0
    if digits < 1:
        raise ValueError("digits must be at least 1")
    sign = 1.0 if value > 0 else -1.0
    magnitude = abs(float(value))
    exp = math.floor(math.log10(magnitude))
    scale = 10.0 ** (exp - digits + 1)
    return sign * round(magnitude / scale) * scale


def format_sig(value: float, digits: int) -> str:
    """Significant-digit number with thousands separators and no scientific notation."""
    rounded = sigfigs(value, digits)
    if rounded == 0:
        return "0"
    exp = math.floor(math.log10(abs(rounded)))
    if exp >= digits - 1:
        return f"{int(round(rounded)):,}"
    decimals = digits - 1 - exp
    return f"{rounded:,.{decimals}f}"


def au_multiple(n: int) -> float:
    return length_m(n) / AU_M


def light_years(n: int) -> float:
    return length_m(n) / LIGHT_YEAR_M


def earth_multiple(n: int) -> float:
    return length_m(n) / EARTH_CIRC_M


def height_km(n: int) -> float:
    return height_m(n) / 1000.0


def length_km(n: int) -> float:
    return length_m(n) / 1000.0
