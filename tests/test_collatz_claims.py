"""Claims lint and the Barina bound parser."""

from fc_sat.collatz_bound import latest_bound
from fc_sat.collatz_claims import evaluate, lint_text, load_claims, number_tokens
from fc_sat.collatz_config import contrast_ratio, load_yaml, validate_palette
from fc_sat.collatz_script import estimate_characters, lint_script, load_script
from fc_sat.easing import ease_in_cubic, ease_in_out_cubic, ease_out_back, ease_out_cubic, monotonic


HTML = """
<p>The convergence of all numbers below 2075 &times; 2<sup>60</sup> (&asymp; 2<sup>71.02</sup>) has been verified.</p>
<p>Progress towards verifying all numbers below 2076 &times; 2<sup>60</sup></p>
<p>Lowest incomplete: 17.8734 % (work unit 2175982616 &times; 2<sup>40</sup>, all work units below are verified)</p>
<ul>
<li>the convergence of all numbers below 1.5 &times; 2<sup>70</sup> is verified
<li>the convergence of all numbers below 2<sup>71</sup> is verified
</ul>
"""


def test_bound_parser_ignores_the_unfinished_target():
    assert latest_bound(HTML) == 2075 * 2**60


def test_computed_claims_match():
    book = load_claims()
    results, note = evaluate(book, html=HTML, fetch=False)
    assert "unchanged" in note
    failed = [item for item in results if not item[1]]
    assert failed == []


def test_digit_without_a_claim_fails():
    book = load_claims()
    errors = lint_text("it takes 42 steps", [], book, "sample")
    assert errors


def test_whitelist_and_claimed_phrase():
    book = load_claims()
    assert lint_text("in 30 seconds", [], book, "title") == []
    assert lint_text("a hundred and eleven steps", ["steps_27"], book, "vo") == []


def test_script_passes():
    book = load_claims()
    lines = load_script()
    assert lint_script(lines, book) == []
    assert estimate_characters(lines) > 0


def test_contrast_floors():
    spec = load_yaml("collatz_timeline.yaml")
    validate_palette(spec)
    assert contrast_ratio(spec["palette"]["text"], spec["palette"]["bg"]) >= 7
    assert contrast_ratio(spec["palette"]["muted"], spec["palette"]["bg"]) >= 4.5


def test_easing():
    cubic = [ease_out_cubic(i / 40) for i in range(41)]
    smooth = [ease_in_out_cubic(i / 40) for i in range(41)]
    entered = [ease_in_cubic(i / 40) for i in range(41)]
    assert monotonic(cubic) and monotonic(smooth) and monotonic(entered)
    assert cubic[0] == 0 and cubic[-1] == 1
    back = [ease_out_back(i / 80) for i in range(81)]
    assert back[0] == 0 and abs(back[-1] - 1) < 1e-9
    assert 1.0 < max(back) <= 1.08


def test_number_tokens_keep_phrases():
    assert "twenty-seven" in [tok.lower() for tok in number_tokens("try twenty-seven")]
    assert "9,232" in number_tokens("peak 9,232")
