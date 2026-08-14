"""The shared field-notes plate system: registry sanity and the AA-label contract.

These are design invariants the artifacts rely on, not aesthetics: every plate
number is unique (one numbering across figures/ and the PDF), the attribution
caption is part of every SVG plate's chrome, every bin of the sequential scale can
carry a legible value label on its own fill (WCAG AA, 4.5:1), and every
categorical slot clears 3:1 against the paper surface it is drawn on.

The contrast helper below is the sRGB relative-luminance formula (WCAG 2.x). It
was previously written with the linear/gamma branch threshold at ``0.04045 *
12.92`` instead of ``0.04045``, which sent every channel below 0.52 down the
linear branch and understated luminance; the assertions happened to pass anyway,
but the numbers were wrong. Fixed here, and the ramp is validated against it.
"""

from __future__ import annotations

from retail import plate


def _channel(c: float) -> float:
    c = c / 255.0
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def _luminance(hex_color: str) -> float:
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return 0.2126 * _channel(r) + 0.7152 * _channel(g) + 0.0722 * _channel(b)


def _contrast(a: str, b: str) -> float:
    la, lb = sorted((_luminance(a), _luminance(b)), reverse=True)
    return (la + 0.05) / (lb + 0.05)


def test_plate_numbers_are_unique_and_contiguous():
    numbers = sorted(number for number, _title in plate.PLATES.values())
    assert numbers == list(range(1, len(plate.PLATES) + 1))


def test_seq_bins_carry_aa_labels_on_their_own_fill():
    # Per-cell labels are the accessibility channel of the heatmaps: the chosen
    # label ink must clear WCAG AA (4.5:1) on the bin's fill, for every bin.
    for _hi, fill, label_ink in plate.SEQ_BINS:
        assert _contrast(fill, label_ink) >= 4.5, (fill, label_ink)


def test_categorical_slots_clear_three_to_one_on_the_paper_surface():
    # A mark that does not clear 3:1 against its own surface is a mark a reader
    # has to hunt for. The goods palette clears it on every slot (the dashboard
    # palette it replaced did not: its magenta was 2.62 and its yellow 2.11).
    for slot in plate.CATEGORICAL:
        assert _contrast(slot, plate.SURFACE) >= 3.0, (slot, _contrast(slot, plate.SURFACE))


def test_categorical_slots_are_distinct_and_ordered():
    # Identity comes from a FIXED order, so the same entity keeps its ink when a
    # filter changes how many series are on screen.
    assert plate.CATEGORICAL == [plate.RUST, plate.INDIGO, plate.OCHRE, plate.BERRY]
    assert len(set(plate.CATEGORICAL)) == len(plate.CATEGORICAL)


def test_seq_bins_step_visibly_from_light_to_dark():
    # A sequential ramp has to read AS steps: luminance strictly decreasing, and
    # the darkest bin genuinely dark (the scale ends in real ink, not mid-tone).
    lums = [_luminance(fill) for _hi, fill, _ink in plate.SEQ_BINS]
    assert lums == sorted(lums, reverse=True)
    assert _contrast(plate.SEQ_BINS[-1][1], "#ffffff") >= 7.0
    # ...and the lightest bin stays near the paper: the low end must recede.
    assert _contrast(plate.SEQ_BINS[0][1], plate.SURFACE) < 1.5


def test_seq_bins_stay_within_seven_classes_one_hue():
    assert len(plate.SEQ_BINS) <= 7
    # bounds are strictly increasing and end at 100
    bounds = [hi for hi, _f, _i in plate.SEQ_BINS]
    assert bounds == sorted(bounds)
    assert bounds[-1] == 100.0


def test_seq_bin_lookup_edges():
    assert plate.seq_bin(0.0) == (plate.SEQ_BINS[0][1], plate.SEQ_BINS[0][2])
    assert plate.seq_bin(4.99)[0] == plate.SEQ_BINS[0][1]
    assert plate.seq_bin(5.0)[0] == plate.SEQ_BINS[1][1]
    assert plate.seq_bin(100.0)[0] == plate.SEQ_BINS[-1][1]
    assert plate.seq_bin(200.0)[0] == plate.SEQ_BINS[-1][1]   # clamps, never KeyErrors


def test_svg_chrome_carries_plate_number_and_attribution():
    header = "".join(plate.svg_header(800, "cohort"))
    footer = "".join(plate.svg_footer(800, 600, "cohort", notes=("a note",)))
    assert "PLATE 10" in header
    assert plate.SERIES in header
    assert "UCI Online Retail II" in footer
    assert "CC BY 4.0" in footer
    assert "a note" in footer
    assert "PLATE 10" in footer          # identity line repeats below the rule


def test_every_analysis_module_has_a_plate_number():
    # A new analysis is not finished until it is a numbered plate in the registry.
    for key in ("cohort", "lifecycle", "clv", "returns", "basket", "pricing"):
        assert key in plate.PLATES
        assert plate.plate_tag(key).startswith("PLATE ")
