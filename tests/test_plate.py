"""The shared field-notes plate system: registry sanity and the AA-label contract.

These are design invariants the artifacts rely on, not aesthetics: every plate
number is unique (one numbering across figures/ and the PDF), the attribution
caption is part of every SVG plate's chrome, and every bin of the sequential
scale can carry a legible value label on its own fill (WCAG AA, 4.5:1).
"""

from __future__ import annotations

from retail import plate


def _channel(c: float) -> float:
    c = c / 255.0
    return c / 12.92 if c <= 0.04045 * 12.92 else ((c + 0.055) / 1.055) ** 2.4


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
