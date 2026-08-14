"""The shared "field notes" plate system for every visual artifact in this repo.

One design language across the committed figures (PNG + hand-drawn SVG) and the
PDF pages, so the whole output reads as a single set of numbered plates from one
field notebook on real data:

* **Numbered plates.** Every artifact carries a plate number from the single
  registry below. The PDF shows a subset of the plates in ascending order; the
  rest of the set lives in ``figures/``. An artifact that appears both as a
  committed figure and as a PDF page carries the *same* number in both places.
* **Hairline chrome.** One hairline rule under the kicker, one above the footer;
  grids and axes are recessive hairlines; marks are thin; the data is the only
  loud thing on the page.
* **Attribution as a designed caption.** "UCI Online Retail II · CC BY 4.0" is
  set into the footer of EVERY plate as part of the design language, never as
  fine print an export could lose.
* **Real ink vs model overlay.** Marks measured from the real transactions are
  drawn in solid ink (the categorical slots below). Anything a model produced
  (a CLV fit, a naive forecast, a fitted curve) is drawn dashed and/or as a gray
  outline, and plates that contain model output say so in the footer -- the
  real-vs-modelled distinction is visual, not a footnote.

The palette is the validated light-mode set documented in docs/BUSINESS_CASE.md
(single-hue blue for magnitude, fixed categorical order for identity). The
sequential scale is BINNED (max 7 classes) so every heatmap cell can carry an
AA-contrast label on its own fill -- the exact value is always printed, the
shade is legibility only.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --------------------------------------------------------------------------- #
# Palette tokens (validated light-mode set; single source of truth)
# --------------------------------------------------------------------------- #
BLUE = "#2a78d6"
GREEN = "#008300"
MAGENTA = "#e87ba4"
YELLOW = "#eda100"
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"

CATEGORICAL = [BLUE, GREEN, MAGENTA, YELLOW]

# Model overlays: dashed and/or gray -- never solid ink (see module docstring).
MODEL_GRAY = MUTED
MODEL_DASH = (0, (4.0, 2.4))          # matplotlib linestyle tuple
MODEL_DASH_SVG = "5,3"                # the same dash for hand-drawn SVG strokes

# Sequential blue, binned. Each bin carries the label ink that clears WCAG AA
# (4.5:1) on its own fill, so a value printed inside a cell is always legible.
# Bounds are retention percentages: [lo, hi) except the last, which includes 100.
SEQ_BINS: list[tuple[float, str, str]] = [
    # upper bound (%), cell fill, AA label ink on that fill
    (5.0, "#cde2fb", INK),
    (10.0, "#9ec5f4", INK),
    (15.0, "#86b6ef", INK),
    (20.0, "#6da7ec", INK),
    (30.0, "#5598e7", INK),
    (50.0, "#3987e5", INK),
    (100.0, "#0d366b", "#ffffff"),    # dark anchor: the scale ends in real ink
]
SEQ_BIN_EDGES = [0.0, *(hi for hi, _f, _i in SEQ_BINS)]


def seq_bin(pct: float) -> tuple[str, str]:
    """(fill, label ink) for a 0..100 percentage on the binned sequential scale."""
    for hi, fill, label_ink in SEQ_BINS:
        if pct < hi:
            return fill, label_ink
    return SEQ_BINS[-1][1], SEQ_BINS[-1][2]


# --------------------------------------------------------------------------- #
# The plate registry -- one numbering across figures/ and the PDF
# --------------------------------------------------------------------------- #
SERIES = "FIELD NOTES · ONLINE RETAIL II"
ATTRIBUTION = "UCI Online Retail II · CC BY 4.0"
ATTRIBUTION_LONG = "UCI Online Retail II · CC BY 4.0 · real transactions, Dec 2009 – Dec 2011"
MODELLED_TAG = "SOLID INK = MEASURED · DASHED / OUTLINE = MODELLED"

PLATES: dict[str, tuple[int, str]] = {
    "cover": (1, "The dataset"),
    "cleaning": (2, "Cleaning ledger"),
    "monthly_revenue": (3, "Monthly revenue"),
    "top_products": (4, "Top products"),
    "top_countries": (5, "Top countries"),
    "order_values": (6, "Order values"),
    "missing_customers": (7, "Missing customer IDs"),
    "forecast": (8, "Weekly revenue forecast"),
    "rfm": (9, "RFM segments"),
    "cohort": (10, "Cohort retention"),
    "lifecycle": (11, "Lifecycle flows"),
    "clv": (12, "Customer lifetime value"),
    "returns_rate": (13, "Returns rate"),
    "returns": (14, "Returns anatomy"),
    "basket": (15, "Co-purchase rules"),
    "sku_forecast": (16, "SKU forecastability"),
}


def plate_tag(key: str) -> str:
    """'PLATE 03 · MONTHLY REVENUE' -- the footer identity line."""
    number, title = PLATES[key]
    return f"PLATE {number:02d} · {title.upper()}"


def plate_no(key: str) -> str:
    return f"PLATE {PLATES[key][0]:02d}"


# --------------------------------------------------------------------------- #
# Matplotlib side of the system
# --------------------------------------------------------------------------- #
def style() -> None:
    """Base rcParams: recessive hairline grid/axes, left-set bold titles."""
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "text.color": INK,
            "axes.labelcolor": INK_2,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "axes.edgecolor": GRID,
            "axes.grid": True,
            "axes.axisbelow": True,   # hairline grid stays behind the data marks
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
        }
    )


def _rule(fig: plt.Figure, y: float, x0: float = 0.045, x1: float = 0.955) -> None:
    fig.add_artist(
        plt.Line2D([x0, x1], [y, y], transform=fig.transFigure, color=GRID, linewidth=0.9)
    )


def chrome(
    fig: plt.Figure,
    key: str,
    notes: tuple[str, ...] = (),
    modelled: bool = False,
) -> tuple[float, float, float, float]:
    """Draw the plate header + footer on ``fig``; return the content rect.

    Header: kicker (series name) left, plate number right, hairline rule below.
    Footer: hairline rule, then the plate identity line left and the attribution
    caption right; ``notes`` (the figure's honesty captions) stack above the
    rule; ``modelled`` adds the measured-vs-modelled convention line.
    The returned ``(left, bottom, right, top)`` rect is for ``tight_layout``.
    """
    _w, h = fig.get_size_inches()

    kicker_y = 1.0 - 0.20 / h
    head_rule_y = 1.0 - 0.30 / h
    fig.text(0.045, kicker_y, SERIES, fontsize=7.5, color=MUTED)
    fig.text(0.955, kicker_y, plate_no(key), fontsize=7.5, color=INK, ha="right",
             fontweight="bold")
    _rule(fig, head_rule_y)

    foot_rule_y = 0.52 / h
    line1_y = 0.34 / h
    fig.text(0.045, line1_y, plate_tag(key), fontsize=7, color=MUTED)
    fig.text(0.955, line1_y, ATTRIBUTION_LONG, fontsize=7, color=INK_2, ha="right")
    if modelled:
        fig.text(0.045, 0.17 / h, MODELLED_TAG, fontsize=6.5, color=MUTED)
    _rule(fig, foot_rule_y)

    note_y = foot_rule_y + 0.10 / h
    for note in reversed(notes):
        fig.text(0.045, note_y, note, fontsize=7.5, color=MUTED, va="bottom")
        note_y += 0.145 / h * (1 + note.count("\n"))

    top = head_rule_y - 0.10 / h
    bottom = note_y + 0.06 / h if notes else foot_rule_y + 0.12 / h
    return (0.0, bottom, 1.0, top)


def save(fig: plt.Figure, out: Path, rect=None) -> Path:
    """tight_layout inside the plate chrome, then write the PNG."""
    out.parent.mkdir(parents=True, exist_ok=True)
    if rect is not None:
        fig.tight_layout(rect=rect)
    fig.savefig(out, dpi=150, metadata={"Software": "retail-analytics-real"})
    plt.close(fig)
    return out


# --------------------------------------------------------------------------- #
# SVG side of the system (used by the hand-drawn, byte-identical figures)
# --------------------------------------------------------------------------- #
SVG_MARGIN = 42          # left/right margin of the hairline rules
SVG_HEADER_H = 64        # header band height: kicker + rule; content starts below


def svg_text(x: float, y: float, s: str, size: float, fill: str, *,
             anchor: str = "start", weight: str = "400", spacing: str = "") -> str:
    a = f' text-anchor="{anchor}"' if anchor != "start" else ""
    w = f' font-weight="{weight}"' if weight != "400" else ""
    ls = f' letter-spacing="{spacing}"' if spacing else ""
    return f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" fill="{fill}"{a}{w}{ls}>{s}</text>'


def svg_header(width: float, key: str) -> list[str]:
    """Kicker + plate number + hairline rule (occupies y 0..SVG_HEADER_H)."""
    return [
        svg_text(SVG_MARGIN, 30, SERIES, 10, MUTED, spacing="2.5"),
        svg_text(width - SVG_MARGIN, 30, plate_no(key), 10, INK,
                 anchor="end", weight="700", spacing="2.5"),
        f'<line x1="{SVG_MARGIN}" y1="44" x2="{width - SVG_MARGIN}" y2="44" '
        f'stroke="{GRID}" stroke-width="1"/>',
    ]


def svg_footer_height(n_notes: int, modelled: bool = False) -> float:
    """Vertical room the footer needs below the last content baseline."""
    return 26 + 15 * n_notes + 40 + (14 if modelled else 0)


def svg_footer(width: float, height: float, key: str, notes: tuple[str, ...] = (),
               modelled: bool = False) -> list[str]:
    """Notes stack above a hairline rule; identity left, attribution right."""
    p: list[str] = []
    rule_y = height - 34 - (14 if modelled else 0)
    note_y = rule_y - 12 - 15 * (len(notes) - 1) if notes else rule_y
    for note in notes:
        p.append(svg_text(SVG_MARGIN, note_y, note, 9, MUTED))
        note_y += 15
    p.append(f'<line x1="{SVG_MARGIN}" y1="{rule_y:.1f}" x2="{width - SVG_MARGIN}" '
             f'y2="{rule_y:.1f}" stroke="{GRID}" stroke-width="1"/>')
    p.append(svg_text(SVG_MARGIN, rule_y + 18, plate_tag(key), 8.5, MUTED, spacing="1.5"))
    p.append(svg_text(width - SVG_MARGIN, rule_y + 18, ATTRIBUTION_LONG, 8.5, INK_2,
                      anchor="end"))
    if modelled:
        p.append(svg_text(SVG_MARGIN, rule_y + 32, MODELLED_TAG, 7.5, MUTED, spacing="1"))
    return p
