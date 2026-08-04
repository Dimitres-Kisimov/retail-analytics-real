"""Acquisition-cohort repeat-purchase retention on cleaned real transactions.

A classic retail analytic, computed from scratch (numpy/pandas only) on the same
cleaned sales frame every other module uses -- no re-cleaning, no re-invention.

Method
------
* A customer's **cohort** is the calendar month of their first purchase (first
  invoice), over customers with a real ``CustomerID`` only (the ``KnownCustomer``
  rows -- guest/unlinked rows carry no customer identity and cannot be tracked).
* A customer is **active at offset k** if they placed at least one invoice in the
  calendar month ``cohort_month + k``. Offset ``0`` is the acquisition month
  itself, so ``retention[c, 0] == 100%`` for every cohort by construction.
* ``retention[c, k] = active_customers[c, k] / cohort_size[c]`` -- the repeat rate.

Honesty / censoring (the two things that make cohort curves lie if ignored)
---------------------------------------------------------------------------
1.  **Right-censoring.** A cohort acquired late can only be observed for a few
    offsets. A cell ``(c, k)`` is populated only when its target month
    ``c + k`` is a *complete* month within the data; otherwise it is left blank
    (``NaN``). The headline curve at offset ``k`` therefore averages only the
    cohorts old enough to have been observed at ``k`` -- never mixing "hasn't
    happened yet" with "didn't come back".
2.  **Incomplete final month.** The dataset ends mid-month (e.g. 2011-12 holds
    only nine trading days). That trailing partial month is treated as *not a
    complete target month*, so no cell counts an absence there as churn. The
    acquisition column (k=0) is definitional 100% regardless of completeness.

Every number is measured from the real data; nothing here is modelled or invented.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from retail.paths import DELIVERABLES, FIGURES

# Palette shared with retail.eda (validated light-mode single-hue blue).
_BLUE = (42, 120, 214)        # #2a78d6, the sequential end
_BLUE_HEX = "#2a78d6"
_TINT = (234, 242, 251)       # pale blue, the low end of the scale
_INK_2 = "#52514e"
_MUTED = "#898781"
_GRID = "#e1e0d9"
_SURFACE = "#fcfcfb"
_INK = "#0b0b0b"

# How many months-since-acquisition the SVG heatmap and the curve display.
DISPLAY_OFFSETS = 12
# Colour shade is proportional to retention**GAMMA for legibility only; the
# exact percentage is printed inside every cell, so no value is misrepresented.
_GAMMA = 0.70


@dataclass
class CohortResult:
    """Everything the report needs, all derived from the cleaned sales frame."""

    triangle: pd.DataFrame        # index=cohort 'YYYY-MM', cols=offset 0..K, retention fraction (NaN=unobserved)
    sizes: pd.Series              # cohort 'YYYY-MM' -> number of customers acquired
    curve: pd.DataFrame           # per-offset headline: avg retention over observable cohorts
    last_complete_month: str      # 'YYYY-MM' -- last fully-covered calendar month
    final_month_partial: bool     # True when the data ends mid-month (see docstring)
    n_customers: int              # distinct identified customers covered

    def headline(self) -> dict:
        """Month-1 and month-6 repeat rates (fractions) plus their cohort counts."""
        c = self.curve.set_index("offset")
        out: dict[str, float | int] = {}
        for k in (1, 6):
            if k in c.index:
                out[f"month_{k}_rate"] = float(c.loc[k, "avg_retention"])
                out[f"month_{k}_cohorts"] = int(c.loc[k, "cohorts_observed"])
        return out


def _ym_index(dates: pd.Series) -> pd.Series:
    """Integer month index year*12 + (month-1) -- ordered, subtractable, gap-free."""
    return dates.dt.year * 12 + (dates.dt.month - 1)


def _ym_label(i: int) -> str:
    return f"{i // 12:04d}-{i % 12 + 1:02d}"


def cohort_retention(sales: pd.DataFrame) -> CohortResult:
    """Compute the full cohort x months-since-acquisition retention triangle."""
    known = sales.loc[sales["CustomerID"].notna(), ["CustomerID", "InvoiceDate"]].copy()
    if known.empty:
        empty = pd.DataFrame()
        return CohortResult(empty, pd.Series(dtype="int64"), empty, "", False, 0)

    known["ym"] = _ym_index(known["InvoiceDate"])
    known["cohort_i"] = known.groupby("CustomerID")["ym"].transform("min")
    known["offset"] = (known["ym"] - known["cohort_i"]).astype("int64")

    # Last COMPLETE calendar month: if the data ends before the final month's last
    # day, that final month is partial and excluded as a valid target month.
    max_date = known["InvoiceDate"].max()
    last_period = max_date.to_period("M")
    final_month_partial = bool(max_date < last_period.end_time)
    last_complete_i = int(last_period.year * 12 + (last_period.month - 1)) - (1 if final_month_partial else 0)

    cust = known.drop_duplicates("CustomerID")
    sizes_i = cust.groupby("cohort_i").size().sort_index()
    active_i = known.groupby(["cohort_i", "offset"])["CustomerID"].nunique()

    cohorts = list(sizes_i.index)
    max_offset = int(last_complete_i - cohorts[0]) if cohorts else 0
    offsets = list(range(max_offset + 1))

    tri = pd.DataFrame(index=cohorts, columns=offsets, dtype="float64")
    for c in cohorts:
        size = int(sizes_i[c])
        for k in offsets:
            if k == 0:
                tri.at[c, k] = 1.0  # acquisition month is definitional 100%
            elif c + k <= last_complete_i:
                tri.at[c, k] = float(active_i.get((c, k), 0)) / size
            # else: leave NaN -- target month not yet observable / incomplete

    # Headline curve: at each offset, pool only the cohorts observable there
    # (weighted by cohort size -> total repeat customers / total cohort customers).
    rows = []
    for k in offsets:
        obs = [c for c in cohorts if k == 0 or c + k <= last_complete_i]
        if not obs:
            continue
        customers = int(sum(int(sizes_i[c]) for c in obs))
        repeat = int(sum(int(active_i.get((c, k), 0)) if k else int(sizes_i[c]) for c in obs))
        rows.append(
            {
                "offset": k,
                "avg_retention": repeat / customers if customers else np.nan,
                "cohorts_observed": len(obs),
                "customers_observed": customers,
                "repeat_customers": repeat,
            }
        )
    curve = pd.DataFrame(rows)

    tri.index = [_ym_label(c) for c in cohorts]
    tri.index.name = "cohort"
    sizes = pd.Series(sizes_i.to_numpy(), index=tri.index, name="customers")

    return CohortResult(
        triangle=tri,
        sizes=sizes,
        curve=curve,
        last_complete_month=_ym_label(last_complete_i),
        final_month_partial=final_month_partial,
        n_customers=int(cust.shape[0]),
    )


def triangle_frame(result: CohortResult, offsets: int | None = None) -> pd.DataFrame:
    """Tidy export frame: cohort, customers, then m0..mK retention percentages."""
    tri = result.triangle
    if offsets is not None:
        cols = [k for k in tri.columns if k <= offsets]
        tri = tri[cols]
    pct = (100.0 * tri).round(2)
    pct.columns = [f"m{k}" for k in pct.columns]
    out = pct.reset_index()
    out.insert(1, "customers", result.sizes.to_numpy())
    return out


# --------------------------------------------------------------------------- #
# Hand-drawn SVG (no plotting library): deterministic, byte-identical, committed
# --------------------------------------------------------------------------- #
def _lerp_color(t: float) -> str:
    t = 0.0 if t < 0 else (1.0 if t > 1 else t)
    r = round(_TINT[0] + t * (_BLUE[0] - _TINT[0]))
    g = round(_TINT[1] + t * (_BLUE[1] - _TINT[1]))
    b = round(_TINT[2] + t * (_BLUE[2] - _TINT[2]))
    return f"#{r:02x}{g:02x}{b:02x}"


def _esc(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def render_svg(result: CohortResult, offsets: int = DISPLAY_OFFSETS) -> str:
    """Build the cohort heatmap triangle + headline retention curve as an SVG string.

    Pure string formatting from the measured triangle -- no timestamps, no RNG,
    so re-running yields a byte-identical file.
    """
    tri = result.triangle
    cohorts = list(tri.index)
    ncol = min(offsets, int(tri.columns.max())) + 1 if len(tri.columns) else 1
    cols = list(range(ncol))

    # Geometry
    cw, ch = 42, 20
    left = 118
    grid_w = cw * ncol
    width = left + grid_w + 34
    top = 150                       # first heatmap row baseline (below title block + col header)
    heat_bottom = top + ch * len(cohorts)
    curve_top = heat_bottom + 78
    curve_h = 150
    curve_bottom = curve_top + curve_h
    height = curve_bottom + 74

    def txt(x, y, s, size, fill, *, anchor="start", weight="400"):
        a = f' text-anchor="{anchor}"' if anchor != "start" else ""
        w = f' font-weight="{weight}"' if weight != "400" else ""
        return f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" fill="{fill}"{a}{w}>{s}</text>'

    p: list[str] = []
    p.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'font-family="Segoe UI, Helvetica, Arial, sans-serif" '
        f'style="background:{_SURFACE}">'
    )
    p.append(f'<rect x="0" y="0" width="{width}" height="{height}" fill="{_SURFACE}"/>')

    # Title block (short lines so nothing overflows the canvas width)
    p.append(txt(left, 34, "Cohort repeat-purchase retention", 18, _INK, weight="700"))
    p.append(txt(left, 55, "% of each first-purchase cohort that buys again m months later "
                 f"(identified customers, n={result.n_customers:,}).", 11, _INK_2))
    p.append(txt(left, 72, "Blank cells are not yet observable (right-censored); "
                 "offset 0 is the acquisition month (100%).", 11, _MUTED))
    p.append(txt(left, 89, f"Cell shade increases with retention^{_GAMMA:g} for legibility; "
                 "the exact % is printed in each cell.", 11, _MUTED))

    # Column headers (months since first purchase)
    p.append(txt(left, top - 30, "months since first purchase &#8594;", 10.5, _INK_2, weight="600"))
    for k in cols:
        p.append(txt(left + k * cw + cw / 2, top - 9, str(k), 10, _MUTED, anchor="middle"))
    p.append(txt(left - 10, top - 9, "cohort", 10.5, _INK_2, anchor="end", weight="600"))
    p.append(txt(left - 64, top - 9, "size", 9, _MUTED, anchor="end", weight="600"))

    # Heatmap cells
    for ri, cohort in enumerate(cohorts):
        cy = top + ri * ch
        size = int(result.sizes.iloc[ri])
        mid = cy + ch / 2 + 3.5
        p.append(txt(left - 10, mid, _esc(cohort), 10, _INK_2, anchor="end"))
        p.append(txt(left - 64, mid, f"{size:,}", 9, _MUTED, anchor="end"))
        for k in cols:
            val = tri.at[cohort, k] if k in tri.columns else np.nan
            x = left + k * cw
            if pd.isna(val):
                p.append(f'<rect x="{x}" y="{cy}" width="{cw}" height="{ch}" '
                         f'fill="{_SURFACE}" stroke="{_GRID}" stroke-width="1"/>')
                continue
            t = float(val) ** _GAMMA
            fill = _lerp_color(t)
            label_fill = "#ffffff" if t > 0.55 else _INK_2
            p.append(f'<rect x="{x}" y="{cy}" width="{cw}" height="{ch}" '
                     f'fill="{fill}" stroke="{_SURFACE}" stroke-width="1"/>')
            p.append(txt(x + cw / 2, mid, str(round(100 * val)), 9.5, label_fill, anchor="middle"))

    # ---- Headline curve panel ----
    curve = result.curve[result.curve["offset"] <= (ncol - 1)]
    p.append(txt(left, curve_top - 20, "Headline retention curve "
                 "(size-weighted across observable cohorts)", 13, _INK, weight="700"))

    plot_w = grid_w

    def _pt(k: float, v: float) -> tuple[float, float]:
        x = left + (k / max(ncol - 1, 1)) * plot_w
        y = curve_bottom - (v / 100.0) * curve_h
        return x, y

    p.append(f'<line x1="{left}" y1="{curve_bottom}" x2="{left + plot_w}" y2="{curve_bottom}" '
             f'stroke="{_GRID}" stroke-width="1"/>')
    for lab in (0, 25, 50, 75, 100):
        gy = curve_bottom - (lab / 100.0) * curve_h
        p.append(f'<line x1="{left}" y1="{gy:.1f}" x2="{left + plot_w}" y2="{gy:.1f}" '
                 f'stroke="{_GRID}" stroke-width="0.7"/>')
        p.append(txt(left - 8, gy + 3.5, f"{lab}%", 9, _MUTED, anchor="end"))

    pts = [_pt(int(r.offset), 100 * float(r.avg_retention)) for r in curve.itertuples(index=False)]
    poly = " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
    p.append(f'<polyline points="{poly}" fill="none" stroke="{_BLUE_HEX}" stroke-width="2.2"/>')
    for (x, y), r in zip(pts, curve.itertuples(index=False), strict=True):
        p.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3" fill="{_BLUE_HEX}"/>')
        p.append(txt(x, curve_bottom + 14, str(int(r.offset)), 9, _MUTED, anchor="middle"))

    # annotate month-1 and month-6
    hl = result.headline()
    for k in (1, 6):
        key = f"month_{k}_rate"
        if key in hl:
            x, y = _pt(k, 100 * hl[key])
            p.append(txt(x, y - 9, f"{100 * hl[key]:.1f}%", 10, _INK, anchor="middle", weight="700"))
    p.append(txt(left + plot_w / 2, curve_bottom + 30, "months since first purchase", 10, _INK_2,
                 anchor="middle"))

    # Footnotes (two fitted lines so nothing overflows the canvas width)
    lc = result.last_complete_month
    partial = "; final month partial, excluded as a target" if result.final_month_partial else ""
    p.append(txt(left, curve_bottom + 48, f"Last complete month {lc}{partial}.", 9, _MUTED))
    p.append(txt(left, curve_bottom + 62,
                 "Data: UCI Online Retail II (CC BY 4.0) - real transactions; retention is measured, not modelled.",
                 9, _MUTED))

    p.append("</svg>")
    return "\n".join(p) + "\n"


def write_svg(result: CohortResult, out_dir: Path = FIGURES, offsets: int = DISPLAY_OFFSETS) -> Path:
    out = out_dir / "cohort_retention.svg"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_svg(result, offsets=offsets), encoding="utf-8", newline="\n")
    return out


def write_csv(result: CohortResult, out_dir: Path = DELIVERABLES) -> Path:
    out = out_dir / "cohort_retention.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    triangle_frame(result).to_csv(out, index=False, encoding="utf-8", lineterminator="\n")
    return out


def headline_text(result: CohortResult) -> str:
    """The one-line headline, e.g. 'month-1 repeat rate 23.3%, holding near 22.2% by month 6'.

    The verb tracks the measured direction: 'decaying'/'rising' only when the
    month-6 rate is more than 2 points off month-1, else 'holding near' -- so a
    near-flat plateau is not dressed up as a decay it isn't.
    """
    hl = result.headline()
    if "month_1_rate" not in hl:
        return "cohort retention: not enough complete months to report a month-1 repeat rate"
    m1 = 100 * hl["month_1_rate"]
    if "month_6_rate" not in hl:
        return f"month-1 repeat rate {m1:.1f}% (fewer than 6 complete offsets available)"
    m6 = 100 * hl["month_6_rate"]
    delta = m1 - m6
    if delta > 2.0:
        verb = f"decaying to {m6:.1f}%"
    elif delta < -2.0:
        verb = f"rising to {m6:.1f}%"
    else:
        verb = f"holding near {m6:.1f}%"
    return f"month-1 repeat rate {m1:.1f}%, {verb} by month 6"
