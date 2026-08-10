"""Customer lifecycle stages and growth accounting on the cleaned real transactions.

The CRM operational view. RFM is a *snapshot* of who is valuable now, cohort
retention is the *acquisition-time* view, CLV is the *predictive* leg -- this
module is the month-by-month *operating* view: every identified customer is
assigned exactly one lifecycle stage per calendar month, and the month-to-month
transitions between stages are counted into a flow matrix. From scratch,
numpy/pandas only, measured -- nothing modelled.

Stages (per complete calendar month, over customers acquired by that month)
---------------------------------------------------------------------------
* ``new``          -- first-ever (in-window) purchase falls in this month.
* ``retained``     -- bought this month AND the month before.
* ``resurrected``  -- bought this month after >= 1 silent month.
* ``at_risk``      -- no purchase this month; last purchase 1..D months ago.
* ``dormant``      -- no purchase this month; last purchase > D months ago.

``D`` (``LifecycleConfig.dormancy_months``, default 3) is a **documented choice,
not a law**: on the real data the median gap between a customer's consecutive
active months is 2 months and the 75th percentile is 3, so "silent for more than
3 months" genuinely separates a routine reorder pause from deep dormancy. The
measured gap distribution ships in the result (``gap_stats``) so the choice can
be audited or re-made.

Growth accounting (the standard decomposition)
----------------------------------------------
``active(m) = new(m) + retained(m) + resurrected(m)`` and
``churned(m) = active(m-1) - retained(m)`` -- the customers who bought last
month but not this month. The **quick ratio** ``(new + resurrected) / churned``
says whether the month's inflow of buyers covered its outflow (>= 1 means grew).

Honesty guards (built into the numbers, not bolted on)
------------------------------------------------------
1.  **Partial final month.** The data ends mid-month (December 2011 holds nine
    trading days). Scoring that month would fake a churn cliff, so stages are
    computed through the last *complete* month only; activity after it is
    ignored and customers first seen after it are excluded (and counted in
    ``n_excluded_partial``).
2.  **Left-censoring.** The window opens 2009-12, so "new" means *first
    purchase inside the observation window* -- the earliest months conflate
    genuinely-new buyers with pre-existing accounts that are simply invisible
    before the window. This is a property of every growth-accounting view and
    is stated, not hidden.
3.  Stage definitions are **definitional choices** (activity = any invoice in
    the calendar month; the threshold D). They are configurable, printed on the
    outputs, and none of them is presented as a behavioural truth.

Identified customers only, as everywhere: rows without a ``CustomerID`` cannot
be tracked and are structurally invisible to lifecycle analytics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from retail.paths import DELIVERABLES, FIGURES

# Stage vocabulary. Order is the canonical reporting order.
STATES = ("new", "retained", "resurrected", "at_risk", "dormant")
ACTIVE_STATES = ("new", "retained", "resurrected")
ENTRY = "(pre-acquisition)"  # flow-matrix source row for first-ever purchases

_CODE = {name: i + 1 for i, name in enumerate(STATES)}  # 0 = not yet acquired

# Palette shared with retail.eda (validated light-mode categorical order).
# Stack adjacency blue->yellow->green (+ magenta across the axis) passes the
# CVD checks; at-risk/dormant are DELIBERATE neutrals -- inactive stages are
# recessive by design, with identity carried by stack order + labels + legend.
_RETAINED = "#2a78d6"   # blue
_RESURRECTED = "#eda100"  # yellow
_NEW = "#008300"        # green
_CHURNED = "#e87ba4"    # magenta, below the axis
_AT_RISK = "#b4b2a9"    # neutral mid
_DORMANT = "#dcdbd3"    # neutral light
_INK = "#0b0b0b"
_INK_2 = "#52514e"
_MUTED = "#898781"
_GRID = "#e1e0d9"
_SURFACE = "#fcfcfb"

STATE_FILL = {
    "new": _NEW,
    "retained": _RETAINED,
    "resurrected": _RESURRECTED,
    "at_risk": _AT_RISK,
    "dormant": _DORMANT,
}


@dataclass(frozen=True)
class LifecycleConfig:
    """The declared definitional choices (printed on every output)."""

    dormancy_months: int = 3  # at_risk gap 1..D months; dormant gap > D


@dataclass
class LifecycleResult:
    """Everything the report needs, all measured from the cleaned sales frame."""

    monthly: pd.DataFrame          # one row per complete month: stage counts, churn, QR, revenue split
    flows: pd.DataFrame            # source stage (rows, incl. ENTRY) x target stage (cols), counts
    config: LifecycleConfig
    last_complete_month: str       # 'YYYY-MM'
    final_month_partial: bool      # True when the data ends mid-month
    n_customers: int               # identified customers active within complete months
    n_excluded_partial: int        # identified customers seen ONLY after the last complete month
    gap_stats: dict = field(default_factory=dict)  # measured comeback-gap distribution

    def headline(self) -> dict:
        """Mean monthly growth accounting over months 2..K (month 1 has no prior month)."""
        m = self.monthly
        if len(m) < 2:
            return {}
        tail = m.iloc[1:]
        out = {
            "mean_active": float(tail["active"].mean()),
            "mean_new": float(tail["new"].mean()),
            "mean_retained": float(tail["retained"].mean()),
            "mean_resurrected": float(tail["resurrected"].mean()),
            "mean_churned": float(tail["churned"].mean()),
            "months_qr_ge_1": int((tail["quick_ratio"] >= 1.0).sum()),
            "months_scored": int(len(tail)),
        }
        churn_total = float(tail["churned"].sum())
        inflow_total = float((tail["new"] + tail["resurrected"]).sum())
        out["overall_quick_ratio"] = inflow_total / churn_total if churn_total else float("nan")
        return out


def _ym_index(dates: pd.Series) -> pd.Series:
    """Integer month index year*12 + (month-1) -- ordered, subtractable, gap-free."""
    return dates.dt.year * 12 + (dates.dt.month - 1)


def _ym_label(i: int) -> str:
    return f"{i // 12:04d}-{i % 12 + 1:02d}"


def _empty_result(config: LifecycleConfig) -> LifecycleResult:
    return LifecycleResult(
        monthly=pd.DataFrame(), flows=pd.DataFrame(), config=config,
        last_complete_month="", final_month_partial=False,
        n_customers=0, n_excluded_partial=0, gap_stats={},
    )


def lifecycle_states(sales: pd.DataFrame, config: LifecycleConfig | None = None) -> LifecycleResult:
    """Assign every identified customer a lifecycle stage per complete month.

    Deterministic: pure aggregation over the cleaned sales frame -- no RNG, no
    wall clock. Same input, same result, byte for byte.
    """
    config = config or LifecycleConfig()
    if config.dormancy_months < 1:
        raise ValueError("dormancy_months must be >= 1")
    known = sales.loc[sales["CustomerID"].notna(), ["CustomerID", "InvoiceDate", "Revenue"]]
    if known.empty:
        return _empty_result(config)

    ym = _ym_index(known["InvoiceDate"])

    # Last COMPLETE calendar month (same rule as retail.cohort): a final month
    # covered only partway is excluded as a scored month.
    max_date = known["InvoiceDate"].max()
    last_period = max_date.to_period("M")
    final_month_partial = bool(max_date < last_period.end_time)
    last_complete_i = int(last_period.year * 12 + (last_period.month - 1)) - (1 if final_month_partial else 0)

    n_identified = int(known["CustomerID"].nunique())
    keep = ym <= last_complete_i
    known = known.loc[keep]
    ym = ym.loc[keep]
    if known.empty:
        return _empty_result(config)

    # customer x month activity/revenue matrices (identified customers are few
    # thousand, months a couple dozen -- dense matrices are trivial and exact).
    act = (
        pd.DataFrame({"CustomerID": known["CustomerID"].to_numpy(), "ym": ym.to_numpy(),
                      "Revenue": known["Revenue"].to_numpy()})
        .groupby(["CustomerID", "ym"], sort=True)["Revenue"].sum().reset_index()
    )
    cust_ids = np.sort(act["CustomerID"].unique())
    n_excluded = n_identified - len(cust_ids)
    first_i = int(act["ym"].min())
    months = list(range(first_i, last_complete_i + 1))
    n_c, n_m = len(cust_ids), len(months)

    ci = pd.Series(np.arange(n_c), index=cust_ids).reindex(act["CustomerID"]).to_numpy()
    mi = (act["ym"].to_numpy() - first_i).astype(np.int64)
    active = np.zeros((n_c, n_m), dtype=bool)
    revenue = np.zeros((n_c, n_m), dtype="float64")
    active[ci, mi] = True
    revenue[ci, mi] = act["Revenue"].to_numpy()

    col = np.arange(n_m)
    first_idx = active.argmax(axis=1)                       # month index of first purchase
    last_seen = np.maximum.accumulate(np.where(active, col[None, :], -1), axis=1)
    gap = col[None, :] - last_seen                          # months since last purchase (0 if active)
    prev_active = np.zeros_like(active)
    prev_active[:, 1:] = active[:, :-1]
    exists = col[None, :] >= first_idx[:, None]

    stage = np.zeros((n_c, n_m), dtype=np.int8)             # 0 = not yet acquired
    is_first = col[None, :] == first_idx[:, None]
    stage[exists & active & is_first] = _CODE["new"]
    stage[exists & active & ~is_first & prev_active] = _CODE["retained"]
    stage[exists & active & ~is_first & ~prev_active] = _CODE["resurrected"]
    stage[exists & ~active & (gap <= config.dormancy_months)] = _CODE["at_risk"]
    stage[exists & ~active & (gap > config.dormancy_months)] = _CODE["dormant"]

    # ---- monthly growth-accounting table --------------------------------- #
    counts = {name: (stage == _CODE[name]).sum(axis=0).astype("int64") for name in STATES}
    active_n = counts["new"] + counts["retained"] + counts["resurrected"]
    base = sum(counts[name] for name in STATES)             # customers acquired by month m
    churned = np.full(n_m, np.nan)
    churned[1:] = (active_n[:-1] - counts["retained"][1:]).astype("float64")
    with np.errstate(invalid="ignore", divide="ignore"):
        quick = np.where(churned > 0, (counts["new"] + counts["resurrected"]) / churned, np.nan)

    rev_by_state = {
        name: np.where(stage == _CODE[name], revenue, 0.0).sum(axis=0) for name in ACTIVE_STATES
    }
    monthly = pd.DataFrame(
        {
            "month": [_ym_label(i) for i in months],
            "new": counts["new"],
            "retained": counts["retained"],
            "resurrected": counts["resurrected"],
            "active": active_n,
            "churned": churned,
            "quick_ratio": quick,
            "at_risk": counts["at_risk"],
            "dormant": counts["dormant"],
            "base": base,
            "new_revenue": rev_by_state["new"].round(2),
            "retained_revenue": rev_by_state["retained"].round(2),
            "resurrected_revenue": rev_by_state["resurrected"].round(2),
            "active_revenue": revenue.sum(axis=0).round(2),
        }
    )

    # ---- aggregate month-to-month flow matrix ---------------------------- #
    flows = pd.DataFrame(0, index=[ENTRY, *STATES], columns=list(STATES), dtype="int64")
    if n_m >= 2:
        src = stage[:, :-1].ravel()
        dst = stage[:, 1:].ravel()
        moved = dst > 0                                     # target month must have a stage
        pair_counts = np.zeros((len(STATES) + 1, len(STATES)), dtype="int64")
        np.add.at(pair_counts, (src[moved], dst[moved] - 1), 1)
        flows.loc[ENTRY] = pair_counts[0]
        for name in STATES:
            flows.loc[name] = pair_counts[_CODE[name]]
    flows.index.name = "from"

    # ---- measured comeback-gap distribution (audits the D choice) -------- #
    gaps: list[np.ndarray] = []
    for i in range(n_c):
        seen = np.flatnonzero(active[i])
        if len(seen) > 1:
            gaps.append(np.diff(seen))
    gap_stats: dict = {}
    if gaps:
        g = np.concatenate(gaps)
        gap_stats = {
            "n_gaps": int(len(g)),
            "p50": float(np.percentile(g, 50)),
            "p75": float(np.percentile(g, 75)),
            "p90": float(np.percentile(g, 90)),
            "share_within_dormancy": float((g <= config.dormancy_months).mean()),
        }

    return LifecycleResult(
        monthly=monthly,
        flows=flows,
        config=config,
        last_complete_month=_ym_label(last_complete_i),
        final_month_partial=final_month_partial,
        n_customers=n_c,
        n_excluded_partial=n_excluded,
        gap_stats=gap_stats,
    )


def monthly_frame(result: LifecycleResult) -> pd.DataFrame:
    """Tidy export frame: rounded ratios, blank (not fake-zero) undefined cells."""
    if result.monthly.empty:
        return pd.DataFrame()
    out = result.monthly.copy()
    out["quick_ratio"] = out["quick_ratio"].round(3)
    return out


def flow_frame(result: LifecycleResult) -> pd.DataFrame:
    """The transition matrix as a flat frame (source stage column + one column per target)."""
    if result.flows.empty:
        return pd.DataFrame()
    return result.flows.reset_index()


def write_csv(result: LifecycleResult, out_dir: Path = DELIVERABLES) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    monthly_path = out_dir / "lifecycle_stages.csv"
    flows_path = out_dir / "lifecycle_flows.csv"
    monthly_frame(result).to_csv(monthly_path, index=False, encoding="utf-8", lineterminator="\n")
    flow_frame(result).to_csv(flows_path, index=False, encoding="utf-8", lineterminator="\n")
    return monthly_path, flows_path


def headline_text(result: LifecycleResult) -> str:
    """One measured line, e.g. 'avg month: 1,041 active buyers = 212 new + 390 retained + 439 resurrected; 620 lapse (quick ratio 1.09)'."""
    hl = result.headline()
    if not hl:
        return "lifecycle: not enough complete months to score month-over-month flows"
    return (
        f"avg month: {hl['mean_active']:,.0f} active buyers = {hl['mean_new']:,.0f} new + "
        f"{hl['mean_retained']:,.0f} retained + {hl['mean_resurrected']:,.0f} resurrected; "
        f"{hl['mean_churned']:,.0f} lapse (quick ratio {hl['overall_quick_ratio']:.2f})"
    )


# --------------------------------------------------------------------------- #
# Hand-drawn SVG (no plotting library): deterministic, byte-identical, committed
# --------------------------------------------------------------------------- #
def _esc(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _nice_step(span: float) -> int:
    """A clean axis step (1/2/5 x 10^k) giving 3-6 gridlines over `span`."""
    if span <= 0:
        return 1
    raw = span / 4
    mag = 10 ** int(np.floor(np.log10(raw)))
    for mult in (1, 2, 5, 10):
        if raw <= mult * mag:
            return int(mult * mag)
    return int(10 * mag)


def render_svg(result: LifecycleResult) -> str:
    """Growth-accounting bars + base-composition area as one SVG string.

    Pure string formatting from the measured monthly table -- no timestamps,
    no RNG, so re-running yields a byte-identical file.
    """
    m = result.monthly
    months = list(m["month"]) if not m.empty else []
    n = len(months)

    # Geometry
    cw = 34                      # px per month column
    left = 74
    plot_w = max(cw * n, cw)
    width = left + plot_w + 150  # right margin carries the direct band labels
    top = 118
    p1_h_up, p1_h_dn = 170, 90   # panel 1: above-axis / below-axis heights
    p1_axis = top + p1_h_up
    p1_bottom = p1_axis + p1_h_dn
    p2_top = p1_bottom + 130
    p2_h = 170
    p2_bottom = p2_top + p2_h
    height = p2_bottom + 96
    seg_gap = 2                  # 2px surface gap between stacked fills

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

    hl = result.headline()
    p.append(txt(left, 34, "Customer lifecycle - monthly growth accounting", 18, _INK, weight="700"))
    p.append(txt(left, 55, f"Identified customers (n={result.n_customers:,}); every buyer-month is "
                 "new, retained (bought last month too) or resurrected (back after a gap).", 11, _INK_2))
    p.append(txt(left, 72, "Churned = bought last month, not this month. Quick ratio = "
                 "(new + resurrected) / churned; >= 1 means the buyer base grew.", 11, _MUTED))

    if m.empty:
        p.append(txt(left, top, "No complete months to display.", 12, _MUTED))
        p.append("</svg>")
        return "\n".join(p) + "\n"

    up_max = float(m["active"].max())
    dn_max = float(np.nanmax(m["churned"].to_numpy())) if n > 1 else 0.0
    step = _nice_step(up_max)

    def y_up(v: float) -> float:
        return p1_axis - (v / up_max) * (p1_h_up - 14) if up_max else p1_axis

    def y_dn(v: float) -> float:
        return p1_axis + (v / dn_max) * (p1_h_dn - 10) if dn_max else p1_axis

    # hairline grid + clean ticks (above axis)
    tick = step
    while tick <= up_max:
        gy = y_up(tick)
        p.append(f'<line x1="{left}" y1="{gy:.1f}" x2="{left + plot_w}" y2="{gy:.1f}" '
                 f'stroke="{_GRID}" stroke-width="0.7"/>')
        p.append(txt(left - 8, gy + 3.5, f"{tick:,}", 9, _MUTED, anchor="end"))
        tick += step

    # stacked bars: retained (bottom) -> resurrected -> new; churned below axis
    bar_w = cw - 8
    order = [("retained", _RETAINED), ("resurrected", _RESURRECTED), ("new", _NEW)]
    for j in range(n):
        x = left + j * cw + 4
        y_cursor = p1_axis
        for name, fill in order:
            v = float(m[name].iloc[j])
            if v <= 0:
                continue
            h = (v / up_max) * (p1_h_up - 14)
            y0 = y_cursor - h
            h_draw = max(h - seg_gap, 0.5)
            p.append(f'<rect x="{x}" y="{y0:.1f}" width="{bar_w}" height="{h_draw:.1f}" '
                     f'rx="1.5" fill="{fill}"/>')
            y_cursor = y0
        ch = m["churned"].iloc[j]
        if not pd.isna(ch) and ch > 0:
            h = (float(ch) / dn_max) * (p1_h_dn - 10)
            p.append(f'<rect x="{x}" y="{p1_axis + seg_gap}" width="{bar_w}" '
                     f'height="{max(h - seg_gap, 0.5):.1f}" rx="1.5" fill="{_CHURNED}"/>')

    # axis line + month labels (every 3rd to avoid collisions)
    p.append(f'<line x1="{left}" y1="{p1_axis}" x2="{left + plot_w}" y2="{p1_axis}" '
             f'stroke="{_MUTED}" stroke-width="1"/>')
    for j in range(0, n, 3):
        p.append(txt(left + j * cw + cw / 2, p1_bottom + 16, _esc(months[j]), 8.5, _MUTED,
                     anchor="middle"))

    # selective direct labels: final month's active total and churned
    xf = left + (n - 1) * cw + cw / 2
    p.append(txt(xf, y_up(float(m["active"].iloc[-1])) - 6,
                 f"{int(m['active'].iloc[-1]):,} active", 10, _INK, anchor="middle", weight="700"))
    ch_last = m["churned"].iloc[-1]
    if not pd.isna(ch_last) and dn_max:
        p.append(txt(xf, y_dn(float(ch_last)) + 14, f"{int(ch_last):,} churned", 9.5, _INK_2,
                     anchor="middle"))

    # legend (color + text token, never color alone)
    lx = left
    ly = p1_bottom + 40
    for label, fill in [("new", _NEW), ("retained", _RETAINED), ("resurrected", _RESURRECTED),
                        ("churned (below axis)", _CHURNED)]:
        p.append(f'<rect x="{lx}" y="{ly - 9}" width="10" height="10" rx="2" fill="{fill}"/>')
        p.append(txt(lx + 15, ly, label, 10, _INK_2))
        lx += 15 + 7.2 * len(label) + 26

    if hl:
        p.append(txt(left, ly + 22,
                     f"Mean month: {hl['mean_active']:,.0f} active = {hl['mean_new']:,.0f} new + "
                     f"{hl['mean_retained']:,.0f} retained + {hl['mean_resurrected']:,.0f} resurrected; "
                     f"{hl['mean_churned']:,.0f} churn out. Overall quick ratio "
                     f"{hl['overall_quick_ratio']:.2f} "
                     f"(&#8805;1 in {hl['months_qr_ge_1']} of {hl['months_scored']} months).",
                     10.5, _INK_2))

    # ---- panel 2: base composition (stacked area, absolute counts) -------- #
    p.append(txt(left, p2_top - 34, "Where the acquired base sits each month", 13, _INK, weight="700"))
    p.append(txt(left, p2_top - 16, "All customers acquired by each month, stacked by stage. "
                 "Neutral bands are the inactive stages; dormant = silent for more than "
                 f"{result.config.dormancy_months} months (documented choice).", 10, _MUTED))

    base_max = float(m["base"].max())
    stack_order = ["retained", "resurrected", "new", "at_risk", "dormant"]

    def y2(v: float) -> float:
        return p2_bottom - (v / base_max) * (p2_h - 12) if base_max else p2_bottom

    def x2(j: int) -> float:
        return left + j * cw + cw / 2 if n > 1 else left + plot_w / 2

    # hairline grid behind the areas
    step2 = _nice_step(base_max)
    tick = step2
    while tick <= base_max:
        gy = y2(tick)
        p.append(f'<line x1="{left}" y1="{gy:.1f}" x2="{left + plot_w}" y2="{gy:.1f}" '
                 f'stroke="{_GRID}" stroke-width="0.7"/>')
        tick += step2

    cum = np.zeros(n)
    layers: list[tuple[str, np.ndarray, np.ndarray]] = []
    for name in stack_order:
        lower = cum.copy()
        cum = cum + m[name].to_numpy(dtype=float)
        layers.append((name, lower, cum.copy()))
    for name, lower, upper in layers:
        pts_up = [f"{x2(j):.1f},{y2(upper[j]):.1f}" for j in range(n)]
        pts_dn = [f"{x2(j):.1f},{y2(lower[j]):.1f}" for j in range(n - 1, -1, -1)]
        p.append(f'<polygon points="{" ".join(pts_up + pts_dn)}" fill="{STATE_FILL[name]}" '
                 f'stroke="{_SURFACE}" stroke-width="2"/>')

    tick = step2
    while tick <= base_max:
        p.append(txt(left - 8, y2(tick) + 3.5, f"{tick:,}", 9, _MUTED, anchor="end"))
        tick += step2
    p.append(f'<line x1="{left}" y1="{p2_bottom}" x2="{left + plot_w}" y2="{p2_bottom}" '
             f'stroke="{_MUTED}" stroke-width="1"/>')
    for j in range(0, n, 3):
        p.append(txt(x2(j), p2_bottom + 16, _esc(months[j]), 8.5, _MUTED, anchor="middle"))

    # Direct band labels at the right edge (identity never rides on color alone).
    # Each label sits at its band's midpoint, then labels are nudged apart
    # bottom-up so neighbours keep >= 13px separation (no stacked-label collisions).
    label_rows = [
        (y2((lower[-1] + upper[-1]) / 2) + 3.5,
         f"{name.replace('_', '-')} {int(upper[-1] - lower[-1]):,}")
        for name, lower, upper in layers
        if upper[-1] - lower[-1] > 0
    ]
    label_rows.sort(key=lambda r: -r[0])          # bottom-most first (largest y)
    min_gap = 13.0
    placed: list[float] = []
    for y_lab, _text in label_rows:
        if placed and placed[-1] - y_lab < min_gap:
            y_lab = placed[-1] - min_gap
        placed.append(y_lab)
    for y_lab, (_y0, text) in zip(placed, label_rows, strict=True):
        p.append(txt(left + plot_w + 10, y_lab, text, 10, _INK_2))

    # footnotes
    partial = (" the final data month is partial and excluded;" if result.final_month_partial else "")
    p.append(txt(left, p2_bottom + 44,
                 f"Complete months through {result.last_complete_month};{partial} "
                 f"stage definitions are documented choices (activity = any invoice in the month; "
                 f"dormancy threshold {result.config.dormancy_months} months).", 9, _MUTED))
    p.append(txt(left, p2_bottom + 58,
                 "'New' is first purchase inside the window - the earliest months conflate truly-new "
                 "buyers with pre-existing accounts (left-censoring).", 9, _MUTED))
    p.append(txt(left, p2_bottom + 72,
                 "Data: UCI Online Retail II (CC BY 4.0) - real transactions; identified customers "
                 "only; measured, not modelled.", 9, _MUTED))

    p.append("</svg>")
    return "\n".join(p) + "\n"


def write_svg(result: LifecycleResult, out_dir: Path = FIGURES) -> Path:
    out = out_dir / "lifecycle_stages.svg"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_svg(result), encoding="utf-8", newline="\n")
    return out
