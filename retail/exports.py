"""Executive deliverables: a PDF briefing (matplotlib PdfPages) and an Excel workbook.

Both are generated end-to-end by ``python -m retail --deliverables`` and land in
``deliverables/`` (git-ignored — they are build artifacts, rebuilt on demand).

The PDF pages are set in the shared "field notes" plate system (``retail.plate``):
every page carries its plate number and the dataset attribution in a designed
footer, pages that also exist as committed figures carry the SAME plate number
in both places, and the PDF's page order follows the plate numbering. Model
output (forecasts, CLV predictions) is drawn dashed / outline; measured marks
are solid ink.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import BoundaryNorm, ListedColormap

from retail import plate
from retail.paths import DELIVERABLES
from retail.plate import (
    BERRY,
    GRID,
    INDIGO,
    INK,
    INK_2,
    MODEL_DASH,
    MUTED,
    RUST,
    SURFACE,
)

PDF_NAME = "retail_analytics_executive.pdf"
XLSX_NAME = "retail_analytics.xlsx"

CITATION = (
    "Data: Chen, D. (2019). Online Retail II [Dataset]. UCI Machine Learning Repository.\n"
    "https://doi.org/10.24432/C5CG6D - https://archive.ics.uci.edu/dataset/502/online+retail+ii\n"
    "License: CC BY 4.0. Real transactions of a UK-based online giftware retailer, Dec 2009 - Dec 2011.\n"
    "The raw data is not redistributed in this repository; scripts/download_data.py fetches it."
)


# --------------------------------------------------------------------------- #
# PDF
# --------------------------------------------------------------------------- #
def _style_table(table, fontsize: float = 8, yscale: float = 1.5) -> None:
    """Field-notes table styling: horizontal hairlines only, bold header row."""
    table.auto_set_font_size(False)
    table.set_fontsize(fontsize)
    table.scale(1.0, yscale)
    for (row, _col), cell in table.get_celld().items():
        cell.set_edgecolor(GRID)
        cell.set_linewidth(0.8)
        cell.visible_edges = "B"
        if row == 0:
            cell.set_text_props(fontweight="bold", color=INK)
            cell.set_facecolor(SURFACE)


def _text_page(pp: PdfPages, key: str, title: str, blocks: list[tuple[str, str]]) -> None:
    plate.style()
    fig = plt.figure(figsize=(11, 8.5))
    plate.chrome(fig, key)
    fig.text(0.07, 0.88, title, fontsize=20, fontweight="bold", color=INK)
    y = 0.80
    for heading, body in blocks:
        if heading:
            fig.text(0.07, y, heading, fontsize=12, fontweight="bold", color=INK)
            y -= 0.035
        fig.text(0.07, y, body, fontsize=9.5, color=INK_2, va="top", family="monospace")
        y -= 0.045 + 0.023 * body.count("\n")
    pp.savefig(fig)
    plt.close(fig)


def _clip(value: object, width: int = 58) -> str:
    text = str(value)
    return text if len(text) <= width else text[: width - 3] + "..."


def _table_page(pp: PdfPages, df: pd.DataFrame, key: str, title: str, note: str = "",
                col_widths=None) -> None:
    plate.style()
    fig, ax = plt.subplots(figsize=(11, 8.5))
    notes = tuple(note.split("\n")) if note else ()
    plate.chrome(fig, key, notes=notes)
    ax.axis("off")
    ax.set_title(title, fontsize=16, loc="left", pad=20, color=INK)
    table = ax.table(
        cellText=[[_clip(v) for v in row] for row in df.itertuples(index=False)],
        colLabels=list(df.columns),
        loc="upper center",
        cellLoc="left",
        colWidths=col_widths,
    )
    _style_table(table)
    fig.subplots_adjust(top=0.86, bottom=0.16, left=0.07, right=0.95)
    pp.savefig(fig)
    plt.close(fig)


def _forecast_page(pp: PdfPages, ctx: dict) -> None:
    plate.style()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8.5), height_ratios=[1, 1])
    rect = plate.chrome(fig, "forecast", modelled=True)

    monthly = ctx["monthly"]
    x = np.arange(len(monthly))
    ax1.plot(x, monthly["Revenue"].to_numpy() / 1e6, color=RUST, linewidth=2, marker="o", markersize=3)
    ax1.set_xticks(x)
    ax1.set_xticklabels(monthly["YearMonth"], rotation=45, ha="right", fontsize=7)
    ax1.set_ylabel("revenue (GBP, m)")
    ax1.set_ylim(bottom=0)
    ax1.grid(axis="x", visible=False)
    ax1.set_title("Monthly revenue (cleaned sales; final month has 9 days of data)")

    final = ctx["final_fold"]
    idx = np.arange(len(final))
    # Real weekly revenue in solid ink; every model overlay dashed, per the
    # plate system's measured-vs-modelled convention.
    ax2.plot(idx, final["actual"], color=INK, linewidth=2.2, label="actual", marker="o", markersize=4)
    model_colors = [MUTED, INDIGO, BERRY, RUST]
    for color, model in zip(model_colors, [c for c in final.columns if c != "actual"], strict=False):
        ax2.plot(idx, final[model], color=color, linewidth=1.6, linestyle=MODEL_DASH, label=model)
    ax2.set_xticks(idx)
    ax2.set_xticklabels([d.strftime("%Y-%m-%d") for d in final.index], rotation=45, ha="right", fontsize=7)
    ax2.set_ylabel("weekly revenue (GBP)")
    ax2.grid(axis="x", visible=False)
    ax2.legend(frameon=False, fontsize=8, loc="lower right")
    summary = ctx["cv_summary"]
    lines = [
        f"{r.model:<16} mean MASE {r.mean_mase:.3f}   worst {r.worst_mase:.3f}   fold wins {r.fold_wins}"
        for r in summary.itertuples(index=False)
    ]
    ax2.text(
        0.015,
        0.97,
        "rolling-origin CV (5 folds, horizon 8 weeks):\n" + "\n".join(lines),
        transform=ax2.transAxes,
        va="top",
        fontsize=8,
        family="monospace",
        color=INK_2,
    )
    ax2.set_title("Final-fold forecast vs actual (weekly revenue)", fontsize=11)
    fig.tight_layout(rect=rect)
    pp.savefig(fig)
    plt.close(fig)


def _rfm_page(pp: PdfPages, ctx: dict) -> None:
    plate.style()
    seg = ctx["rfm_summary"].sort_values("RevenueShare")
    fig, ax = plt.subplots(figsize=(11, 8.5))
    rect = plate.chrome(fig, "rfm")
    ax.grid(axis="y", visible=False)
    ax.barh(seg["Segment"], 100 * seg["RevenueShare"], color=RUST, height=0.62)
    for i, (share, customers) in enumerate(zip(seg["RevenueShare"], seg["Customers"], strict=True)):
        ax.text(100 * share + 0.4, i, f"{100 * share:.1f}%  ({customers:,} customers)",
                va="center", fontsize=9, color=INK_2)
    ax.set_xlabel("share of identified-customer revenue (%)")
    ax.set_xlim(0, 100 * seg["RevenueShare"].max() * 1.28)
    ax.set_title(
        "RFM segments - revenue share (customers with a real CustomerID only; "
        f"{int(seg['Customers'].sum()):,} customers)"
    )
    fig.tight_layout(rect=rect)
    pp.savefig(fig)
    plt.close(fig)


def _cohort_page(pp: PdfPages, ctx: dict) -> None:
    """Cohort retention heatmap (triangle) + size-weighted headline curve."""
    from retail.cohort import DISPLAY_OFFSETS

    result = ctx.get("cohort_result")
    if result is None or result.triangle.empty:
        return
    plate.style()
    tri = result.triangle
    ncol = min(DISPLAY_OFFSETS, int(tri.columns.max())) + 1
    mat = tri[[k for k in tri.columns if k < ncol]].to_numpy(dtype=float)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8.5), height_ratios=[2.4, 1])
    note = (f"Last complete month {result.last_complete_month}; "
            f"blank cells are right-censored (not yet observable). Offset 0 = acquisition month (100%).")
    rect = plate.chrome(fig, "cohort", notes=(
        note,
        "Cell shade is the shared binned single-hue scale (legibility only); "
        "the exact % is printed in each cell.",
    ))

    # The same binned sequential scale as the committed SVG plate: one hue,
    # dark-anchored, and an AA-contrast label ink chosen per bin.
    cmap = ListedColormap([fill for _hi, fill, _ink in plate.SEQ_BINS])
    cmap.set_bad(SURFACE)
    norm = BoundaryNorm([e / 100.0 for e in plate.SEQ_BIN_EDGES], cmap.N)
    ax1.imshow(mat, aspect="auto", cmap=cmap, norm=norm)
    ax1.set_xticks(range(ncol))
    ax1.set_xticklabels(range(ncol), fontsize=8)
    ax1.set_yticks(range(len(tri.index)))
    ax1.set_yticklabels(tri.index, fontsize=7)
    ax1.set_xlabel("months since first purchase")
    ax1.grid(False)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat[i, j]
            if not np.isnan(v):
                _fill, label_ink = plate.seq_bin(100 * v)
                ax1.text(j, i, f"{round(100 * v)}", ha="center", va="center",
                         fontsize=6.5, color=label_ink)
    ax1.set_title(
        f"Cohort repeat-purchase retention (% of each first-purchase cohort active later; "
        f"{result.n_customers:,} identified customers)", fontsize=11)

    curve = result.curve
    curve = curve[curve["offset"] < ncol]
    ax2.plot(curve["offset"], 100 * curve["avg_retention"], color=RUST, linewidth=2, marker="o", markersize=4)
    ax2.set_xticks(range(ncol))
    ax2.set_ylim(0, 100)
    ax2.set_xlabel("months since first purchase")
    ax2.set_ylabel("repeat rate (%)")
    ax2.grid(axis="x", visible=False)
    ax2.set_title("Headline retention curve (size-weighted across observable cohorts)", fontsize=10)
    fig.tight_layout(rect=rect)
    pp.savefig(fig)
    plt.close(fig)


def _lifecycle_page(pp: PdfPages, ctx: dict) -> None:
    """Lifecycle page: growth-accounting bars + the aggregate stage-flow matrix."""
    from retail import lifecycle

    result = ctx.get("lifecycle_result")
    if result is None or result.monthly.empty:
        return
    plate.style()
    m = result.monthly
    fig, ax1 = plt.subplots(figsize=(11, 8.5))
    plate.chrome(fig, "lifecycle")
    x = np.arange(len(m))
    bottom = np.zeros(len(m))
    # Stack order and stage colors are the shared STATE_FILL mapping, so the
    # PDF page and the committed SVG plate read as the same object.
    for name in ("retained", "resurrected", "new"):
        vals = m[name].to_numpy(dtype=float)
        ax1.bar(x, vals, bottom=bottom, color=lifecycle.STATE_FILL[name], width=0.72,
                edgecolor=SURFACE, linewidth=0.8, label=name)
        bottom += vals
    churn = np.nan_to_num(m["churned"].to_numpy(dtype=float), nan=0.0)
    ax1.bar(x, -churn, color=BERRY, width=0.72, edgecolor=SURFACE, linewidth=0.8,
            label="churned")
    ax1.axhline(0, color=INK_2, linewidth=1)
    ax1.set_xticks(x)
    ax1.set_xticklabels(m["month"], rotation=45, ha="right", fontsize=7)
    ax1.set_ylabel("identified customers buying (above) / lapsing (below)")
    ax1.grid(axis="x", visible=False)
    ax1.legend(frameon=False, fontsize=8, loc="upper left", ncols=4)
    ax1.set_title(
        f"Customer lifecycle - monthly growth accounting ({result.n_customers:,} "
        f"identified customers)", fontsize=12)

    hl = result.headline()
    lines = []
    if hl:
        lines.append(
            f"Mean month: {hl['mean_active']:,.0f} active = {hl['mean_new']:,.0f} new + "
            f"{hl['mean_retained']:,.0f} retained + {hl['mean_resurrected']:,.0f} resurrected; "
            f"{hl['mean_churned']:,.0f} churn out. Overall quick ratio "
            f"{hl['overall_quick_ratio']:.2f} (>=1 in {hl['months_qr_ge_1']}/{hl['months_scored']} months)."
        )
    lines.append(
        f"Stage definitions are documented choices: activity = any invoice in the calendar "
        f"month; dormant = silent > {result.config.dormancy_months} months."
    )
    lines.append(
        f"'New' is first purchase inside the window (left-censored); complete months through "
        f"{result.last_complete_month} - the partial final month is excluded."
    )
    fig.text(0.07, 0.335, "\n".join(lines), fontsize=8.5, color=INK_2, va="top")

    if not result.flows.empty:
        fig.text(0.07, 0.25, "Aggregate month-to-month stage flows (customers):",
                 fontsize=9, color=INK, fontweight="bold", va="top")
        flows = lifecycle.flow_frame(result)
        tbl_ax = fig.add_axes((0.07, 0.075, 0.52, 0.16))
        tbl_ax.axis("off")
        table = tbl_ax.table(
            cellText=[[str(v) for v in row] for row in flows.itertuples(index=False)],
            colLabels=list(flows.columns), loc="center", cellLoc="right",
        )
        _style_table(table, fontsize=7.5, yscale=1.2)
    fig.subplots_adjust(top=0.90, bottom=0.46, left=0.09, right=0.96)
    pp.savefig(fig)
    plt.close(fig)


def _clv_page(pp: PdfPages, ctx: dict) -> None:
    """CLV page: out-of-sample validation bars + concentration Lorenz + params/headline."""
    from retail import clv

    result = ctx.get("clv_result")
    if result is None or result.clv_table.empty:
        return
    plate.style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 8.5))
    plate.chrome(fig, "clv", modelled=True)

    v = result.validation
    if v is not None and not v.by_frequency.empty:
        bf = v.by_frequency
        idx = np.arange(len(bf))
        w = 0.4
        ax1.bar(idx - w / 2, bf["actual_mean"], width=w, color=RUST, label="actual")
        ax1.bar(idx + w / 2, bf["predicted_mean"], width=w, facecolor=SURFACE,
                edgecolor=INK_2, linewidth=1.1, linestyle=MODEL_DASH, label="predicted")
        ax1.set_xticks(idx)
        ax1.set_xticklabels(bf["cal_frequency"])
        ax1.set_xlabel("purchases in calibration period")
        ax1.set_ylabel("mean transactions in holdout")
        ax1.grid(axis="x", visible=False)
        ax1.legend(frameon=False, fontsize=9)
        ax1.set_title("Out-of-sample check (predicted vs actual)", fontsize=11)

    cum_x, cum_y = clv.lorenz_points(result.clv_table["clv"].to_numpy())
    ax2.plot(100 * cum_x, 100 * cum_y, color=RUST, linewidth=2, linestyle=MODEL_DASH)
    ax2.plot([0, 100], [0, 100], color=MUTED, linewidth=1, linestyle=":")
    ax2.set_xlabel("share of customers (%, lowest CLV first)")
    ax2.set_ylabel("share of predicted CLV (%)")
    ax2.grid(axis="x", visible=False)
    ax2.set_title(f"Predicted {result.horizon_days}-day CLV concentration", fontsize=11)

    b, g = result.bgnbd, result.gamma_gamma
    conc = result.concentration(0.10)
    lines = [
        "Customer lifetime value - BG/NBD (frequency) x Gamma-Gamma (value), from scratch.",
        f"BG/NBD  r={b.r:.4f}  alpha={b.alpha:.3f}  a={b.a:.4f}  b={b.b:.4f}",
        f"Gamma-Gamma  p={g.p:.4f}  q={g.q:.4f}  v={g.v:.2f}   pop. mean value GBP "
        f"{g.population_mean():,.0f}   freq-value corr {g.freq_value_corr:+.3f}",
        f"{result.n_customers:,} identified customers ({result.n_repeat:,} repeat); predicted "
        f"{result.horizon_days}-day revenue GBP {conc['total_clv']:,.0f}, top 10% hold "
        f"{100 * conc['top_share']:.1f}%.",
    ]
    if v is not None and v.actual_total:
        lines.append(
            f"Holdout ({v.holdout_days} days, {v.n_customers:,} customers): predicted "
            f"{v.predicted_total:,.0f} vs actual {v.actual_total:,} transactions "
            f"({v.ratio:.3f}x), correlation {v.correlation:.2f}."
        )
    lines.append(
        "Gross revenue, not margin (no cost data); finite undiscounted horizon; "
        "identified customers only."
    )
    fig.text(0.07, 0.42, "\n".join(lines), fontsize=9.5, color=INK_2, va="top", family="monospace")

    top = clv.clv_summary_table(result, top=8).copy()
    top["monetary_value"] = top["monetary_value"].round(0)
    tbl_ax = fig.add_axes((0.07, 0.10, 0.86, 0.17))
    tbl_ax.axis("off")
    table = tbl_ax.table(
        cellText=[[_clip(v) for v in row] for row in top.itertuples(index=False)],
        colLabels=list(top.columns), loc="center", cellLoc="center",
    )
    _style_table(table, fontsize=7.5, yscale=1.1)

    fig.suptitle("Predictive customer lifetime value", fontsize=16, x=0.07, ha="left", color=INK,
                 y=0.915)
    fig.subplots_adjust(top=0.85, bottom=0.58, left=0.09, right=0.95, wspace=0.28)
    pp.savefig(fig)
    plt.close(fig)


def _returns_page(pp: PdfPages, ctx: dict) -> None:
    """Returns & cancellations: headline rates + reverse-logistics lag + top returned SKUs."""
    result = ctx.get("returns_result")
    if result is None or result.by_sku.empty:
        return
    o, m, c = result.overview, result.match, result.concentration
    top = result.by_sku.head(12).copy()
    top["Description"] = top["Description"].astype(str).str.slice(0, 30)
    top["returned_value"] = top["returned_value"].round(0)
    top["value_return_rate"] = (100.0 * top["value_return_rate"]).round(1)
    top = top[["StockCode", "Description", "returned_units", "returned_value", "value_return_rate"]]
    top.columns = ["SKU", "Description", "Ret. units", "Ret. value (GBP)", "Value return rate %"]
    lag_line = ""
    if m.get("matched_lines"):
        lag_line = (f"Reverse-logistics lag: {100 * m['matched_value_share']:.1f}% of returned value "
                    f"matches a prior same-customer, same-SKU purchase; median "
                    f"{m['median_lag_days']:.0f} days to return (p25 {m['p25_lag_days']:.0f}, "
                    f"p75 {m['p75_lag_days']:.0f}). Matching is a heuristic, not a linked RMA.")
    note = (
        f"Gross GBP {o['gross_value']:,.0f}  ->  returned GBP {o['returned_value']:,.0f}  "
        f"->  net GBP {o['net_value']:,.0f}.   Return rate {100 * o['value_return_rate']:.2f}% of value, "
        f"{100 * o['unit_return_rate']:.2f}% of units.\n"
        f"{o['return_lines']:,} return lines across {o['cancellation_invoices']:,} cancellation invoices; "
        f"top {c['top_n']} SKUs carry {100 * c['top_n_share']:.1f}% of returned value.\n"
        f"{lag_line}\n"
        "Measured on real cancellation invoices (UCI Online Retail II, CC BY 4.0). A monthly rate can "
        "exceed 100% when a credit note posts in a quieter month than its sale."
    )
    _table_page(
        pp, top, "returns",
        "Returns & cancellations - top returned SKUs by value",
        note=note,
        col_widths=[0.09, 0.34, 0.12, 0.16, 0.18],
    )


def _pricing_page(pp: PdfPages, ctx: dict) -> None:
    """Price ladders: what each SKU actually sold at, and what the slopes do (not) say."""
    result = ctx.get("pricing_result")
    if result is None or result.profile.empty:
        return
    from retail import pricing

    o = result.overview
    top = pricing.ladder_table(result, top=14).copy()
    top.columns = ["SKU", "Description", "Rungs", "Posted (GBP)", "Realized (GBP)",
                   "p90/p10", "Units below posted %", "Market-adj. slope"]
    lines = [
        f"{o['n_multi_price_skus']:,} of {o['n_skus']:,} SKUs ({100 * o['multi_price_share']:.1f}%) sold at "
        f"more than one price. Qualifying ladders: {o['n_qualifying']:,} SKUs "
        f"({100 * o['qualifying_revenue_share']:.1f}% of revenue),",
        f"median {o['median_rungs']:.0f} rungs, p90/p10 spread {o['median_spread']:.2f}x. Units sold below / "
        f"at / above the posted (modal) price: {100 * o['unit_share_below']:.1f}% / "
        f"{100 * o['unit_share_at']:.1f}% / {100 * o['unit_share_above']:.1f}%.",
        f"Price realization {100 * o['realization']:.1f}% - arithmetic on the units that actually sold, "
        f"never a claim about units that did not.",
    ]
    if o["n_fitted"]:
        lines += [
            f"Price/quantity slopes ({o['n_fitted']:,} SKUs, log-log OLS): "
            f"{o['median_slope_within_week']:.2f} from the volume-discount schedule alone (within week), "
            f"{o['median_slope_market_adj']:.2f} market-adjusted",
            f"week to week; {o['n_perm_significant']:,} of {o['n_perm_tested']:,} beat a seeded permutation "
            f"null. The two agree on the median and correlate only r={o['slope_agreement_corr']:.2f} per SKU.",
            f"NOT an elasticity - prices here were never randomised. A slope of "
            f"{o['constant_spend_slope']:.0f} is what a buyer spending the same amount per line produces "
            f"with no demand response at all;",
            "read the numbers against that, not against zero.",
        ]
    lines.append('"-" = the SKU\'s posted price barely moved week to week, so no slope was fitted.')
    _table_page(
        pp, top, "pricing",
        "Price ladders - the same SKU at many prices",
        note="\n".join(lines),
        col_widths=[0.08, 0.28, 0.07, 0.11, 0.12, 0.09, 0.15, 0.13],
    )


def export_pdf(ctx: dict, path: Path | None = None) -> Path:
    path = path or DELIVERABLES / PDF_NAME
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = ctx["raw_report"]
    with PdfPages(path) as pp:
        _text_page(
            pp,
            "cover",
            "Online Retail II - honest analytics on real data",
            [
                ("Dataset & citation", CITATION),
                (
                    "What was measured",
                    f"raw rows {raw['rows']:,} | invoices {raw['invoices']:,} | "
                    f"{raw['date_min']:%Y-%m-%d} to {raw['date_max']:%Y-%m-%d}\n"
                    f"missing CustomerID {100 * raw['missing_customer_id_pct']:.1f}% | "
                    f"cancellation rows {100 * raw['cancellation_rows_pct']:.1f}% | "
                    f"exact duplicates {raw['exact_duplicate_rows']:,}",
                ),
                (
                    "Method summary",
                    "1. Documented cleaning pipeline - every step logs rows in/out (next page).\n"
                    "2. RFM segmentation from scratch (quintile scores, standard segment grid).\n"
                    "3. Weekly revenue forecast: seasonal-naive vs Holt-Winters vs lag-features OLS,\n"
                    "   rolling-origin CV, MASE-scored, leakage-safe (train strictly before origin).\n"
                    "4. Market-basket mining (from-scratch Apriori, FP-growth cross-check) on invoice\n"
                    "   baskets over the top-200 SKUs; all rules in the Excel Rules sheet.\n"
                    "5. All methods implemented with numpy/pandas only - no ML/mining libraries.",
                ),
                (
                    "The plate system",
                    "Every artifact of this project is a numbered plate in one design system;\n"
                    "pages here share their numbers with the committed figures. Measured marks\n"
                    "are solid ink; model output is dashed / outline.",
                ),
                (
                    "Honesty notes",
                    "Single retailer, UK-heavy, gift-season seasonality, final month incomplete.\n"
                    "Where a simple baseline wins a fold, that is reported, not hidden.",
                ),
                ("", f"Generated {datetime.now():%Y-%m-%d %H:%M} by `python -m retail --deliverables`."),
            ],
        )
        _table_page(
            pp,
            ctx["clean_table"],
            "cleaning",
            "Cleaning impact - every decision with its row cost",
            note="Cancellations are separated (kept for returns analysis), not deleted. "
            "Missing CustomerID rows are flagged and kept for revenue, excluded from RFM.",
            col_widths=[0.24, 0.09, 0.09, 0.11, 0.09, 0.38],
        )
        _forecast_page(pp, ctx)
        _rfm_page(pp, ctx)
        _cohort_page(pp, ctx)
        _lifecycle_page(pp, ctx)
        _clv_page(pp, ctx)
        _returns_page(pp, ctx)
        sku = ctx["sku_table"].copy()
        sku["Description"] = sku["Description"].astype(str).str.slice(0, 30)
        sku.columns = ["SKU", "Description", "Units", "Weeks", "Zero-wk share",
                       "MASE naive", "MASE s-naive", "MASE ma4", "Best model"]
        _table_page(
            pp,
            sku,
            "sku_forecast",
            "Top-10 SKU weekly demand - forecastability check",
            note="MASE > 1 means the model failed to beat a one-week naive walk on that SKU. "
            "Zero-wk share = share of zero-demand weeks since first sale (intermittency).",
            col_widths=[0.07, 0.25, 0.08, 0.07, 0.11, 0.09, 0.10, 0.08, 0.15],
        )
        _pricing_page(pp, ctx)
    return path


# --------------------------------------------------------------------------- #
# Excel
# --------------------------------------------------------------------------- #
def _write_quality_sheet(writer, ctx: dict) -> None:
    """Optional data-quality sheet: same numbers the printed report card shows
    (screen == export). Written only when ctx carries a quality card."""
    from retail import quality

    before, after = ctx.get("quality_before"), ctx.get("quality_after")
    if after is None:
        return
    sheet = "DataQuality"
    header = pd.DataFrame({"Data-quality report card": [quality.NOT_A_CERTIFICATION]})
    header.to_excel(writer, sheet_name=sheet, index=False, startrow=0)

    row = 3
    if before is not None:
        summary = pd.DataFrame(
            {
                "Dimension": ["OVERALL", *quality.DIMENSIONS],
                "Before": [before.overall_score, *(before.dimensions[d].score for d in quality.DIMENSIONS)],
                "After": [after.overall_score, *(after.dimensions[d].score for d in quality.DIMENSIONS)],
            }
        )
        summary["Lift"] = (summary["After"] - summary["Before"]).round(1)
        summary.to_excel(writer, sheet_name=sheet, index=False, startrow=row)
        row += len(summary) + 3
    quality.findings_frame(after).to_excel(writer, sheet_name=sheet, index=False, startrow=row)


def _write_cohort_sheet(writer, ctx: dict) -> None:
    """Cohort retention triangle + the size-weighted headline curve (screen == export)."""
    from retail import cohort

    result = ctx.get("cohort_result")
    if result is None or result.triangle.empty:
        return
    sheet = "Cohort"
    header = pd.DataFrame(
        {"Cohort repeat-purchase retention (% of each first-purchase cohort active m months later)": [
            f"identified customers {result.n_customers:,} | last complete month "
            f"{result.last_complete_month} | blank cells are not-yet-observable (right-censored)"
        ]}
    )
    header.to_excel(writer, sheet_name=sheet, index=False, startrow=0)
    cohort.triangle_frame(result).to_excel(writer, sheet_name=sheet, index=False, startrow=2)
    row = len(result.triangle) + 5
    curve = result.curve.copy()
    curve["avg_retention_pct"] = (100.0 * curve["avg_retention"]).round(2)
    curve = curve[["offset", "avg_retention_pct", "cohorts_observed", "customers_observed", "repeat_customers"]]
    curve.to_excel(writer, sheet_name=sheet, index=False, startrow=row)


def _write_lifecycle_sheet(writer, ctx: dict) -> None:
    """Lifecycle sheet: monthly stage counts + growth accounting + the flow matrix (screen == export)."""
    from retail import lifecycle

    result = ctx.get("lifecycle_result")
    if result is None or result.monthly.empty:
        return
    sheet = "Lifecycle"
    g = result.gap_stats
    gap_note = (f"comeback gaps: median {g['p50']:.0f} / p75 {g['p75']:.0f} months, "
                f"{100 * g['share_within_dormancy']:.1f}% within the threshold" if g else "n/a")
    header = pd.DataFrame(
        {"Customer lifecycle stages & growth accounting - one stage per identified customer per complete month": [
            f"customers {result.n_customers:,} | months through {result.last_complete_month} | "
            f"dormancy threshold {result.config.dormancy_months} months (documented choice; {gap_note}) | "
            "churned = active last month, not this month | quick ratio = (new+resurrected)/churned"
        ]}
    )
    header.to_excel(writer, sheet_name=sheet, index=False, startrow=0)
    monthly = lifecycle.monthly_frame(result)
    monthly.to_excel(writer, sheet_name=sheet, index=False, startrow=2)
    row = len(monthly) + 5
    flows_header = pd.DataFrame(
        {"Aggregate month-to-month stage flows (customers, all complete month pairs)": [
            "rows = stage in month m-1 (plus pre-acquisition entries), columns = stage in month m"
        ]}
    )
    flows_header.to_excel(writer, sheet_name=sheet, index=False, startrow=row)
    lifecycle.flow_frame(result).to_excel(writer, sheet_name=sheet, index=False, startrow=row + 2)


def _write_clv_sheet(writer, ctx: dict) -> None:
    """Customer-lifetime-value sheet: fit params, out-of-sample check, top customers."""
    result = ctx.get("clv_result")
    if result is None or result.clv_table.empty:
        return
    from retail import clv

    sheet = "CLV"
    b, g = result.bgnbd, result.gamma_gamma
    conc = result.concentration(0.10)
    header = pd.DataFrame(
        {
            "Customer lifetime value - BG/NBD (frequency) x Gamma-Gamma (value); "
            "predicted GROSS revenue over the horizon, identified customers only": [
                f"horizon {result.horizon_days} days | customers {result.n_customers:,} "
                f"({result.n_repeat:,} repeat) | total predicted revenue GBP "
                f"{conc['total_clv']:,.0f} | top 10% of customers = "
                f"{100 * conc['top_share']:.1f}% of it"
            ]
        }
    )
    header.to_excel(writer, sheet_name=sheet, index=False, startrow=0)

    params = pd.DataFrame(
        {
            "quantity": [
                "BG/NBD r", "BG/NBD alpha", "BG/NBD a", "BG/NBD b",
                "Gamma-Gamma p", "Gamma-Gamma q", "Gamma-Gamma v",
                "population mean value (GBP)", "frequency-value correlation",
            ],
            "value": [
                round(b.r, 5), round(b.alpha, 4), round(b.a, 5), round(b.b, 5),
                round(g.p, 5), round(g.q, 5), round(g.v, 4),
                round(g.population_mean(), 2), round(g.freq_value_corr, 4),
            ],
        }
    )
    params.to_excel(writer, sheet_name=sheet, index=False, startrow=3)
    row = 3 + len(params) + 3

    v = result.validation
    if v is not None:
        val = pd.DataFrame(
            {
                "out_of_sample_holdout": [
                    "calibration end", "holdout end", "holdout days", "customers",
                    "predicted transactions", "actual transactions", "ratio (pred/actual)",
                    "predicted-vs-actual correlation", "per-customer MAE",
                ],
                "value": [
                    v.calibration_end.date().isoformat(), v.holdout_end.date().isoformat(),
                    v.holdout_days, v.n_customers, round(v.predicted_total, 1), v.actual_total,
                    round(v.ratio, 4), round(v.correlation, 4), round(v.mae, 4),
                ],
            }
        )
        val.to_excel(writer, sheet_name=sheet, index=False, startrow=row)
        row += len(val) + 3
        v.by_frequency.to_excel(writer, sheet_name=sheet, index=False, startrow=row)
        row += len(v.by_frequency) + 3

    top = clv.clv_summary_table(result, top=50)
    top.to_excel(writer, sheet_name=sheet, index=False, startrow=row)


def _write_returns_sheet(writer, ctx: dict) -> None:
    """Returns analysis: headline metrics, matching/lag, per-SKU table, monthly net (screen == export)."""
    from retail import returns as returns_mod

    result = ctx.get("returns_result")
    if result is None or result.by_sku.empty:
        return
    sheet = "Returns"
    o, m, c = result.overview, result.match, result.concentration
    header = pd.DataFrame(
        {"Returns & cancellations - measured on real C-invoices; returned value/units as a share of gross": [
            f"return rate {100 * o['value_return_rate']:.2f}% of value / {100 * o['unit_return_rate']:.2f}% "
            f"of units | net GBP {o['net_value']:,.0f} of {o['gross_value']:,.0f} | "
            f"{m['matched_value_share'] * 100:.1f}% of returned value matched to a prior purchase"
        ]}
    )
    header.to_excel(writer, sheet_name=sheet, index=False, startrow=0)

    metrics = pd.DataFrame(
        {
            "metric": [
                "gross value (GBP)", "returned value (GBP)", "net value (GBP)",
                "value return rate %", "unit return rate %",
                "return lines", "cancellation invoices", "returned SKUs",
                "top-N SKU share of returned value %", "SKUs to cover 80% of returns",
                "matched value share %", "matched lines", "median days to return",
            ],
            "value": [
                round(o["gross_value"], 2), round(o["returned_value"], 2), round(o["net_value"], 2),
                round(100 * o["value_return_rate"], 2), round(100 * o["unit_return_rate"], 2),
                o["return_lines"], o["cancellation_invoices"], c["n_returned_skus"],
                round(100 * c["top_n_share"], 1), c["skus_to_80pct"],
                round(100 * m["matched_value_share"], 1), m["matched_lines"],
                (round(m["median_lag_days"], 1) if m.get("matched_lines") else "n/a"),
            ],
        }
    )
    metrics.to_excel(writer, sheet_name=sheet, index=False, startrow=3)
    row = 3 + len(metrics) + 2

    lag = result.lag.copy()
    lag["returned_value"] = lag["returned_value"].round(2)
    lag["line_share_pct"] = (100.0 * lag["line_share"]).round(2)
    lag = lag[["lag_bucket", "lines", "returned_value", "line_share_pct"]]
    lag.to_excel(writer, sheet_name=sheet, index=False, startrow=row)
    row += len(lag) + 2

    returns_mod.sku_export_frame(result).to_excel(writer, sheet_name=sheet, index=False, startrow=row)


def _write_pricing_sheet(writer, ctx: dict) -> None:
    """Price ladders: headline metrics, then the full per-SKU table (screen == export)."""
    result = ctx.get("pricing_result")
    if result is None or result.profile.empty:
        return
    from retail import pricing

    sheet = "PriceLadder"
    o = result.overview
    header = pd.DataFrame(
        {"Price ladders - every price each SKU was actually charged at, measured on the real invoices": [
            f"{100 * o['multi_price_share']:.1f}% of SKUs sold at more than one price | qualifying "
            f"ladders {o['n_qualifying']:,} ({100 * o['qualifying_revenue_share']:.1f}% of revenue) | "
            f"slopes are observational co-movement, NOT causal elasticity (no experiment, no cost data)"
        ]}
    )
    header.to_excel(writer, sheet_name=sheet, index=False, startrow=0)

    metrics = pd.DataFrame(
        {
            "metric": [
                "SKUs in cleaned sales", "SKUs sold at more than one price", "multi-price share %",
                "qualifying SKUs", "qualifying revenue (GBP)", "qualifying revenue share %",
                "median rungs per ladder", "median p90/p10 price spread",
                "units below posted price %", "units at posted price %", "units above posted price %",
                "revenue at posted price (GBP)", "price realization %",
                "SKUs with fitted slopes", "median slope: volume-discount schedule (within week)",
                "median slope: posted price week to week", "median slope: market-adjusted",
                "negative market-adjusted slopes %", "per-SKU agreement between the two slopes (r)",
                "constant-spend benchmark slope", "permutation draws per SKU", "permutation seed",
                "SKUs beating their own permutation null at p<=0.05",
            ],
            "value": [
                o["n_skus"], o["n_multi_price_skus"], round(100 * o["multi_price_share"], 2),
                o["n_qualifying"], round(o["qualifying_revenue"], 2),
                round(100 * o["qualifying_revenue_share"], 2),
                o["median_rungs"], round(o["median_spread"], 3),
                round(100 * o["unit_share_below"], 2), round(100 * o["unit_share_at"], 2),
                round(100 * o["unit_share_above"], 2), round(o["revenue_at_posted"], 2),
                round(100 * o["realization"], 2),
                o["n_fitted"], round(o["median_slope_within_week"], 4),
                round(o["median_slope_between_week"], 4), round(o["median_slope_market_adj"], 4),
                round(100 * o["share_slope_negative"], 2), round(o["slope_agreement_corr"], 4),
                o["constant_spend_slope"], result.config.n_permutations, result.config.seed,
                o["n_perm_significant"],
            ],
        }
    )
    metrics.to_excel(writer, sheet_name=sheet, index=False, startrow=3)
    pricing.export_frame(result).to_excel(
        writer, sheet_name=sheet, index=False, startrow=3 + len(metrics) + 2
    )


def export_excel(ctx: dict, path: Path | None = None) -> Path:
    path = path or DELIVERABLES / XLSX_NAME
    path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        ctx["clean_table"].to_excel(writer, sheet_name="CleaningReport", index=False)
        monthly = ctx["monthly"].copy()
        monthly["ReturnsRate"] = ctx["returns_rate"].to_numpy()
        monthly.to_excel(writer, sheet_name="MonthlyRevenue", index=False)
        ctx["rfm_summary"].to_excel(writer, sheet_name="RFM", index=False)
        _write_cohort_sheet(writer, ctx)
        _write_lifecycle_sheet(writer, ctx)
        _write_clv_sheet(writer, ctx)
        _write_returns_sheet(writer, ctx)
        _write_pricing_sheet(writer, ctx)
        ctx["cv"].to_excel(writer, sheet_name="ForecastCV", index=False)
        ctx["cv_summary"].to_excel(writer, sheet_name="ForecastCV", index=False, startrow=len(ctx["cv"]) + 3)
        ctx["sku_table"].to_excel(writer, sheet_name="TopSKUs", index=False)
        if "rules_table" in ctx:
            ctx["rules_table"].to_excel(writer, sheet_name="Rules", index=False)
        _write_quality_sheet(writer, ctx)
        weekly = ctx["weekly"].rename("Revenue").reset_index()
        weekly.columns = ["WeekEnding", "Revenue"]
        weekly.to_excel(writer, sheet_name="WeeklyRevenue", index=False)
    return path
