"""Executive deliverables: a PDF briefing (matplotlib PdfPages) and an Excel workbook.

Both are generated end-to-end by ``python -m retail --deliverables`` and land in
``deliverables/`` (git-ignored — they are build artifacts, rebuilt on demand).
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

from retail.eda import BLUE, CATEGORICAL, GRID, INK, INK_2, MUTED, SURFACE, _style
from retail.paths import DELIVERABLES

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
def _text_page(pp: PdfPages, title: str, blocks: list[tuple[str, str]]) -> None:
    _style()
    fig = plt.figure(figsize=(11, 8.5))
    fig.text(0.07, 0.90, title, fontsize=20, fontweight="bold", color=INK)
    y = 0.82
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


def _table_page(pp: PdfPages, df: pd.DataFrame, title: str, note: str = "", col_widths=None) -> None:
    _style()
    fig, ax = plt.subplots(figsize=(11, 8.5))
    ax.axis("off")
    ax.set_title(title, fontsize=16, loc="left", pad=20, color=INK)
    table = ax.table(
        cellText=[[_clip(v) for v in row] for row in df.itertuples(index=False)],
        colLabels=list(df.columns),
        loc="upper center",
        cellLoc="left",
        colWidths=col_widths,
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.5)
    for (row, _col), cell in table.get_celld().items():
        cell.set_edgecolor(GRID)
        if row == 0:
            cell.set_text_props(fontweight="bold", color=INK)
            cell.set_facecolor(SURFACE)
    if note:
        fig.text(0.07, 0.06, note, fontsize=8.5, color=MUTED)
    pp.savefig(fig)
    plt.close(fig)


def _forecast_page(pp: PdfPages, ctx: dict) -> None:
    _style()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8.5), height_ratios=[1, 1])

    monthly = ctx["monthly"]
    x = np.arange(len(monthly))
    ax1.plot(x, monthly["Revenue"].to_numpy() / 1e6, color=BLUE, linewidth=2, marker="o", markersize=3)
    ax1.set_xticks(x)
    ax1.set_xticklabels(monthly["YearMonth"], rotation=45, ha="right", fontsize=7)
    ax1.set_ylabel("revenue (GBP, m)")
    ax1.set_ylim(bottom=0)
    ax1.grid(axis="x", visible=False)
    ax1.set_title("Monthly revenue (cleaned sales; final month has 9 days of data)")

    final = ctx["final_fold"]
    idx = np.arange(len(final))
    ax2.plot(idx, final["actual"], color=INK, linewidth=2, label="actual", marker="o", markersize=4)
    for color, model in zip(CATEGORICAL, [c for c in final.columns if c != "actual"], strict=False):
        ax2.plot(idx, final[model], color=color, linewidth=1.6, label=model)
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
    fig.tight_layout()
    pp.savefig(fig)
    plt.close(fig)


def _rfm_page(pp: PdfPages, ctx: dict) -> None:
    _style()
    seg = ctx["rfm_summary"].sort_values("RevenueShare")
    fig, ax = plt.subplots(figsize=(11, 8.5))
    ax.grid(axis="y", visible=False)
    ax.barh(seg["Segment"], 100 * seg["RevenueShare"], color=BLUE, height=0.62)
    for i, (share, customers) in enumerate(zip(seg["RevenueShare"], seg["Customers"], strict=True)):
        ax.text(100 * share + 0.4, i, f"{100 * share:.1f}%  ({customers:,} customers)",
                va="center", fontsize=9, color=INK_2)
    ax.set_xlabel("share of identified-customer revenue (%)")
    ax.set_xlim(0, 100 * seg["RevenueShare"].max() * 1.28)
    ax.set_title(
        "RFM segments - revenue share (customers with a real CustomerID only; "
        f"{int(seg['Customers'].sum()):,} customers)"
    )
    pp.savefig(fig)
    plt.close(fig)


def _cohort_page(pp: PdfPages, ctx: dict) -> None:
    """Cohort retention heatmap (triangle) + size-weighted headline curve."""
    from retail.cohort import DISPLAY_OFFSETS

    result = ctx.get("cohort_result")
    if result is None or result.triangle.empty:
        return
    _style()
    tri = result.triangle
    ncol = min(DISPLAY_OFFSETS, int(tri.columns.max())) + 1
    mat = tri[[k for k in tri.columns if k < ncol]].to_numpy(dtype=float)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8.5), height_ratios=[2.4, 1])
    ax1.imshow(mat, aspect="auto", cmap="Blues", vmin=0.0, vmax=1.0)
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
                ax1.text(j, i, f"{round(100 * v)}", ha="center", va="center",
                         fontsize=6.5, color="white" if v > 0.55 else INK_2)
    ax1.set_title(
        f"Cohort repeat-purchase retention (% of each first-purchase cohort active later; "
        f"{result.n_customers:,} identified customers)", fontsize=11)

    curve = result.curve
    curve = curve[curve["offset"] < ncol]
    ax2.plot(curve["offset"], 100 * curve["avg_retention"], color=BLUE, linewidth=2, marker="o", markersize=4)
    ax2.set_xticks(range(ncol))
    ax2.set_ylim(0, 100)
    ax2.set_xlabel("months since first purchase")
    ax2.set_ylabel("repeat rate (%)")
    ax2.grid(axis="x", visible=False)
    ax2.set_title("Headline retention curve (size-weighted across observable cohorts)", fontsize=10)
    note = (f"Last complete month {result.last_complete_month}; "
            f"blank cells are right-censored (not yet observable). Offset 0 = acquisition month (100%).")
    fig.text(0.07, 0.045, note, fontsize=8, color=MUTED)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    pp.savefig(fig)
    plt.close(fig)


def _clv_page(pp: PdfPages, ctx: dict) -> None:
    """CLV page: out-of-sample validation bars + concentration Lorenz + params/headline."""
    from retail import clv

    result = ctx.get("clv_result")
    if result is None or result.clv_table.empty:
        return
    _style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 8.5))

    v = result.validation
    if v is not None and not v.by_frequency.empty:
        bf = v.by_frequency
        idx = np.arange(len(bf))
        w = 0.4
        ax1.bar(idx - w / 2, bf["actual_mean"], width=w, color=BLUE, label="actual")
        ax1.bar(idx + w / 2, bf["predicted_mean"], width=w, color=CATEGORICAL[1], label="predicted")
        ax1.set_xticks(idx)
        ax1.set_xticklabels(bf["cal_frequency"])
        ax1.set_xlabel("purchases in calibration period")
        ax1.set_ylabel("mean transactions in holdout")
        ax1.grid(axis="x", visible=False)
        ax1.legend(frameon=False, fontsize=9)
        ax1.set_title("Out-of-sample check (predicted vs actual)", fontsize=11)

    cum_x, cum_y = clv.lorenz_points(result.clv_table["clv"].to_numpy())
    ax2.plot(100 * cum_x, 100 * cum_y, color=BLUE, linewidth=2)
    ax2.plot([0, 100], [0, 100], color=MUTED, linewidth=1, linestyle="--")
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
    fig.text(0.07, 0.40, "\n".join(lines), fontsize=9.5, color=INK_2, va="top", family="monospace")

    top = clv.clv_summary_table(result, top=8).copy()
    top["monetary_value"] = top["monetary_value"].round(0)
    tbl_ax = fig.add_axes((0.07, 0.06, 0.86, 0.24))
    tbl_ax.axis("off")
    table = tbl_ax.table(
        cellText=[[_clip(v) for v in row] for row in top.itertuples(index=False)],
        colLabels=list(top.columns), loc="center", cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7.5)
    table.scale(1.0, 1.3)
    for (row, _col), cell in table.get_celld().items():
        cell.set_edgecolor(GRID)
        if row == 0:
            cell.set_text_props(fontweight="bold", color=INK)
            cell.set_facecolor(SURFACE)

    fig.suptitle("Predictive customer lifetime value", fontsize=16, x=0.07, ha="left", color=INK)
    fig.subplots_adjust(top=0.90, bottom=0.55, left=0.09, right=0.95, wspace=0.28)
    pp.savefig(fig)
    plt.close(fig)


def export_pdf(ctx: dict, path: Path | None = None) -> Path:
    path = path or DELIVERABLES / PDF_NAME
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = ctx["raw_report"]
    with PdfPages(path) as pp:
        _text_page(
            pp,
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
            "Cleaning impact - every decision with its row cost",
            note="Cancellations are separated (kept for returns analysis), not deleted. "
            "Missing CustomerID rows are flagged and kept for revenue, excluded from RFM.",
            col_widths=[0.24, 0.09, 0.09, 0.11, 0.09, 0.38],
        )
        _forecast_page(pp, ctx)
        _rfm_page(pp, ctx)
        _cohort_page(pp, ctx)
        _clv_page(pp, ctx)
        sku = ctx["sku_table"].copy()
        sku["Description"] = sku["Description"].astype(str).str.slice(0, 30)
        sku.columns = ["SKU", "Description", "Units", "Weeks", "Zero-wk share",
                       "MASE naive", "MASE s-naive", "MASE ma4", "Best model"]
        _table_page(
            pp,
            sku,
            "Top-10 SKU weekly demand - forecastability check",
            note="MASE > 1 means the model failed to beat a one-week naive walk on that SKU. "
            "Zero-wk share = share of zero-demand weeks since first sale (intermittency).",
            col_widths=[0.07, 0.25, 0.08, 0.07, 0.11, 0.09, 0.10, 0.08, 0.15],
        )
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
        _write_clv_sheet(writer, ctx)
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
