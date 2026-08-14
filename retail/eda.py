"""Honest EDA figures, saved to figures/ as numbered plates.

No cherry-picking: the November/December gift-season spikes are annotated, and
so is the fact that the final month (December 2011) contains only nine days of
trading — a naive month-over-month read of the tail would be wrong.

All figures are light-mode PNGs set in the shared "field notes" plate system
(``retail.plate``): numbered plates, hairline chrome, one validated palette
(single-hue blue for measured magnitude, green for the returns family), and the
dataset attribution designed into every footer. Reference statistics (median,
mean) are drawn dotted so they never read as data marks.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from retail import plate
from retail.paths import FIGURES
from retail.plate import BLUE, GREEN, INK_2, MUTED, SURFACE


def _new_axes(figsize: tuple[float, float] = (9, 4.9)) -> tuple[plt.Figure, plt.Axes]:
    plate.style()
    fig, ax = plt.subplots(figsize=figsize)
    ax.grid(axis="x", visible=False)
    return fig, ax


def monthly_revenue_table(sales: pd.DataFrame) -> pd.DataFrame:
    monthly = sales.groupby("YearMonth", observed=True).agg(
        Revenue=("Revenue", "sum"), Invoices=("Invoice", "nunique"), Rows=("Revenue", "size")
    )
    return monthly.round(2).reset_index()


def fig_monthly_revenue(sales: pd.DataFrame, out_dir: Path = FIGURES) -> Path:
    monthly = sales.groupby("YearMonth", observed=True)["Revenue"].sum()
    fig, ax = _new_axes()
    x = np.arange(len(monthly))
    ax.plot(x, monthly.to_numpy() / 1e6, color=BLUE, linewidth=2, marker="o", markersize=4)
    ax.set_xticks(x)
    ax.set_xticklabels(monthly.index, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("revenue (GBP, millions)")
    ymax = float(monthly.max()) / 1e6
    ax.set_ylim(0, ymax * 1.22)  # headroom so annotations stay inside the axes
    ax.set_title("Monthly revenue, Dec 2009 - Dec 2011 (cleaned sales)")

    # honest annotations: gift-season spikes + incomplete final month
    for peak in ("2010-11", "2011-11"):
        if peak in monthly.index:
            i = monthly.index.get_loc(peak)
            ax.annotate(
                "gift-season peak",
                xy=(i, monthly.iloc[i] / 1e6),
                xytext=(i - 4.5, monthly.iloc[i] / 1e6 + 0.06 * ymax),
                fontsize=8,
                color=INK_2,
                arrowprops={"arrowstyle": "-", "color": MUTED, "linewidth": 0.8},
            )
    if "2011-12" in monthly.index:
        i = monthly.index.get_loc("2011-12")
        ax.annotate(
            "only 9 days of data",
            xy=(i, monthly.iloc[i] / 1e6),
            xytext=(i - 5.5, monthly.iloc[i] / 1e6 - 0.25),
            fontsize=8,
            color=INK_2,
            arrowprops={"arrowstyle": "-", "color": MUTED, "linewidth": 0.8},
        )
    rect = plate.chrome(fig, "monthly_revenue")
    return plate.save(fig, out_dir / "monthly_revenue.png", rect)


def fig_top_products(sales: pd.DataFrame, out_dir: Path = FIGURES, n: int = 10) -> Path:
    top = (
        sales.groupby(["StockCode", "Description"], observed=True)["Revenue"]
        .sum()
        .sort_values(ascending=False)
        .head(n)
    )
    labels = [str(desc)[:32].strip().title() for _, desc in top.index][::-1]
    values = (top.to_numpy() / 1e3)[::-1]
    fig, ax = _new_axes(figsize=(9, 5.2))
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", visible=True)
    ax.barh(labels, values, color=BLUE, height=0.62)
    ax.set_xlabel("revenue (GBP, thousands)")
    ax.set_title(f"Top {n} products by revenue")
    for i, v in enumerate(values):
        ax.text(v, i, f" {v:,.0f}", va="center", fontsize=8, color=INK_2)
    notes: tuple[str, ...] = ()
    if any("PAPER CRAFT" in str(desc) for _, desc in top.index):
        notes = (
            'Honesty note: "Paper Craft, Little Birdie" is a single 80,995-unit order placed and '
            "cancelled the same day (2011-12-09);\nit stays in gross sales because cancellations are "
            "kept separately in the returns frame.",
        )
    rect = plate.chrome(fig, "top_products", notes=notes)
    return plate.save(fig, out_dir / "top_products.png", rect)


def fig_top_countries(sales: pd.DataFrame, out_dir: Path = FIGURES, n: int = 10) -> Path:
    by_country = sales.groupby("Country", observed=True)["Revenue"].sum().sort_values(ascending=False)
    uk_share = by_country.get("United Kingdom", 0.0) / by_country.sum()
    top = by_country.head(n)
    fig, ax = _new_axes(figsize=(9, 5.2))
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", visible=True)
    ax.barh(top.index[::-1], (top.to_numpy() / 1e6)[::-1], color=BLUE, height=0.62)
    ax.set_xlabel("revenue (GBP, millions)")
    ax.set_title(
        f"Top {n} countries by revenue - the UK is {100 * uk_share:.0f}% of the total"
    )
    for i, v in enumerate((top.to_numpy() / 1e6)[::-1]):
        ax.text(v, i, f" {v:,.2f}", va="center", fontsize=8, color=INK_2)
    rect = plate.chrome(fig, "top_countries")
    return plate.save(fig, out_dir / "top_countries.png", rect)


def fig_order_values(sales: pd.DataFrame, out_dir: Path = FIGURES) -> Path:
    order_values = sales.groupby("Invoice", observed=True)["Revenue"].sum()
    order_values = order_values[order_values > 0]
    bins = np.logspace(0, np.log10(order_values.max()), 60)
    fig, ax = _new_axes()
    ax.hist(order_values, bins=bins, color=BLUE, edgecolor=SURFACE, linewidth=0.4)
    ax.set_xscale("log")
    ax.set_xlabel("order value (GBP, log scale)")
    ax.set_ylabel("orders")
    med = float(order_values.median())
    ax.axvline(med, color=MUTED, linewidth=1, linestyle=":")
    ax.text(med * 1.15, ax.get_ylim()[1] * 0.92, f"median {med:,.0f}", fontsize=8, color=INK_2)
    ax.set_title("Order-size distribution (invoice totals, log scale)")
    rect = plate.chrome(fig, "order_values")
    return plate.save(fig, out_dir / "order_values.png", rect)


def returns_rate_table(sales: pd.DataFrame, returns: pd.DataFrame) -> pd.Series:
    """Monthly returned value as a share of gross sales value (by YearMonth)."""
    gross = sales.groupby("YearMonth", observed=True)["Revenue"].sum()
    returned = (-returns.groupby("YearMonth", observed=True)["Revenue"].sum()).reindex(
        gross.index, fill_value=0.0
    )
    return (returned / gross).rename("ReturnsRate")


def fig_returns_rate(sales: pd.DataFrame, returns: pd.DataFrame, out_dir: Path = FIGURES) -> Path:
    rate = returns_rate_table(sales, returns)
    fig, ax = _new_axes()
    x = np.arange(len(rate))
    ax.plot(x, 100 * rate.to_numpy(), color=GREEN, linewidth=2, marker="o", markersize=4)
    ax.set_xticks(x)
    ax.set_xticklabels(rate.index, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("returned value / gross sales (%)")
    ax.set_ylim(bottom=0)
    mean_rate = 100 * float(rate.mean())
    ax.axhline(mean_rate, color=MUTED, linewidth=1, linestyle=":")
    ax.text(0.2, mean_rate + 0.25, f"mean {mean_rate:.1f}%", fontsize=8, color=INK_2)
    if "2011-12" in rate.index and rate["2011-12"] > 3 * rate.mean():
        i = rate.index.get_loc("2011-12")
        ax.annotate(
            "9-day month + one 80,995-unit\ncancellation (see README)",
            xy=(i, 100 * rate.iloc[i]),
            xytext=(i - 7.5, 100 * rate.iloc[i] - 3.5),
            fontsize=8,
            color=INK_2,
            arrowprops={"arrowstyle": "-", "color": MUTED, "linewidth": 0.8},
        )
    ax.set_title("Returns rate over time (cancellation invoices vs gross sales)")
    rect = plate.chrome(fig, "returns_rate")
    return plate.save(fig, out_dir / "returns_rate.png", rect)


def fig_missing_customers(sales: pd.DataFrame, out_dir: Path = FIGURES) -> Path:
    share = 100 * (
        1.0 - sales.groupby("YearMonth", observed=True)["KnownCustomer"].mean()
    )
    fig, ax = _new_axes()
    x = np.arange(len(share))
    ax.plot(x, share.to_numpy(), color=BLUE, linewidth=2, marker="o", markersize=4)
    ax.set_xticks(x)
    ax.set_xticklabels(share.index, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("rows without CustomerID (%)")
    ax.set_ylim(bottom=0)
    ax.set_title("Missing-CustomerID share by month (guest checkouts / unlinked tills)")
    rect = plate.chrome(fig, "missing_customers")
    return plate.save(fig, out_dir / "missing_customer_share.png", rect)


def make_all_figures(
    sales: pd.DataFrame, returns: pd.DataFrame, out_dir: Path = FIGURES
) -> list[Path]:
    return [
        fig_monthly_revenue(sales, out_dir),
        fig_top_products(sales, out_dir),
        fig_top_countries(sales, out_dir),
        fig_order_values(sales, out_dir),
        fig_returns_rate(sales, returns, out_dir),
        fig_missing_customers(sales, out_dir),
    ]
