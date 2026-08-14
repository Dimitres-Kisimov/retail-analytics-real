"""Returns & cancellations analysis on the cleaned real transactions.

The cleaning pipeline (``retail.clean``) does not throw cancellation invoices
away: it *separates* the ``C``-prefixed rows into a first-class ``returns`` frame
(product codes only, signed revenue). This module is what that frame was kept
for -- a proper returns analysis, computed from scratch (numpy / pandas only) on
the same cleaned frames every other module uses. Nothing here is modelled; every
number is measured off the real data.

What it measures
----------------
* **Headline.** Returned value and units as a share of gross sales (the return
  rate), net-of-returns revenue, and the counts behind them.
* **Return-to-sale matching (reverse-logistics lag).** Online Retail II credit
  notes carry *no reference* to the invoice they cancel, so a return is attributed
  to a sale heuristically: the most recent prior sale of the **same StockCode to
  the same CustomerID** on or before the cancellation date. The share of returns
  that can be matched this way, and the distribution of days between purchase and
  return, are reported -- as a **measured heuristic, not a linked RMA**. Returns
  with no CustomerID (they cannot be attributed to a buyer) are counted and
  reported separately, never silently matched.
* **By product.** Per-SKU sold vs returned units and value, and the resulting
  unit/value return rates -- so the products that come back can be ranked.
* **Concentration.** How much of the returned value sits in the top few SKUs.
* **By country and by month.** Return rate by country; gross vs net revenue by
  month.

Honesty guards
--------------
* Cancellation ``Quantity`` and ``Revenue`` are negative in the returns frame;
  every "returned" figure below is their sign-flipped magnitude, so a return rate
  is always a non-negative share.
* A return rate is ``returned / gross`` over the **same cleaned universe** -- it
  is *not* a per-order refund probability, and because a credit note can post in a
  different month (or year) than its sale, the monthly rate can exceed 100% in a
  quiet or truncated month. That is a timing artefact, and it is annotated, not
  hidden.
* Matching is a heuristic attribution; unmatched and unidentified returns are
  first-class rows in the report, not dropped.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from retail.paths import DELIVERABLES, FIGURES

# Ordered reverse-logistics lag buckets (days between sale and its return).
LAG_BUCKETS: tuple[tuple[str, int, float], ...] = (
    ("same day", 0, 0.0),
    ("1-7 days", 1, 7.0),
    ("8-30 days", 8, 30.0),
    ("31-90 days", 31, 90.0),
    ("91-365 days", 91, 365.0),
    ("over 365 days", 366, float("inf")),
)


@dataclass
class ReturnsResult:
    """Everything the report needs, all derived from the cleaned sales + returns frames."""

    overview: dict                 # headline scalars (gross/returned/net, rates, counts)
    by_sku: pd.DataFrame           # per-SKU sold vs returned, return rates (returned SKUs only)
    lag: pd.DataFrame              # reverse-logistics lag buckets: lines + value per bucket
    match: dict                    # matching stats (matched share of lines/value, lag percentiles)
    by_country: pd.DataFrame       # return rate by country (top by gross value)
    monthly: pd.DataFrame          # YearMonth: gross, returned, net, value return rate
    concentration: dict            # top-N SKU share of returned value

    def headline(self) -> dict:
        """The handful of numbers the one-line headline quotes."""
        o = self.overview
        return {
            "value_return_rate": o["value_return_rate"],
            "unit_return_rate": o["unit_return_rate"],
            "matched_value_share": self.match.get("matched_value_share", 0.0),
            "median_lag_days": self.match.get("median_lag_days", float("nan")),
        }


# --------------------------------------------------------------------------- #
# Core measurements
# --------------------------------------------------------------------------- #
def _returned_value(returns: pd.DataFrame) -> float:
    return float(-returns["Revenue"].sum()) if len(returns) else 0.0


def _returned_units(returns: pd.DataFrame) -> int:
    return int(-returns["Quantity"].sum()) if len(returns) else 0


def returns_overview(sales: pd.DataFrame, returns: pd.DataFrame) -> dict:
    """Headline scalars: gross vs returned vs net, return rates, and their counts."""
    gross_value = float(sales["Revenue"].sum()) if len(sales) else 0.0
    gross_units = int(sales["Quantity"].sum()) if len(sales) else 0
    returned_value = _returned_value(returns)
    returned_units = _returned_units(returns)
    return {
        "gross_value": gross_value,
        "gross_units": gross_units,
        "gross_lines": int(len(sales)),
        "gross_invoices": int(sales["Invoice"].nunique()) if len(sales) else 0,
        "returned_value": returned_value,
        "returned_units": returned_units,
        "return_lines": int(len(returns)),
        "cancellation_invoices": int(returns["Invoice"].nunique()) if len(returns) else 0,
        "net_value": gross_value - returned_value,
        "value_return_rate": (returned_value / gross_value) if gross_value else 0.0,
        "unit_return_rate": (returned_units / gross_units) if gross_units else 0.0,
    }


def _sku_descriptions(sales: pd.DataFrame, returns: pd.DataFrame) -> pd.Series:
    """StockCode -> its most frequent non-null description (deterministic tie-break)."""
    parts = []
    for df in (sales, returns):
        if len(df):
            sub = df[["StockCode", "Description"]].copy()
            sub = sub[sub["Description"].notna()]
            parts.append(sub)
    if not parts:
        return pd.Series(dtype="object", name="Description")
    alld = pd.concat(parts, ignore_index=True)
    alld["Description"] = alld["Description"].astype(str).str.strip()
    counts = alld.groupby(["StockCode", "Description"]).size().rename("n").reset_index()
    counts = counts.sort_values(["StockCode", "n", "Description"], ascending=[True, False, True])
    return counts.drop_duplicates("StockCode").set_index("StockCode")["Description"]


def returns_by_sku(sales: pd.DataFrame, returns: pd.DataFrame) -> pd.DataFrame:
    """Per-SKU sold vs returned units/value and the resulting return rates.

    Only SKUs that were actually returned appear (that is the point of the table);
    a returned SKU that never shows up in cleaned sales gets 0 sold and a NaN rate.
    """
    if not len(returns):
        return pd.DataFrame(
            columns=["StockCode", "Description", "sold_units", "sold_value",
                     "returned_units", "returned_value", "unit_return_rate", "value_return_rate"]
        )
    sold = sales.groupby("StockCode", observed=True).agg(
        sold_units=("Quantity", "sum"), sold_value=("Revenue", "sum")
    )
    ret = returns.groupby("StockCode", observed=True).agg(
        returned_units=("Quantity", "sum"), returned_value=("Revenue", "sum")
    )
    ret["returned_units"] = -ret["returned_units"]
    ret["returned_value"] = -ret["returned_value"]

    out = ret.join(sold, how="left")
    out["sold_units"] = out["sold_units"].fillna(0).astype("int64")
    out["sold_value"] = out["sold_value"].fillna(0.0)
    out["returned_units"] = out["returned_units"].astype("int64")

    with np.errstate(divide="ignore", invalid="ignore"):
        out["unit_return_rate"] = np.where(
            out["sold_units"] > 0, out["returned_units"] / out["sold_units"], np.nan
        )
        out["value_return_rate"] = np.where(
            out["sold_value"] > 0, out["returned_value"] / out["sold_value"], np.nan
        )

    desc = _sku_descriptions(sales, returns)
    out = out.reset_index()
    out["Description"] = out["StockCode"].map(desc).fillna("")
    out = out[["StockCode", "Description", "sold_units", "sold_value",
               "returned_units", "returned_value", "unit_return_rate", "value_return_rate"]]
    # Deterministic order: biggest returned value first, StockCode as tie-break.
    return out.sort_values(["returned_value", "StockCode"], ascending=[False, True]).reset_index(drop=True)


def match_returns_to_sales(sales: pd.DataFrame, returns: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Attribute each return to its most recent prior same-customer, same-SKU sale.

    Uses a backward as-of join on the cancellation date within each
    (CustomerID, StockCode) group. Returns the per-line matched frame and a stats
    dict. This is a **heuristic** attribution -- the data carries no link from a
    credit note to its originating invoice -- so matched/unmatched/unidentified
    shares are all reported, never silently assumed.
    """
    stats: dict = {
        "return_lines": int(len(returns)),
        "identified_lines": 0,
        "matched_lines": 0,
        "matched_value_share": 0.0,
        "matched_line_share": 0.0,
        "median_lag_days": float("nan"),
        "mean_lag_days": float("nan"),
        "p25_lag_days": float("nan"),
        "p75_lag_days": float("nan"),
    }
    total_returned_value = _returned_value(returns)
    empty = pd.DataFrame(columns=["StockCode", "CustomerID", "InvoiceDate",
                                   "returned_value", "SaleDate", "days_to_return"])
    if not len(returns):
        return empty, stats

    ret = returns.loc[returns["CustomerID"].notna(),
                      ["CustomerID", "StockCode", "InvoiceDate", "Revenue"]].copy()
    stats["identified_lines"] = int(len(ret))
    if not len(ret):
        return empty, stats
    ret["returned_value"] = -ret["Revenue"]
    ret["CustomerID"] = ret["CustomerID"].astype("int64")

    sale = sales.loc[sales["CustomerID"].notna(), ["CustomerID", "StockCode", "InvoiceDate"]].copy()
    sale["CustomerID"] = sale["CustomerID"].astype("int64")
    sale = sale.rename(columns={"InvoiceDate": "SaleDate"}).sort_values("SaleDate", kind="stable")
    ret = ret.sort_values("InvoiceDate", kind="stable")

    merged = pd.merge_asof(
        ret, sale,
        left_on="InvoiceDate", right_on="SaleDate",
        by=["CustomerID", "StockCode"], direction="backward",
    )
    merged["days_to_return"] = (merged["InvoiceDate"] - merged["SaleDate"]).dt.days
    matched = merged[merged["SaleDate"].notna()].copy()

    stats["matched_lines"] = int(len(matched))
    stats["matched_line_share"] = len(matched) / len(returns) if len(returns) else 0.0
    stats["matched_value_share"] = (
        float(matched["returned_value"].sum()) / total_returned_value if total_returned_value else 0.0
    )
    if len(matched):
        d = matched["days_to_return"].to_numpy(dtype="float64")
        stats["median_lag_days"] = float(np.median(d))
        stats["mean_lag_days"] = float(np.mean(d))
        stats["p25_lag_days"] = float(np.percentile(d, 25))
        stats["p75_lag_days"] = float(np.percentile(d, 75))
    return matched[["StockCode", "CustomerID", "InvoiceDate",
                    "returned_value", "SaleDate", "days_to_return"]].reset_index(drop=True), stats


def lag_buckets(matched: pd.DataFrame) -> pd.DataFrame:
    """Reverse-logistics lag: matched return lines and their value per lag bucket."""
    rows = []
    d = matched["days_to_return"].to_numpy(dtype="float64") if len(matched) else np.array([])
    v = matched["returned_value"].to_numpy(dtype="float64") if len(matched) else np.array([])
    for label, lo, hi in LAG_BUCKETS:
        mask = (d >= lo) & (d <= hi)
        rows.append({
            "lag_bucket": label,
            "lines": int(mask.sum()),
            "returned_value": float(v[mask].sum()) if mask.any() else 0.0,
        })
    out = pd.DataFrame(rows)
    total = out["lines"].sum()
    out["line_share"] = out["lines"] / total if total else 0.0
    return out


def returns_by_country(sales: pd.DataFrame, returns: pd.DataFrame, top: int = 10) -> pd.DataFrame:
    """Return rate by country (top ``top`` countries by gross sales value)."""
    if not len(sales):
        return pd.DataFrame(columns=["Country", "sold_value", "returned_value", "value_return_rate"])
    sold = sales.groupby("Country", observed=True)["Revenue"].sum().rename("sold_value")
    ret = (-returns.groupby("Country", observed=True)["Revenue"].sum()).rename("returned_value")
    out = pd.concat([sold, ret], axis=1).fillna({"returned_value": 0.0})
    out = out.dropna(subset=["sold_value"])
    out["value_return_rate"] = out["returned_value"] / out["sold_value"]
    out = out.sort_values("sold_value", ascending=False).head(top).reset_index()
    return out


def monthly_returns(sales: pd.DataFrame, returns: pd.DataFrame) -> pd.DataFrame:
    """Gross, returned, net revenue and the return rate by calendar month."""
    if not len(sales):
        return pd.DataFrame(columns=["YearMonth", "gross", "returned", "net", "value_return_rate"])
    gross = sales.groupby("YearMonth", observed=True)["Revenue"].sum()
    returned = (-returns.groupby("YearMonth", observed=True)["Revenue"].sum()).reindex(
        gross.index, fill_value=0.0
    ) if len(returns) else pd.Series(0.0, index=gross.index)
    out = pd.DataFrame({"gross": gross, "returned": returned})
    out["net"] = out["gross"] - out["returned"]
    out["value_return_rate"] = out["returned"] / out["gross"]
    return out.reset_index()


def _concentration(by_sku: pd.DataFrame, top: int = 10) -> dict:
    vals = by_sku["returned_value"].to_numpy(dtype="float64")
    total = float(vals.sum())
    if total <= 0:
        return {"top_n": top, "top_n_share": 0.0, "skus_to_80pct": 0, "n_returned_skus": int(len(vals))}
    top_share = float(vals[:top].sum()) / total
    cum = np.cumsum(vals) / total
    skus_to_80 = int(np.searchsorted(cum, 0.80) + 1)
    return {
        "top_n": top,
        "top_n_share": top_share,
        "skus_to_80pct": skus_to_80,
        "n_returned_skus": int(len(vals)),
    }


def run_returns_analysis(sales: pd.DataFrame, returns: pd.DataFrame) -> ReturnsResult:
    """Compute the full returns analysis from the cleaned sales + returns frames."""
    overview = returns_overview(sales, returns)
    by_sku = returns_by_sku(sales, returns)
    matched, match_stats = match_returns_to_sales(sales, returns)
    lag = lag_buckets(matched)
    by_country = returns_by_country(sales, returns)
    monthly = monthly_returns(sales, returns)
    concentration = _concentration(by_sku)
    return ReturnsResult(
        overview=overview,
        by_sku=by_sku,
        lag=lag,
        match=match_stats,
        by_country=by_country,
        monthly=monthly,
        concentration=concentration,
    )


# --------------------------------------------------------------------------- #
# Exports
# --------------------------------------------------------------------------- #
def sku_export_frame(result: ReturnsResult) -> pd.DataFrame:
    """Tidy, rounded per-SKU returns table for the CSV / Excel deliverable."""
    df = result.by_sku.copy()
    df["sold_value"] = df["sold_value"].round(2)
    df["returned_value"] = df["returned_value"].round(2)
    df["unit_return_rate"] = (100.0 * df["unit_return_rate"]).round(2)
    df["value_return_rate"] = (100.0 * df["value_return_rate"]).round(2)
    df = df.rename(columns={"unit_return_rate": "unit_return_rate_pct",
                            "value_return_rate": "value_return_rate_pct"})
    return df


def write_csv(result: ReturnsResult, out_dir: Path = DELIVERABLES) -> Path:
    """Write the per-SKU returns table. Deterministic -> byte-identical on re-run."""
    out = out_dir / "returns_analysis.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    sku_export_frame(result).to_csv(out, index=False, encoding="utf-8", lineterminator="\n")
    return out


def headline_text(result: ReturnsResult) -> str:
    """One measured line, e.g. 'returns run at 3.65% of gross value ...'."""
    hl = result.headline()
    vr = 100 * hl["value_return_rate"]
    ur = 100 * hl["unit_return_rate"]
    mv = 100 * hl["matched_value_share"]
    med = hl["median_lag_days"]
    lag = f", median {med:.0f} days to return" if np.isfinite(med) else ""
    return (f"returns run at {vr:.2f}% of gross value ({ur:.2f}% of units); "
            f"{mv:.1f}% of returned value matches a prior purchase{lag}")


# --------------------------------------------------------------------------- #
# Figure (matplotlib PNG, committed as visual proof like the other analytics)
# --------------------------------------------------------------------------- #
def fig_returns(result: ReturnsResult, out_dir: Path = FIGURES) -> Path | None:
    """Two panels: reverse-logistics lag distribution + top returned SKUs by value.

    Returns ``None`` when there is nothing to plot (no returns in the input).
    """
    if result.overview["returned_value"] <= 0 or result.by_sku.empty:
        return None

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from retail import plate
    from retail.plate import BERRY, INK, INK_2

    plate.style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5.6))

    lag = result.lag
    labels = lag["lag_bucket"].tolist()
    y = np.arange(len(labels))
    # Both panels measure the same quantity (returned value), so both wear the
    # returns family's berry -- color follows the entity across the plate set.
    ax1.barh(y, lag["returned_value"].to_numpy() / 1e3, color=BERRY, height=0.66)
    ax1.set_yticks(y)
    ax1.set_yticklabels(labels, fontsize=9)
    ax1.invert_yaxis()
    ax1.grid(axis="y", visible=False)
    ax1.set_xlabel("matched returned value (GBP, thousands)")
    ax1.set_title("Time from purchase to return", fontsize=11)
    med = result.match.get("median_lag_days", float("nan"))
    mshare = 100 * result.match.get("matched_value_share", 0.0)
    if np.isfinite(med):
        ax1.text(0.98, 0.03,
                 f"median {med:.0f} days  |  {mshare:.0f}% of returned\nvalue matched to a prior sale",
                 transform=ax1.transAxes, ha="right", va="bottom", fontsize=8.5, color=INK_2)

    top = result.by_sku.head(10).iloc[::-1]
    desc = [str(d)[:28].strip().title() if d else str(c)
            for c, d in zip(top["StockCode"], top["Description"], strict=True)]
    ax2.barh(np.arange(len(top)), top["returned_value"].to_numpy() / 1e3, color=BERRY, height=0.66)
    ax2.set_yticks(np.arange(len(top)))
    ax2.set_yticklabels(desc, fontsize=8)
    ax2.grid(axis="y", visible=False)
    ax2.set_xlabel("returned value (GBP, thousands)")
    ax2.set_title("Top 10 SKUs by returned value", fontsize=11)
    for i, (v, r) in enumerate(zip(top["returned_value"] / 1e3, top["value_return_rate"], strict=True)):
        tag = f" {v:,.0f}" + (f"  ({100 * r:.0f}% of sales)" if np.isfinite(r) else "")
        ax2.text(v, i, tag, va="center", fontsize=7.5, color=INK_2)

    o = result.overview
    rect = plate.chrome(
        fig, "returns",
        notes=("Real cancellation invoices; matching is a same-customer/same-SKU heuristic, "
               "not a linked RMA.",),
    )
    fig.suptitle(
        f"Returns & cancellations - {100 * o['value_return_rate']:.2f}% of gross value returned "
        f"(net GBP {o['net_value']:,.0f} of {o['gross_value']:,.0f})",
        fontsize=13, x=0.045, y=rect[3] - 0.012, ha="left", color=INK, fontweight="bold",
    )
    for spine in ("top", "right"):
        for ax in (ax1, ax2):
            ax.spines[spine].set_visible(False)
    return plate.save(fig, out_dir / "returns_analysis.png",
                      (rect[0], rect[1], rect[2], rect[3] - 0.065))
