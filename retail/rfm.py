"""From-scratch RFM segmentation on customers with a real CustomerID.

Recency  = days between a customer's last invoice and the snapshot date
           (one day after the last invoice in the data).
Frequency = number of distinct invoices.
Monetary  = total revenue.

Scores are quintiles 1..5 computed by ranking (rank method="first" breaks ties
deterministically, so exactly ~20% of customers land in each bin even with
heavily tied frequencies). R is reversed: recent purchase = high score.

Segments follow the standard RFM grid on (R, F) — all 25 cells are mapped:

    R 4-5, F 4-5  -> Champions            R 3, F 2-3 -> Need Attention
    R 3,   F 4-5  -> Loyal Customers      R 3, F 1   -> About to Sleep
    R 4-5, F 2-3  -> Potential Loyalist   R 1, F 5   -> Can't Lose Them
    R 4-5, F 1    -> New Customers        R 1-2, F 4-5 -> At Risk
                                          R 1-2, F 2-3 -> Hibernating
                                          R 1-2, F 1   -> Lost
"""

from __future__ import annotations

import pandas as pd

SEGMENT_ORDER = [
    "Champions",
    "Loyal Customers",
    "Potential Loyalist",
    "New Customers",
    "Need Attention",
    "About to Sleep",
    "At Risk",
    "Can't Lose Them",
    "Hibernating",
    "Lost",
]


def rfm_table(sales: pd.DataFrame, snapshot: pd.Timestamp | None = None) -> pd.DataFrame:
    """Per-customer R/F/M values from cleaned sales (KnownCustomer rows only)."""
    known = sales.loc[sales["CustomerID"].notna()]
    if snapshot is None:
        snapshot = known["InvoiceDate"].max() + pd.Timedelta(days=1)
    grouped = known.groupby("CustomerID", observed=True)
    out = pd.DataFrame(
        {
            "RecencyDays": (snapshot - grouped["InvoiceDate"].max()).dt.days,
            "Frequency": grouped["Invoice"].nunique(),
            "Monetary": grouped["Revenue"].sum(),
        }
    )
    out.index.name = "CustomerID"
    return out.reset_index()


def quintile_scores(rfm: pd.DataFrame) -> pd.DataFrame:
    """Add R, F, M quintile scores (1..5). Rank-based so ties split deterministically."""
    rfm = rfm.copy()

    def _quintile(values: pd.Series, ascending: bool) -> pd.Series:
        ranks = values.rank(method="first", ascending=ascending)
        return pd.qcut(ranks, 5, labels=False).astype("int64") + 1

    rfm["R"] = _quintile(rfm["RecencyDays"], ascending=False)  # low recency -> rank high -> 5
    rfm["F"] = _quintile(rfm["Frequency"], ascending=True)
    rfm["M"] = _quintile(rfm["Monetary"], ascending=True)
    return rfm


def segment_name(r: int, f: int) -> str:
    """Map an (R, F) score pair to its named segment. Covers all 25 cells."""
    if r >= 4 and f >= 4:
        return "Champions"
    if r == 3 and f >= 4:
        return "Loyal Customers"
    if r >= 4 and f >= 2:
        return "Potential Loyalist"
    if r >= 4:
        return "New Customers"
    if r == 3 and f >= 2:
        return "Need Attention"
    if r == 3:
        return "About to Sleep"
    if r == 1 and f == 5:
        return "Can't Lose Them"
    if f >= 4:
        return "At Risk"
    if f >= 2:
        return "Hibernating"
    return "Lost"


def assign_segments(rfm: pd.DataFrame) -> pd.DataFrame:
    rfm = rfm.copy()
    rfm["Segment"] = [segment_name(r, f) for r, f in zip(rfm["R"], rfm["F"], strict=True)]
    return rfm


def segment_summary(rfm: pd.DataFrame) -> pd.DataFrame:
    """Customers, revenue and revenue share per segment, ordered by the grid."""
    total_rev = float(rfm["Monetary"].sum())
    grouped = rfm.groupby("Segment", observed=True)
    summary = pd.DataFrame(
        {
            "Customers": grouped.size(),
            "Revenue": grouped["Monetary"].sum().round(2),
            "AvgRecencyDays": grouped["RecencyDays"].mean().round(1),
            "AvgFrequency": grouped["Frequency"].mean().round(2),
            "AvgMonetary": grouped["Monetary"].mean().round(2),
        }
    )
    summary["RevenueShare"] = (summary["Revenue"] / total_rev).round(4)
    summary = summary.reindex([s for s in SEGMENT_ORDER if s in summary.index])
    summary.index.name = "Segment"
    return summary.reset_index()


def run_rfm(sales: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Full RFM pass: (per-customer table with scores+segments, segment summary)."""
    rfm = assign_segments(quintile_scores(rfm_table(sales)))
    return rfm, segment_summary(rfm)
