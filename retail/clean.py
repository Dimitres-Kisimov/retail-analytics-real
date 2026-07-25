"""Documented cleaning pipeline for Online Retail II.

Every step logs rows in -> rows out and the percentage removed; the resulting
table is what appears in the README and in the executive PDF. Decisions:

1.  Exact duplicate rows are dropped (same invoice, code, qty, timestamp, price,
    customer — almost certainly double-entry, not two identical orders logged in
    the same second).
2.  Cancellation invoices (Invoice starting with "C") are SEPARATED, not deleted:
    excluded from sales, kept as the returns frame for the returns analysis.
3.  Non-product StockCodes are removed from both frames. The explicit list
    (NON_PRODUCT_CODES) covers postage, manual entries, bank charges, discounts,
    adjustments, samples, carriage, charity, test rows — plus "gift_*" voucher
    codes by prefix. These are bookkeeping lines, not merchandise.
4.  Zero/negative-price rows are excluded from revenue (zero-price rows are
    mostly stock corrections and damaged/lost notes in the Description field).
5.  Remaining negative/zero-quantity rows (non-cancellation stock adjustments)
    are dropped from sales.
6.  Rows with missing CustomerID are FLAGGED (KnownCustomer=False), never
    dropped: they are real revenue (kept for revenue analyses) but unusable for
    customer analytics (excluded from RFM).

Derived columns: Revenue = Quantity * Price, Date, Year, Month, YearMonth, Week.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

# Bookkeeping StockCodes that are not merchandise. Kept explicit on purpose —
# every removal is attributable to a named code, not a fuzzy rule.
NON_PRODUCT_CODES = {
    "POST",          # postage
    "DOT",           # dotcom postage
    "C2",            # carriage
    "D",             # discount
    "M", "m",        # manual entry
    "S",             # samples
    "B",             # adjust bad debt
    "BANK CHARGES",  # bank charges
    "AMAZONFEE",     # Amazon fees
    "ADJUST", "ADJUST2",  # manual adjustments
    "CRUK",          # charity commission (Cancer Research UK)
    "TEST001", "TEST002",  # literal test rows left in the export
    "PADS",          # "pads to match all cushions" placeholder at GBP 0.001
}
GIFT_VOUCHER_PREFIX = "gift_"  # gift_0001_10 ... gift vouchers, not merchandise

# A duplicate is exact: same invoice, code, description, qty, timestamp, price, customer, country.
DEDUP_COLUMNS = [
    "Invoice", "StockCode", "Description", "Quantity", "InvoiceDate", "Price", "CustomerID", "Country",
]


def is_non_product(stock_code: pd.Series) -> pd.Series:
    """True for bookkeeping codes (explicit list) and gift-voucher codes (prefix)."""
    s = stock_code.astype(str)
    return s.isin(NON_PRODUCT_CODES) | s.str.lower().str.startswith(GIFT_VOUCHER_PREFIX)


@dataclass
class CleanResult:
    sales: pd.DataFrame        # cleaned sales rows (revenue-bearing)
    returns: pd.DataFrame      # cancellation rows, product codes only
    steps: list[dict] = field(default_factory=list)

    def report(self) -> pd.DataFrame:
        return clean_report(self.steps)


def _log(steps: list[dict], step: str, rows_in: int, rows_out: int, note: str) -> None:
    removed = rows_in - rows_out
    steps.append(
        {
            "step": step,
            "rows_in": rows_in,
            "rows_out": rows_out,
            "rows_removed": removed,
            "pct_removed": (removed / rows_in) if rows_in else 0.0,
            "note": note,
        }
    )


def clean(df: pd.DataFrame) -> CleanResult:
    """Run the full documented pipeline. Returns sales + returns + step log."""
    steps: list[dict] = []

    # 1. exact duplicates
    n0 = len(df)
    df = df.drop_duplicates(subset=DEDUP_COLUMNS).copy()
    _log(steps, "drop exact duplicates", n0, len(df), "identical invoice/code/qty/time/price/customer")

    # 2. separate cancellations
    n1 = len(df)
    is_cancel = df["Invoice"].str.startswith("C").fillna(False)
    returns = df.loc[is_cancel].copy()
    df = df.loc[~is_cancel].copy()
    _log(steps, "separate cancellations (C*)", n1, len(df), "kept as returns frame, excluded from sales")

    # 3. non-product StockCodes
    n2 = len(df)
    df = df.loc[~is_non_product(df["StockCode"])].copy()
    _log(steps, "remove non-product StockCodes", n2, len(df), "POST/D/M/DOT/BANK CHARGES/ADJUST/TEST/gift_ etc.")

    # 4. zero/negative prices
    n3 = len(df)
    df = df.loc[df["Price"] > 0].copy()
    _log(steps, "drop zero/negative-price rows", n3, len(df), "no revenue information; mostly stock notes")

    # 5. residual non-positive quantities (negative qty without a C invoice)
    n4 = len(df)
    df = df.loc[df["Quantity"] > 0].copy()
    _log(steps, "drop non-cancellation qty <= 0", n4, len(df), "stock adjustments outside C-invoices")

    # 6. flag (not drop) missing CustomerID
    n5 = len(df)
    df["KnownCustomer"] = df["CustomerID"].notna()
    flagged = int((~df["KnownCustomer"]).sum())
    _log(
        steps,
        "flag missing CustomerID",
        n5,
        len(df),
        f"{flagged:,} rows flagged KnownCustomer=False (kept for revenue, excluded from RFM)",
    )

    df = add_derived_columns(df)

    # returns frame: keep product codes only, add signed revenue
    returns = returns.loc[~is_non_product(returns["StockCode"])].copy()
    returns = add_derived_columns(returns)

    return CleanResult(sales=df.reset_index(drop=True), returns=returns.reset_index(drop=True), steps=steps)


def add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Revenue"] = df["Quantity"].astype("float64") * df["Price"]
    df["Date"] = df["InvoiceDate"].dt.normalize()
    df["Year"] = df["InvoiceDate"].dt.year
    df["Month"] = df["InvoiceDate"].dt.month
    df["YearMonth"] = df["InvoiceDate"].dt.strftime("%Y-%m")
    iso = df["InvoiceDate"].dt.isocalendar()
    df["ISOYear"] = iso["year"].astype("int64")
    df["ISOWeek"] = iso["week"].astype("int64")
    return df


def clean_report(steps: list[dict]) -> pd.DataFrame:
    """The cleaning-impact table: one row per step, rows in/out and % removed."""
    rep = pd.DataFrame(steps)
    rep["pct_removed"] = (100.0 * rep["pct_removed"]).round(2)
    return rep[["step", "rows_in", "rows_out", "rows_removed", "pct_removed", "note"]]
