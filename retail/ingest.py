"""Load the raw Online Retail II workbook and report its (real) quality problems.

The workbook has two sheets — "Year 2009-2010" and "Year 2010-2011" — which we
concatenate into one frame. Reading ~1M rows through openpyxl takes minutes, so
the concatenated, type-cast frame is cached to ``data/interim/`` (parquet when
pyarrow is available, gzipped CSV otherwise). Both cache files are git-ignored.

Column names are normalised to: Invoice, StockCode, Description, Quantity,
InvoiceDate, Price, CustomerID, Country (the raw sheets use "Customer ID").
"""

from __future__ import annotations

import pandas as pd

from retail import clean as _clean
from retail.paths import DATA_INTERIM, INTERIM_CSV, INTERIM_PARQUET, RAW_XLSX

RENAME = {"Customer ID": "CustomerID"}
COLUMNS = ["Invoice", "StockCode", "Description", "Quantity", "InvoiceDate", "Price", "CustomerID", "Country"]


def _type_cast(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns=RENAME)
    out = pd.DataFrame(
        {
            "Invoice": df["Invoice"].astype(str),
            "StockCode": df["StockCode"].astype(str),
            "Description": df["Description"].astype(str).where(df["Description"].notna()),
            "Quantity": pd.to_numeric(df["Quantity"], errors="coerce").astype("int64"),
            "InvoiceDate": pd.to_datetime(df["InvoiceDate"], errors="coerce"),
            "Price": pd.to_numeric(df["Price"], errors="coerce").astype("float64"),
            "CustomerID": pd.to_numeric(df["CustomerID"], errors="coerce").astype("Int64"),
            "Country": df["Country"].astype(str),
        }
    )
    return out


def load_raw(force: bool = False, use_cache: bool = True) -> pd.DataFrame:
    """Return the combined 2009-2011 frame, reading from the interim cache when possible."""
    if use_cache and not force:
        if INTERIM_PARQUET.exists():
            return pd.read_parquet(INTERIM_PARQUET)
        if INTERIM_CSV.exists():
            df = pd.read_csv(INTERIM_CSV, encoding="utf-8")
            df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
            df["CustomerID"] = df["CustomerID"].astype("Int64")
            return df

    if not RAW_XLSX.exists():
        raise FileNotFoundError(
            f"{RAW_XLSX} not found. Run `python scripts/download_data.py` first "
            "(the raw xlsx is deliberately not committed)."
        )

    sheets = pd.read_excel(RAW_XLSX, sheet_name=None, engine="openpyxl")
    frames = []
    for name, sheet in sheets.items():
        cast = _type_cast(sheet)
        cast["SourceSheet"] = name
        frames.append(cast)
    df = pd.concat(frames, ignore_index=True)
    df = df.sort_values("InvoiceDate", kind="stable").reset_index(drop=True)

    if use_cache:
        _write_cache(df)
    return df


def _write_cache(df: pd.DataFrame) -> None:
    DATA_INTERIM.mkdir(parents=True, exist_ok=True)
    try:
        df.to_parquet(INTERIM_PARQUET, index=False)
    except (ImportError, ValueError):
        df.to_csv(INTERIM_CSV, index=False, encoding="utf-8", compression="gzip")


def raw_quality_report(df: pd.DataFrame) -> dict:
    """Quantify the mess before touching it. Every number lands in the README."""
    n = len(df)
    cancellations = df["Invoice"].str.startswith("C").fillna(False)
    non_product = _clean.is_non_product(df["StockCode"])
    return {
        "rows": n,
        "invoices": int(df["Invoice"].nunique()),
        "stock_codes": int(df["StockCode"].nunique()),
        "date_min": df["InvoiceDate"].min(),
        "date_max": df["InvoiceDate"].max(),
        "exact_duplicate_rows": int(df.duplicated(subset=COLUMNS).sum()),
        "missing_customer_id": int(df["CustomerID"].isna().sum()),
        "missing_customer_id_pct": float(df["CustomerID"].isna().mean()),
        "cancellation_rows": int(cancellations.sum()),
        "cancellation_rows_pct": float(cancellations.mean()),
        "negative_quantity_rows": int((df["Quantity"] < 0).sum()),
        "zero_price_rows": int((df["Price"] == 0).sum()),
        "negative_price_rows": int((df["Price"] < 0).sum()),
        "non_product_code_rows": int(non_product.sum()),
        "countries": int(df["Country"].nunique()),
    }


def print_quality_report(report: dict) -> None:
    print("--- raw quality report -------------------------------------")
    print(f"rows                    {report['rows']:>12,}")
    print(f"invoices                {report['invoices']:>12,}")
    print(f"stock codes             {report['stock_codes']:>12,}")
    print(f"date range              {report['date_min']} .. {report['date_max']}")
    print(f"exact duplicate rows    {report['exact_duplicate_rows']:>12,}")
    print(
        f"missing CustomerID      {report['missing_customer_id']:>12,}"
        f"  ({100 * report['missing_customer_id_pct']:.2f}%)"
    )
    print(
        f"cancellation rows (C*)  {report['cancellation_rows']:>12,}"
        f"  ({100 * report['cancellation_rows_pct']:.2f}%)"
    )
    print(f"negative-quantity rows  {report['negative_quantity_rows']:>12,}")
    print(f"zero-price rows         {report['zero_price_rows']:>12,}")
    print(f"negative-price rows     {report['negative_price_rows']:>12,}")
    print(f"non-product code rows   {report['non_product_code_rows']:>12,}")
    print(f"countries               {report['countries']:>12,}")
    print("------------------------------------------------------------")
