"""Full-dataset ingest checks. Skips cleanly when the raw data is not downloaded."""

from __future__ import annotations

import pytest

from retail import ingest
from retail.paths import INTERIM_CSV, INTERIM_PARQUET, RAW_XLSX

HAVE_DATA = RAW_XLSX.exists() or INTERIM_PARQUET.exists() or INTERIM_CSV.exists()


@pytest.mark.skipif(
    not HAVE_DATA,
    reason="full dataset not downloaded - run `python scripts/download_data.py` "
    "(tests are fixture-based by design; this one alone needs data/)",
)
def test_full_dataset_loads_with_expected_shape():
    df = ingest.load_raw()
    assert len(df) > 1_000_000
    assert {"Invoice", "StockCode", "Quantity", "InvoiceDate", "Price", "CustomerID", "Country"} <= set(df.columns)
    report = ingest.raw_quality_report(df)
    assert report["rows"] == len(df)
    assert 0.15 < report["missing_customer_id_pct"] < 0.35  # the documented ~20% gap
    assert report["cancellation_rows"] > 0


def test_quality_report_works_on_fixture(raw_fixture):
    report = ingest.raw_quality_report(raw_fixture)
    assert report["rows"] == len(raw_fixture)
    assert report["missing_customer_id"] == int(raw_fixture["CustomerID"].isna().sum())
    assert report["cancellation_rows"] == int(raw_fixture["Invoice"].str.startswith("C").sum())
