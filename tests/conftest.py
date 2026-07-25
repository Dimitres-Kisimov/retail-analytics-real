"""Shared fixtures.

All tests run on tests/fixtures/sample.csv — 1,950 REAL rows drawn
deterministically (seed 42) from the full dataset by scripts/make_fixture.py,
stratified to include cancellations, non-product codes, zero prices, missing
CustomerIDs and real exact-duplicate groups. The full ~1M-row download is NOT
required (the one test that touches it skips itself when data/ is absent).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from retail import clean

FIXTURE = Path(__file__).parent / "fixtures" / "sample.csv"


@pytest.fixture(scope="session")
def raw_fixture() -> pd.DataFrame:
    df = pd.read_csv(FIXTURE, encoding="utf-8")
    df["Invoice"] = df["Invoice"].astype(str)
    df["StockCode"] = df["StockCode"].astype(str)
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    df["CustomerID"] = df["CustomerID"].astype("Int64")
    return df


@pytest.fixture(scope="session")
def cleaned(raw_fixture: pd.DataFrame) -> clean.CleanResult:
    return clean.clean(raw_fixture)
