"""The cleaning pipeline removes exactly what it claims — verified on real rows."""

from __future__ import annotations

import numpy as np

from retail import clean


def test_fixture_is_real_and_messy(raw_fixture):
    assert len(raw_fixture) > 1500
    assert raw_fixture["Invoice"].str.startswith("C").any()
    assert raw_fixture["CustomerID"].isna().sum() > 100
    assert raw_fixture.duplicated(subset=clean.DEDUP_COLUMNS).any()
    assert (raw_fixture["Price"] <= 0).any()


def test_dedup_removes_exactly_the_duplicates(raw_fixture, cleaned):
    expected = int(raw_fixture.duplicated(subset=clean.DEDUP_COLUMNS).sum())
    step = cleaned.steps[0]
    assert step["step"] == "drop exact duplicates"
    assert step["rows_removed"] == expected
    assert not cleaned.sales.duplicated(subset=clean.DEDUP_COLUMNS).any()


def test_cancellations_separated_not_deleted(cleaned):
    assert not cleaned.sales["Invoice"].str.startswith("C").any()
    assert len(cleaned.returns) > 0
    assert cleaned.returns["Invoice"].str.startswith("C").all()


def test_non_product_codes_removed(cleaned):
    for frame in (cleaned.sales, cleaned.returns):
        assert not frame["StockCode"].isin(clean.NON_PRODUCT_CODES).any()
        assert not frame["StockCode"].str.lower().str.startswith(clean.GIFT_VOUCHER_PREFIX).any()


def test_sales_prices_and_quantities_positive(cleaned):
    assert (cleaned.sales["Price"] > 0).all()
    assert (cleaned.sales["Quantity"] > 0).all()


def test_missing_customer_ids_flagged_not_dropped(cleaned):
    sales = cleaned.sales
    flag_step = next(s for s in cleaned.steps if s["step"] == "flag missing CustomerID")
    assert flag_step["rows_removed"] == 0  # flagged, never dropped
    assert (sales["KnownCustomer"] == sales["CustomerID"].notna()).all()
    assert (~sales["KnownCustomer"]).sum() > 0  # the fixture really has them


def test_revenue_math(cleaned):
    sales = cleaned.sales
    assert np.allclose(sales["Revenue"], sales["Quantity"] * sales["Price"])
    # returns keep their negative sign so returned value is visible
    assert (cleaned.returns["Revenue"] <= 0).all()


def test_clean_report_accounting_is_consistent(cleaned):
    report = cleaned.report()
    assert (report["rows_in"] - report["rows_removed"] == report["rows_out"]).all()
    # each step starts where the previous one ended
    assert (report["rows_in"].iloc[1:].to_numpy() == report["rows_out"].iloc[:-1].to_numpy()).all()
    assert report["rows_out"].iloc[-1] == len(cleaned.sales)
