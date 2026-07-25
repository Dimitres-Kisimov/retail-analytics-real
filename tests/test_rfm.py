"""RFM: valid quintiles, every customer assigned, every grid cell mapped."""

from __future__ import annotations

import pytest

from retail import rfm


@pytest.fixture(scope="module")
def scored(cleaned):
    table, summary = rfm.run_rfm(cleaned.sales)
    return table, summary


def test_quintiles_valid_and_all_customers_assigned(cleaned, scored):
    table, _ = scored
    known_ids = cleaned.sales.loc[cleaned.sales["KnownCustomer"], "CustomerID"].nunique()
    assert len(table) == known_ids
    for col in ("R", "F", "M"):
        assert table[col].between(1, 5).all()
    assert table["Segment"].notna().all()
    assert set(table["Segment"]).issubset(set(rfm.SEGMENT_ORDER))


def test_every_grid_cell_has_a_segment():
    for r in range(1, 6):
        for f in range(1, 6):
            assert rfm.segment_name(r, f) in rfm.SEGMENT_ORDER


def test_segment_summary_shares_sum_to_one(scored):
    table, summary = scored
    assert abs(summary["RevenueShare"].sum() - 1.0) < 0.01  # rounding tolerance
    assert summary["Customers"].sum() == len(table)
    assert (summary["Revenue"] >= 0).all()  # sales rows all carry positive revenue
