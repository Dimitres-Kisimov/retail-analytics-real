"""Lifecycle stages & growth accounting: partition/flow invariants, a fully
hand-checked micro-case, cross-module consistency with cohort sizes, revenue
ties, determinism, edge cases.

Fixture-based tests run on the shared 1,950-row real sample; dataset-specific
magnitudes are NOT asserted there -- only invariants that must hold for any
input. The micro-case asserts exact hand-computed values.
"""

from __future__ import annotations

import xml.dom.minidom as minidom

import numpy as np
import pandas as pd
import pytest

from retail import cohort, lifecycle
from retail.lifecycle import ENTRY, STATES, LifecycleConfig


@pytest.fixture(scope="module")
def result(cleaned):
    return lifecycle.lifecycle_states(cleaned.sales)


# --------------------------------------------------------------------------- #
# invariants on the real fixture
# --------------------------------------------------------------------------- #
def test_stages_partition_the_acquired_base(result):
    # Every customer acquired by month m is in exactly one stage at m.
    m = result.monthly
    assert ((m["new"] + m["retained"] + m["resurrected"] + m["at_risk"] + m["dormant"])
            == m["base"]).all()
    assert (m["base"].diff().dropna() >= 0).all()          # base only grows
    assert int(m["base"].iloc[-1]) == result.n_customers   # everyone acquired by the end


def test_active_decomposition_matches_independent_count(cleaned, result):
    # active == new + retained + resurrected, and equals a from-scratch count of
    # distinct identified buyers per complete calendar month.
    m = result.monthly
    assert ((m["new"] + m["retained"] + m["resurrected"]) == m["active"]).all()

    known = cleaned.sales.loc[cleaned.sales["CustomerID"].notna()]
    per_month = known.groupby(known["InvoiceDate"].dt.strftime("%Y-%m"))["CustomerID"].nunique()
    for row in m.itertuples(index=False):
        assert row.active == int(per_month.get(row.month, 0))


def test_new_counts_match_cohort_sizes(cleaned, result):
    # Cross-module invariant: 'new' per month is exactly the cohort module's
    # acquisition count for that month (both = first purchase in that month).
    sizes = cohort.cohort_retention(cleaned.sales).sizes
    for row in result.monthly.itertuples(index=False):
        assert row.new == int(sizes.get(row.month, 0))


def test_churned_and_quick_ratio_identities(result):
    m = result.monthly
    assert pd.isna(m["churned"].iloc[0])                   # month 1 has no prior month
    for i in range(1, len(m)):
        expected = int(m["active"].iloc[i - 1]) - int(m["retained"].iloc[i])
        assert m["churned"].iloc[i] == expected
        assert expected >= 0                               # retained cannot exceed prior actives
        qr = m["quick_ratio"].iloc[i]
        if expected > 0:
            inflow = int(m["new"].iloc[i]) + int(m["resurrected"].iloc[i])
            assert abs(qr - inflow / expected) < 1e-12
        else:
            assert pd.isna(qr)


def test_revenue_ties_to_identified_monthly_revenue(cleaned, result):
    # Stage revenue split sums to each month's identified revenue, and the total
    # ties to an independent groupby over the cleaned sales (complete months).
    m = result.monthly
    split_sum = m["new_revenue"] + m["retained_revenue"] + m["resurrected_revenue"]
    assert np.allclose(split_sum, m["active_revenue"], atol=0.05)

    known = cleaned.sales.loc[cleaned.sales["CustomerID"].notna()]
    per_month = known.groupby(known["InvoiceDate"].dt.strftime("%Y-%m"))["Revenue"].sum()
    for row in m.itertuples(index=False):
        assert abs(row.active_revenue - float(per_month.get(row.month, 0.0))) < 0.05


def test_flow_matrix_sums(result):
    # Every customer existing at month m-1 flows somewhere at m: the non-entry
    # flows total the base summed over source months. Column totals equal the
    # monthly stage counts summed over target months (2..K).
    m, flows = result.monthly, result.flows
    non_entry = flows.drop(index=ENTRY)
    assert int(non_entry.to_numpy().sum()) == int(m["base"].iloc[:-1].sum())
    assert int(flows.loc[ENTRY, "new"]) == int(m["new"].iloc[1:].sum())
    for state in STATES:
        assert int(flows[state].sum()) == int(m[state].iloc[1:].sum())


def test_flow_matrix_structural_zeros(result):
    flows = result.flows
    # 'new' can only be entered from pre-acquisition.
    assert (flows.loc[list(STATES), "new"] == 0).all()
    # ENTRY leads nowhere but 'new'.
    assert int(flows.loc[ENTRY].drop("new").sum()) == 0
    # An active month followed by an active month is 'retained', never 'resurrected'.
    for src in ("new", "retained", "resurrected"):
        assert int(flows.loc[src, "resurrected"]) == 0
        # gap jumps straight from 0 to 1 <= D, so active never falls to dormant.
        assert int(flows.loc[src, "dormant"]) == 0
    # Inactive customers cannot be 'retained' next month, and dormancy is a
    # one-way gap: it cannot shrink back to at_risk without a purchase.
    assert int(flows.loc["at_risk", "retained"]) == 0
    assert int(flows.loc["dormant", "retained"]) == 0
    assert int(flows.loc["dormant", "at_risk"]) == 0


def test_months_are_complete_ordered_and_end_at_last_complete(cleaned, result):
    m = result.monthly
    months = list(m["month"])
    assert months == sorted(months)
    assert months[-1] == result.last_complete_month
    max_label = cleaned.sales["InvoiceDate"].max().strftime("%Y-%m")
    if result.final_month_partial:
        assert result.last_complete_month < max_label      # partial month never scored


def test_dormancy_threshold_only_moves_the_inactive_split(cleaned, result):
    # Active-stage counts do not depend on D; D only splits inactive into
    # at_risk vs dormant (their sum is invariant).
    alt = lifecycle.lifecycle_states(cleaned.sales, LifecycleConfig(dormancy_months=6))
    for col in ("new", "retained", "resurrected", "active", "base"):
        assert (alt.monthly[col] == result.monthly[col]).all()
    assert ((alt.monthly["at_risk"] + alt.monthly["dormant"])
            == (result.monthly["at_risk"] + result.monthly["dormant"])).all()
    assert (alt.monthly["dormant"] <= result.monthly["dormant"]).all()  # larger D -> never more dormant


def test_gap_stats_audit_the_threshold(result):
    g = result.gap_stats
    if g:                                                  # fixture has repeat buyers
        assert g["n_gaps"] > 0
        assert 0.0 <= g["share_within_dormancy"] <= 1.0
        assert g["p50"] <= g["p75"] <= g["p90"]


def test_determinism_and_svg_wellformed(cleaned, result):
    again = lifecycle.lifecycle_states(cleaned.sales)
    pd.testing.assert_frame_equal(result.monthly, again.monthly)
    pd.testing.assert_frame_equal(result.flows, again.flows)
    svg = lifecycle.render_svg(result)
    minidom.parseString(svg)                               # raises if not well-formed XML
    assert svg.startswith("<svg")
    assert "Customer lifecycle" in svg
    assert lifecycle.render_svg(again) == svg              # byte-identical re-render


def test_empty_and_invalid_inputs(cleaned):
    empty = cleaned.sales.iloc[0:0]
    res = lifecycle.lifecycle_states(empty)
    assert res.n_customers == 0
    assert res.monthly.empty and res.flows.empty
    assert lifecycle.monthly_frame(res).empty and lifecycle.flow_frame(res).empty
    assert "not enough complete months" in lifecycle.headline_text(res)
    with pytest.raises(ValueError):
        lifecycle.lifecycle_states(cleaned.sales, LifecycleConfig(dormancy_months=0))


def test_csv_exports_write_both_files(result, tmp_path):
    monthly_path, flows_path = lifecycle.write_csv(result, out_dir=tmp_path)
    assert monthly_path.exists() and flows_path.exists()
    back = pd.read_csv(monthly_path)
    assert list(back.columns)[:5] == ["month", "new", "retained", "resurrected", "active"]
    assert len(back) == len(result.monthly)
    flows_back = pd.read_csv(flows_path)
    assert list(flows_back.columns) == ["from", *STATES]


# --------------------------------------------------------------------------- #
# hand-checked micro-case (every number verified by hand, D=2)
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def micro_result():
    # Customer 101: buys Jan, Feb, Apr -> new, retained, at_risk, resurrected
    # Customer 102: buys Jan only      -> new, at_risk, at_risk, dormant (gap 3 > 2)
    # Customer 103: buys Mar           -> -, -, new, at_risk
    # Customer 104: buys only in May, the partial final month -> excluded entirely
    rows = [
        (101, "2020-01-15", 10.0), (101, "2020-02-20", 20.0), (101, "2020-04-05", 40.0),
        (102, "2020-01-31", 5.0),
        (103, "2020-03-10", 7.0),
        (104, "2020-05-02", 99.0),
    ]
    frame = pd.DataFrame(rows, columns=["CustomerID", "InvoiceDate", "Revenue"])
    frame["InvoiceDate"] = pd.to_datetime(frame["InvoiceDate"])
    frame["CustomerID"] = frame["CustomerID"].astype("Int64")
    return lifecycle.lifecycle_states(frame, LifecycleConfig(dormancy_months=2))


def test_micro_case_monthly_table(micro_result):
    m = micro_result.monthly.set_index("month")
    assert list(m.index) == ["2020-01", "2020-02", "2020-03", "2020-04"]
    assert micro_result.last_complete_month == "2020-04"
    assert micro_result.final_month_partial is True
    assert micro_result.n_customers == 3
    assert micro_result.n_excluded_partial == 1            # customer 104

    expected = {
        #            new retained resurrected at_risk dormant base active
        "2020-01": (2, 0, 0, 0, 0, 2, 2),
        "2020-02": (0, 1, 0, 1, 0, 2, 1),
        "2020-03": (1, 0, 0, 2, 0, 3, 1),
        "2020-04": (0, 0, 1, 1, 1, 3, 1),
    }
    for month, (new, ret, res, risk, dorm, base, active) in expected.items():
        row = m.loc[month]
        assert (int(row["new"]), int(row["retained"]), int(row["resurrected"]),
                int(row["at_risk"]), int(row["dormant"]), int(row["base"]),
                int(row["active"])) == (new, ret, res, risk, dorm, base, active)

    # churned: Feb 2-1=1, Mar 1-0=1, Apr 1-0=1; quick ratio 0/1, 1/1, 1/1.
    assert pd.isna(m["churned"].iloc[0])
    assert list(m["churned"].iloc[1:]) == [1.0, 1.0, 1.0]
    assert list(m["quick_ratio"].iloc[1:]) == [0.0, 1.0, 1.0]

    # revenue lands with the stage that earned it.
    assert list(m["new_revenue"]) == [15.0, 0.0, 7.0, 0.0]
    assert list(m["retained_revenue"]) == [0.0, 20.0, 0.0, 0.0]
    assert list(m["resurrected_revenue"]) == [0.0, 0.0, 0.0, 40.0]
    assert list(m["active_revenue"]) == [15.0, 20.0, 7.0, 40.0]


def test_micro_case_flow_matrix(micro_result):
    flows = micro_result.flows
    expected_nonzero = {
        (ENTRY, "new"): 1,             # 103 enters in March
        ("new", "retained"): 1,        # 101 Jan->Feb
        ("new", "at_risk"): 2,         # 102 Jan->Feb, 103 Mar->Apr
        ("retained", "at_risk"): 1,    # 101 Feb->Mar
        ("at_risk", "at_risk"): 1,     # 102 Feb->Mar
        ("at_risk", "resurrected"): 1,  # 101 Mar->Apr
        ("at_risk", "dormant"): 1,     # 102 Mar->Apr (gap 3 > 2)
    }
    for src in flows.index:
        for dst in flows.columns:
            assert int(flows.loc[src, dst]) == expected_nonzero.get((src, dst), 0), (src, dst)


def test_micro_case_gap_stats(micro_result):
    # Only 101 has comeback gaps: Jan->Feb (1 month) and Feb->Apr (2 months).
    g = micro_result.gap_stats
    assert g["n_gaps"] == 2
    assert g["share_within_dormancy"] == 1.0               # both gaps <= 2
