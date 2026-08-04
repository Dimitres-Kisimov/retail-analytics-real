"""Cohort retention: definitional invariants, honest censoring, deterministic SVG.

Runs on the shared 1,950-row real fixture. The fixture is a random sample, so
repeat rates are tiny and dataset-specific magnitudes are NOT asserted -- only
the invariants that must hold for any input are.
"""

from __future__ import annotations

import xml.dom.minidom as minidom

import numpy as np
import pytest

from retail import cohort


@pytest.fixture(scope="module")
def result(cleaned):
    return cohort.cohort_retention(cleaned.sales)


def test_acquisition_month_is_full_retention(result):
    # Offset 0 is the first-purchase month -> 100% by definition, for every cohort.
    assert (result.triangle[0] == 1.0).all()
    assert float(result.curve.set_index("offset").loc[0, "avg_retention"]) == 1.0


def test_retention_values_are_fractions(result):
    vals = result.triangle.to_numpy(dtype=float)
    finite = vals[~np.isnan(vals)]
    assert finite.min() >= 0.0
    assert finite.max() <= 1.0


def test_cohort_sizes_match_identified_customers(cleaned, result):
    known_ids = cleaned.sales.loc[cleaned.sales["KnownCustomer"], "CustomerID"].nunique()
    assert int(result.sizes.sum()) == known_ids == result.n_customers


def test_right_censoring_is_blank_not_zero(result):
    # A cell (cohort, k) must be NaN exactly when its target month is beyond the
    # last complete month; observable cells are never NaN. This is the property
    # that stops "hasn't happened yet" from being counted as churn.
    lc = result.last_complete_month
    lc_i = int(lc[:4]) * 12 + (int(lc[5:7]) - 1)
    for cohort_label in result.triangle.index:
        c_i = int(cohort_label[:4]) * 12 + (int(cohort_label[5:7]) - 1)
        for k in result.triangle.columns:
            val = result.triangle.at[cohort_label, k]
            observable = (k == 0) or (c_i + k <= lc_i)
            assert observable != bool(np.isnan(val))


def test_curve_weighting_is_consistent(result):
    for r in result.curve.itertuples(index=False):
        assert 0 <= r.repeat_customers <= r.customers_observed
        expected = r.repeat_customers / r.customers_observed
        assert abs(r.avg_retention - expected) < 1e-12
        # every offset averages only cohorts old enough to be observed there
        assert r.cohorts_observed >= 1


def test_curve_offsets_are_gap_free_and_ordered(result):
    offs = result.curve["offset"].tolist()
    assert offs == sorted(offs)
    assert offs == list(range(len(offs)))


def test_triangle_frame_export_shape(result):
    frame = cohort.triangle_frame(result)
    assert list(frame.columns[:3]) == ["cohort", "customers", "m0"]
    assert (frame["m0"] == 100.0).all()
    assert len(frame) == len(result.triangle)


def test_headline_text_reports_measured_numbers(result):
    hl = result.headline()
    text = cohort.headline_text(result)
    assert "month-1 repeat rate" in text
    m1 = 100 * hl["month_1_rate"]
    assert f"{m1:.1f}%" in text


def test_svg_is_wellformed_and_byte_identical(result):
    svg = cohort.render_svg(result)
    minidom.parseString(svg)  # raises if not well-formed XML
    assert svg.startswith("<svg")
    assert "Cohort repeat-purchase retention" in svg
    # deterministic: no timestamps / RNG -> byte-identical on re-render
    assert cohort.render_svg(result) == svg


def test_empty_input_is_handled(cleaned):
    empty = cleaned.sales.iloc[0:0]
    res = cohort.cohort_retention(empty)
    assert res.n_customers == 0
    assert res.triangle.empty
