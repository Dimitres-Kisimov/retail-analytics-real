"""CLV tests: hand-checked summary, BG/NBD + Gamma-Gamma invariants, honest validation.

Everything runs on the shared 1,950-row real fixture or on tiny hand-built cases
whose statistics are verifiable with pencil and paper. The fixture is a sparse
random sample, so dataset-specific *magnitudes* are not asserted here -- only the
invariants that must hold for any input, the internal math cross-check, and the
determinism of the fit. The one measured-headline assertion lives in the
full-data test, which skips itself cleanly when the dataset is absent.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from retail import clv, ingest, paths
from retail.__main__ import main
from retail.paths import INTERIM_CSV, INTERIM_PARQUET, RAW_XLSX

HAVE_DATA = RAW_XLSX.exists() or INTERIM_PARQUET.exists() or INTERIM_CSV.exists()


# --------------------------------------------------------------------------- #
# Module-scoped fits on the fixture (computed once, reused)
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def result(cleaned):
    return clv.run_clv(cleaned.sales)


@pytest.fixture(scope="module")
def summary(cleaned):
    return clv.summarize_customers(cleaned.sales)


# --------------------------------------------------------------------------- #
# Customer summary: hand-checked
# --------------------------------------------------------------------------- #
def _tiny_sales() -> pd.DataFrame:
    """Two identified customers + one guest, with a same-day double invoice."""
    rows = [
        # customer 1: 3 distinct purchase-days; the 2011-01-15 day has two invoices
        (1.0, "2011-01-01 09:00", 100.0),
        (1.0, "2011-01-15 14:30", 200.0),
        (1.0, "2011-01-15 16:00", 50.0),   # same day -> collapses into 2011-01-15
        (1.0, "2011-02-01 11:00", 300.0),
        # customer 2: single purchase (frequency 0)
        (2.0, "2011-01-10 10:00", 500.0),
        # guest (no CustomerID): must be excluded entirely
        (np.nan, "2011-01-20 12:00", 999.0),
    ]
    return pd.DataFrame(
        {
            "CustomerID": pd.array([r[0] for r in rows], dtype="Float64"),
            "InvoiceDate": pd.to_datetime([r[1] for r in rows]),
            "Revenue": [r[2] for r in rows],
        }
    )


def test_summary_is_hand_checkable():
    s = clv.summarize_customers(_tiny_sales(), observation_end=pd.Timestamp("2011-03-01"))
    assert set(s["CustomerID"]) == {1.0, 2.0}  # the guest row is dropped

    c1 = s.loc[s["CustomerID"] == 1.0].iloc[0]
    assert c1["frequency"] == 2            # 3 distinct days minus the first
    assert c1["recency"] == 31.0           # 2011-01-01 -> 2011-02-01
    assert c1["T"] == 59.0                 # 2011-01-01 -> 2011-03-01
    # repeat revenue = (250 on 01-15) + (300 on 02-01) = 550 over 2 repeat days
    assert c1["monetary_value"] == pytest.approx(275.0)

    c2 = s.loc[s["CustomerID"] == 2.0].iloc[0]
    assert c2["frequency"] == 0
    assert c2["recency"] == 0.0
    assert c2["T"] == 50.0                 # 2011-01-10 -> 2011-03-01
    assert c2["monetary_value"] == 0.0     # no repeat purchases -> 0 by construction


def test_summary_invariants_on_fixture(summary):
    assert (summary["frequency"] >= 0).all()
    assert (summary["recency"] >= 0).all()
    assert (summary["recency"] <= summary["T"]).all()   # last purchase <= observation end
    assert (summary["monetary_value"] >= 0).all()
    # frequency 0 exactly when the customer has a single purchase-day (monetary 0)
    one_timers = summary["frequency"] == 0
    assert (summary.loc[one_timers, "monetary_value"] == 0).all()


def test_summary_empty_input_is_handled(cleaned):
    empty = clv.summarize_customers(cleaned.sales.iloc[0:0])
    assert empty.empty
    assert list(empty.columns)[:4] == ["CustomerID", "frequency", "recency", "T"]


# --------------------------------------------------------------------------- #
# BG/NBD: the internal math cross-check + invariants
# --------------------------------------------------------------------------- #
# A fixed parameter set (no fitting needed) for pure-math assertions.
_PARAMS = clv.BGNBDParams(r=0.55, alpha=12.0, a=0.80, b=2.50, log_likelihood=0.0, converged=True)


@pytest.mark.parametrize("t", [1.0, 7.0, 30.0, 90.0, 180.0, 365.0])
def test_conditional_at_origin_equals_new_customer(t):
    # E[Y(t) | x=0, t_x=0, T=0] must equal the unconditional new-customer E[X(t)].
    # This ties the two independently written formulas together and catches any
    # transcription error in either.
    zero = np.array([0.0])
    cond = clv.conditional_expected_purchases(_PARAMS, t, zero, zero, zero)[0]
    unc = clv.expected_purchases_new_customer(_PARAMS, t)
    assert cond == pytest.approx(unc, rel=1e-12, abs=1e-12)


def test_expected_purchases_are_nonnegative_and_monotonic_in_horizon():
    x = np.array([0.0, 1.0, 5.0, 20.0])
    t_x = np.array([0.0, 10.0, 100.0, 300.0])
    T = np.array([5.0, 40.0, 200.0, 400.0])
    prev = np.zeros_like(x)
    for t in (10.0, 30.0, 90.0, 180.0, 365.0):
        pred = clv.conditional_expected_purchases(_PARAMS, t, x, t_x, T)
        assert (pred >= 0).all()
        assert (pred >= prev - 1e-9).all()   # longer horizon never predicts fewer
        prev = pred


def test_prob_alive_is_a_probability_and_one_timers_are_alive():
    x = np.array([0.0, 1.0, 5.0, 50.0])
    t_x = np.array([0.0, 5.0, 200.0, 700.0])
    T = np.array([10.0, 400.0, 400.0, 720.0])
    alive = clv.prob_alive(_PARAMS, x, t_x, T)
    assert (alive >= 0).all() and (alive <= 1).all()
    assert alive[0] == 1.0  # x=0 never had a chance to churn under BG/NBD


def test_recent_active_buyer_scores_above_dormant_twin():
    # Same frequency and age, but one shopped recently (t_x near T) and one long ago.
    x = np.array([8.0, 8.0])
    T = np.array([300.0, 300.0])
    t_x = np.array([290.0, 40.0])          # recent vs dormant
    pred = clv.conditional_expected_purchases(_PARAMS, 180.0, x, t_x, T)
    alive = clv.prob_alive(_PARAMS, x, t_x, T)
    assert pred[0] > pred[1]
    assert alive[0] > alive[1]


def test_bgnbd_fit_converges_and_is_deterministic(summary):
    p1 = clv.fit_bgnbd(summary)
    p2 = clv.fit_bgnbd(summary)
    assert p1.converged
    assert all(v > 0 for v in p1.as_tuple())
    assert p1.as_tuple() == p2.as_tuple()   # byte-identical refit (fixed start)


# --------------------------------------------------------------------------- #
# Gamma-Gamma
# --------------------------------------------------------------------------- #
def test_gamma_gamma_fit_and_scale_invariance(summary):
    gg = clv.fit_gamma_gamma(summary)
    assert gg.converged
    assert gg.p > 0 and gg.q > 0 and gg.v > 0
    assert gg.population_mean() > 0

    # p and q are scale-free; multiplying every monetary value by 10 must leave
    # them put and scale v by 10 (this is the reparametrisation the fit relies on).
    scaled = summary.copy()
    scaled["monetary_value"] = scaled["monetary_value"] * 10.0
    gg10 = clv.fit_gamma_gamma(scaled)
    assert gg10.p == pytest.approx(gg.p, rel=1e-3)
    assert gg10.q == pytest.approx(gg.q, rel=1e-3)
    assert gg10.v == pytest.approx(gg.v * 10.0, rel=1e-3)
    assert gg10.population_mean() == pytest.approx(gg.population_mean() * 10.0, rel=1e-3)


def test_expected_value_blends_between_population_and_individual(summary):
    gg = clv.fit_gamma_gamma(summary)
    pop = gg.population_mean()
    x = np.array([0.0, 1.0, 5.0, 30.0])
    m = np.array([0.0, 50.0, 800.0, 250.0])
    val = clv.conditional_expected_average_value(gg, x, m)
    assert val[0] == pytest.approx(pop)     # one-timers fall back to the population mean
    for i in (1, 2, 3):                      # repeat customers: bounded by pop and own mean
        lo, hi = sorted((pop, m[i]))
        assert lo - 1e-6 <= val[i] <= hi + 1e-6
    assert (val > 0).all()


# --------------------------------------------------------------------------- #
# CLV table + concentration + validation
# --------------------------------------------------------------------------- #
def test_clv_table_is_sane_and_sorted(result):
    ct = result.clv_table
    assert len(ct) == result.n_customers
    assert (ct["clv"] >= 0).all()
    assert (ct["pred_purchases"] >= 0).all()
    assert (ct["prob_alive"] >= 0).all() and (ct["prob_alive"] <= 1).all()
    assert ct["clv"].is_monotonic_decreasing            # sorted, best first
    assert ct["clv"].to_numpy()[0] == ct["clv"].max()


def test_concentration_is_a_share(result):
    conc = result.concentration(0.10)
    assert 0.0 <= conc["top_share"] <= 1.0
    assert conc["top_customers"] == max(1, round(0.10 * result.n_customers))
    assert conc["total_clv"] == pytest.approx(result.clv_table["clv"].sum())


def test_lorenz_points_are_well_formed(result):
    cum_x, cum_y = clv.lorenz_points(result.clv_table["clv"].to_numpy())
    assert cum_x[0] == 0.0 and cum_y[0] == 0.0
    assert cum_x[-1] == pytest.approx(1.0) and cum_y[-1] == pytest.approx(1.0)
    assert np.all(np.diff(cum_x) >= 0) and np.all(np.diff(cum_y) >= -1e-12)
    # concentration: the curve never rises above the equality diagonal
    assert np.all(cum_y <= cum_x + 1e-9)


def test_validation_runs_and_is_consistent(result):
    v = result.validation
    assert v is not None
    assert v.holdout_days > 0
    assert v.predicted_total >= 0 and v.actual_total >= 0
    assert np.isfinite(v.ratio)
    # by-frequency table: means are non-negative and buckets cover the customers
    bf = v.by_frequency
    assert (bf["predicted_mean"] >= 0).all() and (bf["actual_mean"] >= 0).all()
    assert int(bf["customers"].sum()) == v.n_customers


def test_headline_reports_measured_numbers(result):
    text = clv.headline_text(result)
    assert "BG/NBD + Gamma-Gamma" in text
    assert f"{result.n_customers:,}" in text


# --------------------------------------------------------------------------- #
# Figure + CLI
# --------------------------------------------------------------------------- #
def test_fig_clv_writes_png(result, tmp_path):
    path = clv.fig_clv(result, out_dir=tmp_path)
    assert path is not None and path.name == "clv_validation.png"
    assert path.stat().st_size > 10_000


def test_cli_clv_skips_gracefully_without_raw_data(monkeypatch, capsys):
    def _absent(force=False):
        raise FileNotFoundError("raw data not downloaded")

    monkeypatch.setattr(ingest, "load_raw", _absent)
    assert main(["--clv"]) == 0
    out = capsys.readouterr().out
    assert "[SKIP]" in out
    assert "download_data.py" in out


def test_cli_clv_runs_on_fixture(monkeypatch, capsys, tmp_path, raw_fixture):
    monkeypatch.setattr(ingest, "load_raw", lambda force=False: raw_fixture.copy())
    monkeypatch.setattr(paths, "FIGURES", tmp_path)
    monkeypatch.setattr(paths, "DELIVERABLES", tmp_path)
    assert main(["--clv"]) == 0
    out = capsys.readouterr().out
    assert "BG/NBD" in out
    assert "out-of-sample holdout" in out
    assert (tmp_path / "clv_validation.png").exists()
    assert (tmp_path / "customer_lifetime_value.csv").exists()


# --------------------------------------------------------------------------- #
# Full-data headline (skips cleanly without the dataset)
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(
    not HAVE_DATA,
    reason="full dataset not downloaded - run `python scripts/download_data.py` "
    "(tests are fixture-based by design; this one alone needs data/)",
)
def test_full_data_clv_headline():
    from retail import clean

    sales = clean.clean(ingest.load_raw()).sales
    res = clv.run_clv(sales)

    # counts are exact and measured
    assert res.n_customers == 5852
    assert res.n_repeat == 4179
    assert res.bgnbd.converged and res.gamma_gamma.converged
    # fitted parameters land in their measured neighbourhoods
    assert 0.60 < res.bgnbd.r < 0.75
    assert 55.0 < res.bgnbd.alpha < 72.0
    # Gamma-Gamma independence assumption genuinely holds on this data
    assert abs(res.gamma_gamma.freq_value_corr) < 0.10
    assert 350.0 < res.gamma_gamma.population_mean() < 450.0

    # the honest headline: predicted holdout transactions within a few % of actual
    v = res.validation
    assert v is not None
    assert 0.95 < v.ratio < 1.05
    assert v.correlation > 0.75

    # revenue concentration is real, not synthetic
    conc = res.concentration(0.10)
    assert 0.50 < conc["top_share"] < 0.65
