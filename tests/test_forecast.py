"""Forecast math: hand-checkable MASE, leak-free folds, sane from-scratch models."""

from __future__ import annotations

import numpy as np
import pandas as pd

from retail import forecast


def test_mase_hand_checked():
    # naive scale: train diffs are all 1 -> denom 1.0; forecast errors 1 and 2 -> MAE 1.5
    train = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    assert forecast.mase(np.array([6.0, 8.0]), np.array([5.0, 6.0]), train, m=1) == 1.5
    # seasonal scale m=2: |2-1|,|4-3|,|3-2|,|5-4| -> denom 1.0; errors 0.5 -> MASE 0.5
    train2 = np.array([1.0, 3.0, 2.0, 4.0, 3.0, 5.0])
    got = forecast.mase(np.array([4.0]), np.array([3.5]), train2, m=2)
    assert abs(got - 0.5) < 1e-12


def test_seasonal_naive_repeats_last_season():
    y = np.arange(1.0, 11.0)  # 1..10
    pred = forecast.seasonal_naive(y, 6, m=4)
    assert pred.tolist() == [7.0, 8.0, 9.0, 10.0, 7.0, 8.0]
    # shorter than one season -> falls back to naive
    assert forecast.seasonal_naive(np.array([3.0, 4.0]), 3, m=4).tolist() == [4.0, 4.0, 4.0]


def test_rolling_origin_folds_never_leak():
    n, horizon = 104, 8
    folds = forecast.rolling_origin_folds(n, horizon=horizon, n_folds=5, step=8)
    assert len(folds) == 5
    dates = pd.date_range("2010-01-03", periods=n, freq="W-SUN")
    for fold in folds:
        assert fold.train_idx[0] == 0
        assert fold.train_idx.max() < fold.test_idx.min()  # strict time order
        assert len(fold.test_idx) == horizon
        assert dates[fold.train_idx].max() < dates[fold.test_idx].min()
    # folds requiring less history than min_train are dropped, not padded
    assert forecast.rolling_origin_folds(60, horizon=8, n_folds=5, step=8, min_train=56) == []


def test_holt_winters_beats_naive_on_seasonal_series():
    t = np.arange(120, dtype=float)
    y = 10.0 + 0.05 * t + 5.0 * np.sin(2 * np.pi * t / 12)
    train, test = y[:108], y[108:]
    hw = forecast.holt_winters(train, 12, m=12)
    nv = forecast.naive(train, 12)
    assert np.isfinite(hw).all() and len(hw) == 12
    assert forecast.mase(test, hw, train, m=1) < forecast.mase(test, nv, train, m=1)


def test_lag_linear_output_is_finite():
    rng = np.random.default_rng(0)
    y = 100.0 + np.cumsum(rng.normal(0, 1, 80))
    pred = forecast.lag_linear(y, 6)
    assert pred.shape == (6,)
    assert np.isfinite(pred).all()


def test_cross_validate_runs_on_fixture_weekly_series(cleaned):
    weekly = forecast.weekly_revenue(cleaned.sales)
    cv = forecast.cross_validate(weekly, horizon=8, n_folds=3, step=8)
    assert set(cv["model"]) == set(forecast.MODELS)
    assert cv["fold"].nunique() >= 1
    assert np.isfinite(cv["mae"]).all()
    summary = forecast.cv_summary(cv)
    assert len(summary) == len(forecast.MODELS)
    assert summary["fold_wins"].sum() == cv["fold"].nunique()
