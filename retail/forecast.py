"""Weekly revenue forecasting, from scratch, with rolling-origin cross-validation.

Models (no scikit-learn / statsmodels available — implemented directly):

* seasonal-naive     y_hat(t+i) = y(t+i-52)  (falls back to naive if history < 52)
* Holt-Winters       additive level/trend/seasonality, smoothing params picked by
                     a small grid search on the tail of the TRAINING window only
* lag-linear         ordinary least squares (numpy lstsq) on lag features
                     (t-1..t-4), an annual sin/cos pair and a linear trend,
                     forecast recursively over the horizon

Evaluation is MASE per fold (scaled by the in-sample MAE of the seasonal-naive
walk on that fold's training data; falls back to the non-seasonal naive scale
when the training window is shorter than one season). Rolling-origin CV is
leakage-safe by construction: each fold trains strictly before its origin and
predicts the weeks after it. If Holt-Winters loses to seasonal-naive on a fold,
that is reported plainly — real weekly revenue with one strong yearly peak is
exactly where simple baselines are hard to beat.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

SEASON_WEEKS = 52
WEEKS_PER_YEAR = 365.25 / 7.0


# --------------------------------------------------------------------------- #
# Weekly series
# --------------------------------------------------------------------------- #
def weekly_revenue(sales: pd.DataFrame) -> pd.Series:
    """Weekly (W-SUN) revenue. The two edge weeks are partial and are dropped."""
    weekly = sales.set_index("InvoiceDate")["Revenue"].resample("W-SUN").sum()
    return weekly.iloc[1:-1]  # first and last calendar weeks are incomplete


# --------------------------------------------------------------------------- #
# Models: f(train_values, horizon) -> np.ndarray of length horizon
# --------------------------------------------------------------------------- #
def naive(y: np.ndarray, h: int) -> np.ndarray:
    return np.full(h, y[-1], dtype=float)


def seasonal_naive(y: np.ndarray, h: int, m: int = SEASON_WEEKS) -> np.ndarray:
    if len(y) < m:
        return naive(y, h)
    return np.array([y[-m + (i % m)] for i in range(h)], dtype=float)


def holt_winters(
    y: np.ndarray,
    h: int,
    m: int = SEASON_WEEKS,
    alpha: float = 0.3,
    beta: float = 0.05,
    gamma: float = 0.1,
) -> np.ndarray:
    """Additive Holt-Winters. Needs at least one full season of history."""
    n = len(y)
    if n < m + 4:  # not enough history to estimate a seasonal profile
        return naive(y, h)
    level = float(np.mean(y[:m]))
    if n >= 2 * m:
        trend = (float(np.mean(y[m : 2 * m])) - level) / m
    else:  # single season: per-step slope of a straight-line fit on it
        t = np.arange(m, dtype=float)
        trend = float(np.polyfit(t, y[:m], 1)[0])
    season = (y[:m] - level).astype(float)
    for t_i in range(n):
        si = t_i % m
        last_level = level
        level = alpha * (y[t_i] - season[si]) + (1 - alpha) * (level + trend)
        trend = beta * (level - last_level) + (1 - beta) * trend
        season[si] = gamma * (y[t_i] - level) + (1 - gamma) * season[si]
    return np.array(
        [level + (i + 1) * trend + season[(n + i) % m] for i in range(h)], dtype=float
    )


HW_GRID = [
    (a, b, g)
    for a in (0.1, 0.3, 0.5)
    for b in (0.01, 0.1)
    for g in (0.05, 0.2)
]


def holt_winters_tuned(y: np.ndarray, h: int, m: int = SEASON_WEEKS) -> np.ndarray:
    """Grid-search (alpha, beta, gamma) on the last min(h, 8) points of TRAIN only."""
    h_val = min(h, 8)
    if len(y) < m + 4 + h_val:
        return holt_winters(y, h, m)
    fit, val = y[:-h_val], y[-h_val:]
    best, best_mae = (0.3, 0.05, 0.1), np.inf
    for a, b, g in HW_GRID:
        pred = holt_winters(fit, h_val, m, a, b, g)
        mae = float(np.mean(np.abs(val - pred)))
        if mae < best_mae:
            best, best_mae = (a, b, g), mae
    a, b, g = best
    return holt_winters(y, h, m, a, b, g)


def lag_linear(y: np.ndarray, h: int, lags: tuple[int, ...] = (1, 2, 3, 4)) -> np.ndarray:
    """OLS (numpy lstsq) on lag features + annual Fourier pair + trend; recursive forecast."""
    max_lag = max(lags)
    n = len(y)
    if n < max_lag + 8:
        return naive(y, h)

    def features(t: int, history: np.ndarray) -> list[float]:
        angle = 2.0 * np.pi * t / WEEKS_PER_YEAR
        row = [1.0, t / n, np.sin(angle), np.cos(angle)]
        row.extend(history[t - lag] for lag in lags)
        return row

    x_rows = [features(t, y) for t in range(max_lag, n)]
    coef, *_ = np.linalg.lstsq(np.asarray(x_rows), y[max_lag:], rcond=None)

    history = y.astype(float).copy()
    preds = []
    for i in range(h):
        t = n + i
        row = np.asarray(features(t, history))
        pred = float(row @ coef)
        preds.append(pred)
        history = np.append(history, pred)
    return np.array(preds)


MODELS = {
    "seasonal_naive": seasonal_naive,
    "holt_winters": holt_winters_tuned,
    "lag_linear": lag_linear,
}


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def mae(actual: np.ndarray, pred: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(actual, float) - np.asarray(pred, float))))


def mase(actual: np.ndarray, pred: np.ndarray, train: np.ndarray, m: int = SEASON_WEEKS) -> float:
    """MAE scaled by the in-sample MAE of the (seasonal-)naive walk on `train`."""
    train = np.asarray(train, float)
    if len(train) <= m:
        m = 1
    denom = float(np.mean(np.abs(train[m:] - train[:-m])))
    if denom == 0.0 or not np.isfinite(denom):
        return float("nan")
    return mae(actual, pred) / denom


# --------------------------------------------------------------------------- #
# Rolling-origin cross-validation
# --------------------------------------------------------------------------- #
@dataclass
class Fold:
    train_idx: np.ndarray
    test_idx: np.ndarray


def rolling_origin_folds(
    n: int, horizon: int = 8, n_folds: int = 5, step: int = 8, min_train: int = 56
) -> list[Fold]:
    """Fold k trains on [0, origin_k) and tests on [origin_k, origin_k + horizon).

    Origins step back from the end of the series; folds whose training window
    would fall under `min_train` are discarded (never padded with future data).
    """
    last_origin = n - horizon
    folds = []
    for k in range(n_folds):
        origin = last_origin - (n_folds - 1 - k) * step
        if origin < min_train:
            continue
        folds.append(Fold(train_idx=np.arange(0, origin), test_idx=np.arange(origin, origin + horizon)))
    return folds


def cross_validate(
    series: pd.Series, horizon: int = 8, n_folds: int = 5, step: int = 8
) -> pd.DataFrame:
    """Rolling-origin CV of every model. Returns one row per (model, fold)."""
    values = series.to_numpy(dtype=float)
    folds = rolling_origin_folds(len(values), horizon=horizon, n_folds=n_folds, step=step)
    rows = []
    for fold_no, fold in enumerate(folds, start=1):
        train, test = values[fold.train_idx], values[fold.test_idx]
        for name, model in MODELS.items():
            pred = model(train, len(test))
            rows.append(
                {
                    "model": name,
                    "fold": fold_no,
                    "origin": series.index[fold.test_idx[0]],
                    "train_weeks": len(train),
                    "mase": round(mase(test, pred, train), 4),
                    "mae": round(mae(test, pred), 2),
                }
            )
    return pd.DataFrame(rows)


def cv_summary(cv: pd.DataFrame) -> pd.DataFrame:
    """Mean/worst MASE per model plus fold wins (honest tie-break by mean)."""
    summary = (
        cv.groupby("model", observed=True)
        .agg(folds=("fold", "nunique"), mean_mase=("mase", "mean"), worst_mase=("mase", "max"), mean_mae=("mae", "mean"))
        .round(4)
        .sort_values("mean_mase")
        .reset_index()
    )
    wins = cv.loc[cv.groupby("fold", observed=True)["mase"].idxmin(), "model"].value_counts()
    summary["fold_wins"] = summary["model"].map(wins).fillna(0).astype(int)
    return summary


def final_fold_forecast(series: pd.Series, horizon: int = 8) -> pd.DataFrame:
    """Forecasts of every model on the LAST fold, for the chart in the PDF/README."""
    values = series.to_numpy(dtype=float)
    train, test = values[:-horizon], values[-horizon:]
    idx = series.index[-horizon:]
    out = pd.DataFrame({"actual": test}, index=idx)
    for name, model in MODELS.items():
        out[name] = model(train, horizon)
    return out


# --------------------------------------------------------------------------- #
# Per-SKU weekly demand (top 10 by units)
# --------------------------------------------------------------------------- #
def moving_average(y: np.ndarray, h: int, k: int = 4) -> np.ndarray:
    return np.full(h, float(np.mean(y[-k:])), dtype=float)


def top_sku_forecasts(sales: pd.DataFrame, n_sku: int = 10, holdout: int = 12) -> pd.DataFrame:
    """Naive / seasonal-naive / 4-week-MA on weekly units for the top SKUs.

    Intermittency (share of zero-demand weeks since the SKU first sold) is
    reported because it is the honest caveat: weekly demand for a single SKU is
    lumpy, and MASE > 1 means the model failed to beat a one-week naive walk.
    """
    top = (
        sales.groupby("StockCode", observed=True)["Quantity"]
        .sum()
        .sort_values(ascending=False)
        .head(n_sku)
    )
    descriptions = (
        sales.loc[sales["StockCode"].isin(top.index)]
        .groupby("StockCode", observed=True)["Description"]
        .agg(lambda s: s.mode().iloc[0] if len(s.mode()) else "")
    )
    weekly_all = (
        sales.set_index("InvoiceDate")
        .groupby("StockCode", observed=True)["Quantity"]
        .resample("W-SUN")
        .sum()
    )
    full_index = weekly_revenue(sales).index

    rows = []
    for code in top.index:
        weekly = weekly_all.loc[code].reindex(full_index, fill_value=0).astype(float)
        active = weekly.loc[weekly.ne(0).idxmax() :]  # from first selling week
        zero_share = float((active == 0).mean())
        values = active.to_numpy()
        if len(values) <= holdout + 4:
            continue
        train, test = values[:-holdout], values[-holdout:]
        row = {
            "StockCode": code,
            "Description": str(descriptions.get(code, "")),
            "TotalUnits": int(top[code]),
            "WeeksActive": len(active),
            "ZeroWeekShare": round(zero_share, 3),
            "mase_naive": round(mase(test, naive(train, holdout), train, m=1), 3),
            "mase_seasonal_naive": round(mase(test, seasonal_naive(train, holdout), train, m=1), 3),
            "mase_ma4": round(mase(test, moving_average(train, holdout), train, m=1), 3),
        }
        candidates = {
            k: row[k]
            for k in ("mase_naive", "mase_seasonal_naive", "mase_ma4")
            if np.isfinite(row[k])
        }
        # all-NaN happens for real: SKU 23843 is one 80,995-unit order placed and
        # cancelled the same day, landing in the dropped partial final week.
        row["best_model"] = (
            min(candidates, key=lambda k: candidates[k]).removeprefix("mase_")
            if candidates
            else "n/a (single-spike demand)"
        )
        rows.append(row)
    return pd.DataFrame(rows)
