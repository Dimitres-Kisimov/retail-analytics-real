"""Predictive Customer Lifetime Value on the cleaned real transactions.

Where RFM is a *snapshot* of who is valuable now and cohort retention is the
*descriptive* time dimension, CLV is the *predictive* leg: a probabilistic model
that turns each customer's purchase history into an expected number of future
transactions and an expected value per transaction, and multiplies them into an
expected forward revenue. It is fit and validated out-of-sample on the same
cleaned sales frame every other module uses -- no re-cleaning, no re-invention.

Two from-scratch models, the standard pairing, implemented with numpy / pandas /
scipy only (no ``lifetimes``, no ML library):

* **BG/NBD** (Fader, Hardie & Lee 2005, "Counting Your Customers the Easy Way")
  models *how many* future transactions a customer will make. Per customer it
  uses three sufficient statistics measured from the data:
    - ``frequency`` x   = number of *repeat* purchase-days (first day excluded);
    - ``recency``  t_x  = days between the first and the last purchase;
    - ``T``             = days between the first purchase and the observation end.
  Multiple invoices on the same calendar day count as one transaction (the model
  is a day-grained counting process), and the four population parameters
  (r, alpha, a, b) are fit by maximum likelihood (``scipy.optimize.minimize``,
  deterministic from a fixed start, so the fit reproduces run to run).

* **Gamma-Gamma** (Fader & Hardie 2013) models *how much* each transaction is
  worth, given the (documented) assumption that a customer's average transaction
  value is independent of their purchase frequency -- an assumption this module
  *measures* (the frequency-vs-value correlation is reported, not assumed away)
  rather than claims. It is fit on the repeat customers only (x >= 1).

CLV(horizon) = E[transactions over horizon | x, t_x, T] * E[value per transaction].

Honesty (on brand with the rest of the repo)
--------------------------------------------
* **Out-of-sample validation is the headline, not the fit.** The observation
  window is split into a *calibration* period and a later *holdout* period; the
  model is fit on calibration only and its predicted holdout transactions are
  compared with what actually happened. The incomplete final month is excluded
  from the holdout so no absence in a partial month is scored as a lost sale.
* **Gross revenue, not margin.** No cost data exists in this dataset, so CLV is
  expected *revenue*, never profit. The horizon is finite (a documented number
  of days) and undiscounted -- a "next-N-days expected revenue", not an
  infinite-horizon discounted figure that would need a made-up discount rate.
* **Identified customers only.** Like RFM and cohort, the ~22.8% of rows with no
  ``CustomerID`` cannot be attributed and are structurally invisible here.

Every reported number is measured from the real data or produced by a
deterministic fit; nothing is invented.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln, hyp2f1

from retail.paths import DELIVERABLES, FIGURES

# Deliverable CLV horizon: the next 180 days (~6 months). Chosen to match the
# length of the holdout window used for validation, so the number people read in
# the CLV table is over the same horizon the model was checked on.
DEFAULT_HORIZON_DAYS = 180


# --------------------------------------------------------------------------- #
# Customer-level sufficient statistics (RFM-for-BG/NBD summary)
# --------------------------------------------------------------------------- #
def summarize_customers(
    sales: pd.DataFrame,
    observation_end: pd.Timestamp | None = None,
    calibration_end: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Per-customer (frequency, recency, T, monetary_value) from cleaned sales.

    Identified customers only. Transactions are collapsed to one per customer per
    calendar day (BG/NBD is a day-grained counter). When ``calibration_end`` is
    given, only purchases on or before it are used to derive the statistics (for
    the calibration/holdout validation); ``observation_end`` (default: the last
    purchase date in the data) sets each customer's age ``T``.

    * frequency x   = number of repeat purchase-days (distinct days minus one);
    * recency  t_x  = days from first to last purchase (0 for one-day customers);
    * T             = days from first purchase to ``observation_end``;
    * monetary_value = mean invoice-day revenue over the *repeat* purchase-days
      (the first day is excluded, matching what Gamma-Gamma models); 0 when x = 0.
    """
    known = sales.loc[sales["CustomerID"].notna(), ["CustomerID", "InvoiceDate", "Revenue"]].copy()
    known["Day"] = known["InvoiceDate"].dt.normalize()
    if calibration_end is not None:
        known = known.loc[known["Day"] <= calibration_end]
    if known.empty:
        return _empty_summary()
    if observation_end is None:
        observation_end = known["Day"].max()

    daily = known.groupby(["CustomerID", "Day"], observed=True)["Revenue"].sum().reset_index()
    grouped = daily.groupby("CustomerID", observed=True)
    first = grouped["Day"].min()
    last = grouped["Day"].max()
    n_days = grouped["Day"].size()

    frequency = (n_days - 1).astype("int64")
    recency = (last - first).dt.days.astype("float64")
    age = (observation_end - first).dt.days.astype("float64")

    # monetary_value: mean over repeat days only (exclude the first purchase-day).
    total_rev = grouped["Revenue"].sum()
    first_rev = daily.loc[daily.groupby("CustomerID", observed=True)["Day"].idxmin()].set_index(
        "CustomerID"
    )["Revenue"]
    repeat_rev = (total_rev - first_rev).where(frequency > 0, 0.0)
    monetary_value = np.where(frequency > 0, repeat_rev / frequency.replace(0, 1), 0.0)

    out = pd.DataFrame(
        {
            "CustomerID": frequency.index,
            "frequency": frequency.to_numpy(),
            "recency": recency.to_numpy(),
            "T": age.to_numpy(),
            "monetary_value": monetary_value,
            "first_purchase": first.to_numpy(),
            "last_purchase": last.to_numpy(),
        }
    )
    return out.reset_index(drop=True)


def _empty_summary() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "CustomerID": pd.Series(dtype="Int64"),
            "frequency": pd.Series(dtype="int64"),
            "recency": pd.Series(dtype="float64"),
            "T": pd.Series(dtype="float64"),
            "monetary_value": pd.Series(dtype="float64"),
            "first_purchase": pd.Series(dtype="datetime64[ns]"),
            "last_purchase": pd.Series(dtype="datetime64[ns]"),
        }
    )


# --------------------------------------------------------------------------- #
# BG/NBD: how many future transactions
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class BGNBDParams:
    r: float
    alpha: float
    a: float
    b: float
    log_likelihood: float
    converged: bool

    def as_tuple(self) -> tuple[float, float, float, float]:
        return (self.r, self.alpha, self.a, self.b)


def bgnbd_negative_log_likelihood(
    log_params: np.ndarray, x: np.ndarray, t_x: np.ndarray, T: np.ndarray
) -> float:
    """Summed negative log-likelihood of BG/NBD (params passed in log space).

    Log space keeps r, alpha, a, b strictly positive during optimisation. The
    log-sum-exp guard keeps the two-term mixture stable when x = 0 (the second
    term is switched off) and for large counts.
    """
    r, alpha, a, b = np.exp(log_params)
    x = np.asarray(x, dtype=float)
    t_x = np.asarray(t_x, dtype=float)
    T = np.asarray(T, dtype=float)

    with np.errstate(all="ignore"):
        a1 = gammaln(r + x) - gammaln(r) + r * np.log(alpha)
        a2 = gammaln(a + b) + gammaln(b + x) - gammaln(b) - gammaln(a + b + x)
        a3 = -(r + x) * np.log(alpha + T)
        a4 = np.log(a) - np.log(b + np.maximum(x, 1) - 1) - (r + x) * np.log(alpha + t_x)

        max34 = np.maximum(a3, a4)
        positive = (x > 0).astype(float)
        mixture = np.exp(a3 - max34) + positive * np.exp(a4 - max34)
        ll = a1 + a2 + np.log(mixture) + max34
        return -float(ll.sum())


def fit_bgnbd(summary: pd.DataFrame) -> BGNBDParams:
    """Maximum-likelihood BG/NBD fit. Deterministic (fixed start, Nelder-Mead)."""
    x = summary["frequency"].to_numpy(dtype=float)
    t_x = summary["recency"].to_numpy(dtype=float)
    T = summary["T"].to_numpy(dtype=float)
    if len(x) == 0:
        raise ValueError("cannot fit BG/NBD on an empty summary")

    x0 = np.zeros(4)  # exp(0) = 1 for every parameter
    res = minimize(
        bgnbd_negative_log_likelihood,
        x0,
        args=(x, t_x, T),
        method="Nelder-Mead",
        options={"maxiter": 20000, "xatol": 1e-9, "fatol": 1e-9},
    )
    r, alpha, a, b = np.exp(res.x)
    return BGNBDParams(
        r=float(r), alpha=float(alpha), a=float(a), b=float(b),
        log_likelihood=float(-res.fun), converged=bool(res.success),
    )


def conditional_expected_purchases(
    params: BGNBDParams,
    t: float,
    x: np.ndarray,
    t_x: np.ndarray,
    T: np.ndarray,
) -> np.ndarray:
    """E[transactions in the next ``t`` days | x, t_x, T] (Fader-Hardie-Lee eq. 10)."""
    r, alpha, a, b = params.as_tuple()
    x = np.asarray(x, dtype=float)
    t_x = np.asarray(t_x, dtype=float)
    T = np.asarray(T, dtype=float)

    with np.errstate(all="ignore"):
        z = t / (alpha + T + t)
        hyp = hyp2f1(r + x, b + x, a + b + x - 1, z)
        first = (a + b + x - 1) / (a - 1)
        bracket = 1.0 - ((alpha + T) / (alpha + T + t)) ** (r + x) * hyp
        numerator = first * bracket

        # The x>0-only term. np.maximum(x, 1) keeps the log argument positive for
        # the x=0 rows it is masked from (they add nothing to the denominator).
        log_ratio = np.log(a) - np.log(b + np.maximum(x, 1) - 1) + (r + x) * (
            np.log(alpha + T) - np.log(alpha + t_x)
        )
        denom = 1.0 + np.where(x > 0, np.exp(log_ratio), 0.0)
        return numerator / denom


def expected_purchases_new_customer(params: BGNBDParams, t: float) -> float:
    """Unconditional E[transactions in ``t`` days] for a just-acquired customer.

    Equals :func:`conditional_expected_purchases` evaluated at (x=0, t_x=0, T=0);
    the test suite asserts that identity as an internal cross-check of the math.
    """
    r, alpha, a, b = params.as_tuple()
    hyp = hyp2f1(r, b, a + b - 1, t / (alpha + t))
    return float((a + b - 1) / (a - 1) * (1.0 - (alpha / (alpha + t)) ** r * hyp))


def prob_alive(params: BGNBDParams, x: np.ndarray, t_x: np.ndarray, T: np.ndarray) -> np.ndarray:
    """P(customer is still active | x, t_x, T). One-time buyers (x = 0) are 1.0."""
    r, alpha, a, b = params.as_tuple()
    x = np.asarray(x, dtype=float)
    t_x = np.asarray(t_x, dtype=float)
    T = np.asarray(T, dtype=float)
    with np.errstate(all="ignore"):
        log_ratio = np.log(a) - np.log(b + np.maximum(x, 1) - 1) + (r + x) * (
            np.log(alpha + T) - np.log(alpha + t_x)
        )
        return np.where(x > 0, 1.0 / (1.0 + np.exp(log_ratio)), 1.0)


# --------------------------------------------------------------------------- #
# Gamma-Gamma: how much each transaction is worth
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class GammaGammaParams:
    p: float
    q: float
    v: float
    log_likelihood: float
    converged: bool
    freq_value_corr: float  # measured frequency-vs-value correlation (assumption check)

    def population_mean(self) -> float:
        """E[average transaction value] across the population (needs q > 1)."""
        return float(self.v * self.p / (self.q - 1.0)) if self.q > 1.0 else float("nan")


def gamma_gamma_negative_log_likelihood(
    log_params: np.ndarray, x: np.ndarray, m: np.ndarray
) -> float:
    """Summed negative log-likelihood of the Gamma-Gamma model (log-space params).

    ``x`` is repeat frequency (>= 1) and ``m`` the observed mean repeat value.
    """
    p, q, v = np.exp(log_params)
    x = np.asarray(x, dtype=float)
    m = np.asarray(m, dtype=float)
    with np.errstate(all="ignore"):
        ll = (
            gammaln(p * x + q)
            - gammaln(p * x)
            - gammaln(q)
            + q * np.log(v)
            + (p * x - 1.0) * np.log(m)
            + p * x * np.log(x)
            - (p * x + q) * np.log(x * m + v)
        )
        return -float(ll.sum())


def fit_gamma_gamma(summary: pd.DataFrame) -> GammaGammaParams:
    """MLE Gamma-Gamma fit on the repeat customers (frequency >= 1, value > 0).

    Monetary values are mean-normalised before fitting. The Gamma-Gamma
    likelihood is scale-invariant in ``p`` and ``q`` (only the scale ``v`` moves,
    linearly), so fitting on values of order 1 removes the ill-conditioning that
    would otherwise let the optimiser wander off along a flat ridge (p -> inf,
    v -> 0). ``v`` is rescaled back to the original currency afterward, so the
    reported parameters and the population mean are in GBP.
    """
    repeat = summary.loc[(summary["frequency"] > 0) & (summary["monetary_value"] > 0)]
    x = repeat["frequency"].to_numpy(dtype=float)
    m = repeat["monetary_value"].to_numpy(dtype=float)
    if len(x) == 0:
        raise ValueError("cannot fit Gamma-Gamma without repeat customers")
    corr = float(np.corrcoef(x, m)[0, 1]) if len(x) > 1 else 0.0

    scale = float(m.mean())
    m_scaled = m / scale
    x0 = np.zeros(3)
    res = minimize(
        gamma_gamma_negative_log_likelihood,
        x0,
        args=(x, m_scaled),
        method="Nelder-Mead",
        options={"maxiter": 20000, "xatol": 1e-9, "fatol": 1e-9},
    )
    p, q, v = np.exp(res.x)
    # NLL(m, v) = NLL(m/scale, v/scale) - N*log(scale): p, q invariant, v scales.
    return GammaGammaParams(
        p=float(p), q=float(q), v=float(v * scale),
        log_likelihood=float(-res.fun - len(x) * np.log(scale)),
        converged=bool(res.success), freq_value_corr=corr,
    )


def conditional_expected_average_value(
    gg: GammaGammaParams, x: np.ndarray, m: np.ndarray
) -> np.ndarray:
    """E[value per transaction | x, observed mean m] (Bayesian shrinkage toward the population).

    A credibility-weighted blend of the customer's own mean and the population
    mean; the more repeat purchases a customer has, the more weight their own
    average carries. Customers with x = 0 fall back to the population mean.
    """
    x = np.asarray(x, dtype=float)
    m = np.asarray(m, dtype=float)
    population = gg.population_mean()
    weight = (gg.p * x) / (gg.p * x + gg.q - 1.0)
    blended = (1.0 - weight) * population + weight * m
    return np.where(x > 0, blended, population)


# --------------------------------------------------------------------------- #
# CLV assembly
# --------------------------------------------------------------------------- #
def customer_lifetime_value(
    summary: pd.DataFrame,
    bgnbd: BGNBDParams,
    gg: GammaGammaParams,
    horizon_days: int = DEFAULT_HORIZON_DAYS,
) -> pd.DataFrame:
    """Per-customer predicted purchases, value/transaction, P(alive) and CLV.

    CLV = predicted transactions over the horizon * predicted value per
    transaction -- expected *revenue* over the next ``horizon_days`` days.
    """
    x = summary["frequency"].to_numpy(dtype=float)
    t_x = summary["recency"].to_numpy(dtype=float)
    T = summary["T"].to_numpy(dtype=float)
    m = summary["monetary_value"].to_numpy(dtype=float)

    pred_txn = conditional_expected_purchases(bgnbd, horizon_days, x, t_x, T)
    pred_value = conditional_expected_average_value(gg, x, m)
    alive = prob_alive(bgnbd, x, t_x, T)
    clv = pred_txn * pred_value

    out = summary[["CustomerID", "frequency", "recency", "T", "monetary_value"]].copy()
    out["prob_alive"] = np.round(alive, 4)
    out["pred_purchases"] = np.round(pred_txn, 4)
    out["pred_avg_value"] = np.round(pred_value, 2)
    out["clv"] = np.round(clv, 2)
    return out.sort_values("clv", ascending=False, kind="stable").reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Out-of-sample calibration / holdout validation (the honest headline)
# --------------------------------------------------------------------------- #
@dataclass
class ValidationResult:
    calibration_end: pd.Timestamp
    holdout_end: pd.Timestamp
    holdout_days: int
    n_customers: int
    predicted_total: float       # sum of predicted holdout transactions
    actual_total: int            # sum of actual holdout transactions
    mae: float                   # per-customer mean absolute error
    correlation: float           # predicted vs actual, across customers
    by_frequency: pd.DataFrame   # predicted vs actual grouped by calibration frequency

    @property
    def ratio(self) -> float:
        return self.predicted_total / self.actual_total if self.actual_total else float("nan")


def calibration_holdout_validation(
    sales: pd.DataFrame,
    calibration_end: pd.Timestamp,
    holdout_end: pd.Timestamp,
) -> tuple[ValidationResult, BGNBDParams]:
    """Fit BG/NBD on the calibration window, score its holdout predictions.

    ``holdout_end`` should be a *complete* month boundary so no absence in a
    partial month is counted as a lost sale.
    """
    calibration_end = pd.Timestamp(calibration_end)
    holdout_end = pd.Timestamp(holdout_end)
    holdout_days = int((holdout_end - calibration_end).days)

    cal = summarize_customers(sales, observation_end=calibration_end, calibration_end=calibration_end)
    params = fit_bgnbd(cal)

    predicted = conditional_expected_purchases(
        params, holdout_days,
        cal["frequency"].to_numpy(), cal["recency"].to_numpy(), cal["T"].to_numpy(),
    )

    # Actual holdout transactions: distinct purchase-days in (calibration_end, holdout_end].
    known = sales.loc[sales["CustomerID"].notna(), ["CustomerID", "InvoiceDate"]].copy()
    known["Day"] = known["InvoiceDate"].dt.normalize()
    hold = known.loc[(known["Day"] > calibration_end) & (known["Day"] <= holdout_end)]
    actual_days = hold.groupby("CustomerID", observed=True)["Day"].nunique()
    actual = cal["CustomerID"].map(actual_days).fillna(0).to_numpy(dtype=float)

    frame = pd.DataFrame({"frequency": cal["frequency"].to_numpy(), "predicted": predicted, "actual": actual})
    by_freq = _validation_by_frequency(frame)

    return (
        ValidationResult(
            calibration_end=calibration_end,
            holdout_end=holdout_end,
            holdout_days=holdout_days,
            n_customers=int(len(cal)),
            predicted_total=float(predicted.sum()),
            actual_total=int(actual.sum()),
            mae=float(np.mean(np.abs(predicted - actual))),
            correlation=float(np.corrcoef(predicted, actual)[0, 1]) if len(actual) > 1 else float("nan"),
            by_frequency=by_freq,
        ),
        params,
    )


def _validation_by_frequency(frame: pd.DataFrame) -> pd.DataFrame:
    """Predicted vs actual holdout transactions, grouped by calibration frequency.

    Frequencies of 5+ are pooled into one '5+' bucket so no bucket is a handful of
    customers. This is the classic BG/NBD calibration chart in table form.
    """
    bucket = np.where(frame["frequency"] >= 5, 5, frame["frequency"]).astype(int)
    frame = frame.assign(bucket=bucket)
    grouped = frame.groupby("bucket", observed=True)
    out = pd.DataFrame(
        {
            "customers": grouped.size(),
            "predicted_mean": grouped["predicted"].mean().round(4),
            "actual_mean": grouped["actual"].mean().round(4),
        }
    ).reset_index()
    out["cal_frequency"] = out["bucket"].map(lambda k: "5+" if k >= 5 else str(k))
    return out[["cal_frequency", "customers", "predicted_mean", "actual_mean"]]


# --------------------------------------------------------------------------- #
# One-call orchestration
# --------------------------------------------------------------------------- #
@dataclass
class CLVResult:
    summary: pd.DataFrame
    bgnbd: BGNBDParams
    gamma_gamma: GammaGammaParams
    clv_table: pd.DataFrame
    horizon_days: int
    observation_end: pd.Timestamp
    validation: ValidationResult | None

    @property
    def n_customers(self) -> int:
        return int(len(self.summary))

    @property
    def n_repeat(self) -> int:
        return int((self.summary["frequency"] > 0).sum())

    def concentration(self, top_frac: float = 0.10) -> dict:
        """Share of total predicted CLV held by the top ``top_frac`` of customers."""
        clv = self.clv_table["clv"].to_numpy()
        total = float(clv.sum())
        k = max(1, int(round(len(clv) * top_frac)))
        top_share = float(clv[:k].sum()) / total if total else float("nan")
        return {"top_frac": top_frac, "top_customers": k, "top_share": top_share, "total_clv": total}


def run_clv(
    sales: pd.DataFrame,
    horizon_days: int = DEFAULT_HORIZON_DAYS,
    calibration_end: pd.Timestamp | None = None,
    holdout_end: pd.Timestamp | None = None,
    validate: bool = True,
) -> CLVResult:
    """Full CLV pass: summary -> BG/NBD + Gamma-Gamma fit -> CLV table -> validation.

    ``calibration_end`` / ``holdout_end`` default to a 6-month holdout ending at
    the last *complete* month in the data (the partial final month is excluded).
    """
    summary = summarize_customers(sales)
    bgnbd = fit_bgnbd(summary)
    gg = fit_gamma_gamma(summary)
    observation_end = summary["last_purchase"].max()
    clv_table = customer_lifetime_value(summary, bgnbd, gg, horizon_days)

    validation = None
    if validate:
        cal_end, hold_end = _default_validation_window(sales, calibration_end, holdout_end)
        if cal_end is not None:
            validation, _ = calibration_holdout_validation(sales, cal_end, hold_end)

    return CLVResult(
        summary=summary, bgnbd=bgnbd, gamma_gamma=gg, clv_table=clv_table,
        horizon_days=horizon_days, observation_end=observation_end, validation=validation,
    )


def _default_validation_window(
    sales: pd.DataFrame,
    calibration_end: pd.Timestamp | None,
    holdout_end: pd.Timestamp | None,
) -> tuple[pd.Timestamp | None, pd.Timestamp | None]:
    """A 6-month holdout ending at the last complete month, unless overridden."""
    if calibration_end is not None and holdout_end is not None:
        return pd.Timestamp(calibration_end), pd.Timestamp(holdout_end)
    dates = sales.loc[sales["CustomerID"].notna(), "InvoiceDate"]
    if dates.empty:
        return None, None
    max_date = dates.max()
    last_period = max_date.to_period("M")
    partial = max_date < last_period.end_time
    last_complete = (last_period - 1) if partial else last_period
    hold_end = last_complete.end_time.normalize()
    cal_end = (last_complete - 6).end_time.normalize()
    # Guard: need calibration transactions before cal_end.
    if (dates.dt.normalize() <= cal_end).sum() == 0:
        return None, None
    return cal_end, hold_end


# --------------------------------------------------------------------------- #
# Reporting / export
# --------------------------------------------------------------------------- #
def headline_text(result: CLVResult) -> str:
    """One-line headline: validation accuracy + CLV concentration."""
    conc = result.concentration(0.10)
    parts = [
        f"BG/NBD + Gamma-Gamma on {result.n_customers:,} identified customers "
        f"({result.n_repeat:,} repeat); top 10% hold {100 * conc['top_share']:.1f}% of predicted "
        f"{result.horizon_days}-day revenue"
    ]
    v = result.validation
    if v is not None and v.actual_total:
        over = 100 * (v.ratio - 1.0)
        direction = "over" if over >= 0 else "under"
        parts.append(
            f"holdout check: predicted {v.predicted_total:,.0f} vs actual {v.actual_total:,} "
            f"transactions ({abs(over):.1f}% {direction}), correlation {v.correlation:.2f}"
        )
    return "; ".join(parts)


def clv_summary_table(result: CLVResult, top: int = 10) -> pd.DataFrame:
    """Top-``top`` customers by predicted CLV, for the CLI print and the PDF."""
    cols = ["CustomerID", "frequency", "recency", "T", "monetary_value",
            "prob_alive", "pred_purchases", "pred_avg_value", "clv"]
    return result.clv_table[cols].head(top).reset_index(drop=True)


def write_csv(result: CLVResult, out_dir=DELIVERABLES):
    """Write the full per-customer CLV table (deterministic, rounded)."""
    out = out_dir / "customer_lifetime_value.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    result.clv_table.to_csv(out, index=False, encoding="utf-8", lineterminator="\n")
    return out


def lorenz_points(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Cumulative customer share (x) vs cumulative CLV share (y), ascending order.

    Both start at (0, 0). A diagonal would be perfect equality; the real curve
    bows below it, and how far it bows is the concentration story.
    """
    v = np.sort(np.asarray(values, dtype=float))
    total = v.sum()
    n = len(v)
    cum_x = np.arange(0, n + 1) / n
    cum_y = np.concatenate([[0.0], np.cumsum(v) / total]) if total > 0 else np.zeros(n + 1)
    return cum_x, cum_y


def fig_clv(result: CLVResult, out_dir=FIGURES):
    """Two-panel committed figure: out-of-sample validation + CLV concentration.

    Deterministic (no timestamps, no RNG): re-running writes the same PNG bytes.
    Returns None when there is nothing to plot (empty/degenerate input).
    """
    import matplotlib.pyplot as plt  # noqa: F401  (Agg backend set in retail.eda)

    from retail.eda import BLUE, GREEN, INK, INK_2, MUTED, _save, _style

    if result.clv_table.empty:
        return None
    _style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.6))

    # Panel A: predicted vs actual mean holdout transactions, by calibration frequency
    v = result.validation
    if v is not None and not v.by_frequency.empty:
        bf = v.by_frequency
        idx = np.arange(len(bf))
        w = 0.4
        ax1.bar(idx - w / 2, bf["actual_mean"], width=w, color=BLUE, label="actual")
        ax1.bar(idx + w / 2, bf["predicted_mean"], width=w, color=GREEN, label="predicted")
        ax1.set_xticks(idx)
        ax1.set_xticklabels(bf["cal_frequency"])
        ax1.set_xlabel("purchases in calibration period")
        ax1.set_ylabel("mean transactions in holdout")
        ax1.grid(axis="x", visible=False)
        ax1.legend(frameon=False, fontsize=8)
        ax1.set_title(
            f"Out-of-sample check: predicted vs actual\n"
            f"({v.holdout_days}-day holdout, {v.n_customers:,} customers, corr {v.correlation:.2f})",
            fontsize=10.5,
        )
    else:
        ax1.axis("off")
        ax1.set_title("Out-of-sample validation unavailable", fontsize=10.5)

    # Panel B: Lorenz curve of predicted CLV
    cum_x, cum_y = lorenz_points(result.clv_table["clv"].to_numpy())
    ax2.plot(100 * cum_x, 100 * cum_y, color=BLUE, linewidth=2)
    ax2.plot([0, 100], [0, 100], color=MUTED, linewidth=1, linestyle="--")
    ax2.set_xlabel("share of customers (%, lowest CLV first)")
    ax2.set_ylabel("share of predicted CLV (%)")
    ax2.grid(axis="x", visible=False)
    conc = result.concentration(0.10)
    top_pct = 100 * (1.0 - conc["top_frac"])  # top 10% are the rightmost 10% of the curve
    y_at = 100 * (1.0 - conc["top_share"])
    ax2.axvline(top_pct, color=INK_2, linewidth=0.8, linestyle=":")
    ax2.annotate(
        f"top 10% of customers\n= {100 * conc['top_share']:.0f}% of predicted CLV",
        xy=(top_pct, y_at), xytext=(8, 62), fontsize=8, color=INK_2,
        arrowprops={"arrowstyle": "-", "color": MUTED, "linewidth": 0.8},
    )
    ax2.set_title(
        f"Predicted {result.horizon_days}-day CLV concentration (Lorenz)", fontsize=10.5
    )

    fig.suptitle(
        "Customer lifetime value - BG/NBD (frequency) x Gamma-Gamma (value), real transactions",
        fontsize=12, fontweight="bold", x=0.02, ha="left", color=INK,
    )
    fig.tight_layout(rect=(0, 0.02, 1, 0.94))
    return _save(fig, out_dir / "clv_validation.png", tight=False)
