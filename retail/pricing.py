"""Product-level price ladders and an observational price/quantity slope.

Every SKU in Online Retail II has a *ladder*: the set of unit prices it was
actually charged at across the real invoices. 88.7% of SKUs sold at more than one
price, so the ladder is not an edge case -- it is how this business prices. This
module measures that ladder and then asks the obvious next question as honestly
as observational data allows: **when the price was lower, did more units move?**

What it measures (all from the cleaned sales frame, numpy/pandas only)
---------------------------------------------------------------------
* **The ladder.** Per SKU: every distinct price it sold at, the units and revenue
  on each rung, the **posted price** (the modal price by *lines* -- the rung most
  invoice lines were written at; ties break to the lower price), the **realized
  price** (revenue / units), and the p90/p10 spread of charged prices.
* **Price realization.** Realized price over posted price, and the arithmetic
  difference between the revenue that was booked and the revenue the *same units*
  would have carried at the posted price. That is arithmetic, not a forecast: it
  does **not** claim those units would have sold at the posted price.
* **Two independent price/quantity slopes**, both fitted per SKU as a log-log OLS
  slope, and deliberately kept apart because they are not the same measurement:

  1. ``slope_within_week`` -- line-level, with the SKU-week mean removed. All the
     variation left is *cross-buyer, same week*: one buyer paid less than another
     in the same week. That is the seller's own volume-discount schedule, so this
     slope is a **price-schedule descriptor, not a demand response**.
  2. ``slope_market_adj`` -- week-level: the SKU's posted price that week against
     its units that week, after subtracting the whole assortment's weekly log-unit
     index (a market control, so a Christmas week does not read as a price effect).
     This is the closest thing here to "the price moved and demand answered", and
     it is still observational.

  A **permutation null** (units reshuffled across that SKU's own weeks, seeded per
  SKU, deterministic) says whether each SKU's market-adjusted slope is bigger than
  its own noise. The share of SKUs that clear it is reported, never assumed.

Honesty guards
--------------
* **This is not an elasticity and no causal claim is made anywhere.** Prices here
  were not assigned by an experiment; they move with order size, customer type,
  season and promotions at once. The word used throughout is *proxy*.
* **-1.0 is the mechanical benchmark**: a buyer spending the same amount per line
  regardless of price produces a slope of exactly -1 with zero demand response.
  The measured numbers are quoted against that benchmark, not against zero.
* **The two slopes are reported side by side precisely because they disagree.**
  They agree on the assortment median and correlate weakly per SKU -- which is the
  finding: this data supports an assortment-level statement, not a per-SKU price
  recommendation. ``overview["slope_agreement_corr"]`` is that correlation, and it
  is printed, not buried.
* Only SKUs with enough real price variation and enough lines qualify (see
  ``PricingConfig``); the qualifying set and its revenue share are reported so the
  reader knows what fraction of the business the numbers speak for.
* No cost data exists in this dataset, so nothing here is a margin statement.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from retail.paths import DELIVERABLES, FIGURES

# A buyer who spends the same amount per line whatever the price produces exactly
# this slope with no demand response at all. Every measured slope is read against it.
CONSTANT_SPEND_SLOPE = -1.0


@dataclass(frozen=True)
class PricingConfig:
    """Documented thresholds. Defaults are tuned for the full dataset; the tests
    lower them so the same code runs on the small committed fixture."""

    min_distinct_prices: int = 4     # a SKU needs a real ladder, not two rungs
    min_lines: int = 200             # ...and enough invoice lines to fit anything on
    min_weeks: int = 12              # weeks with sales, for the week-level slope
    min_week_prices: int = 3         # distinct posted prices across those weeks
    n_permutations: int = 199        # permutation null draws per SKU
    seed: int = 42                   # per-SKU child seeds derive from this
    top_ladders: int = 10            # SKUs drawn on the plate / listed in the PDF


@dataclass
class PricingResult:
    """Everything the report needs, all derived from the cleaned sales frame."""

    ladder: pd.DataFrame       # one row per (StockCode, Price) rung
    profile: pd.DataFrame      # one row per qualifying SKU: ladder shape + realization
    slopes: pd.DataFrame       # one row per SKU that could be fitted: the two slopes
    overview: dict             # headline scalars
    config: PricingConfig

    def headline(self) -> dict:
        o = self.overview
        return {
            "multi_price_share": o["multi_price_share"],
            "median_rungs": o["median_rungs"],
            "median_spread": o["median_spread"],
            "median_slope_market_adj": o["median_slope_market_adj"],
            "median_slope_within_week": o["median_slope_within_week"],
            "slope_agreement_corr": o["slope_agreement_corr"],
        }


# --------------------------------------------------------------------------- #
# Small shared helpers
# --------------------------------------------------------------------------- #
def week_key(df: pd.DataFrame) -> pd.Series:
    """ISO year+week as one sortable integer (2011-W07 -> 201107)."""
    return df["ISOYear"].astype("int64") * 100 + df["ISOWeek"].astype("int64")


def ols_slope_by(frame: pd.DataFrame, by: str, x: str, y: str) -> pd.Series:
    """Per-group OLS slope of ``y`` on ``x``, vectorised over groups.

    Groups whose ``x`` has no variance return NaN rather than a fabricated slope.
    """
    if frame.empty:
        return pd.Series(dtype="float64", name="slope")
    t = frame[[by, x, y]].copy()
    t["_xx"] = t[x] * t[x]
    t["_xy"] = t[x] * t[y]
    a = t.groupby(by, observed=True).agg(
        n=(x, "size"), sx=(x, "sum"), sy=(y, "sum"), sxx=("_xx", "sum"), sxy=("_xy", "sum")
    )
    num = a["sxy"] - a["sx"] * a["sy"] / a["n"]
    den = a["sxx"] - a["sx"] * a["sx"] / a["n"]
    return (num / den).where(den > 1e-12).rename("slope")


def _posted_price(frame: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """Modal price by number of invoice lines within ``keys``; ties -> lower price."""
    counts = (
        frame.groupby([*keys, "Price"], observed=True)
        .size()
        .rename("lines")
        .reset_index()
        .sort_values([*keys, "lines", "Price"], ascending=[*([True] * len(keys)), False, True])
    )
    return counts.drop_duplicates(keys)[[*keys, "Price"]].rename(columns={"Price": "posted_price"})


def qualifying_skus(sales: pd.DataFrame, config: PricingConfig) -> pd.Index:
    """SKUs with a real ladder (>= min_distinct_prices) and enough lines to fit on."""
    if sales.empty:
        return pd.Index([], name="StockCode")
    g = sales.groupby("StockCode", observed=True)
    stats = pd.DataFrame({"prices": g["Price"].nunique(), "lines": g.size()})
    keep = stats[(stats["prices"] >= config.min_distinct_prices) & (stats["lines"] >= config.min_lines)]
    return pd.Index(sorted(keep.index), name="StockCode")


def _descriptions(sales: pd.DataFrame) -> pd.Series:
    """StockCode -> its most frequent non-null description (deterministic tie-break)."""
    sub = sales.loc[sales["Description"].notna(), ["StockCode", "Description"]].copy()
    if sub.empty:
        return pd.Series(dtype="object", name="Description")
    sub["Description"] = sub["Description"].astype(str).str.strip()
    counts = sub.groupby(["StockCode", "Description"], observed=True).size().rename("n").reset_index()
    counts = counts.sort_values(["StockCode", "n", "Description"], ascending=[True, False, True])
    return counts.drop_duplicates("StockCode").set_index("StockCode")["Description"]


# --------------------------------------------------------------------------- #
# The ladder
# --------------------------------------------------------------------------- #
LADDER_COLUMNS = ["StockCode", "Price", "lines", "units", "revenue", "unit_share", "is_posted"]


def price_ladder(sales: pd.DataFrame, config: PricingConfig, skus: pd.Index | None = None) -> pd.DataFrame:
    """One row per (SKU, price actually charged): lines, units, revenue, unit share."""
    skus = qualifying_skus(sales, config) if skus is None else skus
    sub = sales[sales["StockCode"].isin(skus)]
    if sub.empty:
        return pd.DataFrame(columns=LADDER_COLUMNS)
    rungs = (
        sub.groupby(["StockCode", "Price"], observed=True)
        .agg(lines=("Quantity", "size"), units=("Quantity", "sum"), revenue=("Revenue", "sum"))
        .reset_index()
    )
    total_units = rungs.groupby("StockCode", observed=True)["units"].transform("sum")
    rungs["unit_share"] = rungs["units"] / total_units
    posted = _posted_price(sub, ["StockCode"])
    rungs = rungs.merge(posted, on="StockCode", how="left")
    rungs["is_posted"] = np.isclose(rungs["Price"], rungs["posted_price"])
    rungs = rungs.drop(columns="posted_price")
    return rungs.sort_values(["StockCode", "Price"]).reset_index(drop=True)[LADDER_COLUMNS]


PROFILE_COLUMNS = [
    "StockCode", "Description", "rungs", "lines", "units", "revenue",
    "posted_price", "realized_price", "realization", "price_p10", "price_p90", "spread",
    "unit_share_below", "unit_share_at", "unit_share_above",
]


def sku_profile(sales: pd.DataFrame, config: PricingConfig, ladder: pd.DataFrame | None = None,
                skus: pd.Index | None = None) -> pd.DataFrame:
    """Per-SKU ladder shape: rungs, posted vs realized price, spread, unit split."""
    skus = qualifying_skus(sales, config) if skus is None else skus
    sub = sales[sales["StockCode"].isin(skus)]
    if sub.empty:
        return pd.DataFrame(columns=PROFILE_COLUMNS)
    lad = price_ladder(sales, config, skus) if ladder is None else ladder

    agg = sub.groupby("StockCode", observed=True).agg(
        lines=("Quantity", "size"), units=("Quantity", "sum"), revenue=("Revenue", "sum"),
        price_p10=("Price", lambda s: s.quantile(0.10)), price_p90=("Price", lambda s: s.quantile(0.90)),
    )
    posted = _posted_price(sub, ["StockCode"]).set_index("StockCode")["posted_price"]
    out = agg.join(posted)
    out["rungs"] = lad.groupby("StockCode", observed=True).size()
    out["realized_price"] = out["revenue"] / out["units"]
    out["realization"] = out["realized_price"] / out["posted_price"]
    out["spread"] = out["price_p90"] / out["price_p10"]

    # unit split against the posted rung, straight off the ladder
    j = lad.merge(out[["posted_price"]], left_on="StockCode", right_index=True)
    side = np.where(np.isclose(j["Price"], j["posted_price"]), "at",
                    np.where(j["Price"] < j["posted_price"], "below", "above"))
    j = j.assign(side=side)
    split = j.pivot_table(index="StockCode", columns="side", values="unit_share",
                          aggfunc="sum", observed=True).fillna(0.0)
    for name in ("below", "at", "above"):
        out[f"unit_share_{name}"] = split[name] if name in split.columns else 0.0

    out["Description"] = out.index.map(_descriptions(sales)).fillna("")
    out = out.reset_index()
    return out.sort_values(["revenue", "StockCode"], ascending=[False, True]).reset_index(drop=True)[
        PROFILE_COLUMNS
    ]


# --------------------------------------------------------------------------- #
# The two slopes
# --------------------------------------------------------------------------- #
def weekly_frame(sales: pd.DataFrame, config: PricingConfig, skus: pd.Index | None = None) -> pd.DataFrame:
    """SKU-week panel: posted price that week, units that week, market-adjusted units.

    ``log_units_adj`` subtracts the whole cleaned frame's weekly log-unit index, so a
    gift-season week lifts every SKU's control, not one SKU's "price effect".
    """
    skus = qualifying_skus(sales, config) if skus is None else skus
    sub = sales[sales["StockCode"].isin(skus)]
    if sub.empty:
        return pd.DataFrame(columns=["StockCode", "week", "posted_price", "units",
                                     "log_price", "log_units", "log_units_adj"])
    sub = sub.assign(week=week_key(sub))
    posted = _posted_price(sub, ["StockCode", "week"])
    units = (
        sub.groupby(["StockCode", "week"], observed=True)["Quantity"].sum().rename("units").reset_index()
    )
    panel = posted.merge(units, on=["StockCode", "week"])
    panel = panel[panel["units"] > 0].copy()

    market = sales.assign(week=week_key(sales)).groupby("week", observed=True)["Quantity"].sum()
    log_market = np.log(market.astype("float64"))
    log_market = log_market - log_market.mean()

    panel["log_price"] = np.log(panel["posted_price"].to_numpy(dtype="float64"))
    panel["log_units"] = np.log(panel["units"].to_numpy(dtype="float64"))
    panel["log_units_adj"] = panel["log_units"] - panel["week"].map(log_market).to_numpy(dtype="float64")
    return panel.sort_values(["StockCode", "week"]).reset_index(drop=True)


def within_week_slopes(sales: pd.DataFrame, config: PricingConfig,
                       skus: pd.Index | None = None) -> pd.Series:
    """Line-level log-log slope with the SKU-week mean removed.

    What is left is cross-buyer variation inside one week -- the seller's own
    volume-discount schedule. Named for what it is, never called a demand curve.
    """
    skus = qualifying_skus(sales, config) if skus is None else skus
    sub = sales[sales["StockCode"].isin(skus)]
    if sub.empty:
        return pd.Series(dtype="float64", name="slope_within_week")
    sub = sub.assign(
        week=week_key(sub),
        log_price=np.log(sub["Price"].to_numpy(dtype="float64")),
        log_units=np.log(sub["Quantity"].to_numpy(dtype="float64")),
    )
    key = sub["StockCode"].astype(str) + "|" + sub["week"].astype(str)
    sub = sub.assign(_key=key)
    means = sub.groupby("_key", observed=True)[["log_price", "log_units"]].transform("mean")
    sub = sub.assign(
        log_price_d=sub["log_price"] - means["log_price"],
        log_units_d=sub["log_units"] - means["log_units"],
    )
    return ols_slope_by(sub, "StockCode", "log_price_d", "log_units_d").rename("slope_within_week")


def _permutation_p(log_price: np.ndarray, log_units: np.ndarray, observed: float,
                   n_perm: int, rng: np.random.Generator) -> float:
    """Two-sided p-value of ``observed`` against slopes from reshuffled units.

    The x values (and therefore the OLS denominator) are held fixed; only the
    pairing is destroyed, which is exactly the null "this SKU's weekly units are
    unrelated to the price posted that week".
    """
    xc = log_price - log_price.mean()
    den = float((xc * xc).sum())
    if den <= 1e-12 or not np.isfinite(observed):
        return float("nan")
    draws = np.array([rng.permutation(log_units) for _ in range(n_perm)])
    null = (draws - draws.mean(axis=1, keepdims=True)) @ xc / den
    return float((np.sum(np.abs(null) >= abs(observed)) + 1) / (n_perm + 1))


SLOPE_COLUMNS = ["StockCode", "weeks", "week_prices", "slope_within_week",
                 "slope_between_week", "slope_market_adj", "perm_p"]


def price_quantity_slopes(sales: pd.DataFrame, config: PricingConfig,
                          skus: pd.Index | None = None) -> pd.DataFrame:
    """Both slopes per SKU plus the seeded permutation p-value of the adjusted one."""
    skus = qualifying_skus(sales, config) if skus is None else skus
    panel = weekly_frame(sales, config, skus)
    if panel.empty:
        return pd.DataFrame(columns=SLOPE_COLUMNS)

    counts = panel.groupby("StockCode", observed=True).agg(
        weeks=("week", "size"), week_prices=("posted_price", "nunique")
    )
    fittable = counts[(counts["weeks"] >= config.min_weeks)
                      & (counts["week_prices"] >= config.min_week_prices)]
    panel = panel[panel["StockCode"].isin(fittable.index)]
    if panel.empty:
        return pd.DataFrame(columns=SLOPE_COLUMNS)

    out = fittable.loc[sorted(fittable.index)].copy()
    out["slope_within_week"] = within_week_slopes(sales, config, pd.Index(out.index))
    out["slope_between_week"] = ols_slope_by(panel, "StockCode", "log_price", "log_units")
    out["slope_market_adj"] = ols_slope_by(panel, "StockCode", "log_price", "log_units_adj")

    # Permutation null, one deterministic child seed per SKU: the p-values do not
    # depend on iteration order, so a re-run reproduces them exactly.
    groups = dict(tuple(panel.groupby("StockCode", observed=True)))
    p_values = []
    for i, code in enumerate(out.index):
        d = groups[code]
        rng = np.random.default_rng([config.seed, i])
        p_values.append(
            _permutation_p(
                d["log_price"].to_numpy(dtype="float64"),
                d["log_units_adj"].to_numpy(dtype="float64"),
                float(out.loc[code, "slope_market_adj"]),
                config.n_permutations,
                rng,
            )
        )
    out["perm_p"] = p_values
    return out.reset_index()[SLOPE_COLUMNS]


# --------------------------------------------------------------------------- #
# Assembly
# --------------------------------------------------------------------------- #
def _overview(sales: pd.DataFrame, profile: pd.DataFrame, slopes: pd.DataFrame,
              config: PricingConfig) -> dict:
    all_prices = sales.groupby("StockCode", observed=True)["Price"].nunique() if len(sales) else pd.Series(dtype=int)
    n_skus = int(len(all_prices))
    n_multi = int((all_prices > 1).sum())
    total_revenue = float(sales["Revenue"].sum()) if len(sales) else 0.0

    units = profile["units"].to_numpy(dtype="float64") if len(profile) else np.array([])
    revenue = profile["revenue"].to_numpy(dtype="float64") if len(profile) else np.array([])
    at_posted = (profile["posted_price"].to_numpy(dtype="float64") * units) if len(profile) else np.array([])

    def _wmean(col: str) -> float:
        if not len(profile) or units.sum() <= 0:
            return 0.0
        return float((profile[col].to_numpy(dtype="float64") * units).sum() / units.sum())

    fitted = slopes.dropna(subset=["slope_within_week", "slope_market_adj"]) if len(slopes) else slopes
    corr = float("nan")
    if len(fitted) >= 3:
        corr = float(np.corrcoef(fitted["slope_within_week"], fitted["slope_market_adj"])[0, 1])
    tested = slopes["perm_p"].notna().sum() if len(slopes) else 0

    return {
        "n_skus": n_skus,
        "n_multi_price_skus": n_multi,
        "multi_price_share": (n_multi / n_skus) if n_skus else 0.0,
        "n_qualifying": int(len(profile)),
        "qualifying_revenue": float(revenue.sum()),
        "qualifying_revenue_share": (float(revenue.sum()) / total_revenue) if total_revenue else 0.0,
        "median_rungs": float(profile["rungs"].median()) if len(profile) else float("nan"),
        "median_spread": float(profile["spread"].median()) if len(profile) else float("nan"),
        "unit_share_below": _wmean("unit_share_below"),
        "unit_share_at": _wmean("unit_share_at"),
        "unit_share_above": _wmean("unit_share_above"),
        "revenue_at_posted": float(at_posted.sum()),
        "realization": (float(revenue.sum()) / float(at_posted.sum())) if at_posted.sum() else float("nan"),
        "n_fitted": int(len(fitted)),
        "median_slope_within_week": float(fitted["slope_within_week"].median()) if len(fitted) else float("nan"),
        "median_slope_between_week": float(fitted["slope_between_week"].median()) if len(fitted) else float("nan"),
        "median_slope_market_adj": float(fitted["slope_market_adj"].median()) if len(fitted) else float("nan"),
        "share_slope_negative": float((fitted["slope_market_adj"] < 0).mean()) if len(fitted) else float("nan"),
        "slope_agreement_corr": corr,
        "n_perm_tested": int(tested),
        "n_perm_significant": int((slopes["perm_p"] <= 0.05).sum()) if len(slopes) else 0,
        "perm_significant_share": (float((slopes["perm_p"] <= 0.05).sum()) / int(tested)) if tested else float("nan"),
        "constant_spend_slope": CONSTANT_SPEND_SLOPE,
    }


def run_pricing_analysis(sales: pd.DataFrame, config: PricingConfig | None = None) -> PricingResult:
    """Compute ladders, realization and both price/quantity slopes from cleaned sales."""
    config = config or PricingConfig()
    skus = qualifying_skus(sales, config)
    ladder = price_ladder(sales, config, skus)
    profile = sku_profile(sales, config, ladder, skus)
    slopes = price_quantity_slopes(sales, config, skus)
    return PricingResult(
        ladder=ladder,
        profile=profile,
        slopes=slopes,
        overview=_overview(sales, profile, slopes, config),
        config=config,
    )


def headline_text(result: PricingResult) -> str:
    """One measured line: how wide the ladders are and what the two slopes say."""
    o = result.overview
    if not o["n_qualifying"]:
        return "no SKU carried enough price variation to measure a ladder on this input"
    slope = o["median_slope_market_adj"]
    if not np.isfinite(slope):
        return (f"{100 * o['multi_price_share']:.1f}% of SKUs sold at more than one price "
                f"(median {o['median_rungs']:.0f} rungs); too few weeks to fit a slope")
    return (
        f"{100 * o['multi_price_share']:.1f}% of SKUs sold at more than one price "
        f"(median {o['median_rungs']:.0f} rungs, p90/p10 spread {o['median_spread']:.2f}x); "
        f"median market-adjusted price/quantity slope {slope:.2f} vs "
        f"{o['median_slope_within_week']:.2f} from the volume-discount schedule alone "
        f"(per-SKU agreement r={o['slope_agreement_corr']:.2f} - an assortment-level "
        f"reading, not a per-SKU price recommendation)"
    )


# --------------------------------------------------------------------------- #
# Exports
# --------------------------------------------------------------------------- #
EXPORT_COLUMNS = [
    "StockCode", "Description", "rungs", "lines", "units", "revenue",
    "posted_price", "realized_price", "realization_pct", "price_p10", "price_p90", "spread",
    "unit_share_below_pct", "unit_share_at_pct", "unit_share_above_pct",
    "weeks", "slope_within_week", "slope_between_week", "slope_market_adj", "perm_p",
]


def export_frame(result: PricingResult) -> pd.DataFrame:
    """Tidy, rounded per-SKU table for the CSV / Excel deliverable."""
    if result.profile.empty:
        return pd.DataFrame(columns=EXPORT_COLUMNS)
    df = result.profile.merge(
        result.slopes[["StockCode", "weeks", "slope_within_week", "slope_between_week",
                       "slope_market_adj", "perm_p"]],
        on="StockCode", how="left",
    )
    df["revenue"] = df["revenue"].round(2)
    df["realization_pct"] = (100.0 * df["realization"]).round(2)
    for col in ("posted_price", "realized_price", "price_p10", "price_p90"):
        df[col] = df[col].round(4)
    df["spread"] = df["spread"].round(3)
    for name in ("below", "at", "above"):
        df[f"unit_share_{name}_pct"] = (100.0 * df[f"unit_share_{name}"]).round(2)
    for col in ("slope_within_week", "slope_between_week", "slope_market_adj"):
        df[col] = df[col].round(4)
    df["perm_p"] = df["perm_p"].round(4)
    df["weeks"] = df["weeks"].astype("Int64")
    return df[EXPORT_COLUMNS]


def write_csv(result: PricingResult, out_dir: Path = DELIVERABLES) -> Path:
    """Write the per-SKU price-ladder table. Deterministic -> byte-identical on re-run."""
    out = out_dir / "price_ladder.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    export_frame(result).to_csv(out, index=False, encoding="utf-8", lineterminator="\n")
    return out


def ladder_table(result: PricingResult, top: int | None = None) -> pd.DataFrame:
    """Compact top-SKU ladder table for the console and the PDF page."""
    top = result.config.top_ladders if top is None else top
    if result.profile.empty:
        return pd.DataFrame(columns=["StockCode", "Description", "rungs", "posted", "realized",
                                     "spread", "units_below_pct", "slope_market_adj"])
    df = result.profile.head(top).merge(
        result.slopes[["StockCode", "slope_market_adj"]], on="StockCode", how="left"
    )
    slope = df["slope_market_adj"].round(2)
    out = pd.DataFrame({
        "StockCode": df["StockCode"],
        "Description": df["Description"].astype(str).str.slice(0, 30),
        "rungs": df["rungs"].astype(int),
        "posted": df["posted_price"].round(2),
        "realized": df["realized_price"].round(2),
        "spread": df["spread"].round(2),
        "units_below_pct": (100.0 * df["unit_share_below"]).round(1),
        # "-" where no slope was fitted: the SKU's posted price barely moved week
        # to week, and a fragile number is worse than an honest blank.
        "slope_market_adj": [f"{v:.2f}" if np.isfinite(v) else "-" for v in slope],
    })
    return out


# --------------------------------------------------------------------------- #
# Figure (plate 17)
# --------------------------------------------------------------------------- #
def fig_price_ladder(result: PricingResult, out_dir: Path = FIGURES) -> Path | None:
    """Two panels: the measured ladders (solid ink) and the fitted slopes (outline).

    Returns ``None`` when there is nothing to draw (no SKU qualified).
    """
    if result.profile.empty:
        return None

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from retail import plate
    from retail.plate import GRID, INK, INK_2, MODEL_DASH, MODEL_GRAY, MUTED, OCHRE, RUST, SURFACE

    plate.style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.6, 5.9), width_ratios=[1.12, 1])

    # ---- panel 1: price ladders, measured, solid ink -------------------------
    # Rungs are drawn RELATIVE to each SKU's posted price, so ladders of a 0.85
    # popcorn holder and a 12.75 cakestand are on one comparable scale; the posted
    # price itself is printed in GBP against each row.
    top = result.profile.head(result.config.top_ladders).iloc[::-1].reset_index(drop=True)
    lad = result.ladder[result.ladder["StockCode"].isin(top["StockCode"])]
    for i, row in top.iterrows():
        rungs = lad[lad["StockCode"] == row["StockCode"]]
        rel = rungs["Price"].to_numpy(dtype="float64") / float(row["posted_price"])
        ax1.plot([rel.min(), rel.max()], [i, i], color=GRID, linewidth=1.2, zorder=1,
                 solid_capstyle="round")
        ax1.scatter(rel, np.full(len(rel), i),
                    s=10.0 + 150.0 * rungs["unit_share"].to_numpy(dtype="float64"),
                    color=RUST, alpha=0.6, edgecolors="none", zorder=2)
        ax1.scatter([1.0], [i], s=40, facecolor=SURFACE, edgecolor=INK, linewidth=1.2, zorder=4)
        ax1.scatter([row["realized_price"] / row["posted_price"]], [i], s=150, marker="|",
                    color=OCHRE, linewidth=2.4, zorder=3)
    ax1.axvline(1.0, color=INK, linewidth=0.9, zorder=0)
    ax1.set_yticks(range(len(top)))
    ax1.set_yticklabels(
        [f"{str(d)[:26].strip().title() if d else c}  {p:,.2f}"
         for c, d, p in zip(top["StockCode"], top["Description"], top["posted_price"], strict=True)],
        fontsize=8,
    )
    ax1.set_xscale("log", base=2)
    ticks = [0.25, 0.5, 1.0, 2.0, 3.0]
    ax1.set_xticks(ticks)
    ax1.set_xticklabels([f"{t:g}x" for t in ticks], fontsize=8.5)
    ax1.minorticks_off()
    ax1.grid(axis="y", visible=False)
    ax1.set_xlabel("price charged, relative to this SKU's posted price\n"
                   "(the posted price in GBP follows each row label)", labelpad=4)
    ax1.set_title(f"Price ladders - top {len(top)} SKUs by revenue", fontsize=11)
    ax1.scatter([], [], s=46, color=RUST, alpha=0.6, edgecolors="none", label="rung (area = unit share)")
    ax1.scatter([], [], s=40, facecolor=SURFACE, edgecolor=INK, linewidth=1.2, label="posted price (modal)")
    ax1.scatter([], [], s=150, marker="|", color=OCHRE, linewidth=2.4, label="realized price")
    ax1.legend(frameon=False, fontsize=7.5, loc="lower left", handletextpad=0.5,
               labelspacing=0.5, borderaxespad=0.6)

    # ---- panel 2: the two fitted slopes disagree, drawn as model output ------
    o = result.overview
    fitted = result.slopes.dropna(subset=["slope_within_week", "slope_market_adj"])
    ax2.scatter(fitted["slope_within_week"], fitted["slope_market_adj"], s=13,
                facecolor="none", edgecolor=MODEL_GRAY, linewidth=0.7, alpha=0.75, zorder=2)
    if len(fitted):
        x, y = fitted["slope_within_week"].to_numpy(), fitted["slope_market_adj"].to_numpy()
        lo = float(min(np.percentile(x, 1), np.percentile(y, 1)))
        hi = float(max(np.percentile(x, 99), np.percentile(y, 99)))
        pad = 0.08 * (hi - lo)
        lo, hi = lo - pad, hi + pad
        ax2.plot([lo, hi], [lo, hi], color=MUTED, linewidth=1, linestyle=":", zorder=1)
        ax2.axvline(o["median_slope_within_week"], color=INK, linewidth=1.1, linestyle=MODEL_DASH, zorder=3)
        ax2.axhline(o["median_slope_market_adj"], color=INK, linewidth=1.1, linestyle=MODEL_DASH, zorder=3)
        ax2.set_xlim(lo, hi)
        ax2.set_ylim(lo, hi)
        outside = int(((x < lo) | (x > hi) | (y < lo) | (y > hi)).sum())
        note = (f"medians {o['median_slope_within_week']:.2f} (dashed) / "
                f"{o['median_slope_market_adj']:.2f}\n"
                f"per-SKU agreement r = {o['slope_agreement_corr']:.2f}   n = {o['n_fitted']:,}\n"
                f"{CONSTANT_SPEND_SLOPE:.0f} = same spend per line, no demand response")
        if outside:
            note += f"\n{outside} SKU(s) fall outside this frame"
        ax2.text(0.03, 0.03, note, transform=ax2.transAxes, fontsize=8, color=INK_2,
                 va="bottom",
                 bbox={"facecolor": SURFACE, "edgecolor": "none", "alpha": 0.85, "pad": 3.0})
    ax2.set_xlabel("slope from the volume-discount schedule (within week)")
    ax2.set_ylabel("market-adjusted slope (posted price, week to week)")
    ax2.set_title("Two measurements of the same SKU", fontsize=11)

    rect = plate.chrome(
        fig, "pricing",
        notes=("Ladders are measured: every rung is a price the real invoices were written at.",
               "Slopes are observational co-movement, not causal elasticity - no experiment, "
               "no cost data, no competitor prices."),
        modelled=True,
    )
    fig.suptitle(
        f"Price ladders - {100 * o['multi_price_share']:.1f}% of SKUs sold at more than one price "
        f"(median {o['median_rungs']:.0f} rungs, {o['median_spread']:.2f}x p90/p10 spread)",
        fontsize=12.5, x=0.045, y=rect[3] - 0.012, ha="left", color=INK, fontweight="bold",
    )
    for ax in (ax1, ax2):
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
    return plate.save(fig, out_dir / "price_ladder.png",
                      (rect[0], rect[1], rect[2], rect[3] - 0.06))
