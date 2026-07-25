# retail-analytics-real

End-to-end analytics on a **real, public, messy** dataset: the UCI *Online Retail II*
transactions of a UK online giftware retailer, December 2009 - December 2011.

## Why this repo exists

The rest of my portfolio is built on synthetic data and says so on the label. This repo is
the real-data counterpart: 1,067,371 genuine transaction rows with everything real data
brings — cancellation invoices, negative quantities, ~23% missing customer IDs, bookkeeping
rows pretending to be products, literal `TEST001` rows, and an 80,995-unit order that was
cancelled the same day it was placed. Every cleaning decision below is documented with its
row cost, every model result is out-of-sample, and where a simple baseline wins, I say so.

## The dataset

> Chen, D. (2019). **Online Retail II** [Dataset]. UCI Machine Learning Repository.
> https://doi.org/10.24432/C5CG6D — https://archive.ics.uci.edu/dataset/502/online+retail+ii
> License: **CC BY 4.0**.

The raw xlsx (~45 MB, two sheets: 2009-2010 and 2010-2011) is **not committed** — committing
raw data is bad practice, and this repo does not redistribute the dataset. Fetch it once:

```bash
python scripts/download_data.py
# or:
curl -L -o online_retail_ii.zip "https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip" \
  && unzip online_retail_ii.zip -d data/raw/
```

### Raw quality report (measured, not estimated)

| metric | value |
|---|---|
| rows | 1,067,371 |
| invoices | 53,628 |
| stock codes | 5,305 |
| countries | 43 |
| date range | 2009-12-01 to 2011-12-09 |
| exact duplicate rows | 34,335 |
| missing CustomerID | 243,007 (**22.77%**) |
| cancellation invoices (`C*`) | 19,494 rows (1.83%) |
| negative-quantity rows | 22,950 |
| zero-price rows | 6,202 (plus 5 negative-price) |
| non-product StockCode rows | 5,930 (POST, D, M, DOT, BANK CHARGES, TEST001, gift vouchers, ...) |

## Cleaning: every decision with its row cost

| step | rows in | rows out | removed | % removed |
|---|---:|---:|---:|---:|
| drop exact duplicates | 1,067,371 | 1,033,036 | 34,335 | 3.22% |
| separate cancellations (`C*`) — kept as returns frame | 1,033,036 | 1,013,932 | 19,104 | 1.85% |
| remove non-product StockCodes (explicit list) | 1,013,932 | 1,009,304 | 4,628 | 0.46% |
| drop zero/negative-price rows | 1,009,304 | 1,003,340 | 5,964 | 0.59% |
| drop non-cancellation qty <= 0 | 1,003,340 | 1,003,340 | 0 | 0.00% |
| flag missing CustomerID (kept, never dropped) | 1,003,340 | 1,003,340 | 0 (226,761 flagged) | 0.00% |

Result: **1,003,340 sales rows** (94.0% of raw), **17,914 returns rows**, **GBP 19,643,862**
gross product revenue. Two honest footnotes:

- The "qty <= 0" step removes nothing *on this dataset* — every non-cancellation
  negative-quantity row also had a zero price, so the previous step already caught them.
  The step stays because the pipeline shouldn't rely on that coincidence.
- Missing-CustomerID rows are **flagged, not dropped**: they are real revenue (13.1% of it)
  but can't be attributed to a customer, so they count for revenue/EDA and are excluded
  from RFM only.

## What the data shows (EDA)

![Monthly revenue](figures/monthly_revenue.png)

Gift-season peaks in November of both years are annotated — and so is the fact that
December 2011 contains only nine days of data. A naive month-over-month read of the tail
would call it a crash; it's a truncated month.

![Top products](figures/top_products.png)

The number-3 product by gross revenue, "Paper Craft, Little Birdie", is a **single
80,995-unit order placed and cancelled the same day** (2011-12-09). It stays in gross sales
because its cancellation lives in the returns frame — a textbook example of why "top
products" charts on this dataset need footnotes.

![Top countries](figures/top_countries.png)
![Order sizes](figures/order_values.png)
![Returns rate](figures/returns_rate.png)
![Missing CustomerID share](figures/missing_customer_share.png)

Returns run at **3.65% of gross value overall** (monthly mean 4.0%). The January 2011
(13.6%) and December 2011 (28.3%) spikes are single large cancellations against quiet or
truncated months, not a returns crisis.

## RFM segmentation (from scratch)

Quintile R/F/M scores over the **5,852 customers with a real CustomerID**, mapped to the
standard segment grid. Measured segment table (revenue = identified-customer revenue,
GBP 17.07M = 86.9% of total):

| segment | customers | revenue share |
|---|---:|---:|
| Champions | 1,464 | **69.0%** |
| Loyal Customers | 517 | 9.8% |
| At Risk | 344 | 5.6% |
| Hibernating | 1,172 | 5.3% |
| Potential Loyalist | 710 | 4.8% |
| Need Attention | 458 | 2.7% |
| Lost | 809 | 1.6% |
| About to Sleep | 195 | 0.5% |
| Can't Lose Them | 16 | 0.5% |
| New Customers | 167 | 0.3% |

A quarter of identified customers (Champions) carry 69% of identified revenue — a real
concentration number, not a synthetic one.

## Forecasting: honest, leakage-safe, MASE-scored

Weekly revenue (104 complete weeks; the two partial edge weeks are dropped), rolling-origin
CV with 5 folds, horizon 8 weeks, training strictly before each origin. All models
implemented from scratch (numpy/pandas only). MASE is scaled by the in-sample one-step
error of the seasonal-naive walk (m=52) on each fold's training window.

**MASE per fold (lower is better):**

| fold (origin, train weeks) | seasonal-naive | Holt-Winters | lag-features OLS |
|---|---:|---:|---:|
| 1 (2011-03-06, 64) | **0.651** | 0.702 | 1.515 |
| 2 (2011-05-01, 72) | **1.404** | 1.745 | 1.990 |
| 3 (2011-06-26, 80) | **0.887** | 0.952 | 0.907 |
| 4 (2011-08-21, 88) | 1.732 | **1.410** | 1.558 |
| 5 (2011-10-16, 96) | **0.796** | 1.126 | 1.981 |
| **mean** | **1.094** | 1.187 | 1.590 |

**The honest headline: seasonal-naive wins.** It takes 4 of 5 folds and the best mean MASE.
Holt-Winters (additive, m=52, smoothing tuned on the training tail only) beats it on one
fold and loses overall; the lag-features linear model never wins. With exactly one full
seasonal cycle available for training, "same week last year" is genuinely hard to beat —
a modest, real result, which is the point of this repo. A mean MASE around 1.1 means
8-week-ahead forecasts carry about the same error as a one-week seasonal walk does
in-sample; nobody should claim precision forecasting on 104 weekly points with one
Christmas per training window.

Per-SKU weekly demand for the top 10 SKUs is in the deliverables, with intermittency
(zero-week share) noted per SKU. A 4-week moving average or the naive walk wins on most of
them; SKU 23843 is unforecastable by construction (one giant cancelled order).

## Deliverables

```bash
python -m retail --deliverables
```

produces `deliverables/retail_analytics_executive.pdf` (5-page executive briefing: citation,
cleaning-impact table, forecast + CV results, RFM, top SKUs) and
`deliverables/retail_analytics.xlsx` (sheets: CleaningReport, MonthlyRevenue, RFM,
ForecastCV, TopSKUs, WeeklyRevenue).

## Reproduce

```bash
pip install -r requirements.txt
python scripts/download_data.py    # one-time, ~45 MB from UCI
python -m retail --deliverables    # full pipeline: ~1 min first run (xlsx parse), ~15 s after
pytest -q                          # 20 tests, fixture-based, no download needed
ruff check .
```

Tests run on `tests/fixtures/sample.csv` — 1,950 real rows drawn deterministically from the
dataset (seed 42, stratified to include every kind of mess). CI does not download the full
dataset; the one full-data test skips itself cleanly when `data/raw/` is absent.

## Limitations (read before reusing any number)

- **Single retailer, single channel** — one UK-based online giftware wholesaler/retailer.
  Nothing here generalises to other retailers without re-measurement.
- **UK-centric**: 86% of revenue is domestic UK.
- **Gift-heavy seasonality with only one full cycle in training** — the main reason
  seasonal-naive is the strongest forecaster; two years of data cannot validate a
  second seasonal cycle.
- **The final month is incomplete** (9 days of December 2011) and is annotated, not hidden.
- **22.77% of rows have no CustomerID**, so RFM covers the 86.9% of revenue that is
  attributable; the rest is structurally invisible to customer analytics.
- **Wholesale and retail orders are mixed** and not separable with certainty (many
  customers are wholesalers, hence 80k-unit orders).
- Prices are nominal GBP; no inflation or FX adjustment.

## License

Code: MIT (see `LICENSE`). Data: **CC BY 4.0**, © the dataset authors via the UCI Machine
Learning Repository — cited above and **not redistributed in this repository**.
