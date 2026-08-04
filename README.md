# retail-analytics-real

End-to-end analytics on a **real, public, messy** dataset: the UCI *Online Retail II*
transactions of a UK online giftware retailer, December 2009 - December 2011.

![Monthly revenue](figures/monthly_revenue.png)

Gift-season peaks in November of both years are annotated — and so is the fact that
December 2011 contains only nine days of data. A naive month-over-month read of the tail
would call it a crash; it's a truncated month.

The short version: **1,067,371** raw rows, **94.0% retained** after a cleaning pipeline
that logs every step's row cost, **GBP 19,643,862** of revenue analyzed — and the honest
forecasting headline is that **seasonal-naive wins** (mean MASE 1.094 vs 1.187 for
Holt-Winters), because one seasonal training cycle is what this data honestly supports.

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

Monthly revenue is the chart at the top of this page; the rest of the picture:

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

## Cohort retention (repeat purchase)

```bash
python -m retail --cohort     # cohort triangle + SVG heatmap/curve + CSV
```

Where RFM is a *snapshot* of who's valuable now, cohort retention is the *time* dimension:
group the **5,852 identified customers** by their first-purchase month, then track what
share of each cohort comes back to buy in each later month. A customer is "retained at
month _k_" if they placed **any** invoice in the calendar month _k_ months after their first
(so this is month-by-month activity retention, not cumulative — skipping a month and
returning the next counts as retained in the month they bought). Offset 0 is the
acquisition month, 100% by definition. Everything is measured from the cleaned real
transactions — nothing modelled.

![Cohort retention](figures/cohort_retention.svg)

Two honesty guards are built into the numbers, not bolted on after:

- **Right-censoring.** A cohort acquired late can only be observed for a few months. Each
  cell is filled only when its target month is a *complete* month in the data; otherwise
  it's left blank. The headline curve at month _k_ therefore averages only the cohorts old
  enough to have been observed at _k_ — it never counts "hasn't happened yet" as churn.
- **Incomplete final month.** December 2011 holds nine trading days, so it is excluded as a
  target month (last complete month: **2011-11**). No absence in a partial month is scored
  as a lost customer.

**Headline: month-1 repeat rate 23.3%, holding near 22.2% by month 6** (size-weighted across
the cohorts observable at each offset):

| months since first purchase | 1 | 2 | 3 | 4 | 5 | 6 | 9 | 12 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| repeat rate | **23.3%** | 23.6% | 25.0% | 22.8% | 21.9% | **22.2%** | 19.4% | 22.7% |

**Plain-language reading.** The striking thing is what *doesn't* happen: the curve barely
decays. For a consumer shop you'd expect month-1 retention to fall off a cliff; here it
plateaus in the low-20s and even **rebounds to 22.7% by month 12**. That is the wholesale
nature of this retailer showing through — a large share of customers are shops that
**re-order on an annual rhythm**, so the year-anniversary month lifts retention back up
rather than letting it fade. The shallow trough is around month 9 (**19.4%**). The launch
cohort (December 2009, 951 customers) is the strongest by far: 35.0% back at month 1,
peaking at **49.6% in November 2010** (its first gift season) and still at 40.6% two years
on — the long-standing wholesale accounts that anchor the base. Later cohorts settle lower
(~15–23% at month 1), which is why the size-weighted average sits at 23.3%.

**Caveats.** Identified customers only — the 22.77% of rows without a `CustomerID` are
structurally invisible here, exactly as in RFM, so this covers the 86.9% of revenue that is
attributable. Single UK gift retailer with wholesale-heavy baskets, so these repeat rates
should not be read as consumer-ecommerce benchmarks. Retention is measured co-purchase
behaviour, not a fitted survival model; the full cohort×month triangle (with cohort sizes
and every censored cell blank) is written to `deliverables/cohort_retention.csv` and the
**Cohort** sheet of the Excel workbook.

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

## What actually sells together

```bash
python -m retail --basket
```

Market-basket analysis on the real invoices. The mining engine — from-scratch Apriori with
downward-closure pruning, plus an independent from-scratch FP-growth implementation as a
cross-check — is adapted from my
[market-basket-analysis](https://github.com/Dimitres-Kisimov/market-basket-analysis) repo,
which ran it on 14 synthetic categories. Here it runs on real SKUs, so the item universe is
a documented choice, not a default: **the top 200 stock codes by cleaned gross revenue**
(5,305 codes total; product level would drown in rare items), each labelled with its most
frequent normalised description, codes with identical descriptions merged. A basket is one
sales invoice reduced to tracked items; invoices with fewer than two tracked items cannot
express a co-purchase and are excluded. Survivors, measured:

- 39,516 sales invoices -> 34,394 contain at least one top-200 SKU -> **29,639 baskets**
  with >= 2 tracked items (mean basket size 8.6 tracked items — wholesale-sized).
- Min support **1% of baskets (>= 297 invoices)**: 197 frequent single items, 859 pairs,
  278 triples. At 2% only 133 pairs survive; at 0.5% it balloons to 4,196 pairs and the
  tail turns noisy — 1% is the documented middle, not a magic number.
- **1,222 directed rules** (565 distinct item sets) kept at confidence >= 30% and
  lift >= 1.10. Support/confidence/lift/leverage/conviction all computed from scratch.
- FP-growth on a seeded 5,000-basket sample returns **identical itemsets and supports**
  to Apriori (also asserted in the test suite on the fixture).
- Rules backed by < 30 invoices get flagged `thin_support`. At these thresholds **none
  trigger** — min support already guarantees >= 297 invoices; the flag only matters if
  you lower support below ~0.1%.

![Top co-purchase rules](figures/basket_top_rules.png)

Top 10 item sets by lift (higher-confidence direction shown; both directions of a set
share support and lift):

| # | rule | support | invoices | confidence | lift |
|---|---|---:|---:|---:|---:|
| 1 | PINK + RED 3 PIECE MINI DOTS CUTLERY SET -> BLUE | 1.35% | 399 | 74.3% | **23.4** |
| 2 | BLUE HAPPY BIRTHDAY BUNTING -> PINK HAPPY BIRTHDAY BUNTING | 1.51% | 448 | 66.3% | **22.9** |
| 3 | PINK REGENCY TEACUP AND SAUCER -> GREEN + ROSES REGENCY TEACUP AND SAUCER | 2.48% | 735 | 71.8% | **20.6** |
| 4 | BLACK/BLUE POLKADOT UMBRELLA -> RED RETROSPOT UMBRELLA | 1.39% | 411 | 60.5% | **20.1** |
| 5 | GREEN REGENCY TEACUP AND SAUCER + REGENCY CAKESTAND 3 TIER -> PINK REGENCY TEACUP AND SAUCER | 1.67% | 495 | 68.7% | **19.9** |
| 6 | EDWARDIAN PARASOL RED -> EDWARDIAN PARASOL BLACK | 1.34% | 396 | 55.8% | **19.5** |
| 7 | BLUE 3 PIECE MINI DOTS CUTLERY SET -> PINK 3 PIECE MINI DOTS CUTLERY SET | 2.20% | 651 | 69.3% | **19.1** |
| 8 | WOOD S/3 CABINET ANT WHITE FINISH + WOODEN PICTURE FRAME WHITE FINISH -> WOOD 2 DRAWER CABINET WHITE FINISH | 1.10% | 325 | 70.0% | **18.7** |
| 9 | PINK REGENCY TEACUP AND SAUCER -> GREEN REGENCY TEACUP AND SAUCER | 2.92% | 864 | 84.4% | **18.6** |
| 10 | REGENCY CAKESTAND 3 TIER + ROSES REGENCY TEACUP AND SAUCER -> PINK REGENCY TEACUP AND SAUCER | 1.58% | 469 | 63.0% | **18.2** |

**Plain-language reading.** This is a gift retailer selling to (largely) wholesale buyers,
and the top of the lift table says one thing loudly: **buyers stock colour and design
variants of the same product together**. The mini-dots cutlery sets in blue/pink/red, the
happy-birthday bunting in blue/pink, the Edwardian parasols, the polkadot/retrospot
umbrellas, the white-finish wooden furniture — all variant families. The Regency tea-set
family (teacups in pink/green/roses, the 3-tier cakestand, teapot) is the closest thing to
a true cross-product bundle, and even that is one design line being bought as a set. A
shop buying the pink Regency teacup goes on to take the green one 84% of the time.

**Honest comparison with the synthetic repo.** The synthetic generator planted category
bundles (fasteners + power tools + gloves) and mining recovered them at lifts of about
1.5-2.4 with 30-80% confidence — realistic for 14 broad *categories*, where baselines are
high. On real SKUs the same engine reports lifts of **18-23**, roughly ten times larger,
simply because each single item appears in only 3-5% of baskets, so co-occurring at 1.5-3%
is dozens of times above independence. What genuinely surprised me: I expected classic
complement pairs (teapot -> teacups is the shape of story the synthetic data planted);
instead the real signal is almost entirely **variant collecting within an assortment**,
plus one strong asymmetry the synthetic data never produced — rules point from the rarer
variant to the more popular one with much higher confidence than the reverse (pink Regency
-> green at 84%, green -> pink at only 64%), which is exactly what a "core colour +
optional extra colour" buying pattern looks like.

**Caveats before anyone reprices a shelf:** single UK gift retailer, wholesale quantities
(one basket is often a shop's stock order, not a consumer's), gift-season concentration,
and support fractions are shares of *multi-item invoices over the tracked top-200
assortment*, not of all invoices. Above all: lift is co-purchase frequency versus
independence — **correlation, not causation**. Nothing here proves that stocking blue
cutlery *makes* anyone buy pink cutlery.

## Data-quality report card

```bash
python -m retail --report-card     # scores the data BEFORE and AFTER cleaning
```

A **generic, config-driven** module (`retail/quality.py`) that scores any pandas
DataFrame across five standard dimensions and renders a report card. It is
dataset-agnostic — give it a frame and an optional config (per-column expected
type, range/regex/domain, key columns, a few cross-field rules) — and here it is
pointed at the real UCI data to make the cleaning pipeline's value *measurable*.

**It is a heuristic scorecard with stated weights and stated rules, NOT a
certification of data correctness.** A high score means the data *passed the
declared checks*, not that the data is true. That caveat is printed on the card
header and in the module docstring.

Every check scores `100 * (1 - violations / evaluated)`; a dimension's score is
the mean of its checks, and the overall score is a weighted mean with **explicit,
documented weights** — Completeness **0.30**, Validity **0.25**, Consistency
**0.20**, Uniqueness **0.15**, Plausibility **0.10** (renormalised over the
dimensions that actually run). A dimension's status is its *worst* check, and the
letter grade is **capped** by the worst hard-dimension status (a failing hard
check caps at C, a warning at B). Plausibility is an **outlier heuristic**
(3×IQR) — it never hard-fails and never caps the grade.

Measured on the full dataset (same config applied to both frames):

| dimension | raw | cleaned | what moved |
|---|---:|---:|---|
| Completeness | 100.0 [OK] | 100.0 [OK] | unchanged by design (see note) |
| Validity | 100.0 [OK] | 100.0 [OK] | types/ranges already conform per-row |
| Uniqueness | 96.8 **[FAIL]** | 100.0 [OK] | **34,335 exact duplicate rows removed** |
| Consistency | 99.1 [WARN] | 100.0 [OK] | **22,950 non-positive qty + 6,207 bad prices resolved** |
| Plausibility (heuristic) | 96.7 [WARN] | 97.1 [WARN] | extreme outliers remain (not errors) |
| **Overall grade** | **C (99.0)** | **A (99.7)** | **raw C → cleaned A** |

**Honest reading.** The per-*row* scores are all high because real-data problems
are concentrated in a *minority* of rows — which is exactly what the pipeline
removes. The letter grade, not the raw score, tells the story: raw data fails on
duplicates and warns on sign errors, so it is capped to **C**; the cleaned frame
passes every hard check and earns an **A**. Two honest footnotes, on brand:

- **Completeness does not move.** Missing `CustomerID` (declared *nullable*) is
  kept by design — flagged, never dropped — so cleaning can't and shouldn't lift
  completeness. The lift is entirely in Uniqueness and Consistency.
- **The cleaned data is not "perfect".** Plausibility still warns: ~2–4% of
  `Quantity`/`Price` values are extreme outliers by the 3×IQR rule. Those are
  large legitimate wholesale orders, not errors — so the check is labelled a
  heuristic and the card reports them rather than claiming a spotless dataset.

The full before/after card is written to `deliverables/data_quality_report_card.md`
and, inside the pipeline, to the **DataQuality** sheet of the Excel workbook
(screen == export). The module is self-contained and reusable on any dataset.

## Deliverables

```bash
python -m retail --deliverables
```

produces `deliverables/retail_analytics_executive.pdf` (6-page executive briefing: citation,
cleaning-impact table, forecast + CV results, RFM, cohort retention, top SKUs),
`deliverables/retail_analytics.xlsx` (sheets: CleaningReport, MonthlyRevenue, RFM, Cohort —
the full cohort×month retention triangle with sizes and the size-weighted curve —
ForecastCV, TopSKUs, Rules — all 1,222 association rules with metrics and the thin-support
flag — DataQuality — the raw-vs-cleaned report card with per-dimension lift and the flat
findings list — and WeeklyRevenue), and `deliverables/cohort_retention.csv` (the same
triangle as a flat CSV). The basket, cohort and report-card steps run inside the pipeline,
so every sheet is always current.

## Reproduce

```bash
pip install -r requirements.txt
python scripts/download_data.py    # one-time, ~45 MB from UCI
python -m retail --deliverables    # full pipeline: ~1.5 min first run (xlsx parse), ~45 s after
python -m retail --basket          # market-basket analysis only (skips politely if data absent)
python -m retail --cohort          # cohort repeat-purchase retention only (skips politely if absent)
pytest -q                          # 57 tests, fixture-based, no download needed
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

Code: © 2026 Dimitres Kisimov — all rights reserved; published for portfolio review. See LICENSE. Data: **CC BY 4.0**, © the dataset authors via the UCI Machine
Learning Repository — cited above and **not redistributed in this repository**.
