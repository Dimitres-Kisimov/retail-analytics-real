# What these analyses are worth to a retailer

Every number in this document that comes from the data is measured; every monetary
"worth" figure is an **illustrative estimate, labelled as such** — this retailer's real
margins, costs and retention economics are not in the dataset.

Context (measured): GBP 19.64M gross product revenue over 24.3 months, ~GBP 9.7M/year.
5,852 identified customers carrying 86.9% of revenue; 13.1% of revenue unattributed.

## 1. Revenue concentration -> retention economics

**Measured:** the Champions segment — 1,464 customers, 25% of identified customers —
carries **69.0%** of identified revenue (GBP 11.77M over the period). The top four
segments carry ~83%.

**Why it matters:** with concentration this extreme, small retention changes in one
segment dominate everything else. *Illustrative estimate:* if churn among Champions
were reduced by 2 percentage points, at the segment's average value of GBP 8,038 per
customer over the period, that is on the order of GBP 235k of protected revenue per
period (2% x 1,464 x 8,038). The precise figure depends on churn and margin data the
dataset does not contain; the segmentation tells you *where* to spend retention budget,
not the exact ROI.

**Deliverable:** the RFM sheet ranks every identified customer; "At Risk" (344
customers, 5.6% of revenue) and "Can't Lose Them" (16 high-value lapsed accounts) are
the actionable call lists.

## 2. Returns visibility

**Measured:** returns run at 3.65% of gross value overall, with single-event spikes
(January 2011: 13.6%; December 2011: 28.3% — one 80,995-unit same-day cancellation
against a 9-day month). The monthly series separates process-level returns (~2-4%)
from one-off events.

**Why it matters:** a returns dashboard that doesn't separate one-off cancellations
from systemic returns will trigger false alarms. *Illustrative estimate:* at this
revenue scale, each percentage point of avoidable returns is roughly GBP 100k/year
gross — worth monitoring, not worth panicking over a single spike.

## 3. Forecasting for stock and cash planning

**Measured:** on 104 weekly revenue points, seasonal-naive ("same week last year")
achieves mean MASE 1.094 across five rolling-origin folds and beats Holt-Winters and a
lag-features linear model on 4 of 5 folds.

**Why it matters (honestly):** with one full seasonal cycle of training data, the
defensible planning baseline is *last year's week, adjusted by recent level* — not a
complex model. The business value here is negative knowledge: it prevents spending on
forecasting sophistication the data cannot yet support. Two more years of history
would change that calculus. Per-SKU weekly demand for the top sellers is mostly
naive/moving-average territory (MASE 0.6-1.1), fine for coarse reorder planning.

## 4. The missing-CustomerID gap as an opportunity

**Measured:** 22.77% of rows and 13.1% of revenue (~GBP 2.57M over the period) have no
customer identity — guest checkouts or unlinked channels, elevated around the
2010-2011 holiday season (peaking near 38% of rows in January 2011).

**Why it matters:** that revenue is invisible to segmentation, retention and lifetime
value. *Illustrative estimate:* converting even a fifth of unattributed revenue into
identified accounts would grow the addressable RFM base by ~3% of total revenue —
before any behaviour change — simply by making existing revenue targetable.

## 5. Making the cleaning pipeline's value measurable

**Measured:** the generic data-quality report card (`retail/quality.py`,
`python -m retail --report-card`) scores the data before and after cleaning on the
same declared checks and lifts the grade from **C (uniqueness failing on 34,335
duplicate rows, consistency warning on 22,950 non-positive quantities) to A**.

**Why it matters:** "we cleaned the data" is an assertion; a report card with
stated weights and rules turns it into a number a stakeholder can audit. The
module is dataset-agnostic and config-driven, so the same scorecard is reusable
on any tabular dataset (a supplier feed, a CRM export) — the retail run is just
the demo. It is explicitly a heuristic scorecard, **not** a certification that
the data is true; a high score means the declared checks passed.

## 6. Lifecycle flows → where retention effort actually goes

**Measured:** in the average complete month, 1,041 identified customers buy — 212 new,
390 retained from last month and 439 *resurrected after a silent gap* — while 620 lapse
(overall quick ratio 1.05). At-risk customers (silent 1–3 months) resurrect at 24.1% the
next month; even dormant customers (> 3 silent months) come back at 8.9% per month, and
29.4% of all resurrections are from deep dormancy. Revenue splits 51.2% retained / 31.2%
resurrected / 17.6% new.

**Why it matters:** the flow matrix says this base's biggest recurring revenue engine
after steady repeats is *comebacks*, not new acquisition — so a churn dashboard that
flags every one-month absence would mislabel roughly half the base as lost while they
are mid-cycle. The operational move the numbers support is timing outreach to the
measured reorder rhythm (median gap 2 months, p75 3) rather than reacting to a single
silent month. *Illustrative estimate:* December 2010 alone put 1,082 customers into the
at-risk pool; at the measured 24.1% monthly comeback rate, most of the following
quarter's ~1,220 resurrections were this pool returning on its own — a baseline any
win-back campaign must beat before claiming credit.

**Deliverable:** the Lifecycle sheet (monthly stage counts, churn, quick ratio, the flow
matrix), `lifecycle_stages.csv`, `lifecycle_flows.csv` and the committed
`figures/lifecycle_stages.svg`.

## 7. Price ladders → what the transactions can and cannot say about pricing

**Measured:** 4,342 of 4,895 SKUs (88.7%) sold at more than one price. On the 1,342 SKUs
with a real ladder (≥ 4 distinct prices, ≥ 200 lines — 79.7% of revenue), the median SKU
has 6 rungs and a p90/p10 spread of 2.00×; 39.5% of units went below the posted (modal)
price, 51.5% at it, 9.0% above it, for a price realization of 98.6%. Two independent
log-log price/quantity slopes land at -1.71 (within-week, i.e. the volume-discount
schedule alone) and -1.67 (market-adjusted, week to week); 825 of 1,113 SKUs beat their
own seeded permutation null.

**Why it matters:** the first operational value here is *visibility* — a business that
believes it has one price per SKU actually has six, and the gap between posted and
realized price is measurable per SKU rather than argued about. The second is a guard
rail. The arithmetic gap between booked revenue and revenue-at-posted-price is GBP
216,608 (1.4%), and it would be easy to present that as recoverable margin. It is not:
the same data says quantity falls as price rises, the within-week slope shows a number
near -1.7 arises from the discount schedule with *no* demand response at all, and the two
slopes correlate at only r = 0.20 per SKU. *What the numbers support* is picking the
handful of SKUs whose ladder is widest and whose posted price is followed least, and
testing a deliberate price change on them — with a holdout — because **nothing here is
causal and no per-SKU elasticity is claimed**.

**Deliverable:** the PriceLadder sheet (headline metrics + the full per-SKU table with
both slopes and the permutation p-value), `price_ladder.csv`, and the committed
`figures/price_ladder.png` (plate 17).

## Caveats that bound all of the above

Single retailer, UK-heavy, gift-seasonal, 2009-2011 nominal GBP, wholesale/retail mix,
final month truncated. These analyses transfer as *method*, not as *numbers*.
