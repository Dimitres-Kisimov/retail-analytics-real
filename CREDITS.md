# Credits

## Data

- **Chen, D. (2019). Online Retail II [Dataset]. UCI Machine Learning Repository.**
  https://doi.org/10.24432/C5CG6D — https://archive.ics.uci.edu/dataset/502/online+retail+ii
  Licensed CC BY 4.0. Real transactions of a UK-based online giftware retailer,
  December 2009 - December 2011. The dataset is downloaded at build time and not
  redistributed here; the 1,950-row test fixture is sampled from it under CC BY 4.0.

## Methods

- **MASE** (mean absolute scaled error) and the seasonal-naive scaling convention:
  Hyndman, R.J. & Koehler, A.B. (2006), "Another look at measures of forecast accuracy";
  Hyndman, R.J. & Athanasopoulos, G., *Forecasting: Principles and Practice*.
- **Holt-Winters additive smoothing**: Holt (1957), Winters (1960) — implemented from
  scratch in `retail/forecast.py`.
- **RFM segmentation** and the standard (R, F) segment grid (Champions / Loyal /
  At Risk / Hibernating / ...): common industry practice; implemented from scratch in
  `retail/rfm.py` with rank-based quintiles.

## Tools

- Python, numpy, scipy, pandas, matplotlib, openpyxl — the entire analysis stack.
  No ML libraries were used; every model is implemented in this repository.
- pytest and ruff for the quality gates; GitHub Actions for CI.

## Authorship

Analysis, code and documentation: Dimitres Kisimov. The forecasting and segmentation
modules follow the same from-scratch pattern as my other analytics repositories
(sales-kpi-analytics, market-basket-analysis), applied here to real data.
