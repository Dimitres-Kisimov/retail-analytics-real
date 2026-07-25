"""End-to-end pipeline entry point.

    python -m retail --deliverables     # ingest -> clean -> EDA -> RFM -> forecast -> PDF/Excel
    python -m retail --quality          # just the raw-quality report

Requires data/raw/online_retail_II.xlsx (run scripts/download_data.py once).
Console output is UTF-8-safe and uses ASCII status markers only.
"""

from __future__ import annotations

import argparse
import sys
import time

from retail import clean, eda, exports, forecast, ingest, rfm
from retail.util import configure_stdout, fmt_int, fmt_pct

MIN_DELIVERABLE_BYTES = 10_000


def run_pipeline(force_reingest: bool = False) -> int:
    t0 = time.time()

    print("[1/6] ingest: loading raw workbook (cached to data/interim/ after first run)")
    df = ingest.load_raw(force=force_reingest)
    raw_report = ingest.raw_quality_report(df)
    ingest.print_quality_report(raw_report)

    print("[2/6] clean: documented pipeline")
    result = clean.clean(df)
    clean_table = result.report()
    print(clean_table.to_string(index=False))
    sales, returns = result.sales, result.returns
    print(
        f"      sales rows {fmt_int(len(sales))} | returns rows {fmt_int(len(returns))} | "
        f"revenue GBP {sales['Revenue'].sum():,.0f}"
    )

    print("[3/6] eda: writing figures/")
    for fig_path in eda.make_all_figures(sales, returns):
        print(f"      [OK] {fig_path.name}")

    print("[4/6] rfm: segmentation on identified customers")
    rfm_customers, rfm_summary = rfm.run_rfm(sales)
    print(f"      customers scored: {fmt_int(len(rfm_customers))}")
    print(rfm_summary.to_string(index=False))

    print("[5/6] forecast: rolling-origin CV on weekly revenue")
    weekly = forecast.weekly_revenue(sales)
    cv = forecast.cross_validate(weekly)
    cv_summary = forecast.cv_summary(cv)
    print(cv_summary.to_string(index=False))
    _honesty_check(cv, cv_summary)
    final_fold = forecast.final_fold_forecast(weekly)
    sku_table = forecast.top_sku_forecasts(sales)

    print("[6/6] exports: PDF + Excel deliverables")
    ctx = {
        "raw_report": raw_report,
        "clean_table": clean_table,
        "monthly": eda.monthly_revenue_table(sales),
        "returns_rate": eda.returns_rate_table(sales, returns),
        "weekly": weekly,
        "cv": cv,
        "cv_summary": cv_summary,
        "final_fold": final_fold,
        "rfm_summary": rfm_summary,
        "sku_table": sku_table,
    }
    pdf_path = exports.export_pdf(ctx)
    xlsx_path = exports.export_excel(ctx)

    status = 0
    for path in (pdf_path, xlsx_path):
        size = path.stat().st_size
        if size >= MIN_DELIVERABLE_BYTES:
            print(f"      [OK] {path.name} ({size / 1024:.1f} KB)")
        else:
            print(f"      [FAIL] {path.name} is only {size} bytes (expected >= {MIN_DELIVERABLE_BYTES})")
            status = 1

    print(f"[DONE] pipeline finished in {time.time() - t0:.0f}s")
    return status


def _honesty_check(cv, cv_summary) -> None:
    """Say it plainly when the fancy model loses to the baseline."""
    means = dict(zip(cv_summary["model"], cv_summary["mean_mase"], strict=True))
    hw, snaive = means.get("holt_winters"), means.get("seasonal_naive")
    if hw is None or snaive is None:
        return
    lost_folds = 0
    pivot = cv.pivot_table(index="fold", columns="model", values="mase", observed=True)
    if {"holt_winters", "seasonal_naive"} <= set(pivot.columns):
        lost_folds = int((pivot["holt_winters"] > pivot["seasonal_naive"]).sum())
    if hw > snaive:
        print(
            f"      [HONEST] Holt-Winters (mean MASE {hw:.3f}) did NOT beat "
            f"seasonal-naive ({snaive:.3f}) overall."
        )
    elif lost_folds:
        print(
            f"      [HONEST] Holt-Winters wins on average but loses to seasonal-naive "
            f"on {lost_folds} fold(s)."
        )


def main(argv: list[str] | None = None) -> int:
    configure_stdout()
    parser = argparse.ArgumentParser(prog="python -m retail", description=__doc__)
    parser.add_argument("--deliverables", action="store_true", help="run the full pipeline and write PDF + Excel")
    parser.add_argument("--quality", action="store_true", help="print the raw-quality report only")
    parser.add_argument("--force-reingest", action="store_true", help="ignore the interim cache")
    args = parser.parse_args(argv)

    if args.quality:
        report = ingest.raw_quality_report(ingest.load_raw(force=args.force_reingest))
        ingest.print_quality_report(report)
        print(f"missing CustomerID: {fmt_pct(report['missing_customer_id_pct'])}")
        return 0
    if args.deliverables:
        return run_pipeline(force_reingest=args.force_reingest)
    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
