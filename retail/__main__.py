"""End-to-end pipeline entry point.

    python -m retail --deliverables     # ingest -> clean -> EDA -> RFM -> forecast -> basket -> PDF/Excel
    python -m retail --basket           # market-basket analysis only (top rules + figure)
    python -m retail --quality          # just the raw-quality report

Requires data/raw/online_retail_II.xlsx (run scripts/download_data.py once).
Console output is UTF-8-safe and uses ASCII status markers only.
"""

from __future__ import annotations

import argparse
import sys
import time

from retail import basket, clean, eda, exports, forecast, ingest, paths, rfm
from retail.util import configure_stdout, fmt_int, fmt_pct

MIN_DELIVERABLE_BYTES = 10_000


def run_pipeline(force_reingest: bool = False) -> int:
    t0 = time.time()

    print("[1/7] ingest: loading raw workbook (cached to data/interim/ after first run)")
    df = ingest.load_raw(force=force_reingest)
    raw_report = ingest.raw_quality_report(df)
    ingest.print_quality_report(raw_report)

    print("[2/7] clean: documented pipeline")
    result = clean.clean(df)
    clean_table = result.report()
    print(clean_table.to_string(index=False))
    sales, returns = result.sales, result.returns
    print(
        f"      sales rows {fmt_int(len(sales))} | returns rows {fmt_int(len(returns))} | "
        f"revenue GBP {sales['Revenue'].sum():,.0f}"
    )

    print("[3/7] eda: writing figures/")
    for fig_path in eda.make_all_figures(sales, returns):
        print(f"      [OK] {fig_path.name}")

    print("[4/7] rfm: segmentation on identified customers")
    rfm_customers, rfm_summary = rfm.run_rfm(sales)
    print(f"      customers scored: {fmt_int(len(rfm_customers))}")
    print(rfm_summary.to_string(index=False))

    print("[5/7] forecast: rolling-origin CV on weekly revenue")
    weekly = forecast.weekly_revenue(sales)
    cv = forecast.cross_validate(weekly)
    cv_summary = forecast.cv_summary(cv)
    print(cv_summary.to_string(index=False))
    _honesty_check(cv, cv_summary)
    final_fold = forecast.final_fold_forecast(weekly)
    sku_table = forecast.top_sku_forecasts(sales)

    print("[6/7] basket: market-basket mining (Apriori + FP-growth cross-check)")
    basket_result = basket.run_basket_analysis(sales)
    _print_basket_summary(basket_result)
    fig_path = basket.fig_top_rules(basket_result.rules, out_dir=paths.FIGURES)
    if fig_path is not None:
        print(f"      [OK] {fig_path.name}")

    print("[7/7] exports: PDF + Excel deliverables")
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
        "rules_table": basket.rules_table(basket_result.rules),
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


def _print_basket_summary(result) -> None:
    """ASCII summary of the basket run: survivor counts, thresholds, cross-check, top rules."""
    cfg, stats = result.config, result.stats
    print(
        f"      invoices {fmt_int(stats['invoices_total'])} -> "
        f"{fmt_int(stats['invoices_with_tracked_item'])} touch the top-{cfg.top_n_items} SKUs -> "
        f"{fmt_int(stats['baskets_mined'])} baskets with >= {cfg.min_basket_items} tracked items "
        f"(mean size {stats['mean_basket_size']:.1f})"
    )
    sizes = result.itemset_counts()
    print(
        f"      min support {fmt_pct(cfg.min_support)} of baskets "
        f"(>= {basket.min_count_for_support(cfg.min_support, result.n_baskets) if result.n_baskets else 0} invoices) | "
        f"frequent itemsets: " + ", ".join(f"{v} of size {k}" for k, v in sizes.items())
    )
    thin = sum(r.thin_support for r in result.rules)
    print(
        f"      rules kept at confidence >= {fmt_pct(cfg.min_confidence)} and lift >= {cfg.min_lift}: "
        f"{fmt_int(len(result.rules))} ({thin} thin-support, flagged)"
    )
    if result.crosscheck_ok is not None:
        marker = "[OK]" if result.crosscheck_ok else "[FAIL]"
        print(
            f"      {marker} FP-growth cross-check on {fmt_int(result.crosscheck_baskets)} "
            f"sampled baskets: itemsets and supports "
            + ("identical to Apriori" if result.crosscheck_ok else "DIFFER from Apriori")
        )
    if result.rules:
        print("      top 10 rules by lift:")
        print(basket.rules_table(result.rules, top=10).to_string(index=False))
    else:
        print("      no rules survived the thresholds on this input")


def run_basket(force_reingest: bool = False) -> int:
    """Standalone market-basket run: mining summary + top-rules figure."""
    try:
        df = ingest.load_raw(force=force_reingest)
    except FileNotFoundError:
        print(
            "[SKIP] raw data not found - run `python scripts/download_data.py` first "
            "(the dataset is deliberately not committed)."
        )
        return 0
    print("[1/3] clean: documented pipeline")
    sales = clean.clean(df).sales
    print(f"      sales rows {fmt_int(len(sales))}")
    print("[2/3] basket: market-basket mining (Apriori + FP-growth cross-check)")
    result = basket.run_basket_analysis(sales)
    _print_basket_summary(result)
    if result.crosscheck_ok is False:
        return 1
    print("[3/3] figure: figures/basket_top_rules.png")
    fig_path = basket.fig_top_rules(result.rules, out_dir=paths.FIGURES)
    if fig_path is None:
        print("      [WARN] no solid rules to plot (all thin-support or none mined)")
    else:
        print(f"      [OK] {fig_path.name}")
    return 0


def main(argv: list[str] | None = None) -> int:
    configure_stdout()
    parser = argparse.ArgumentParser(prog="python -m retail", description=__doc__)
    parser.add_argument("--deliverables", action="store_true", help="run the full pipeline and write PDF + Excel")
    parser.add_argument("--basket", action="store_true", help="market-basket analysis: top co-purchase rules + figure")
    parser.add_argument("--quality", action="store_true", help="print the raw-quality report only")
    parser.add_argument("--force-reingest", action="store_true", help="ignore the interim cache")
    args = parser.parse_args(argv)

    if args.basket:
        return run_basket(force_reingest=args.force_reingest)
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
