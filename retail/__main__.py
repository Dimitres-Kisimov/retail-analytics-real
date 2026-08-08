"""End-to-end pipeline entry point.

    python -m retail --deliverables     # ingest -> clean -> EDA -> returns -> RFM -> cohort -> CLV -> forecast -> basket -> PDF/Excel
    python -m retail --basket           # market-basket analysis only (top rules + figure)
    python -m retail --cohort           # cohort repeat-purchase retention only (triangle + SVG + CSV)
    python -m retail --clv              # customer lifetime value only (BG/NBD + Gamma-Gamma + holdout check)
    python -m retail --returns          # returns & cancellations analysis only (rates, lag, per-SKU CSV)
    python -m retail --quality          # just the raw-quality report
    python -m retail --report-card      # data-quality report card: raw vs cleaned (measured lift)

Requires data/raw/online_retail_II.xlsx (run scripts/download_data.py once).
Console output is UTF-8-safe and uses ASCII status markers only.
"""

from __future__ import annotations

import argparse
import sys
import time

import pandas as pd

from retail import basket, clean, clv, cohort, eda, exports, forecast, ingest, paths, quality, rfm
from retail import returns as returns_mod
from retail.util import configure_stdout, fmt_int, fmt_pct

MIN_DELIVERABLE_BYTES = 10_000

# The committed real sample used when the full ~1M-row dataset is not downloaded.
FIXTURE_SAMPLE = paths.ROOT / "tests" / "fixtures" / "sample.csv"


def run_pipeline(force_reingest: bool = False) -> int:
    t0 = time.time()

    print("[1/10] ingest: loading raw workbook (cached to data/interim/ after first run)")
    df = ingest.load_raw(force=force_reingest)
    raw_report = ingest.raw_quality_report(df)
    ingest.print_quality_report(raw_report)

    print("[2/10] clean: documented pipeline")
    result = clean.clean(df)
    clean_table = result.report()
    print(clean_table.to_string(index=False))
    sales, returns_frame = result.sales, result.returns
    print(
        f"      sales rows {fmt_int(len(sales))} | returns rows {fmt_int(len(returns_frame))} | "
        f"revenue GBP {sales['Revenue'].sum():,.0f}"
    )
    q_cfg = retail_quality_config()
    quality_before = quality.assess(df, q_cfg, label="RAW")
    quality_after = quality.assess(sales, q_cfg, label="CLEANED sales")
    print(f"      quality {quality.compare(quality_before, quality_after)}")

    print("[3/10] eda: writing figures/")
    for fig_path in eda.make_all_figures(sales, returns_frame):
        print(f"      [OK] {fig_path.name}")

    print("[4/10] returns: cancellations analysis (return rates, reverse-logistics lag, per-SKU)")
    returns_result = returns_mod.run_returns_analysis(sales, returns_frame)
    _print_returns_summary(returns_result)
    returns_fig = returns_mod.fig_returns(returns_result, out_dir=paths.FIGURES)
    returns_csv = returns_mod.write_csv(returns_result)
    if returns_fig is not None:
        print(f"      [OK] {returns_fig.name} (committed figure) | {returns_csv.name} (deliverable)")

    print("[5/10] rfm: segmentation on identified customers")
    rfm_customers, rfm_summary = rfm.run_rfm(sales)
    print(f"      customers scored: {fmt_int(len(rfm_customers))}")
    print(rfm_summary.to_string(index=False))

    print("[6/10] cohort: repeat-purchase retention by acquisition month")
    cohort_result = cohort.cohort_retention(sales)
    print(f"      identified customers in cohorts: {fmt_int(cohort_result.n_customers)} | "
          f"cohorts: {len(cohort_result.triangle)} | last complete month {cohort_result.last_complete_month}")
    print(f"      [HEADLINE] {cohort.headline_text(cohort_result)}")
    svg_path = cohort.write_svg(cohort_result)
    csv_path = cohort.write_csv(cohort_result)
    print(f"      [OK] {svg_path.name} (committed figure) | {csv_path.name} (deliverable)")

    print("[7/10] clv: predictive lifetime value (BG/NBD + Gamma-Gamma, out-of-sample check)")
    clv_result = clv.run_clv(sales)
    _print_clv_summary(clv_result)
    clv_fig = clv.fig_clv(clv_result, out_dir=paths.FIGURES)
    clv_csv = clv.write_csv(clv_result)
    if clv_fig is not None:
        print(f"      [OK] {clv_fig.name} (committed figure) | {clv_csv.name} (deliverable)")

    print("[8/10] forecast: rolling-origin CV on weekly revenue")
    weekly = forecast.weekly_revenue(sales)
    cv = forecast.cross_validate(weekly)
    cv_summary = forecast.cv_summary(cv)
    print(cv_summary.to_string(index=False))
    _honesty_check(cv, cv_summary)
    final_fold = forecast.final_fold_forecast(weekly)
    sku_table = forecast.top_sku_forecasts(sales)

    print("[9/10] basket: market-basket mining (Apriori + FP-growth cross-check)")
    basket_result = basket.run_basket_analysis(sales)
    _print_basket_summary(basket_result)
    fig_path = basket.fig_top_rules(basket_result.rules, out_dir=paths.FIGURES)
    if fig_path is not None:
        print(f"      [OK] {fig_path.name}")

    print("[10/10] exports: PDF + Excel deliverables")
    ctx = {
        "raw_report": raw_report,
        "clean_table": clean_table,
        "monthly": eda.monthly_revenue_table(sales),
        "returns_rate": eda.returns_rate_table(sales, returns_frame),
        "returns_result": returns_result,
        "weekly": weekly,
        "cv": cv,
        "cv_summary": cv_summary,
        "final_fold": final_fold,
        "rfm_summary": rfm_summary,
        "cohort_result": cohort_result,
        "clv_result": clv_result,
        "sku_table": sku_table,
        "rules_table": basket.rules_table(basket_result.rules),
        "quality_before": quality_before,
        "quality_after": quality_after,
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


def _print_clv_summary(result) -> None:
    """ASCII summary of the CLV run: fit params, concentration, out-of-sample check."""
    b, g = result.bgnbd, result.gamma_gamma
    conc = result.concentration(0.10)
    print(
        f"      customers {fmt_int(result.n_customers)} "
        f"({fmt_int(result.n_repeat)} repeat) | horizon {result.horizon_days} days"
    )
    print(
        f"      BG/NBD  r={b.r:.4f} alpha={b.alpha:.3f} a={b.a:.4f} b={b.b:.4f} "
        f"(converged={b.converged})"
    )
    print(
        f"      Gamma-Gamma p={g.p:.4f} q={g.q:.4f} v={g.v:.2f} | "
        f"pop. mean value GBP {g.population_mean():,.2f} | "
        f"freq-value corr {g.freq_value_corr:+.3f} (assumption check)"
    )
    print(
        f"      predicted {result.horizon_days}-day revenue GBP {conc['total_clv']:,.0f} | "
        f"top 10% of customers hold {fmt_pct(conc['top_share'], 1)}"
    )
    v = result.validation
    if v is not None and v.actual_total:
        marker = "[OK]" if 0.8 <= v.ratio <= 1.2 else "[WARN]"
        print(
            f"      {marker} out-of-sample holdout ({v.holdout_days} days, "
            f"{fmt_int(v.n_customers)} customers): predicted {v.predicted_total:,.0f} vs "
            f"actual {fmt_int(v.actual_total)} transactions "
            f"({v.ratio:.3f}x), correlation {v.correlation:.2f}, per-customer MAE {v.mae:.3f}"
        )
        print("      predicted vs actual holdout transactions by calibration frequency:")
        print(v.by_frequency.to_string(index=False))
    print(f"      [HEADLINE] {clv.headline_text(result)}")


def _print_returns_summary(result) -> None:
    """ASCII summary of the returns run: rates, matching, lag, top returned SKUs."""
    o = result.overview
    print(
        f"      gross GBP {o['gross_value']:,.0f} | returned GBP {o['returned_value']:,.0f} | "
        f"net GBP {o['net_value']:,.0f}"
    )
    print(
        f"      return rate {fmt_pct(o['value_return_rate'])} of value, "
        f"{fmt_pct(o['unit_return_rate'])} of units | "
        f"{fmt_int(o['return_lines'])} return lines across "
        f"{fmt_int(o['cancellation_invoices'])} cancellation invoices"
    )
    m = result.match
    med = m.get("median_lag_days", float("nan"))
    lag = f", median {med:.0f} days to return (p25 {m['p25_lag_days']:.0f}, p75 {m['p75_lag_days']:.0f})" \
        if m.get("matched_lines") else ""
    print(
        f"      matched to a prior purchase: {fmt_pct(m['matched_value_share'], 1)} of returned value "
        f"({fmt_int(m['matched_lines'])}/{fmt_int(m['return_lines'])} lines){lag}"
    )
    c = result.concentration
    print(
        f"      concentration: top {c['top_n']} SKUs = {fmt_pct(c['top_n_share'], 1)} of returned value; "
        f"{fmt_int(c['skus_to_80pct'])} SKUs cover 80%"
    )
    if not result.by_sku.empty:
        top = result.by_sku.head(10).copy()
        top["returned_value"] = top["returned_value"].round(0)
        top["value_return_rate"] = (100.0 * top["value_return_rate"]).round(1)
        top["Description"] = top["Description"].astype(str).str.slice(0, 34)
        print("      top 10 returned SKUs by value:")
        print(top[["StockCode", "Description", "returned_units", "returned_value",
                   "value_return_rate"]].to_string(index=False))
    print(f"      [HEADLINE] {returns_mod.headline_text(result)}")


def run_returns(force_reingest: bool = False) -> int:
    """Standalone returns run: return rates + reverse-logistics lag + per-SKU CSV/figure."""
    try:
        df = ingest.load_raw(force=force_reingest)
    except FileNotFoundError:
        print(
            "[SKIP] raw data not found - run `python scripts/download_data.py` first "
            "(the dataset is deliberately not committed)."
        )
        return 0
    print("[1/3] clean: documented pipeline (cancellations separated as the returns frame)")
    cleaned = clean.clean(df)
    sales, returns_frame = cleaned.sales, cleaned.returns
    print(f"      sales rows {fmt_int(len(sales))} | returns rows {fmt_int(len(returns_frame))}")
    print("[2/3] returns: rates, reverse-logistics lag, per-SKU return analysis")
    result = returns_mod.run_returns_analysis(sales, returns_frame)
    _print_returns_summary(result)
    print("[3/3] outputs: figures/returns_analysis.png + deliverables/returns_analysis.csv")
    fig_path = returns_mod.fig_returns(result, out_dir=paths.FIGURES)
    csv_path = returns_mod.write_csv(result, out_dir=paths.DELIVERABLES)
    if fig_path is not None:
        print(f"      [OK] {fig_path.name} ({fig_path.stat().st_size / 1024:.1f} KB, committed)")
    print(f"      [OK] {csv_path.name} ({csv_path.stat().st_size / 1024:.1f} KB, deliverable)")
    return 0


def run_clv(force_reingest: bool = False) -> int:
    """Standalone CLV run: BG/NBD + Gamma-Gamma fit, holdout check, figure + CSV."""
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
    print("[2/3] clv: BG/NBD (frequency) x Gamma-Gamma (value), out-of-sample validated")
    result = clv.run_clv(sales)
    _print_clv_summary(result)
    print("      top 10 customers by predicted CLV:")
    print(clv.clv_summary_table(result, top=10).to_string(index=False))
    print("[3/3] outputs: figures/clv_validation.png + deliverables/customer_lifetime_value.csv")
    fig_path = clv.fig_clv(result, out_dir=paths.FIGURES)
    csv_path = clv.write_csv(result, out_dir=paths.DELIVERABLES)
    if fig_path is not None:
        print(f"      [OK] {fig_path.name} ({fig_path.stat().st_size / 1024:.1f} KB, committed)")
    print(f"      [OK] {csv_path.name} ({csv_path.stat().st_size / 1024:.1f} KB, deliverable)")
    return 0


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


def run_cohort(force_reingest: bool = False) -> int:
    """Standalone cohort-retention run: triangle + committed SVG + CSV + headline."""
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
    print("[2/3] cohort: repeat-purchase retention by acquisition month")
    result = cohort.cohort_retention(sales)
    print(f"      identified customers in cohorts: {fmt_int(result.n_customers)} | "
          f"cohorts: {len(result.triangle)} | last complete month {result.last_complete_month}")
    print("      headline retention curve (size-weighted across observable cohorts):")
    curve = result.curve.copy()
    curve["avg_retention"] = (100.0 * curve["avg_retention"]).round(1)
    print(curve.head(13).to_string(index=False))
    print(f"      [HEADLINE] {cohort.headline_text(result)}")
    print("[3/3] outputs: figures/cohort_retention.svg + deliverables/cohort_retention.csv")
    svg_path = cohort.write_svg(result)
    csv_path = cohort.write_csv(result)
    print(f"      [OK] {svg_path.name} ({svg_path.stat().st_size / 1024:.1f} KB, committed)")
    print(f"      [OK] {csv_path.name} ({csv_path.stat().st_size / 1024:.1f} KB, deliverable)")
    return 0


def retail_quality_config() -> quality.QualityConfig:
    """The declared quality expectations for the Online Retail II schema.

    Deliberately the SAME config is applied to the raw frame and to the cleaned
    sales frame, so the before/after scores are directly comparable. CustomerID
    and Description are declared NULLABLE on purpose -- the cleaning pipeline
    keeps missing-CustomerID rows by design, so completeness is honestly not
    where the lift comes from; uniqueness and consistency are.
    """
    return quality.QualityConfig(
        columns={
            "Invoice": quality.ColumnRule(expected_type="string", required=True, regex=r"[A-Za-z]?\d+"),
            "StockCode": quality.ColumnRule(expected_type="string", required=True),
            "Description": quality.ColumnRule(expected_type="string", required=False),
            "Quantity": quality.ColumnRule(expected_type="integer", required=True),
            "InvoiceDate": quality.ColumnRule(expected_type="datetime", required=True),
            "Price": quality.ColumnRule(expected_type="numeric", required=True, min_value=0.0),
            "CustomerID": quality.ColumnRule(expected_type="numeric", required=False),
            "Country": quality.ColumnRule(expected_type="string", required=True),
        },
        rules=(
            quality.Rule("Quantity is positive", "Quantity", ">", 0),
            quality.Rule("Price is positive", "Price", ">", 0),
            quality.Rule("InvoiceDate within 2009-2011 window", "InvoiceDate", "between",
                         ("2009-01-01", "2012-01-01")),
        ),
        dedup_columns=tuple(clean.DEDUP_COLUMNS),
        plausibility_columns=("Quantity", "Price"),
    )


def _load_frame_or_fixture(force_reingest: bool = False) -> tuple[pd.DataFrame, bool]:
    """Return (raw_frame, used_fixture). Degrades to the committed sample if the
    full dataset is not present, and says so -- never downloads anything."""
    try:
        return ingest.load_raw(force=force_reingest), False
    except FileNotFoundError:
        df = pd.read_csv(FIXTURE_SAMPLE, encoding="utf-8")
        df["Invoice"] = df["Invoice"].astype(str)
        df["StockCode"] = df["StockCode"].astype(str)
        df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
        df["CustomerID"] = df["CustomerID"].astype("Int64")
        return df, True


def run_report_card(force_reingest: bool = False, write: bool = True) -> int:
    """Score data quality BEFORE and AFTER the cleaning pipeline and report the lift."""
    raw, used_fixture = _load_frame_or_fixture(force_reingest)
    source = "committed 1,950-row sample fixture" if used_fixture else "full Online Retail II dataset"
    if used_fixture:
        print(f"[INFO] full dataset not found -> running on the {source} (no download).")
    else:
        print(f"[INFO] running on the {source}.")

    cfg = retail_quality_config()
    print("[1/3] scoring RAW data")
    before = quality.assess(raw, cfg, label=f"RAW ({source})")

    print("[2/3] clean: documented pipeline")
    sales = clean.clean(raw).sales
    print(f"      raw rows {fmt_int(len(raw))} -> cleaned sales rows {fmt_int(len(sales))}")

    print("[3/3] scoring CLEANED sales")
    after = quality.assess(sales, cfg, label=f"CLEANED sales ({source})")

    print()
    print(quality.render_report_card(before))
    print()
    print(quality.render_report_card(after))
    print()
    print(quality.compare(before, after))
    # Honest note on what did NOT move, if completeness barely changed.
    b_comp, a_comp = before.dimensions["Completeness"].score, after.dimensions["Completeness"].score
    if abs(a_comp - b_comp) < 1.0:
        print("[HONEST] Completeness barely moved: missing CustomerID/Description are kept by "
              "design (flagged, not dropped), so the lift is in Uniqueness and Consistency.")
    # Report remaining problems in the cleaned data honestly.
    residual = [f for f in after.findings if f.status != "pass"]
    if residual:
        print(f"[HONEST] {len(residual)} check(s) still not 'pass' AFTER cleaning "
              "(the card does not claim a perfect dataset):")
        for f in residual[:8]:
            print(f"         - {f.dimension}/{f.check}: {f.explanation}")

    if write:
        out = paths.DELIVERABLES / "data_quality_report_card.md"
        out.parent.mkdir(parents=True, exist_ok=True)
        body = "\n\n".join([
            "# Data-quality report card -- Online Retail II (raw vs cleaned)",
            "```",
            quality.render_report_card(before),
            "```",
            "```",
            quality.render_report_card(after),
            "```",
            quality.compare(before, after),
        ])
        out.write_text(body, encoding="utf-8")
        print(f"      [OK] wrote {out.name}")
    return 0


def main(argv: list[str] | None = None) -> int:
    configure_stdout()
    parser = argparse.ArgumentParser(prog="python -m retail", description=__doc__)
    parser.add_argument("--deliverables", action="store_true", help="run the full pipeline and write PDF + Excel")
    parser.add_argument("--basket", action="store_true", help="market-basket analysis: top co-purchase rules + figure")
    parser.add_argument("--cohort", action="store_true",
                        help="cohort repeat-purchase retention: triangle + SVG heatmap/curve + CSV")
    parser.add_argument("--clv", action="store_true",
                        help="customer lifetime value: BG/NBD + Gamma-Gamma, out-of-sample holdout check, figure + CSV")
    parser.add_argument("--returns", action="store_true",
                        help="returns & cancellations analysis: return rates, reverse-logistics lag, per-SKU CSV")
    parser.add_argument("--quality", action="store_true", help="print the raw-quality report only")
    parser.add_argument("--report-card", action="store_true",
                        help="data-quality report card: score raw vs cleaned and report the lift")
    parser.add_argument("--force-reingest", action="store_true", help="ignore the interim cache")
    args = parser.parse_args(argv)

    if args.report_card:
        return run_report_card(force_reingest=args.force_reingest)
    if args.basket:
        return run_basket(force_reingest=args.force_reingest)
    if args.cohort:
        return run_cohort(force_reingest=args.force_reingest)
    if args.clv:
        return run_clv(force_reingest=args.force_reingest)
    if args.returns:
        return run_returns(force_reingest=args.force_reingest)
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
