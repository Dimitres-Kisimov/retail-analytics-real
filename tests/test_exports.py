"""Deliverables build from fixture-sized inputs and are non-trivial files."""

from __future__ import annotations

from retail import eda, exports, forecast, ingest, rfm


def test_pdf_and_excel_exports_nonempty(tmp_path, raw_fixture, cleaned):
    sales, returns = cleaned.sales, cleaned.returns
    weekly = forecast.weekly_revenue(sales)
    cv = forecast.cross_validate(weekly, horizon=8, n_folds=3, step=8)
    _, rfm_summary = rfm.run_rfm(sales)
    ctx = {
        "raw_report": ingest.raw_quality_report(raw_fixture),
        "clean_table": cleaned.report(),
        "monthly": eda.monthly_revenue_table(sales),
        "returns_rate": eda.returns_rate_table(sales, returns),
        "weekly": weekly,
        "cv": cv,
        "cv_summary": forecast.cv_summary(cv),
        "final_fold": forecast.final_fold_forecast(weekly),
        "rfm_summary": rfm_summary,
        "sku_table": forecast.top_sku_forecasts(sales),
    }
    pdf = exports.export_pdf(ctx, tmp_path / "exec.pdf")
    xlsx = exports.export_excel(ctx, tmp_path / "workbook.xlsx")
    assert pdf.stat().st_size > 5_000
    assert xlsx.stat().st_size > 5_000
