"""Market-basket tests: fixture baskets, hand-checked metrics, miner agreement, CLI.

Everything runs on tests/fixtures/sample.csv or on a ten-basket hand case whose
metrics are verifiable with pencil and paper. The full dataset is never needed;
the CLI test for the no-data path forces absence via monkeypatching.
"""

from __future__ import annotations

import pytest

from retail import basket, ingest, paths
from retail.__main__ import main

# Ten baskets, hand-checkable: tea in 4, saucer in 4, both in 3, box in 5.
TINY_BASKETS = (
    [frozenset({"tea", "saucer"})] * 3
    + [frozenset({"tea"})]
    + [frozenset({"saucer"})]
    + [frozenset({"box"})] * 5
)


def test_build_baskets_from_fixture(cleaned):
    baskets, stats = basket.build_baskets(cleaned.sales)
    assert stats["invoices_total"] == cleaned.sales["Invoice"].nunique()
    assert stats["baskets_mined"] == len(baskets) > 0
    assert stats["baskets_mined"] <= stats["invoices_with_tracked_item"] <= stats["invoices_total"]
    # every basket holds >= 2 tracked items, and only labels from the catalog
    labels = set(basket.item_catalog(cleaned.sales, 200))
    assert all(len(b) >= 2 for b in baskets)
    assert all(item in labels for b in baskets for item in b)


def test_metrics_hand_checked():
    itemsets = basket.apriori(TINY_BASKETS, min_support=0.2, max_len=2)
    assert itemsets[frozenset({"tea"})] == pytest.approx(0.4)
    assert itemsets[frozenset({"saucer"})] == pytest.approx(0.4)
    assert itemsets[frozenset({"box"})] == pytest.approx(0.5)
    assert itemsets[frozenset({"tea", "saucer"})] == pytest.approx(0.3)
    assert len(itemsets) == 4  # no other pair reaches 2 of 10 baskets

    rules = basket.generate_rules(
        itemsets, n_baskets=10, min_confidence=0.3, min_lift=1.1, thin_support_count=4
    )
    rule = next(r for r in rules if r.antecedent == frozenset({"tea"}))
    # by hand: support 3/10, confidence 3/4, lift 0.75/0.4, leverage 0.3 - 0.4*0.4
    assert rule.support == pytest.approx(0.3)
    assert rule.confidence == pytest.approx(0.75)
    assert rule.lift == pytest.approx(1.875)
    assert rule.leverage == pytest.approx(0.14)
    assert rule.conviction == pytest.approx((1 - 0.4) / (1 - 0.75))
    assert rule.support_count == 3
    assert rule.thin_support  # 3 invoices < threshold of 4 -> flagged, not dropped


def test_apriori_equals_fpgrowth_on_fixture(cleaned):
    baskets, _ = basket.build_baskets(cleaned.sales)
    from_apriori = basket.apriori(baskets, min_support=0.05, max_len=3)
    from_fpgrowth = basket.fpgrowth(baskets, min_support=0.05, max_len=3)
    assert len(from_apriori) > 0
    assert from_apriori.keys() == from_fpgrowth.keys()
    for itemset, support in from_apriori.items():
        assert support == pytest.approx(from_fpgrowth[itemset], abs=1e-12)
    # the same equality is what run_basket_analysis reports as its cross-check
    result = basket.run_basket_analysis(
        cleaned.sales, basket.BasketConfig(min_support=0.05, thin_support_count=3)
    )
    assert result.crosscheck_ok is True


def test_threshold_config_sanity():
    basket.BasketConfig().validate()  # shipped defaults must be valid
    assert basket.min_count_for_support(0.02, 6000) == 120
    assert basket.min_count_for_support(0.01, 29_639) == 297
    with pytest.raises(ValueError):
        basket.BasketConfig(min_support=0.0).validate()
    with pytest.raises(ValueError):
        basket.BasketConfig(min_lift=0.9).validate()
    with pytest.raises(ValueError):
        basket.BasketConfig(min_basket_items=1).validate()
    with pytest.raises(ValueError):
        basket.apriori(TINY_BASKETS, min_support=1.5)
    with pytest.raises(ValueError):
        basket.fpgrowth(TINY_BASKETS, min_support=0.0)


def test_cli_basket_skips_gracefully_without_raw_data(monkeypatch, capsys):
    def _absent(force=False):
        raise FileNotFoundError("raw data not downloaded")

    monkeypatch.setattr(ingest, "load_raw", _absent)
    assert main(["--basket"]) == 0
    out = capsys.readouterr().out
    assert "[SKIP]" in out
    assert "download_data.py" in out


def test_cli_basket_runs_on_fixture(monkeypatch, capsys, tmp_path, raw_fixture):
    monkeypatch.setattr(ingest, "load_raw", lambda force=False: raw_fixture.copy())
    monkeypatch.setattr(paths, "FIGURES", tmp_path)
    assert main(["--basket"]) == 0
    out = capsys.readouterr().out
    assert "FP-growth cross-check" in out
    assert "[FAIL]" not in out


def test_fig_top_rules_writes_png(tmp_path):
    itemsets = basket.apriori(TINY_BASKETS, min_support=0.2, max_len=2)
    rules = basket.generate_rules(
        itemsets, n_baskets=10, min_confidence=0.3, min_lift=1.1, thin_support_count=2
    )
    path = basket.fig_top_rules(rules, out_dir=tmp_path)
    assert path is not None and path.name == "basket_top_rules.png"
    assert path.stat().st_size > 5_000
    # a rule set with nothing solid yields no figure, not a crash
    all_thin = basket.generate_rules(itemsets, n_baskets=10, thin_support_count=99)
    assert basket.fig_top_rules(all_thin, out_dir=tmp_path / "empty") is None
