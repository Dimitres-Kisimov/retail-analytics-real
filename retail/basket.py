"""Market-basket analysis on the real invoice data (what actually sells together).

The mining engine (Apriori, FP-growth) and the rule metrics below are adapted
from my market-basket-analysis repo (basket/apriori.py, basket/fpgrowth.py,
basket/rules.py) — that repo mined 14 synthetic categories; this one mines real
SKUs, so the only functional change is the Apriori support-counting step (see
``apriori``). Everything is still from scratch: no mlxtend, no mining library.

Item universe — the honest choice, documented:

Product level would explode (5,305 stock codes, most of them rare), so baskets
are built over the TOP ``top_n_items`` (default 200) stock codes by cleaned
gross revenue. Each code is labelled with its most frequent normalised
description; codes whose normalised descriptions collide (recoded duplicates of
the same product) are MERGED into one item. A basket is one sales invoice,
reduced to the tracked items it contains; invoices with fewer than two tracked
items are excluded, because they cannot express a co-purchase — all support
fractions therefore read "share of multi-item invoices over the tracked
assortment", not "share of all invoices". Both restrictions are reported by
:func:`build_baskets` and printed by the CLI rather than hidden.

Interpretation caveat (also in the README): these are observational co-purchase
statistics on a single UK gift retailer with wholesale-sized orders. High lift
means "bought together more often than independence predicts" — never causation.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path

import pandas as pd

from retail.paths import FIGURES


# --------------------------------------------------------------------------- #
# Apriori (adapted from my market-basket-analysis repo: basket/apriori.py)
# --------------------------------------------------------------------------- #
def min_count_for_support(min_support: float, n_baskets: int) -> int:
    """Smallest absolute count c such that c / n_baskets >= min_support.

    The tiny epsilon guards against float products like 0.02 * 6000 landing an
    ulp above the exact integer and inflating the ceiling by one.
    """
    return max(1, math.ceil(min_support * n_baskets - 1e-9))


def _generate_candidates(
    previous_frequent: set[frozenset[str]], k: int
) -> set[frozenset[str]]:
    """Join step + prune step (downward closure) of Apriori."""
    sorted_itemsets = sorted(tuple(sorted(itemset)) for itemset in previous_frequent)
    candidates: set[frozenset[str]] = set()
    for i, a in enumerate(sorted_itemsets):
        for b in sorted_itemsets[i + 1 :]:
            if a[: k - 2] != b[: k - 2]:
                continue  # prefix join: only merge itemsets sharing the first k-2 items
            candidate = frozenset(a) | frozenset(b)
            if len(candidate) != k:
                continue
            # Prune: every (k-1)-subset must itself be frequent.
            if all(
                frozenset(subset) in previous_frequent
                for subset in combinations(sorted(candidate), k - 1)
            ):
                candidates.add(candidate)
    return candidates


def apriori(
    baskets: Sequence[Iterable[str]],
    min_support: float = 0.01,
    max_len: int = 3,
) -> dict[frozenset[str], float]:
    """Mine frequent itemsets up to ``max_len`` items.

    Returns ``itemset -> support`` where support is count / n_baskets, exactly
    comparable with :func:`fpgrowth` (a test asserts equality).

    Adaptation from the synthetic repo: the original counted candidate support
    by scanning every basket against every candidate — fine for 14 categories,
    quadratic pain for ~200 SKUs (tens of thousands of pair candidates times
    tens of thousands of baskets). Here each frequent itemset carries its
    basket-id set (tidset) and a candidate's count is one set intersection.
    The counts are identical; only the counting order changed.
    """
    if not 0.0 < min_support <= 1.0:
        raise ValueError("min_support must be in (0, 1]")
    basket_sets = [frozenset(b) for b in baskets]
    n = len(basket_sets)
    if n == 0:
        return {}
    min_count = min_count_for_support(min_support, n)

    item_tidsets: dict[str, set[int]] = defaultdict(set)
    for tid, basket in enumerate(basket_sets):
        for item in basket:
            item_tidsets[item].add(tid)

    current: dict[frozenset[str], set[int]] = {
        frozenset((item,)): tids
        for item, tids in item_tidsets.items()
        if len(tids) >= min_count
    }
    all_frequent: dict[frozenset[str], int] = {
        itemset: len(tids) for itemset, tids in current.items()
    }

    k = 2
    while current and k <= max_len:
        candidates = _generate_candidates(set(current), k)
        next_level: dict[frozenset[str], set[int]] = {}
        for candidate in candidates:
            # Any item can be split off: the (k-1)-remainder is frequent by the
            # prune step, so its tidset is in ``current``.
            item = next(iter(candidate))
            tids = current[candidate - {item}] & item_tidsets[item]
            if len(tids) >= min_count:
                next_level[candidate] = tids
        current = next_level
        all_frequent.update({c: len(tids) for c, tids in current.items()})
        k += 1

    return {itemset: count / n for itemset, count in all_frequent.items()}


# --------------------------------------------------------------------------- #
# FP-growth (adapted from my market-basket-analysis repo: basket/fpgrowth.py)
# --------------------------------------------------------------------------- #
class _FPNode:
    __slots__ = ("children", "count", "item", "parent")

    def __init__(self, item: str | None, parent: _FPNode | None):
        self.item = item
        self.parent = parent
        self.count = 0
        self.children: dict[str, _FPNode] = {}


def _build_tree(
    weighted_transactions: list[tuple[list[str], int]], min_count: int
) -> tuple[dict[str, list[_FPNode]], dict[str, int]]:
    """Build an FP-tree; return its header table and per-item frequent counts."""
    item_counts: dict[str, int] = defaultdict(int)
    for items, weight in weighted_transactions:
        for item in items:
            item_counts[item] += weight
    frequent = {item: c for item, c in item_counts.items() if c >= min_count}
    # Descending count, name as tie-break: the canonical FP-tree insertion order.
    rank = {
        item: position
        for position, item in enumerate(
            sorted(frequent, key=lambda it: (-frequent[it], it))
        )
    }

    root = _FPNode(None, None)
    header: dict[str, list[_FPNode]] = defaultdict(list)
    for items, weight in weighted_transactions:
        path = sorted((i for i in items if i in frequent), key=rank.__getitem__)
        node = root
        for item in path:
            child = node.children.get(item)
            if child is None:
                child = _FPNode(item, node)
                node.children[item] = child
                header[item].append(child)
            child.count += weight
            node = child
    return header, frequent


def _mine(
    weighted_transactions: list[tuple[list[str], int]],
    min_count: int,
    suffix: frozenset[str],
    results: dict[frozenset[str], int],
    max_len: int | None,
) -> None:
    header, frequent = _build_tree(weighted_transactions, min_count)
    # Mine items from least to most frequent (bottom of the tree upward).
    for item in sorted(frequent, key=lambda it: (frequent[it], it)):
        itemset = frozenset(suffix | {item})
        results[itemset] = frequent[item]
        if max_len is not None and len(itemset) >= max_len:
            continue
        conditional_base: list[tuple[list[str], int]] = []
        for node in header[item]:
            path: list[str] = []
            parent = node.parent
            while parent is not None and parent.item is not None:
                path.append(parent.item)
                parent = parent.parent
            if path:
                conditional_base.append((path, node.count))
        if conditional_base:
            _mine(conditional_base, min_count, itemset, results, max_len)


def fpgrowth(
    baskets: Sequence[Iterable[str]],
    min_support: float = 0.01,
    max_len: int | None = 3,
) -> dict[frozenset[str], float]:
    """Mine frequent itemsets with FP-growth; same contract as ``apriori``."""
    if not 0.0 < min_support <= 1.0:
        raise ValueError("min_support must be in (0, 1]")
    basket_sets = [frozenset(b) for b in baskets]
    n = len(basket_sets)
    if n == 0:
        return {}
    min_count = min_count_for_support(min_support, n)
    weighted = [(sorted(b), 1) for b in basket_sets]
    results: dict[frozenset[str], int] = {}
    _mine(weighted, min_count, frozenset(), results, max_len)
    return {itemset: count / n for itemset, count in results.items()}


# --------------------------------------------------------------------------- #
# Rules (adapted from my market-basket-analysis repo: basket/rules.py)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Rule:
    """An association rule ``antecedent -> consequent`` with its metrics."""

    antecedent: frozenset[str]
    consequent: frozenset[str]
    support: float  # fraction of baskets containing antecedent AND consequent
    confidence: float  # P(consequent | antecedent)
    lift: float  # confidence / P(consequent)
    leverage: float  # support - P(antecedent) * P(consequent)
    conviction: float  # (1 - P(consequent)) / (1 - confidence); inf if confidence == 1
    support_count: int  # absolute number of baskets behind the rule
    thin_support: bool  # True when support_count is below the stability threshold


def format_itemset(itemset: frozenset[str]) -> str:
    return " + ".join(sorted(itemset))


def format_rule(rule: Rule) -> str:
    return f"{format_itemset(rule.antecedent)} -> {format_itemset(rule.consequent)}"


def generate_rules(
    itemsets: Mapping[frozenset[str], float],
    n_baskets: int,
    min_confidence: float = 0.30,
    min_lift: float = 1.10,
    thin_support_count: int = 30,
) -> list[Rule]:
    """Derive filtered, lift-ranked rules from mined frequent itemsets.

    Every antecedent/consequent split of every itemset with two or more items
    is scored; rules below ``min_confidence`` or ``min_lift`` are dropped.
    Rules backed by fewer than ``thin_support_count`` baskets are kept but
    flagged ``thin_support=True`` so downstream consumers can exclude them.
    """
    if n_baskets < 1:
        raise ValueError("n_baskets must be positive")
    rules: list[Rule] = []
    for itemset, support in itemsets.items():
        if len(itemset) < 2:
            continue
        items = sorted(itemset)
        for split_size in range(1, len(items)):
            for antecedent_items in combinations(items, split_size):
                antecedent = frozenset(antecedent_items)
                consequent = itemset - antecedent
                support_antecedent = itemsets.get(antecedent)
                support_consequent = itemsets.get(consequent)
                if support_antecedent is None or support_consequent is None:
                    # Cannot happen for complete downward-closed input; guard anyway.
                    continue
                confidence = support / support_antecedent
                lift = confidence / support_consequent
                leverage = support - support_antecedent * support_consequent
                if confidence >= 1.0:
                    conviction = math.inf
                else:
                    conviction = (1.0 - support_consequent) / (1.0 - confidence)
                if confidence < min_confidence or lift < min_lift:
                    continue
                support_count = round(support * n_baskets)
                rules.append(
                    Rule(
                        antecedent=antecedent,
                        consequent=consequent,
                        support=support,
                        confidence=confidence,
                        lift=lift,
                        leverage=leverage,
                        conviction=conviction,
                        support_count=support_count,
                        thin_support=support_count < thin_support_count,
                    )
                )
    rules.sort(
        key=lambda r: (
            -r.lift,
            -r.confidence,
            format_itemset(r.antecedent),
            format_itemset(r.consequent),
        )
    )
    return rules


# --------------------------------------------------------------------------- #
# Real-data basket construction (new code, specific to Online Retail II)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class BasketConfig:
    """Thresholds for the real-data run. Every choice is documented where used."""

    top_n_items: int = 200  # tracked assortment: top-N stock codes by revenue
    min_basket_items: int = 2  # invoices with < 2 tracked items cannot co-purchase
    min_support: float = 0.01  # >= 1% of multi-item invoices (documented in README)
    max_len: int = 3  # pairs and triples, like the synthetic repo
    min_confidence: float = 0.30
    min_lift: float = 1.10
    thin_support_count: int = 30  # flag rules backed by < 30 invoices
    fpgrowth_sample: int = 5000  # cross-check sample size
    fpgrowth_seed: int = 42

    def validate(self) -> None:
        if self.top_n_items < 2:
            raise ValueError("top_n_items must be at least 2")
        if self.min_basket_items < 2:
            raise ValueError("min_basket_items must be at least 2")
        if not 0.0 < self.min_support <= 1.0:
            raise ValueError("min_support must be in (0, 1]")
        if self.max_len < 2:
            raise ValueError("max_len must be at least 2")
        if not 0.0 < self.min_confidence <= 1.0:
            raise ValueError("min_confidence must be in (0, 1]")
        if self.min_lift < 1.0:
            raise ValueError("min_lift below 1.0 would admit negative associations")
        if self.thin_support_count < 1:
            raise ValueError("thin_support_count must be positive")


#: Shared default thresholds (frozen, so safe as a module-level singleton).
DEFAULT_CONFIG = BasketConfig()


def normalize_description(desc: pd.Series) -> pd.Series:
    """Uppercase, trim, collapse internal whitespace (real descriptions vary in all three)."""
    return desc.astype(str).str.upper().str.strip().str.replace(r"\s+", " ", regex=True)


def item_catalog(sales: pd.DataFrame, top_n: int = 200) -> pd.Series:
    """Map StockCode -> item label for the top ``top_n`` codes by gross revenue.

    The label is the code's most frequent normalised description. Distinct codes
    that end up with the same label (the dataset recodes some products) are
    thereby merged into one item — deliberate: the analysis is about products,
    not warehouse codes.
    """
    frame = sales[["StockCode", "Description", "Revenue"]].copy()
    frame["Item"] = normalize_description(frame["Description"])
    frame = frame[frame["Item"].str.len() > 0]
    top_codes = (
        frame.groupby("StockCode", observed=True)["Revenue"].sum().nlargest(top_n).index
    )
    frame = frame[frame["StockCode"].isin(top_codes)]
    modal = (
        frame.groupby(["StockCode", "Item"], observed=True)
        .size()
        .rename("rows")
        .reset_index()
        .sort_values(["StockCode", "rows", "Item"], ascending=[True, False, True])
        .drop_duplicates("StockCode")
    )
    return modal.set_index("StockCode")["Item"]


def build_baskets(
    sales: pd.DataFrame, config: BasketConfig = DEFAULT_CONFIG
) -> tuple[list[frozenset[str]], dict]:
    """Invoice-level baskets over the tracked assortment, plus honest survivor stats.

    Returns ``(baskets, stats)`` where stats reports how many invoices and items
    survived each restriction — the CLI and README print these numbers.
    """
    config.validate()
    catalog = item_catalog(sales, config.top_n_items)
    tracked = sales[sales["StockCode"].isin(catalog.index)].copy()
    tracked["Item"] = tracked["StockCode"].map(catalog)
    grouped = tracked.groupby("Invoice", observed=True)["Item"]
    all_baskets = [frozenset(items) for _, items in grouped]
    baskets = [b for b in all_baskets if len(b) >= config.min_basket_items]
    stats = {
        "invoices_total": int(sales["Invoice"].nunique()),
        "invoices_with_tracked_item": len(all_baskets),
        "baskets_mined": len(baskets),
        "stock_codes_tracked": int(len(catalog)),
        "items_after_description_merge": int(catalog.nunique()),
        "mean_basket_size": (
            float(sum(len(b) for b in baskets) / len(baskets)) if baskets else 0.0
        ),
    }
    return baskets, stats


@dataclass
class BasketResult:
    config: BasketConfig
    stats: dict
    itemsets: dict[frozenset[str], float]
    rules: list[Rule] = field(default_factory=list)
    crosscheck_ok: bool | None = None  # FP-growth == Apriori on the sample
    crosscheck_baskets: int = 0

    @property
    def n_baskets(self) -> int:
        return self.stats["baskets_mined"]

    def itemset_counts(self) -> dict[int, int]:
        counts: dict[int, int] = {}
        for itemset in self.itemsets:
            counts[len(itemset)] = counts.get(len(itemset), 0) + 1
        return dict(sorted(counts.items()))


def run_basket_analysis(
    sales: pd.DataFrame, config: BasketConfig = DEFAULT_CONFIG
) -> BasketResult:
    """Baskets -> Apriori itemsets -> rules, with an FP-growth cross-check.

    Apriori mines the full basket list; the independently implemented FP-growth
    mines a seeded sample and must return identical itemsets and supports on it
    (``crosscheck_ok``). On fixtures/small inputs the sample is the full list.
    """
    baskets, stats = build_baskets(sales, config)
    result = BasketResult(config=config, stats=stats, itemsets={})
    if not baskets:
        return result

    result.itemsets = apriori(baskets, config.min_support, config.max_len)
    result.rules = generate_rules(
        result.itemsets,
        n_baskets=len(baskets),
        min_confidence=config.min_confidence,
        min_lift=config.min_lift,
        thin_support_count=config.thin_support_count,
    )

    if len(baskets) <= config.fpgrowth_sample:
        sample = baskets
    else:
        # Seeded, order-independent sample; both miners see the same baskets.
        import random

        rng = random.Random(config.fpgrowth_seed)
        sample = rng.sample(baskets, config.fpgrowth_sample)
    result.crosscheck_baskets = len(sample)
    apriori_sample = apriori(sample, config.min_support, config.max_len)
    fpgrowth_sample = fpgrowth(sample, config.min_support, config.max_len)
    result.crosscheck_ok = _itemsets_equal(apriori_sample, fpgrowth_sample)
    return result


def _itemsets_equal(
    a: Mapping[frozenset[str], float], b: Mapping[frozenset[str], float]
) -> bool:
    if set(a) != set(b):
        return False
    return all(math.isclose(a[k], b[k], rel_tol=0, abs_tol=1e-12) for k in a)


def rules_table(rules: Sequence[Rule], top: int | None = None) -> pd.DataFrame:
    """Rules as a flat frame for the Excel Rules sheet and the CLI print."""
    rows = [
        {
            "Antecedent": format_itemset(r.antecedent),
            "Consequent": format_itemset(r.consequent),
            "Support": round(r.support, 4),
            "Invoices": r.support_count,
            "Confidence": round(r.confidence, 3),
            "Lift": round(r.lift, 2),
            "Leverage": round(r.leverage, 4),
            "ThinSupport": r.thin_support,
        }
        for r in (rules[:top] if top else rules)
    ]
    columns = [
        "Antecedent", "Consequent", "Support", "Invoices",
        "Confidence", "Lift", "Leverage", "ThinSupport",
    ]
    return pd.DataFrame(rows, columns=columns)


# --------------------------------------------------------------------------- #
# Figure
# --------------------------------------------------------------------------- #
def dedupe_by_itemset(rules: Sequence[Rule]) -> list[Rule]:
    """One rule per co-purchase item set (both directions share support and lift).

    ``generate_rules`` sorts by lift then confidence, so the first direction
    encountered is the higher-confidence one — that is the direction kept.
    """
    seen: set[frozenset[str]] = set()
    unique: list[Rule] = []
    for rule in rules:
        key = rule.antecedent | rule.consequent
        if key not in seen:
            seen.add(key)
            unique.append(rule)
    return unique


def fig_top_rules(
    rules: Sequence[Rule], out_dir: Path = FIGURES, n: int = 10
) -> Path | None:
    """Top-``n`` co-purchase item sets by lift as a single-hue horizontal bar chart.

    Thin-support rules are excluded (their lift is unstable) and each item set
    appears once, in its higher-confidence direction; returns None when nothing
    solid survives. Single series measuring one magnitude (lift), so one hue and
    direct labels — same validated palette as the other figures.
    """
    import matplotlib.pyplot as plt  # noqa: F401  (backend configured in retail.eda)

    from retail.eda import BLUE, INK_2, MUTED, _new_axes, _save

    solid = dedupe_by_itemset([r for r in rules if not r.thin_support])[:n]
    if not solid:
        return None
    labels = [f"{_short(format_itemset(r.antecedent))}\n-> {_short(format_itemset(r.consequent))}"
              for r in solid][::-1]
    lifts = [r.lift for r in solid][::-1]
    fig, ax = _new_axes(figsize=(9.5, 6.2))
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", visible=True)
    ax.barh(range(len(solid)), lifts, color=BLUE, height=0.62)
    ax.set_yticks(range(len(solid)))
    ax.set_yticklabels(labels, fontsize=7.5)
    ax.set_xlabel("lift (co-purchase frequency vs independence; 1 = independent)")
    ax.axvline(1.0, color=MUTED, linewidth=1, linestyle="--")
    for i, (lift, rule) in enumerate(zip(lifts, solid[::-1], strict=True)):
        ax.text(lift, i, f" {lift:.1f}x ({rule.support_count} inv.)",
                va="center", fontsize=7.5, color=INK_2)
    ax.set_xlim(0, max(lifts) * 1.30)
    ax.set_title(f"Top {len(solid)} co-purchase item sets by lift")
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.text(
        0.02,
        0.012,
        "Observational co-purchase frequency, not causation. Each item set shown once "
        "(higher-confidence direction); thin-support rules excluded.",
        fontsize=7.5,
        color=MUTED,
    )
    return _save(fig, out_dir / "basket_top_rules.png", tight=False)


def _short(text: str, width: int = 46) -> str:
    return text if len(text) <= width else text[: width - 3] + "..."
