"""Generic, config-driven data-quality report card for any pandas DataFrame.

This module scores the quality of a tabular dataset across five standard
dimensions and renders a human-readable report card. It is deliberately
dataset-agnostic: give it a DataFrame and (optionally) a :class:`QualityConfig`
declaring per-column expected types, ranges/regex/domains, key columns and a few
cross-field rules, and it returns a structured :class:`ReportCard`. With no
config it falls back to sensible dtype-inferred defaults and *says so* on the
card.

HONESTY -- READ THIS FIRST
--------------------------
This is a **heuristic scorecard with stated weights and stated rules, NOT a
certification of data correctness.** A high score means the data *passed the
declared checks*; it does NOT mean the data is true, accurate, or fit for any
particular purpose. Only the checks you declare are checked; everything else is
invisible to the score. The plausibility dimension is an outlier *heuristic*,
not a judgement that any row is wrong. Treat the grade as a triage signal, not
a warranty.

How the score is built (no magic formula)
-----------------------------------------
Every individual check produces a 0-100 score from a violation fraction::

    score = 100 * (1 - violations / evaluated)

and a pass/warn/fail status from two documented thresholds
(:data:`WARN_BELOW`, :data:`FAIL_BELOW` -- pass means < 0.5% of evaluated rows
violate the check, warn means 0.5-3%, fail means > 3%). A dimension's *score* is
the plain mean of its checks' scores; a dimension's *status* is the WORST status
among its checks (a diluting mean must not hide one broken check). The overall
score is the weighted mean of the dimension scores using the explicit weights in
:data:`DIMENSION_WEIGHTS`, **renormalised over the dimensions that actually ran**
(a dimension with no applicable checks is dropped, not scored as zero). The
letter grade comes from :data:`GRADE_CUTS`, then is **capped by the worst status
among the four non-heuristic dimensions** (:data:`HARD_DIMENSIONS`): a failing
hard check caps the grade at C, a warning caps it at B. Plausibility is a
heuristic and never caps the grade nor hard-fails (its status is clamped to at
most 'warn'). All weights and cut-offs are module constants you can read and
change; nothing is hidden.

Determinism: pure functions, no network, no I/O, no randomness, no wall-clock in
the result or the rendered card -- the same frame in gives the same card out.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Stated constants -- the whole scoring model lives here, in the open.
# --------------------------------------------------------------------------- #
DIMENSIONS = ("Completeness", "Validity", "Uniqueness", "Consistency", "Plausibility")

#: The four non-heuristic dimensions. Plausibility is excluded because it is an
#: outlier heuristic, not a correctness check; it never caps the letter grade.
HARD_DIMENSIONS = ("Completeness", "Validity", "Uniqueness", "Consistency")

#: Explicit dimension weights for the overall score. They sum to 1.0 and are
#: renormalised over whichever dimensions actually run on a given frame/config.
DIMENSION_WEIGHTS: dict[str, float] = {
    "Completeness": 0.30,
    "Validity": 0.25,
    "Consistency": 0.20,
    "Uniqueness": 0.15,
    "Plausibility": 0.10,
}

#: Status thresholds applied to any 0-100 score. score >= WARN_BELOW -> pass;
#: FAIL_BELOW <= score < WARN_BELOW -> warn; score < FAIL_BELOW -> fail.
#: Equivalently: pass < 0.5% violations, warn 0.5-3%, fail > 3% of evaluated rows.
WARN_BELOW = 99.5
FAIL_BELOW = 97.0

#: Letter-grade cut-offs on the overall 0-100 score (checked high to low).
GRADE_CUTS: tuple[tuple[float, str], ...] = (
    (95.0, "A"),
    (90.0, "B"),
    (80.0, "C"),
    (70.0, "D"),
    (0.0, "F"),
)
_GRADE_ORDER = ("A", "B", "C", "D", "F")

#: A failing hard check caps the grade at C; a warning caps it at B.
_STATUS_GRADE_CAP = {"pass": None, "warn": "B", "fail": "C"}

#: The caveat printed on every card header. Its presence is asserted in tests.
NOT_A_CERTIFICATION = (
    "This is a heuristic scorecard based on the declared checks and stated weights. "
    "A high score means the data PASSED the declared checks -- it is NOT a certification "
    "of data correctness, accuracy, or fitness for use."
)

_STATUS_MARKER = {"pass": "[OK]", "warn": "[WARN]", "fail": "[FAIL]"}
_STATUS_RANK = {"pass": 0, "warn": 1, "fail": 2}


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ColumnRule:
    """Declared expectations for a single column.

    ``expected_type`` is one of ``numeric``/``integer``/``float``/``string``/
    ``datetime``/``boolean`` (drives Validity type conformance). ``required``
    marks the column as non-null-expected (drives Completeness). ``min_value``/
    ``max_value``/``allowed``/``regex`` drive Validity range/domain/pattern
    checks. Only the attributes you set are checked.
    """

    expected_type: str | None = None
    required: bool = False
    min_value: float | None = None
    max_value: float | None = None
    allowed: tuple | None = None
    regex: str | None = None


@dataclass(frozen=True)
class Rule:
    """A declared consistency rule (single-column or cross-field).

    ``op`` is one of ``>`` ``>=`` ``<`` ``<=`` ``==`` ``!=`` ``between`` ``in``
    ``regex`` ``notnull``. Compare a column against a scalar ``value`` (or a
    ``(lo, hi)`` tuple for ``between`` / an iterable for ``in``), or against
    another column via ``other_column`` (for the arithmetic ops). Rows where the
    operand(s) are null are not evaluated -- completeness owns nulls.
    """

    name: str
    column: str
    op: str
    value: object = None
    other_column: str | None = None
    kind: str = "consistency"  # label only; shown in explanations


@dataclass(frozen=True)
class QualityConfig:
    """Declares what "good" means for a dataset. Everything is optional."""

    columns: Mapping[str, ColumnRule] = field(default_factory=dict)
    rules: tuple[Rule, ...] = ()
    key_columns: tuple[str, ...] | None = None
    dedup_columns: tuple[str, ...] | None = None
    plausibility_columns: tuple[str, ...] | None = None
    weights: Mapping[str, float] = field(default_factory=lambda: dict(DIMENSION_WEIGHTS))
    warn_below: float = WARN_BELOW
    fail_below: float = FAIL_BELOW


# --------------------------------------------------------------------------- #
# Result structure
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Finding:
    """One check's outcome: what was checked, how it scored, and why."""

    dimension: str
    check: str
    status: str  # "pass" | "warn" | "fail"
    score: float  # 0-100
    count: int  # offending rows/cells
    total: int  # rows/cells evaluated (the denominator)
    column: str | None
    explanation: str


@dataclass(frozen=True)
class DimensionResult:
    dimension: str
    score: float
    status: str
    findings: tuple[Finding, ...]
    ran: bool  # False when no applicable checks -> excluded from the overall


@dataclass(frozen=True)
class ReportCard:
    """The structured report card. ``findings`` is the flat cross-dimension list."""

    overall_score: float
    grade: str
    dimensions: dict[str, DimensionResult]
    findings: tuple[Finding, ...]
    weights: dict[str, float]  # the renormalised weights actually used
    n_rows: int
    n_cols: int
    used_defaults: bool
    label: str
    caveat: str = NOT_A_CERTIFICATION

    @property
    def worst_status(self) -> str:
        if not self.findings:
            return "pass"
        return max((f.status for f in self.findings), key=_STATUS_RANK.get)


# --------------------------------------------------------------------------- #
# Scoring primitives
# --------------------------------------------------------------------------- #
def _score(n_violations: int, n_total: int) -> float:
    """0-100 from a violation fraction. Empty denominator -> 100 (vacuously)."""
    if n_total <= 0:
        return 100.0
    return round(100.0 * (1.0 - n_violations / n_total), 1)


def _status(score: float, warn_below: float, fail_below: float) -> str:
    if score < fail_below:
        return "fail"
    if score < warn_below:
        return "warn"
    return "pass"


def _grade(score: float) -> str:
    for cut, letter in GRADE_CUTS:
        if score >= cut:
            return letter
    return "F"


def _cap_grade(grade: str, worst_hard_status: str) -> str:
    """Cap a letter grade by the worst status among the hard dimensions."""
    cap = _STATUS_GRADE_CAP.get(worst_hard_status)
    if cap is None:
        return grade
    return cap if _GRADE_ORDER.index(grade) < _GRADE_ORDER.index(cap) else grade


def _worst(statuses: Iterable[str]) -> str:
    ranked = [s for s in statuses]
    return max(ranked, key=_STATUS_RANK.get) if ranked else "pass"


# --------------------------------------------------------------------------- #
# Type / value helpers (operate on non-null values only)
# --------------------------------------------------------------------------- #
def _nonnull(s: pd.Series) -> pd.Series:
    return s[s.notna()]


def _type_violations(s: pd.Series, expected: str) -> int:
    nn = _nonnull(s)
    if len(nn) == 0:
        return 0
    exp = expected.lower()
    if exp in ("numeric", "float", "integer", "int", "number"):
        coerced = pd.to_numeric(nn, errors="coerce")
        bad = int(coerced.isna().sum())
        if exp in ("integer", "int"):
            ok = coerced.dropna()
            bad += int((ok != ok.round()).sum())
        return bad
    if exp in ("datetime", "date", "timestamp"):
        coerced = pd.to_datetime(nn, errors="coerce")
        return int(coerced.isna().sum())
    if exp in ("boolean", "bool"):
        return int((~nn.isin([True, False, 0, 1])).sum())
    if exp in ("string", "str", "object", "text"):
        return 0  # any non-null value has a string form
    return 0


def _range_violations(s: pd.Series, lo: float | None, hi: float | None) -> tuple[int, int]:
    v = pd.to_numeric(_nonnull(s), errors="coerce").dropna()
    if len(v) == 0:
        return 0, 0
    mask = pd.Series(False, index=v.index)
    if lo is not None:
        mask = mask | (v < lo)
    if hi is not None:
        mask = mask | (v > hi)
    return int(mask.sum()), int(len(v))


def _regex_violations(s: pd.Series, pattern: str) -> tuple[int, int]:
    nn = _nonnull(s)
    if len(nn) == 0:
        return 0, 0
    rx = re.compile(pattern)
    as_str = nn.astype(str)
    bad = int((~as_str.map(lambda x: rx.fullmatch(x) is not None)).sum())
    return bad, int(len(nn))


def _domain_violations(s: pd.Series, allowed: Iterable) -> tuple[int, int]:
    nn = _nonnull(s)
    if len(nn) == 0:
        return 0, 0
    allowed_set = set(allowed)
    bad = int((~nn.isin(allowed_set)).sum())
    return bad, int(len(nn))


def _iqr_outliers(s: pd.Series) -> tuple[int, int]:
    """Extreme outliers by the 3xIQR rule. HEURISTIC -- not a correctness claim."""
    v = pd.to_numeric(_nonnull(s), errors="coerce").dropna()
    if len(v) < 4:
        return 0, int(len(v))
    q1, q3 = v.quantile(0.25), v.quantile(0.75)
    iqr = q3 - q1
    if iqr == 0:
        return 0, int(len(v))
    lo, hi = q1 - 3 * iqr, q3 + 3 * iqr
    out = int(((v < lo) | (v > hi)).sum())
    return out, int(len(v))


# --------------------------------------------------------------------------- #
# Default config inference (used when no config is supplied)
# --------------------------------------------------------------------------- #
def _infer_type(dtype: object) -> str:
    if pd.api.types.is_bool_dtype(dtype):
        return "boolean"
    if pd.api.types.is_integer_dtype(dtype):
        return "integer"
    if pd.api.types.is_float_dtype(dtype) or pd.api.types.is_numeric_dtype(dtype):
        return "numeric"
    if pd.api.types.is_datetime64_any_dtype(dtype):
        return "datetime"
    return "string"


def _default_config(df: pd.DataFrame) -> QualityConfig:
    """Sensible dtype-inferred defaults: every column required & type-checked,
    all columns form the dedup key, numeric columns get the outlier heuristic.
    Consistency defaults to 'numeric columns are finite'. Flagged as defaults."""
    cols = {c: ColumnRule(expected_type=_infer_type(df[c].dtype), required=True) for c in df.columns}
    numeric_cols = tuple(c for c in df.columns if pd.api.types.is_numeric_dtype(df[c].dtype))
    rules = tuple(
        Rule(name=f"{c} is finite", column=c, op="finite", kind="consistency (dtype-inferred default)")
        for c in numeric_cols
    )
    return QualityConfig(
        columns=cols,
        rules=rules,
        key_columns=None,
        dedup_columns=tuple(df.columns),
        plausibility_columns=numeric_cols,
    )


# --------------------------------------------------------------------------- #
# Per-dimension assessment
# --------------------------------------------------------------------------- #
def _mk(dim: str, check: str, count: int, total: int, column: str | None,
        expl: str, cfg: QualityConfig, *, score: float | None = None) -> Finding:
    sc = _score(count, total) if score is None else round(score, 1)
    return Finding(dim, check, _status(sc, cfg.warn_below, cfg.fail_below), sc, count, total, column, expl)


def _completeness(df: pd.DataFrame, cfg: QualityConfig, used_defaults: bool) -> list[Finding]:
    dim = "Completeness"
    n = len(df)
    findings: list[Finding] = []
    required = [c for c, r in cfg.columns.items() if r.required and c in df.columns]
    nullable = [c for c, r in cfg.columns.items() if (not r.required) and c in df.columns]

    total_cells = 0
    total_null = 0
    for col in required:
        nulls = int(df[col].isna().sum())
        total_cells += n
        total_null += nulls
        findings.append(_mk(
            dim, f"non-null: {col}", nulls, n, col,
            f"{nulls:,} of {n:,} values missing in required column '{col}'.", cfg,
        ))
    # overall completeness across in-scope (required) columns
    if required:
        findings.append(_mk(
            dim, "overall (required columns)", total_null, total_cells, None,
            f"{total_null:,} of {total_cells:,} cells missing across required columns.", cfg,
        ))
    # nullable columns reported for transparency but NOT scored into the dimension
    for col in nullable:
        nulls = int(df[col].isna().sum())
        if nulls:
            findings.append(Finding(
                dim, f"nullable (informational): {col}", "pass", 100.0, nulls, n, col,
                f"{nulls:,} of {n:,} missing in '{col}' -- allowed by config (nullable), not penalised.",
            ))
    return findings


def _validity(df: pd.DataFrame, cfg: QualityConfig, used_defaults: bool) -> list[Finding]:
    dim = "Validity"
    findings: list[Finding] = []
    default_note = " (dtype-inferred default)" if used_defaults else ""
    for col, rule in cfg.columns.items():
        if col not in df.columns:
            findings.append(Finding(
                dim, f"column present: {col}", "fail", 0.0, 1, 1, col,
                f"declared column '{col}' is not in the frame.",
            ))
            continue
        s = df[col]
        nn = int(s.notna().sum())
        if rule.expected_type:
            bad = _type_violations(s, rule.expected_type)
            findings.append(_mk(
                dim, f"type={rule.expected_type}: {col}", bad, nn, col,
                f"{bad:,} of {nn:,} non-null values do not conform to type "
                f"'{rule.expected_type}'{default_note}.", cfg,
            ))
        if rule.min_value is not None or rule.max_value is not None:
            bad, tot = _range_violations(s, rule.min_value, rule.max_value)
            bounds = f"[{rule.min_value}, {rule.max_value}]"
            findings.append(_mk(
                dim, f"range {bounds}: {col}", bad, tot, col,
                f"{bad:,} of {tot:,} numeric values fall outside {bounds}.", cfg,
            ))
        if rule.allowed is not None:
            bad, tot = _domain_violations(s, rule.allowed)
            findings.append(_mk(
                dim, f"domain: {col}", bad, tot, col,
                f"{bad:,} of {tot:,} values are outside the allowed set "
                f"({len(set(rule.allowed))} permitted).", cfg,
            ))
        if rule.regex is not None:
            bad, tot = _regex_violations(s, rule.regex)
            findings.append(_mk(
                dim, f"regex: {col}", bad, tot, col,
                f"{bad:,} of {tot:,} values do not match /{rule.regex}/.", cfg,
            ))
    return findings


def _uniqueness(df: pd.DataFrame, cfg: QualityConfig, used_defaults: bool) -> list[Finding]:
    dim = "Uniqueness"
    n = len(df)
    findings: list[Finding] = []
    # dedup_columns=None -> use all columns; an explicit () -> skip the row-dup check.
    dedup = list(df.columns) if cfg.dedup_columns is None else list(cfg.dedup_columns)
    dedup = [c for c in dedup if c in df.columns]
    if dedup and n:
        dup_rows = int(df.duplicated(subset=dedup).sum())
        scope = "all columns" if len(dedup) == len(df.columns) else f"{len(dedup)} key column(s)"
        findings.append(_mk(
            dim, "duplicate rows", dup_rows, n, None,
            f"{dup_rows:,} of {n:,} rows are exact duplicates (subset: {scope}).", cfg,
        ))
    if cfg.key_columns:
        keys = [c for c in cfg.key_columns if c in df.columns]
        if keys and n:
            dup_keys = int(df.duplicated(subset=keys).sum())
            findings.append(_mk(
                dim, f"unique key: {'+'.join(keys)}", dup_keys, n, "+".join(keys),
                f"{dup_keys:,} of {n:,} rows repeat a key value (key: {'+'.join(keys)}).", cfg,
            ))
    return findings


def _eval_consistency(df: pd.DataFrame, rule: Rule) -> tuple[int, int, str]:
    """Return (violations, evaluated, detail). Evaluated excludes null operands."""
    if rule.column not in df.columns:
        return len(df), len(df), f"declared column '{rule.column}' not present"
    left = df[rule.column]
    op = rule.op
    if op == "notnull":
        return int(left.isna().sum()), len(df), "must be non-null"
    if op == "finite":
        num = pd.to_numeric(left, errors="coerce")
        evaluated = num.notna()
        bad = int((~np.isfinite(num.to_numpy(dtype="float64", na_value=np.nan)) & evaluated.to_numpy()).sum())
        return bad, int(evaluated.sum()), "must be finite (no inf)"
    if op == "regex":
        bad, tot = _regex_violations(left, str(rule.value))
        return bad, tot, f"must match /{rule.value}/"
    if op == "in":
        bad, tot = _domain_violations(left, rule.value)  # type: ignore[arg-type]
        return bad, tot, "must be in the allowed set"
    if op == "between":
        lo, hi = rule.value  # type: ignore[misc]
        if pd.api.types.is_datetime64_any_dtype(left):
            lts, hts = pd.Timestamp(lo), pd.Timestamp(hi)
            evaluated = left.notna()
            bad = int((((left < lts) | (left > hts)) & evaluated).sum())
            return bad, int(evaluated.sum()), f"must be within [{lo}, {hi}]"
        bad, tot = _range_violations(left, lo, hi)
        return bad, tot, f"must be within [{lo}, {hi}]"

    # arithmetic comparisons: column vs scalar or column vs other column
    lnum = pd.to_numeric(left, errors="coerce")
    if rule.other_column is not None:
        if rule.other_column not in df.columns:
            return len(df), len(df), f"declared column '{rule.other_column}' not present"
        rnum = pd.to_numeric(df[rule.other_column], errors="coerce")
        detail = f"must be {op} {rule.other_column}"
    else:
        rnum = pd.Series(pd.to_numeric([rule.value], errors="coerce")[0], index=left.index)
        detail = f"must be {op} {rule.value}"
    evaluated = lnum.notna() & rnum.notna()
    ops = {
        ">": lnum > rnum, ">=": lnum >= rnum, "<": lnum < rnum,
        "<=": lnum <= rnum, "==": lnum == rnum, "!=": lnum != rnum,
    }
    if op not in ops:
        raise ValueError(f"unknown consistency op: {op!r}")
    ok = ops[op]
    bad = int((~ok & evaluated).sum())
    return bad, int(evaluated.sum()), detail


def _consistency(df: pd.DataFrame, cfg: QualityConfig, used_defaults: bool) -> list[Finding]:
    dim = "Consistency"
    findings: list[Finding] = []
    tag = " (dtype-inferred default)" if used_defaults else ""
    for rule in cfg.rules:
        bad, tot, detail = _eval_consistency(df, rule)
        findings.append(_mk(
            dim, rule.name, bad, tot, rule.column,
            f"rule '{rule.name}': {rule.column} {detail}{tag} -- {bad:,} of {tot:,} rows violate it.", cfg,
        ))
    return findings


def _plausibility(df: pd.DataFrame, cfg: QualityConfig, used_defaults: bool) -> list[Finding]:
    dim = "Plausibility"
    findings: list[Finding] = []
    cols = cfg.plausibility_columns
    if cols is None:
        cols = tuple(c for c in df.columns if pd.api.types.is_numeric_dtype(df[c].dtype))
    for col in cols:
        if col not in df.columns:
            continue
        out, tot = _iqr_outliers(df[col])
        f = _mk(
            dim, f"outliers (3xIQR, heuristic): {col}", out, tot, col,
            f"HEURISTIC: {out:,} of {tot:,} values are extreme outliers by the 3xIQR rule "
            f"in '{col}'. This flags spread, not wrongness.", cfg,
        )
        # A heuristic must never hard-fail: clamp its status to at most 'warn'.
        if f.status == "fail":
            f = replace(f, status="warn")
        findings.append(f)
    return findings


_DIMENSION_FUNCS = {
    "Completeness": _completeness,
    "Validity": _validity,
    "Uniqueness": _uniqueness,
    "Consistency": _consistency,
    "Plausibility": _plausibility,
}


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def assess(df: pd.DataFrame, config: QualityConfig | None = None, *, label: str = "") -> ReportCard:
    """Assess a DataFrame and return a structured :class:`ReportCard`.

    With no ``config``, dtype-inferred defaults are used and the card is marked
    ``used_defaults=True``. Deterministic and side-effect-free.
    """
    used_defaults = config is None
    cfg = _default_config(df) if config is None else config
    n_rows, n_cols = int(len(df)), int(df.shape[1])

    # Empty frame: an honest card cannot pass "no data".
    if n_rows == 0:
        empty = Finding(
            "Completeness", "non-empty frame", "fail", 0.0, 0, 0, None,
            "the frame has 0 rows -- there is nothing to assess; an empty dataset cannot pass.",
        )
        dims = {
            d: DimensionResult(d, 0.0 if d == "Completeness" else 100.0, "fail" if d == "Completeness" else "pass",
                               (empty,) if d == "Completeness" else (), d == "Completeness")
            for d in DIMENSIONS
        }
        return ReportCard(0.0, "F", dims, (empty,), {"Completeness": 1.0}, 0, n_cols,
                          used_defaults, label)

    dimensions: dict[str, DimensionResult] = {}
    flat: list[Finding] = []
    for dim in DIMENSIONS:
        checks = _DIMENSION_FUNCS[dim](df, cfg, used_defaults)
        # scored checks = those that count toward the dimension score
        scored = [f for f in checks if not f.check.startswith("nullable (informational)")]
        if scored:
            dscore = round(float(np.mean([f.score for f in scored])), 1)
            # dimension status is the WORST check status -- a mean must not hide a break
            dstatus = _worst(f.status for f in scored)
            dimensions[dim] = DimensionResult(dim, dscore, dstatus, tuple(checks), True)
        else:
            dimensions[dim] = DimensionResult(dim, 100.0, "pass", tuple(checks), False)
        flat.extend(checks)

    # overall = weighted mean over dimensions that ran, weights renormalised
    ran = {d: dimensions[d] for d in DIMENSIONS if dimensions[d].ran}
    raw_w = {d: float(cfg.weights.get(d, 0.0)) for d in ran}
    wsum = sum(raw_w.values())
    if wsum <= 0:
        norm_w = {d: 1.0 / len(ran) for d in ran} if ran else {}
    else:
        norm_w = {d: w / wsum for d, w in raw_w.items()}
    overall = round(sum(dimensions[d].score * norm_w[d] for d in ran), 1) if ran else 0.0

    # Grade from the score, then capped by the worst HARD-dimension status.
    worst_hard = _worst(dimensions[d].status for d in HARD_DIMENSIONS if dimensions[d].ran)
    grade = _cap_grade(_grade(overall), worst_hard)

    return ReportCard(
        overall_score=overall,
        grade=grade,
        dimensions=dimensions,
        findings=tuple(flat),
        weights=norm_w,
        n_rows=n_rows,
        n_cols=n_cols,
        used_defaults=used_defaults,
        label=label,
    )


def findings_frame(card: ReportCard) -> pd.DataFrame:
    """Flat findings as a DataFrame (used by the Excel export; screen == export)."""
    rows = [
        {
            "Dimension": f.dimension,
            "Check": f.check,
            "Status": f.status,
            "Score": f.score,
            "Count": f.count,
            "Evaluated": f.total,
            "Column": f.column if f.column is not None else "",
            "Explanation": f.explanation,
        }
        for f in card.findings
    ]
    return pd.DataFrame(rows, columns=["Dimension", "Check", "Status", "Score", "Count",
                                       "Evaluated", "Column", "Explanation"])


def scores_frame(card: ReportCard) -> pd.DataFrame:
    """Per-dimension scores + the overall row (with the renormalised weight)."""
    rows = []
    for d in DIMENSIONS:
        dr = card.dimensions[d]
        rows.append({
            "Dimension": d,
            "Score": dr.score if dr.ran else np.nan,
            "Status": dr.status if dr.ran else "n/a",
            "Weight": round(card.weights.get(d, 0.0), 3),
            "Ran": dr.ran,
        })
    rows.append({"Dimension": "OVERALL", "Score": card.overall_score, "Status": card.worst_status,
                 "Weight": 1.0, "Ran": True})
    return pd.DataFrame(rows, columns=["Dimension", "Score", "Status", "Weight", "Ran"])


def render_report_card(card: ReportCard, *, max_findings: int = 40) -> str:
    """Render a plain-text/markdown report card. Deterministic; ASCII markers."""
    lines: list[str] = []
    title = f"DATA-QUALITY REPORT CARD{f'  --  {card.label}' if card.label else ''}"
    lines.append("=" * 72)
    lines.append(title)
    lines.append("=" * 72)
    lines.append(f"CAVEAT: {NOT_A_CERTIFICATION}")
    lines.append("")
    worst_hard = _worst(card.dimensions[d].status for d in HARD_DIMENSIONS if card.dimensions[d].ran)
    uncapped = _grade(card.overall_score)
    cap_note = ""
    if card.grade != uncapped:
        cap_note = (f"   (capped from {uncapped} by a {worst_hard.upper()} in a hard dimension; "
                    f"score alone would read {uncapped})")
    lines.append(f"Overall grade: {card.grade}   score {card.overall_score:.1f}/100   "
                 f"(worst hard check: {_STATUS_MARKER[worst_hard]}){cap_note}")
    lines.append(f"Rows: {card.n_rows:,}   Columns: {card.n_cols}")
    if card.used_defaults:
        lines.append("Config: NONE supplied -> dtype-inferred DEFAULT checks were used "
                     "(every column treated as required & type-checked).")
    else:
        lines.append("Config: user-supplied (only declared checks were run).")
    lines.append("")
    lines.append("Dimension scores (weights renormalised over dimensions that ran):")
    for d in DIMENSIONS:
        dr = card.dimensions[d]
        if dr.ran:
            w = card.weights.get(d, 0.0)
            lines.append(f"  {_STATUS_MARKER[dr.status]:<7} {d:<13} {dr.score:6.1f}/100   weight {w:5.2f}")
        else:
            lines.append(f"  {'[--]':<7} {d:<13}    n/a       (no applicable checks -> excluded)")
    lines.append("")

    # findings worth showing: warn/fail first, then the rest; stable order.
    problems = [f for f in card.findings if f.status != "pass"]
    lines.append(f"Findings that need attention ({len(problems)}):")
    if not problems:
        lines.append("  (none -- every declared check passed)")
    else:
        shown = sorted(problems, key=lambda f: (-_STATUS_RANK[f.status], f.dimension, f.check))[:max_findings]
        for f in shown:
            col = f" [{f.column}]" if f.column else ""
            lines.append(f"  {_STATUS_MARKER[f.status]} {f.dimension}/{f.check}{col}: "
                         f"score {f.score:.1f} -- {f.explanation}")
        if len(problems) > max_findings:
            lines.append(f"  ... and {len(problems) - max_findings} more (see the full findings table).")
    lines.append("")
    lines.append(f"Notes: Plausibility is an outlier HEURISTIC (3xIQR), not a correctness claim. "
                 f"Scores are 100*(1-violations/evaluated); pass>={WARN_BELOW:.1f}, "
                 f"warn>={FAIL_BELOW:.1f}, else fail. Grade capped by worst hard-dimension status "
                 f"(fail->C, warn->B); Plausibility never caps.")
    lines.append("=" * 72)
    return "\n".join(lines)


def compare(before: ReportCard, after: ReportCard) -> str:
    """A short honest before/after summary line (the 'quality lift')."""
    lift = round(after.overall_score - before.overall_score, 1)
    arrow = "+" if lift >= 0 else ""
    return (f"Quality lift: {before.grade} ({before.overall_score:.1f}) -> "
            f"{after.grade} ({after.overall_score:.1f})   [{arrow}{lift:.1f} points]")


# Re-export for callers building custom configs.
__all__ = [
    "ColumnRule", "Rule", "QualityConfig", "Finding", "DimensionResult", "ReportCard",
    "assess", "render_report_card", "findings_frame", "scores_frame", "compare",
    "DIMENSION_WEIGHTS", "WARN_BELOW", "FAIL_BELOW", "GRADE_CUTS", "NOT_A_CERTIFICATION",
]
