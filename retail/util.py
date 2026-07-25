"""Console / IO helpers.

The dataset contains unicode product descriptions; the Windows console often
does not. Every entry point calls :func:`configure_stdout` before printing.
Status markers stay ASCII-only ([OK] / [WARN] / [FAIL]).
"""

from __future__ import annotations

import sys


def configure_stdout() -> None:
    """Make stdout/stderr UTF-8-safe (replace on encode errors) on any console."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="replace")


def fmt_int(n: int | float) -> str:
    """1234567 -> '1,234,567' (ASCII thousands separators)."""
    return f"{int(n):,}"


def fmt_pct(x: float, digits: int = 2) -> str:
    return f"{100.0 * x:.{digits}f}%"
