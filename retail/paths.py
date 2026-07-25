"""Project paths, resolved relative to the repository root."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_INTERIM = ROOT / "data" / "interim"
FIGURES = ROOT / "figures"
DELIVERABLES = ROOT / "deliverables"

RAW_XLSX = DATA_RAW / "online_retail_II.xlsx"
INTERIM_PARQUET = DATA_INTERIM / "combined_raw.parquet"
INTERIM_CSV = DATA_INTERIM / "combined_raw.csv.gz"
