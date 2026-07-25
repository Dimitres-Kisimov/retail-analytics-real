"""Regenerate tests/fixtures/sample.csv from the real data, deterministically.

The fixture is ~2,000 REAL rows drawn with a fixed seed from the cached
combined frame, stratified so every mess the cleaning pipeline handles is
present: cancellations, non-product StockCodes, zero/negative prices, missing
CustomerIDs, and real exact-duplicate groups (both copies included).

Run only when the fixture needs refreshing (requires the raw data / cache):

    python scripts/make_fixture.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from retail import clean, ingest  # noqa: E402

SEED = 42
OUT = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "sample.csv"
COLUMNS = ["Invoice", "StockCode", "Description", "Quantity", "InvoiceDate", "Price", "CustomerID", "Country"]


def main() -> None:
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    df = ingest.load_raw()[COLUMNS]

    random_rows = df.sample(n=1600, random_state=SEED)
    cancellations = df.loc[df["Invoice"].str.startswith("C")].sample(n=150, random_state=SEED)
    non_product = df.loc[clean.is_non_product(df["StockCode"])].sample(n=100, random_state=SEED)
    zero_price = df.loc[df["Price"] <= 0].sample(n=60, random_state=SEED)

    # real exact-duplicate groups, keeping every copy so dedup has work to do
    dup_mask = df.duplicated(subset=clean.DEDUP_COLUMNS, keep=False)
    dup_keys = (
        df.loc[dup_mask, clean.DEDUP_COLUMNS]
        .drop_duplicates()
        .sample(n=20, random_state=SEED)
    )
    dup_rows = df.loc[dup_mask].merge(dup_keys, on=clean.DEDUP_COLUMNS, how="inner")

    import pandas as pd

    fixture = pd.concat([random_rows, cancellations, non_product, zero_price, dup_rows])
    fixture = fixture.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fixture.to_csv(OUT, index=False, encoding="utf-8")
    print(f"[OK] wrote {OUT} ({len(fixture):,} rows)")
    print(f"     cancellations: {int(fixture['Invoice'].str.startswith('C').sum())}")
    print(f"     non-product:   {int(clean.is_non_product(fixture['StockCode']).sum())}")
    print(f"     price <= 0:    {int((fixture['Price'] <= 0).sum())}")
    print(f"     missing ID:    {int(fixture['CustomerID'].isna().sum())}")
    print(f"     exact dupes:   {int(fixture.duplicated(subset=clean.DEDUP_COLUMNS).sum())}")


if __name__ == "__main__":
    main()
