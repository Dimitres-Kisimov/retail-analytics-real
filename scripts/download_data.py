"""Download the UCI Online Retail II dataset (id 502) into data/raw/.

The raw xlsx (~45 MB) is deliberately NOT committed to this repository.
Run this once before the pipeline:

    python scripts/download_data.py

Equivalent curl one-liner:

    curl -L -o online_retail_ii.zip \
        "https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip" \
        && unzip online_retail_ii.zip -d data/raw/

Dataset: Chen, D. (2019). Online Retail II [Dataset]. UCI Machine Learning
Repository. https://doi.org/10.24432/C5CG6D — licensed CC BY 4.0.
"""

from __future__ import annotations

import sys
import time
import urllib.request
import zipfile
from pathlib import Path

URL = "https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip"
ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
ZIP_PATH = ROOT / "data" / "online_retail_ii.zip"
ATTEMPTS = 3


def download(url: str = URL, dest: Path = ZIP_PATH, attempts: int = ATTEMPTS) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    last_err: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            print(f"[{attempt}/{attempts}] downloading {url} ...")
            urllib.request.urlretrieve(url, dest)  # noqa: S310 - fixed https URL
            size = dest.stat().st_size
            if size < 1_000_000:
                raise OSError(f"downloaded file suspiciously small ({size} bytes)")
            print(f"[OK] {dest} ({size / 1e6:.1f} MB)")
            return dest
        except Exception as err:  # noqa: BLE001 - report and retry
            last_err = err
            print(f"[WARN] attempt {attempt} failed: {err}")
            time.sleep(5)
    raise SystemExit(f"[FAIL] download failed after {attempts} attempts: {last_err}")


def extract(zip_path: Path = ZIP_PATH, out_dir: Path = RAW_DIR) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(out_dir)
    print(f"[OK] extracted to {out_dir}")


def main() -> None:
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    xlsx = RAW_DIR / "online_retail_II.xlsx"
    if xlsx.exists():
        print(f"[OK] already present: {xlsx}")
        return
    download()
    extract()


if __name__ == "__main__":
    main()
