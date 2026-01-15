# eia_international_supply.py
from __future__ import annotations

import os
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import requests


# =========================
# Config
# =========================
EIA_BASE = "https://api.eia.gov/v2/international/data/"

@dataclass(frozen=True)
class Config:
    api_key: str
    # ---- Choose ONE series definition (set these from EIA API Browser) ----
    activity_id: int         # e.g. Production
    product_id: int          # e.g. Crude oil incl lease condensate
    unit: str                # e.g. MK (often "thousand barrels per day" style code)
    frequency: str = "monthly"

    # Time window: last 20 years by default
    start: str = "2006-01"   # inclusive (YYYY-MM)
    end: Optional[str] = None  # inclusive-ish; EIA uses period sorting. Leave None to get latest.

    # Paging
    length: int = 5000       # max rows per page (API caps overall; we page via offset)
    sleep_s: float = 0.15    # be polite


@dataclass(frozen=True)
class Paths:
    base: Path
    curated: Path

    @staticmethod
    def from_base(base_dir: str | Path) -> "Paths":
        base = Path(base_dir).resolve()
        curated = base / "data" / "curated"
        curated.mkdir(parents=True, exist_ok=True)
        return Paths(base=base, curated=curated)


# =========================
# Low-level EIA client
# =========================
def _eia_get(params: Dict[str, Any], api_key: str, timeout_s: int = 60) -> Dict[str, Any]:
    """
    EIA v2: api_key must be in URL (not headers). :contentReference[oaicite:5]{index=5}
    """
    url = f"{EIA_BASE}?api_key={api_key}"
    r = requests.get(url, params=params, timeout=timeout_s)
    r.raise_for_status()
    return r.json()


def fetch_all_pages(cfg: Config) -> pd.DataFrame:
    """
    Fetches ALL rows for the given facets by paging with offset. :contentReference[oaicite:6]{index=6}
    """
    rows: List[Dict[str, Any]] = []
    offset = 0

    while True:
        params: Dict[str, Any] = {
            "frequency": cfg.frequency,
            "data[0]": "value",

            # Facets (filters):
            "facets[activityId][]": cfg.activity_id,
            "facets[productId][]": cfg.product_id,
            "facets[unit][]": cfg.unit,

            # Sorting: newest first
            "sort[0][column]": "period",
            "sort[0][direction]": "desc",

            # Paging
            "offset": offset,
            "length": cfg.length,
        }

        # Optional start/end constraints (recommended to keep payload smaller)
        # EIA typically supports 'start'/'end' on period in many routes.
        # If your route rejects these, remove them.
        if cfg.start:
            params["start"] = cfg.start
        if cfg.end:
            params["end"] = cfg.end

        js = _eia_get(params=params, api_key=cfg.api_key)

        data = js.get("response", {}).get("data", [])
        if not data:
            break

        rows.extend(data)

        # If fewer than page size, we’re done
        if len(data) < cfg.length:
            break

        offset += cfg.length
        time.sleep(cfg.sleep_s)

    if not rows:
        raise RuntimeError(
            "No data returned. Most common causes:\n"
            "- Wrong activityId/productId/unit\n"
            "- start/end not accepted for this route\n"
            "- API key missing/invalid\n"
        )

    df = pd.DataFrame(rows)
    return df


# =========================
# Transform
# =========================
def normalize_monthly(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize column names for fact_supply_country_month.
    EIA typically returns keys like: period, value, activityId, productId, countryRegionId, unit
    (exact set can vary).
    """
    df = df_raw.copy()

    # Ensure expected columns exist
    expected = ["period", "value", "activityId", "productId", "unit", "countryRegionId"]
    missing = [c for c in expected if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns from EIA response: {missing}. Got: {list(df.columns)}")

    # Keep only what we need (+ any helpful descriptors if present)
    keep = expected + [c for c in ["countryRegionName", "productName", "activityName"] if c in df.columns]
    df = df[keep].copy()

    # Types
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["value"])

    # Parse period to a monthly timestamp
    # Many EIA monthly series use "YYYY-MM"
    df["period"] = df["period"].astype(str)
    df["date"] = pd.to_datetime(df["period"] + "-01", errors="coerce")
    df = df.dropna(subset=["date"])

    df["ingested_at"] = pd.Timestamp(datetime.now())
    df["source_id"] = "EIA"

    return df


def build_annual_from_monthly(df_m: pd.DataFrame) -> pd.DataFrame:
    df = df_m.copy()
    df["year"] = df["date"].dt.year

    gcols = ["year", "countryRegionId", "activityId", "productId", "unit"]
    out = (
        df.groupby(gcols, as_index=False)["value"]
          .mean()
          .rename(columns={"value": "value_avg_daily"})
    )
    out["method"] = "mean_of_months"
    out["ingested_at"] = pd.Timestamp(datetime.now())
    out["source_id"] = "EIA"
    return out


# =========================
# Persist
# =========================
def write_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


# =========================
# Main
# =========================
def run(base_dir: str | Path) -> None:
    api_key = os.environ.get("EIA_API_KEY", "").strip()
    if not api_key:
        raise SystemExit(
            "Missing EIA_API_KEY env var.\n"
            "Get a key here: https://www.eia.gov/opendata/register.php :contentReference[oaicite:7]{index=7}\n"
            "Then set it in your shell, e.g.:\n"
            "  export EIA_API_KEY='...'\n"
        )

    # >>> Set these 3 from the EIA API Browser for your chosen series <<<
    cfg = Config(
        api_key=api_key,
        activity_id=7,     # TODO: replace
        product_id=2,      # TODO: replace
        unit="MK",         # TODO: replace
        frequency="monthly",
        start="2006-01",
        end=None,
        length=5000
    )

    paths = Paths.from_base(base_dir)

    print("Downloading monthly data from EIA…")
    df_raw = fetch_all_pages(cfg)
    df_month = normalize_monthly(df_raw)

    # Save monthly
    p_month = paths.curated / "fact_supply_country_month.parquet"
    write_parquet(df_month, p_month)

    # Build + save annual
    df_year = build_annual_from_monthly(df_month)
    p_year = paths.curated / "fact_supply_country_year.parquet"
    write_parquet(df_year, p_year)

    print("✅ Done")
    print(f"- Monthly rows: {len(df_month):,} -> {p_month}")
    print(f"- Annual rows : {len(df_year):,} -> {p_year}")


if __name__ == "__main__":
    run(base_dir="oil_model")
