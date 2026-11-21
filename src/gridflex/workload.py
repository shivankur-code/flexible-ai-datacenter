"""Interactive LLM workload: the Azure LLM inference trace 2024, turned into hourly profiles.

Source: Azure Public Dataset, "Azure LLM inference trace 2024" (one week each of
code-assistant and conversation traffic, May 2024). Released with Stojkovic et
al., *DynamoLLM*, HPCA 2025, under CC-BY.
``data/processed/`` holds per-minute aggregates and a token-size sample built
by :func:`gridflex.datasets.prepare_azure_llm`.

The week is mapped onto any year by (local day-of-week, hour). For each slot
we keep the mean request rate and the *peak-minute factor* (max minute rate /
hourly mean). Replicas are provisioned for the peak minute, energy follows the
mean.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
PROCESSED = REPO_ROOT / "data" / "processed"


def load_tokens(path: Path | str | None = None) -> pd.DataFrame:
    return pd.read_csv(path or PROCESSED / "azure_llm_2024_token_sample.csv")


def weekly_profile(utc_offset_h: int = -7, path: Path | str | None = None) -> pd.DataFrame:
    """Mean req/s and peak-minute factor per (service, local day-of-week, hour)."""
    m = pd.read_csv(path or PROCESSED / "azure_llm_2024_per_minute.csv")
    m["t"] = pd.to_datetime(m.minute) + pd.Timedelta(hours=utc_offset_h)
    out = []
    for svc, g in m.groupby("service"):
        g = g.set_index("t").requests
        hourly = pd.DataFrame(dict(mean=g.resample("h").mean() / 60.0, peak=g.resample("h").max() / 60.0)).dropna()
        hourly["dow"], hourly["hour"] = hourly.index.dayofweek, hourly.index.hour
        prof = hourly.groupby(["dow", "hour"]).agg(rate=("mean", "mean"), peak=("peak", "mean")).reset_index()
        prof["peak_factor"] = prof.peak / prof.rate
        prof["service"] = svc
        out.append(prof[["service", "dow", "hour", "rate", "peak_factor"]])
    return pd.concat(out, ignore_index=True)


def year_profile(timestamps: pd.DatetimeIndex, profile: pd.DataFrame) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Map the weekly profile onto ``timestamps``: {service: (req/s, peak factor)} per hour."""
    key = pd.DataFrame(dict(dow=timestamps.dayofweek, hour=timestamps.hour))
    res = {}
    for svc, p in profile.groupby("service"):
        j = key.merge(p, on=["dow", "hour"], how="left")
        res[svc] = (j.rate.to_numpy(), j.peak_factor.to_numpy())
    return res
