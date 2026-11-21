"""Download public datasets into ``data/raw/`` (nothing is committed to the repo).

Usage::

    python -m gridflex.datasets            # downloads RTS-GMLC into data/raw/rts_gmlc
    python -m gridflex.datasets --azure    # also rebuilds data/processed/ from the 1.8 GB Azure LLM trace
"""
from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

# Pinned commit of https://github.com/GridMod/RTS-GMLC so results are reproducible.
RTS_GMLC_COMMIT = "3ece0d3725c844056132393ee252b3083dd4eab4"
RTS_GMLC_BASE = f"https://raw.githubusercontent.com/GridMod/RTS-GMLC/{RTS_GMLC_COMMIT}/RTS_Data"

RTS_GMLC_FILES = [
    "SourceData/bus.csv",
    "SourceData/branch.csv",
    "SourceData/gen.csv",
    "timeseries_data_files/Load/DAY_AHEAD_regional_Load.csv",
    "timeseries_data_files/WIND/DAY_AHEAD_wind.csv",
    "timeseries_data_files/PV/DAY_AHEAD_pv.csv",
    "timeseries_data_files/RTPV/DAY_AHEAD_rtpv.csv",
    "timeseries_data_files/Hydro/DAY_AHEAD_hydro.csv",
]

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RTS_DIR = REPO_ROOT / "data" / "raw" / "rts_gmlc"

# Azure LLM inference trace 2024 (CC-BY; Stojkovic et al., DynamoLLM, HPCA 2025). ~1.8 GB raw.
AZURE_LLM_2024 = "https://github.com/Azure/AzurePublicDataset/releases/download/dataset-llm-2024/AzureLLMInferenceTrace_{svc}_1week.csv"
DEFAULT_AZURE_DIR = REPO_ROOT / "data" / "raw" / "azure_llm"
PROCESSED_DIR = REPO_ROOT / "data" / "processed"


def download_rts_gmlc(dest: Path | str = DEFAULT_RTS_DIR, force: bool = False) -> Path:
    """Fetch the RTS-GMLC files used by :func:`gridflex.grid.load_rts_gmlc`."""
    dest = Path(dest)
    for rel in RTS_GMLC_FILES:
        out = dest / Path(rel).name
        if out.exists() and not force:
            continue
        out.parent.mkdir(parents=True, exist_ok=True)
        url = f"{RTS_GMLC_BASE}/{rel}"
        print(f"downloading {url}", file=sys.stderr)
        with urllib.request.urlopen(url, timeout=120) as r:
            out.write_bytes(r.read())
    return dest


def prepare_azure_llm(raw: Path | str = DEFAULT_AZURE_DIR, out: Path | str = PROCESSED_DIR, seed: int = 0) -> Path:
    """Download the 2024 Azure LLM trace and reduce it to the committed files in ``data/processed/``.

    * ``azure_llm_2024_per_minute.csv``: requests, context and generated tokens per minute and service
    * ``azure_llm_2024_token_sample.csv``: 4,000 (ctx, gen) pairs sampled from each 2M-row chunk
    """
    import numpy as np
    import pandas as pd

    raw, out = Path(raw), Path(out)
    raw.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    mins, samples = [], []
    for svc in ("code", "conv"):
        f = raw / f"{svc}_1week.csv"
        if not f.exists():
            print(f"downloading {svc} trace", file=sys.stderr)
            urllib.request.urlretrieve(AZURE_LLM_2024.format(svc=svc), f)
        agg = None
        for ch in pd.read_csv(f, chunksize=2_000_000, dtype={"ContextTokens": "int32", "GeneratedTokens": "int32"}):
            ch["minute"] = ch.TIMESTAMP.str.slice(0, 16)
            g = ch.groupby("minute").agg(requests=("ContextTokens", "size"), ctx_tokens=("ContextTokens", "sum"),
                                         gen_tokens=("GeneratedTokens", "sum"))
            agg = g if agg is None else agg.add(g, fill_value=0)
            samples.append(ch.sample(n=min(len(ch), 4000), random_state=int(rng.integers(1 << 30)))[["ContextTokens", "GeneratedTokens"]].assign(service=svc))
        mins.append(agg.reset_index().assign(service=svc))
    pd.concat(mins).astype({"requests": int, "ctx_tokens": int, "gen_tokens": int}).to_csv(out / "azure_llm_2024_per_minute.csv", index=False)
    s = pd.concat(samples)
    s.columns = ["ctx_tokens", "gen_tokens", "service"]
    s.to_csv(out / "azure_llm_2024_token_sample.csv", index=False)
    return out


if __name__ == "__main__":
    print(download_rts_gmlc())
    if "--azure" in sys.argv:
        print(prepare_azure_llm())
