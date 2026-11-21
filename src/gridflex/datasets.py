"""Download public datasets into ``data/raw/`` (nothing is committed to the repo).

Usage::

    python -m gridflex.datasets            # downloads RTS-GMLC into data/raw/rts_gmlc
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


if __name__ == "__main__":
    print(download_rts_gmlc())
