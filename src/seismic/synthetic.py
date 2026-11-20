"""Synthetic earthquake catalog with *known* sequences, used only by the tests and for smoke runs.

Background events follow Gutenberg-Richter magnitudes at random times in a few regions.
Planted sequences follow Omori decay in time and Gaussian scatter in space, so the correct
clustering is known ('true_cluster'). Edge cases are deliberately included: one sequence
straddles the antimeridian (lon 180) and one sits near the North Pole.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

# (lat, lon, magnitude, days after catalog start)
DEFAULT_SEQUENCES: List[Tuple[float, float, float, float]] = [
    (35.5, -117.5, 7.0, 400.0),
    (-20.2, -69.8, 6.4, 900.0),
    (-18.0, 179.8, 7.3, 1300.0),   # crosses the antimeridian
    (84.0, 20.0, 6.6, 1700.0),     # near the pole
]
REGIONS = [(35.0, -117.0), (-20.0, -70.0), (38.0, 142.0), (60.0, -150.0), (-18.0, 179.9), (84.0, 20.0)]


def _km_offset(lat, lon, dx_km, dy_km):
    dlat = dy_km / 111.195
    dlon = dx_km / (111.195 * np.cos(np.radians(np.clip(lat, -89.0, 89.0))))
    new_lat = np.clip(lat + dlat, -89.9, 89.9)
    new_lon = (lon + dlon + 180.0) % 360.0 - 180.0
    return new_lat, new_lon


def make_catalog(seed: int = 0, years: float = 6.0, n_background: int = 1200, mmin: float = 4.5,
                 b: float = 1.0, sequences: Optional[List[Tuple[float, float, float, float]]] = None,
                 start: str = "2015-01-01") -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    seqs = DEFAULT_SEQUENCES if sequences is None else sequences
    t0 = pd.Timestamp(start, tz="UTC")
    span_days = years * 365.25
    rows = []

    def gr_mag(n, upper=8.5):
        m = mmin + rng.exponential(1.0 / (b * np.log(10)), size=n)
        return np.minimum(m, upper)

    # background
    reg = rng.integers(0, len(REGIONS), n_background)
    lat0 = np.array([REGIONS[i][0] for i in reg])
    lon0 = np.array([REGIONS[i][1] for i in reg])
    lat, lon = _km_offset(lat0, lon0, rng.normal(0, 150, n_background), rng.normal(0, 150, n_background))
    tday = rng.uniform(0, span_days, n_background)
    mag = gr_mag(n_background, upper=6.4)
    for i in range(n_background):
        rows.append((tday[i], lat[i], lon[i], mag[i], -1))

    # planted sequences
    for sid, (la, lo, M, day0) in enumerate(seqs):
        rows.append((day0, la, lo, M, sid))
        n_after = int(10 ** (0.9 * (M - mmin)))
        c, p, tmax = 0.01, 1.1, 200.0
        u = rng.uniform(size=n_after)
        a, bb = c ** (1 - p), (tmax + c) ** (1 - p)          # inverse-CDF of truncated Omori
        delay = (a + u * (bb - a)) ** (1 / (1 - p)) - c
        sigma = 0.5 * 10 ** (0.5 * M - 1.85)
        a_lat, a_lon = _km_offset(la, lo, rng.normal(0, sigma, n_after), rng.normal(0, sigma, n_after))
        a_mag = np.minimum(gr_mag(n_after), M - 0.3)
        for i in range(n_after):
            rows.append((day0 + delay[i], a_lat[i], a_lon[i], a_mag[i], sid))

    df = pd.DataFrame(rows, columns=["tday", "latitude", "longitude", "magnitude", "true_cluster"])
    df = df.sort_values("tday").reset_index(drop=True)
    df["event_time"] = [(t0 + pd.Timedelta(days=float(d))).round("us") for d in df.tday]  # Spark keeps microseconds
    df["event_id"] = [f"ev{i:06d}" for i in range(len(df))]
    df["depth_km"] = rng.uniform(5, 60, len(df))
    df["place"] = "synthetic"
    return df[["event_id", "event_time", "latitude", "longitude", "depth_km", "magnitude", "place",
               "true_cluster"]]
