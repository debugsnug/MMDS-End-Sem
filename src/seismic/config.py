"""Central configuration. Every tunable in the project plan lives here.

Values can be overridden from a JSON file (config/default.json) and/or from the
command line with `--set key=value` (value parsed as JSON, falling back to str).
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

@dataclass
class Config:
    # ---------------- paths ----------------
    # Assumption A1: the Assignment 2 CSV (USGS ComCat export) is reused unchanged.
    catalog_csv: str = os.path.join(
        PROJECT_ROOT, "dataset", "usgs_earthquakes_historical.csv"
    )
    
    work_dir: str = os.path.join(
        PROJECT_ROOT, "outputs", "real_run"
    )

    # ---------------- Spark ----------------
    master: str = "local[*]"
    shuffle_partitions: int = 8
    driver_memory: str = "6g"
    use_arrow: bool = False            # faster toPandas on Java 8/11/17; off by default (breaks on Java 21)

    # ---------------- catalog ----------------
    mc: float = 4.5                    # completeness cutoff used for clustering and Part B
    mainshock_min_mag: float = 6.0     # Gardner-Knopoff mainshock candidates (as in Assignment 2)

    # ---------------- nearest-neighbour clustering (Zaliapin & Ben-Zion) ----------------
    nn_b: float = 1.0                  # Gutenberg-Richter b-value used in eta
    nn_df: float = 1.6                 # fractal dimension of epicentres
    nn_max_days: float = 365.0         # search-box truncation in time (see README, "truncation")
    nn_radius_scale: float = 3.0       # search radius = scale * Gardner-Knopoff radius(parent magnitude)
    nn_min_km: float = 0.1             # floor on distance (duplicate locations)
    nn_eta0_log10: float = -5.0        # default threshold when it cannot be estimated
    nn_estimate_eta0: bool = True      # estimate threshold with a 2-component MLlib GMM
    nn_salt_buckets: int = 1           # >1 enables salting of hot blocks
    eta0_sweep: List[float] = field(default_factory=lambda: [-7.0, -6.0, -5.5, -5.0, -4.5, -4.0, -3.0])

    # ---------------- zones (MLlib clustering on the unit sphere) ----------------
    zone_kinds: List[str] = field(default_factory=lambda: ["kmeans", "bisecting", "gmm"])
    zone_k_candidates: List[int] = field(default_factory=lambda: [10, 20, 40])
    silhouette_sample: float = 0.3

    # ---------------- DBSCAN single-machine reference (optional) ----------------
    dbscan_eps_km: float = 50.0
    dbscan_eps_days: float = 7.0
    dbscan_min_samples: int = 5

    # ---------------- Part B label / features / split ----------------
    trigger_min_mag: float = 4.5
    label_horizon_days: float = 7.0
    label_radius_km: float = 50.0
    activity_radius_km: float = 100.0
    activity_lookback_days: float = 30.0
    cell_deg: float = 10.0             # "location cell" size for set A
    train_end: str = "2021-01-01"      # train: [catalog start, train_end)
    val_end: str = "2023-01-01"        # validation: [train_end, val_end); test: [val_end, end]
    class_weights_sensitivity: bool = True

    # ---------------- models ----------------
    models: List[str] = field(default_factory=lambda: ["lr", "dt", "rf", "gbt"])
    feature_sets: List[str] = field(default_factory=lambda: ["A", "B", "C"])
    seeds: List[int] = field(default_factory=lambda: [1, 2, 3])
    grid_lr: List[float] = field(default_factory=lambda: [0.001, 0.01, 0.1])
    grid_dt: List[int] = field(default_factory=lambda: [4, 8, 12])
    grid_rf: List[int] = field(default_factory=lambda: [8, 12])
    rf_num_trees: int = 100
    grid_gbt: List[int] = field(default_factory=lambda: [30, 60])
    gbt_max_depth: int = 5
    n_boot: int = 1000
    master_seed: int = 42

    # ---------------- scalability ----------------
    scale_fractions: List[float] = field(default_factory=lambda: [0.25, 0.5, 0.75, 1.0])
    scale_cores: List[int] = field(default_factory=lambda: [1, 2, 4])
    scale_partitions: List[int] = field(default_factory=lambda: [8, 32, 64])
    scale_salt: List[int] = field(default_factory=lambda: [1, 8])
    stress_factor: int = 1             # >1 replicates the catalog (systems stress test only)

    # ---------------- streaming / case studies ----------------
    replay_name: str = "ridgecrest_2019"
    tick_sim_hours: float = 3.0
    tick_real_seconds: float = 0.5

    # ------------------------------------------------------------------
    @classmethod
    def load(cls, path: Optional[str] = None, overrides: Optional[Dict[str, Any]] = None) -> "Config":
        data: Dict[str, Any] = {}
        if path and os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                data.update(json.load(f))
        data.update(overrides or {})
        known = {f.name for f in dataclasses.fields(cls)}
        unknown = set(data) - known
        if unknown:
            raise KeyError(f"Unknown config keys: {sorted(unknown)}")
        return cls(**data)

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    def path(self, *parts: str) -> str:
        p = os.path.join(self.work_dir, *parts)
        os.makedirs(os.path.dirname(p) if os.path.splitext(p)[1] else p, exist_ok=True)
        return p


# Replay presets reused from Assignment 2 (same events as the streaming demo).
REPLAY_PRESETS = {
    "ridgecrest_2019": {"title": "2019 Ridgecrest, California", "start": "2019-07-04 00:00:00", "days": 8,
                        "center": (35.77, -117.60), "radius_km": 150},
    "tohoku_2011": {"title": "2011 Tohoku, Japan", "start": "2011-03-10 00:00:00", "days": 8,
                    "center": (38.30, 142.37), "radius_km": 300},
    "maule_2010": {"title": "2010 Maule, Chile", "start": "2010-02-26 00:00:00", "days": 8,
                   "center": (-35.91, -72.73), "radius_km": 300},
}


def add_common_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--config", default=None, help="JSON file with config overrides")
    parser.add_argument("--set", dest="sets", action="append", default=[], metavar="KEY=VALUE",
                        help="override one config value, e.g. --set mc=4.0 --set seeds=[1,2]")
    return parser


def cfg_from_args(args: argparse.Namespace) -> Config:
    overrides: Dict[str, Any] = {}
    for item in args.sets:
        key, _, raw = item.partition("=")
        try:
            overrides[key] = json.loads(raw)
        except json.JSONDecodeError:
            overrides[key] = raw
    return Config.load(args.config, overrides)
