"""Scalability experiments (plan section 5).

* size: 25/50/75/100 % of the catalog, grown by TIME WINDOW (random row sampling would destroy the
  neighbour structure the clustering depends on)
* parallelism: cores (each core count needs a fresh JVM, so the script launches one process per
  setting) and shuffle partitions
* pair reduction: blocked candidate pairs vs the naive number of pairs
* skew: size of the biggest blocks (hot blocks such as Ridgecrest become straggler tasks) and the
  effect of salting
All timings are wall-clock seconds measured in local mode; they say nothing about a real cluster.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np
from pyspark.sql import DataFrame, SparkSession, functions as F

from .blocking import block_sizes
from .common import StageTimer
from .data import columns_for_clustering
from .geo import gk_radius_km_np
from .gk import run_gk
from .nn_cluster import compute_nearest_neighbors, count_blocked_pairs, link_and_find_roots


def time_fraction(events: DataFrame, fraction: float) -> DataFrame:
    """Events in the first `fraction` of the catalog's time span."""
    r = events.agg(F.min("t_sec").alias("a"), F.max("t_sec").alias("b")).first()
    return events.filter(F.col("t_sec") <= r["a"] + fraction * (r["b"] - r["a"]))


def skew_report(events_mc: DataFrame, cfg) -> Dict[str, float]:
    """Distribution of children per (space, time) block: max / mean ratio flags hot blocks."""
    mmax = events_mc.agg(F.max("magnitude")).first()[0]
    r_max = float(cfg.nn_radius_scale * gk_radius_km_np(mmax))
    sizes = np.array([r["count"] for r in block_sizes(events_mc, r_max, cfg.nn_max_days).collect()], float)
    if len(sizes) == 0:
        return {"n_blocks": 0}
    return {"n_blocks": int(len(sizes)), "block_max": int(sizes.max()), "block_mean": float(sizes.mean()),
            "block_p99": float(np.percentile(sizes, 99)), "max_over_mean": float(sizes.max() / sizes.mean())}


def run_one(spark: SparkSession, events_all: DataFrame, cfg, fraction: float, partitions: int, salt: int,
            with_pairs: bool = True, with_gk: bool = True) -> Dict[str, object]:
    """One timed pass: [G-K] + NN search + root finding on a time-window fraction of the catalog."""
    spark.conf.set("spark.sql.shuffle.partitions", str(partitions))
    sub = time_fraction(events_all, fraction).persist()
    n_all = sub.count()                                   # materialise so load time is not charged to a stage
    mc = columns_for_clustering(sub.filter(F.col("magnitude") >= cfg.mc)).persist()
    n_mc = mc.count()
    timer = StageTimer()
    row: Dict[str, object] = {"fraction": fraction, "partitions": partitions, "salt": salt, "n_events_all": n_all,
                              "n_events_mc": n_mc}
    if with_gk:
        with timer.stage("gk_baseline"):
            try:
                _, after, _ = run_gk(spark, sub, cfg.mainshock_min_mag)
                row["n_gk_aftershocks"] = after.count()
            except ValueError:               # a very short window may contain no M>=6 candidate
                row["n_gk_aftershocks"] = None
    with timer.stage("nn_search"):
        nn = compute_nearest_neighbors(mc, None, cfg, salt_buckets=salt).persist()
        row["n_nn_with_parent"] = nn.filter(F.col("parent_id").isNotNull()).count()
    with timer.stage("root_finding"):
        row["n_nn_dependent"] = link_and_find_roots(spark, nn, cfg.nn_eta0_log10).filter("nn_is_dependent").count()
    if with_pairs:
        blocked, naive = count_blocked_pairs(mc, None, cfg, salt_buckets=salt)
        row.update({"pairs_blocked": blocked, "pairs_naive": naive,
                    "pairs_blocked_pct_of_naive": 100.0 * blocked / max(1, naive)})
        row.update(skew_report(mc, cfg))
    row.update({f"sec_{k}": v for k, v in timer.seconds.items()})
    row["sec_total"] = float(sum(timer.seconds.values()))
    for d in (nn, mc, sub):
        d.unpersist()
    return row


def experiment_plan(cfg) -> List[Dict[str, object]]:
    """The run list: size sweep at every core count; partition and salt sweeps at the largest core count."""
    top = max(cfg.scale_cores)
    base_p = cfg.shuffle_partitions
    plan = []
    for c in cfg.scale_cores:
        for f in cfg.scale_fractions:
            plan.append({"exp": "size", "cores": c, "fraction": f, "partitions": base_p, "salt": 1})
    for p in cfg.scale_partitions:
        plan.append({"exp": "partitions", "cores": top, "fraction": 1.0, "partitions": p, "salt": 1})
    for s in cfg.scale_salt:
        plan.append({"exp": "salt", "cores": top, "fraction": 1.0, "partitions": base_p, "salt": s})
    seen, out = set(), []
    for r in plan:                       # identical settings appear in several sweeps: run them once
        key = (r["cores"], r["fraction"], r["partitions"], r["salt"])
        if key not in seen:
            seen.add(key)
            out.append(r)
    return out


def run_id(r: Dict[str, object], stress: int) -> str:
    return f"S|cores{r['cores']}|frac{r['fraction']}|part{r['partitions']}|salt{r['salt']}|stress{stress}"
