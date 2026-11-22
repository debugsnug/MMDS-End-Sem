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
from .nn_cluster import compute_nearest_neighbors, count_pairs, link_and_find_roots


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


def _one_pass(spark, sub: DataFrame, cfg, salt: int, with_gk: bool) -> Dict[str, object]:
    """One timed pass over an already materialised catalog window: [G-K] + NN search + root finding."""
    mc = columns_for_clustering(sub.filter(F.col("magnitude") >= cfg.mc)).persist()
    n_mc = mc.count()
    timer = StageTimer()
    row: Dict[str, object] = {"n_events_mc": n_mc}
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
    row.update({f"sec_{k}": v for k, v in timer.seconds.items()})
    row["sec_total"] = float(sum(timer.seconds.values()))
    # comparable across ALL runs (stress runs skip the G-K step): nearest-neighbour search + cluster assembly only
    row["sec_nn_total"] = float(timer.seconds.get("nn_search", 0.0) + timer.seconds.get("root_finding", 0.0))
    nn.unpersist()
    mc.unpersist()
    return row


def run_one(spark: SparkSession, events_all: DataFrame, cfg, fraction: float, partitions: int, salt: int,
            with_pairs: bool = True, with_gk: bool = True, repeats: int = 1) -> Dict[str, object]:
    """Timed pass(es) on a time-window fraction of the catalog.

    With repeats > 1 the pass is repeated on the same materialised data and the MEDIAN of every sec_* column is
    reported, with min/max of the total, so that single-run noise (JIT, GC) is visible. Pair counts and skew are
    measured once. The nearest-neighbour threshold is the FIXED cfg.nn_eta0_log10 (recorded as eta0_log10_used), so
    n_nn_dependent here differs from Part A, which uses the estimated threshold.
    """
    spark.conf.set("spark.sql.shuffle.partitions", str(partitions))
    sub = time_fraction(events_all, fraction).persist()
    n_all = sub.count()                                   # materialise so load time is not charged to a stage
    runs = [_one_pass(spark, sub, cfg, salt, with_gk) for _ in range(max(1, repeats))]
    row: Dict[str, object] = {"fraction": fraction, "partitions": partitions, "salt": salt, "n_events_all": n_all,
                              "eta0_log10_used": cfg.nn_eta0_log10, "repeats": len(runs)}
    for k in runs[0]:
        if k.startswith("sec_"):
            row[k] = float(np.median([r[k] for r in runs]))
        else:
            row[k] = runs[0][k]
    totals = [r["sec_total"] for r in runs]
    row.update({"sec_total_min": float(min(totals)), "sec_total_max": float(max(totals)), "sec_total_runs": [round(t, 3) for t in totals]})
    if with_pairs:
        mc = columns_for_clustering(sub.filter(F.col("magnitude") >= cfg.mc)).persist()
        pc = count_pairs(mc, None, cfg, salt_buckets=salt)
        row.update({"pairs_blocked": pc["blocked"], "pairs_exact": pc["exact"], "pairs_naive": pc["naive"],
                    "pairs_blocked_pct_of_naive": 100.0 * pc["blocked"] / max(1, pc["naive"]),
                    "pairs_exact_pct_of_naive": 100.0 * pc["exact"] / max(1, pc["naive"])})
        row.update(skew_report(mc, cfg))
        mc.unpersist()
    sub.unpersist()
    return row


def experiment_plan(cfg) -> List[Dict[str, object]]:
    """The run list: size sweep at every core count; partition, salt and stress sweeps at the largest core count.

    'stress' entries replicate the full catalog k times (runtime only; replicated data says nothing about accuracy)."""
    top = max(cfg.scale_cores)
    base_p = cfg.shuffle_partitions
    plan = []
    for c in cfg.scale_cores:
        for f in cfg.scale_fractions:
            plan.append({"exp": "size", "cores": c, "fraction": f, "partitions": base_p, "salt": 1, "stress": 1})
    for p in cfg.scale_partitions:
        plan.append({"exp": "partitions", "cores": top, "fraction": 1.0, "partitions": p, "salt": 1, "stress": 1})
    for s in cfg.scale_salt:
        plan.append({"exp": "salt", "cores": top, "fraction": 1.0, "partitions": base_p, "salt": s, "stress": 1})
    for k in cfg.scale_stress:
        plan.append({"exp": "stress", "cores": top, "fraction": 1.0, "partitions": base_p, "salt": 1, "stress": int(k)})
    seen, out = set(), []
    for r in plan:                       # identical settings appear in several sweeps: run them once
        key = (r["cores"], r["fraction"], r["partitions"], r["salt"], r["stress"])
        if key not in seen:
            seen.add(key)
            out.append(r)
    return out


def run_id(r: Dict[str, object], stress: int = None) -> str:
    st = r.get("stress", 1) if stress is None else stress
    return f"S|cores{r['cores']}|frac{r['fraction']}|part{r['partitions']}|salt{r['salt']}|stress{st}"
