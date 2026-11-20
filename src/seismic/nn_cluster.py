"""Nearest-neighbour clustering of seismicity (Zaliapin & Ben-Zion) in Spark.

For a child event j and an earlier event i the proximity is

    eta_ij = t_ij * r_ij^df * 10^(-b * m_i)        (t in years, r in km, m_i = parent magnitude)

Each event's *nearest neighbour* is the earlier event with the smallest eta. If that eta is
below a threshold eta0 the child is 'dependent' (clustered) on that parent; following parent
links gives a forest, and every tree is one cluster (sequence). Events that stay roots are
background (or the first event of a sequence).

Truncation (important, disclosed in the README): candidate parents are searched only inside a
box (at most `nn_max_days` earlier and within `nn_radius_scale` x the Gardner-Knopoff radius of the
parent). Events with no parent in the box are treated as background (eta = +inf). Blocking never
changes the result inside the box; the box itself is the only approximation.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from pyspark.ml.clustering import GaussianMixture
from pyspark.ml.linalg import Vectors
from pyspark.sql import DataFrame, SparkSession, functions as F

from .blocking import blocked_pairs, refine_pairs
from .geo import gk_radius_km_col, gk_radius_km_np

SECONDS_PER_YEAR = 365.25 * 86400.0


def _search_radius(mag_col, cfg):
    return cfg.nn_radius_scale * gk_radius_km_col(mag_col)


def compute_nearest_neighbors(children: DataFrame, parents: Optional[DataFrame], cfg,
                              salt_buckets: Optional[int] = None) -> DataFrame:
    """Nearest earlier neighbour of every child.

    children / parents need columns event_id, t_sec, latitude, longitude, magnitude.
    Returns one row per child: event_id, parent_id, log10_eta, log10_t_years, log10_r_km
    (nulls when no parent lies inside the search box).
    """
    parents = children if parents is None else parents
    salt = cfg.nn_salt_buckets if salt_buckets is None else salt_buckets
    mmax = parents.agg(F.max("magnitude")).first()[0]
    if mmax is None:
        raise ValueError("Empty parent set")
    r_max = float(cfg.nn_radius_scale * gk_radius_km_np(mmax))

    pairs = blocked_pairs(children, parents, max_radius_km=r_max,
                          parent_radius_km=_search_radius(F.col("p_magnitude"), cfg),
                          lookback_days=cfg.nn_max_days, salt_buckets=salt)
    pairs = refine_pairs(pairs, cfg.nn_max_days, _search_radius(F.col("p_magnitude"), cfg))

    log_t = F.log10(F.col("dt_sec") / SECONDS_PER_YEAR)
    log_r = F.log10(F.greatest(F.col("dist_km"), F.lit(cfg.nn_min_km)))
    log_eta = log_t + cfg.nn_df * log_r - cfg.nn_b * F.col("p_magnitude")
    # min over a struct orders by its first field: the smallest eta wins, ties broken by parent id
    best = (pairs.groupBy("c_event_id")
                 .agg(F.min(F.struct(log_eta.alias("log10_eta"), F.col("p_event_id").alias("parent_id"),
                                     log_t.alias("log10_t_years"), log_r.alias("log10_r_km"))).alias("b"))
                 .select(F.col("c_event_id").alias("event_id"), "b.log10_eta", "b.parent_id",
                         "b.log10_t_years", "b.log10_r_km"))
    return (children.select("event_id").join(best, "event_id", "left"))


def count_blocked_pairs(children: DataFrame, parents: Optional[DataFrame], cfg,
                        salt_buckets: Optional[int] = None) -> Tuple[int, int]:
    """(candidate pairs after blocking, naive pair count) for the scalability report."""
    parents = children if parents is None else parents
    salt = cfg.nn_salt_buckets if salt_buckets is None else salt_buckets
    mmax = parents.agg(F.max("magnitude")).first()[0]
    r_max = float(cfg.nn_radius_scale * gk_radius_km_np(mmax))
    n_blocked = blocked_pairs(children, parents, max_radius_km=r_max,
                              parent_radius_km=_search_radius(F.col("p_magnitude"), cfg),
                              lookback_days=cfg.nn_max_days, salt_buckets=salt).count()
    return n_blocked, children.count() * parents.count()


def estimate_eta0_log10(nn: DataFrame, default: float, seed: int = 7, min_events: int = 200) -> Tuple[float, dict]:
    """Threshold between the 'clustered' and 'background' modes of log10(eta).

    Fits a 2-component Gaussian mixture (MLlib) to log10(eta) and returns the point between
    the two means where the weighted component densities are equal. Falls back to `default`
    when there is too little data or the fit does not show two separated modes.
    """
    vals = [r[0] for r in nn.filter(F.col("log10_eta").isNotNull()).select("log10_eta").collect()]
    info = {"n": len(vals), "method": "default"}
    if len(vals) < min_events:
        return default, info
    df = nn.sparkSession.createDataFrame([(Vectors.dense([float(v)]),) for v in vals], ["features"])
    try:
        model = GaussianMixture(k=2, seed=seed, maxIter=100).fit(df)
    except Exception as exc:  # degenerate fit
        info["error"] = str(exc)
        return default, info
    w = np.array(model.weights)
    mu = np.array([g.mean.toArray()[0] for g in model.gaussians])
    sd = np.sqrt(np.array([g.cov.toArray()[0, 0] for g in model.gaussians]))
    lo, hi = int(np.argmin(mu)), int(np.argmax(mu))
    info.update({"weights": w.tolist(), "means": mu.tolist(), "sds": sd.tolist()})
    if mu[hi] - mu[lo] < 0.5 or np.any(sd <= 0):
        return default, info
    grid = np.linspace(mu[lo], mu[hi], 2000)

    def dens(i):
        return w[i] * np.exp(-0.5 * ((grid - mu[i]) / sd[i]) ** 2) / sd[i]

    idx = int(np.argmin(np.abs(dens(lo) - dens(hi))))
    info["method"] = "gmm"
    return float(grid[idx]), info


def link_and_find_roots(spark: SparkSession, nn: DataFrame, eta0_log10: float, max_iter: int = 60) -> DataFrame:
    """Turn nearest-neighbour links into clusters.

    A child is linked to its parent only if log10_eta < eta0. Roots are found with pointer
    jumping (root[x] <- root[root[x]]), which needs only O(log depth) joins even for very long
    chains. localCheckpoint cuts the lineage each round.
    Returns event_id, nn_parent, nn_is_dependent, nn_cluster.
    """
    linked = nn.withColumn("nn_parent", F.when(F.col("log10_eta") < eta0_log10, F.col("parent_id")))
    cur = linked.select("event_id", F.coalesce("nn_parent", "event_id").alias("root")).localCheckpoint()
    for _ in range(max_iter):
        a, b = cur.alias("a"), cur.alias("b")
        nxt = (a.join(b, F.col("a.root") == F.col("b.event_id"), "left")
                .select(F.col("a.event_id").alias("event_id"),
                        F.coalesce(F.col("b.root"), F.col("a.root")).alias("root"))
                .localCheckpoint())
        changed = (nxt.alias("n").join(cur.alias("c"), "event_id")
                      .filter(F.col("n.root") != F.col("c.root")).count())
        cur = nxt
        if changed == 0:
            break
    return (linked.select("event_id", "nn_parent").join(cur, "event_id")
                  .select("event_id", "nn_parent", F.col("nn_parent").isNotNull().alias("nn_is_dependent"),
                          F.col("root").alias("nn_cluster")))


def nn_cluster_events(spark: SparkSession, events: DataFrame, cfg, eta0_log10: Optional[float] = None):
    """Convenience wrapper: NN search -> threshold -> clusters.

    Returns (labels_df, nn_df, eta0_used, eta0_info).
    """
    nn = compute_nearest_neighbors(events, None, cfg).persist()
    info = {"method": "given"}
    if eta0_log10 is None:
        if cfg.nn_estimate_eta0:
            eta0_log10, info = estimate_eta0_log10(nn, cfg.nn_eta0_log10)
        else:
            eta0_log10, info = cfg.nn_eta0_log10, {"method": "default"}
    labels = link_and_find_roots(spark, nn, eta0_log10)
    return labels, nn, eta0_log10, info
