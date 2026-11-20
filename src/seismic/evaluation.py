"""Part A evaluation. There is NO ground truth for aftershock labels, so nothing here is called
'accuracy'. The checks are: overlap with G-K, partition agreement (ARI), and physical regularities
(Bath's law, Omori decay, Poisson-like background), plus sensitivity and case studies.
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
from pyspark.sql import DataFrame, functions as F
from scipy.optimize import minimize_scalar


# ---------------------------------------------------------------- overlap / agreement
def flag_overlap(df: DataFrame, flag_a: str, flag_b: str) -> Dict[str, float]:
    """Counts and Jaccard index of two boolean 'is aftershock/dependent' flags."""
    r = df.agg(F.sum((F.col(flag_a) & F.col(flag_b)).cast("long")).alias("both"),
               F.sum((F.col(flag_a) & ~F.col(flag_b)).cast("long")).alias("a_only"),
               F.sum((~F.col(flag_a) & F.col(flag_b)).cast("long")).alias("b_only"),
               F.sum((~F.col(flag_a) & ~F.col(flag_b)).cast("long")).alias("neither")).first()
    both, a_only, b_only, neither = (int(r[k] or 0) for k in ("both", "a_only", "b_only", "neither"))
    union = both + a_only + b_only
    return {"both": both, "a_only": a_only, "b_only": b_only, "neither": neither,
            "jaccard": both / union if union else float("nan"),
            "share_a": (both + a_only) / max(1, both + a_only + b_only + neither),
            "share_b": (both + b_only) / max(1, both + a_only + b_only + neither)}


def adjusted_rand_index(df: DataFrame, col_a: str, col_b: str) -> float:
    """ARI between two partitions, computed with group-by counts (scales to millions of events)."""
    comb2 = lambda c: c * (c - 1) / 2.0  # noqa: E731
    n = df.count()
    if n < 2:
        return float("nan")
    cont = df.groupBy(col_a, col_b).count()
    sum_ij = cont.select(F.sum(F.col("count") * (F.col("count") - 1) / 2.0)).first()[0] or 0.0
    sum_a = df.groupBy(col_a).count().select(F.sum(F.col("count") * (F.col("count") - 1) / 2.0)).first()[0] or 0.0
    sum_b = df.groupBy(col_b).count().select(F.sum(F.col("count") * (F.col("count") - 1) / 2.0)).first()[0] or 0.0
    expected = sum_a * sum_b / comb2(n)
    maximum = 0.5 * (sum_a + sum_b)
    if maximum == expected:
        return 1.0
    return float((sum_ij - expected) / (maximum - expected))


# ---------------------------------------------------------------- Bath's law
def bath_table(events: DataFrame, cluster_col: str, min_size: int = 5) -> pd.DataFrame:
    """Per cluster: largest event ('mainshock'), largest event *after* it, and their difference.

    events needs event_id, t_sec, magnitude and `cluster_col`. Bath's law predicts a mean difference
    of about 1.2 magnitude units.
    """
    g = events.groupBy(cluster_col).agg(
        F.count("*").alias("size"),
        F.max(F.struct(F.col("magnitude"), (-F.col("t_sec")).alias("neg_t"),
                       F.col("event_id"), F.col("t_sec"))).alias("m"))
    main = g.select(cluster_col, "size", F.col("m.magnitude").alias("main_mag"),
                    F.col("m.t_sec").alias("main_t"), F.col("m.event_id").alias("main_id")).filter(F.col("size") >= min_size)
    after = (events.join(main.select(cluster_col, "main_t", "main_id"), cluster_col)
                   .filter((F.col("t_sec") > F.col("main_t")) & (F.col("event_id") != F.col("main_id")))
                   .groupBy(cluster_col).agg(F.max("magnitude").alias("largest_after")))
    out = main.join(after, cluster_col).withColumn("delta_m", F.col("main_mag") - F.col("largest_after"))
    return out.select(F.col(cluster_col).alias("cluster"), "size", "main_mag", "largest_after", "delta_m").toPandas()


def bath_summary(tab: pd.DataFrame) -> Dict[str, float]:
    if tab.empty:
        return {"n_clusters": 0, "mean_delta": float("nan"), "median_delta": float("nan")}
    return {"n_clusters": int(len(tab)), "mean_delta": float(tab.delta_m.mean()),
            "median_delta": float(tab.delta_m.median())}


# ---------------------------------------------------------------- Omori decay
def omori_mle_p(delays_days: np.ndarray, t_min: float = 0.05, t_max: Optional[float] = None) -> float:
    """Maximum-likelihood exponent p of a truncated power-law rate ~ t^-p on [t_min, t_max] (c = 0).

    The density is t^-p / Z(p) with Z(p) = (b^(1-p) - a^(1-p)) / (1-p). NaN if fewer than 30 delays.
    """
    d = np.asarray(delays_days, float)
    hi = float(d.max()) if t_max is None else float(t_max)
    d = d[(d >= t_min) & (d <= hi)]
    if len(d) < 30 or hi <= t_min:
        return float("nan")
    a, b = t_min, hi
    sum_ln = np.log(d).sum()

    def nll(p):
        z = np.log(b / a) if abs(p - 1.0) < 1e-9 else (b ** (1 - p) - a ** (1 - p)) / (1 - p)
        return p * sum_ln + len(d) * np.log(z)

    res = minimize_scalar(nll, bounds=(0.1, 3.0), method="bounded")
    return float(res.x)


def omori_table(events: DataFrame, cluster_col: str, top_k: int = 20, t_min: float = 0.05) -> pd.DataFrame:
    """Omori p for the `top_k` largest clusters (collected to the driver: they are small)."""
    sizes = events.groupBy(cluster_col).count().orderBy(F.desc("count")).limit(top_k)
    pdf = (events.join(F.broadcast(sizes), cluster_col)
                 .select(cluster_col, "t_sec", "magnitude", "event_id", "count").toPandas())
    rows = []
    for cl, g in pdf.groupby(cluster_col):
        main = g.loc[g.magnitude.idxmax()]
        d = (g.t_sec[g.t_sec > main.t_sec].values - main.t_sec) / 86400.0
        rows.append({"cluster": cl, "size": len(g), "main_mag": float(main.magnitude),
                     "n_after": len(d), "p": omori_mle_p(d, t_min=t_min, t_max=min(365.0, d.max()) if len(d) else None)})
    return pd.DataFrame(rows).sort_values("size", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------- Poisson-like background
def dispersion_stats(events: DataFrame, bin_days: float = 1.0, bbox: Optional[tuple] = None) -> Dict[str, float]:
    """How Poisson-like is a set of event times?

    Index of dispersion of counts per time bin (variance / mean) and the coefficient of
    variation of inter-event times. Both equal 1 for a homogeneous Poisson process and exceed 1
    for clustered catalogs, so a good declustering moves both towards 1.
    bbox = (lat_min, lat_max, lon_min, lon_max) restricts to a region (declustering is normally
    judged regionally; a global catalog is spatially inhomogeneous).
    """
    ev = events
    if bbox:
        la0, la1, lo0, lo1 = bbox
        ev = ev.filter(F.col("latitude").between(la0, la1) & F.col("longitude").between(lo0, lo1))
    t = np.sort(np.array([r[0] for r in ev.select("t_sec").collect()], float))
    n = len(t)
    if n < 10:
        return {"n": n, "dispersion_index": float("nan"), "cv_interevent": float("nan")}
    width = bin_days * 86400.0
    nb = int(np.floor((t[-1] - t[0]) / width)) + 1
    counts = np.bincount(((t - t[0]) // width).astype(int), minlength=nb)
    gaps = np.diff(t)
    return {"n": n, "dispersion_index": float(counts.var(ddof=1) / counts.mean()),
            "cv_interevent": float(gaps.std(ddof=1) / gaps.mean())}


# ---------------------------------------------------------------- sensitivity
def sensitivity_eta0(labels_with_nn: DataFrame, eta_values: Iterable[float], gk_flag: str = "gk_is_aftershock") -> pd.DataFrame:
    """How the NN labels change with the threshold eta0 (one Spark pass for all thresholds).

    labels_with_nn needs log10_eta and the G-K flag. Reports the dependent share and the Jaccard
    overlap with G-K for each eta0.
    """
    etas = list(eta_values)
    aggs = [F.count("*").alias("n")]
    for i, e in enumerate(etas):
        dep = F.col("log10_eta") < F.lit(float(e))
        aggs += [F.sum(dep.cast("long")).alias(f"dep_{i}"),
                 F.sum((dep & F.col(gk_flag)).cast("long")).alias(f"both_{i}")]
    aggs.append(F.sum(F.col(gk_flag).cast("long")).alias("gk"))
    r = labels_with_nn.agg(*aggs).first()
    rows = []
    for i, e in enumerate(etas):
        dep, both, gk = int(r[f"dep_{i}"] or 0), int(r[f"both_{i}"] or 0), int(r["gk"] or 0)
        union = dep + gk - both
        rows.append({"eta0_log10": e, "n_dependent": dep, "share_dependent": dep / r["n"],
                     "jaccard_vs_gk": both / union if union else float("nan")})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- case studies
def case_study(labels: pd.DataFrame, preset: dict) -> Dict[str, object]:
    """One famous sequence: how many events does each method attach to the biggest event?

    labels (driver-side, restricted to the window by the caller) needs event_id, magnitude,
    gk_cluster, nn_cluster, gk_is_aftershock, nn_is_dependent.
    """
    if labels.empty:
        return {"title": preset.get("title"), "n_events": 0}
    big = labels.loc[labels.magnitude.idxmax()]
    gk_members = set(labels.event_id[labels.gk_cluster == big.gk_cluster])
    nn_members = set(labels.event_id[labels.nn_cluster == big.nn_cluster])
    union = gk_members | nn_members
    return {"title": preset.get("title"), "n_events": int(len(labels)), "largest_mag": float(big.magnitude),
            "gk_cluster_size": len(gk_members), "nn_cluster_size": len(nn_members),
            "member_jaccard": len(gk_members & nn_members) / len(union) if union else float("nan"),
            "gk_aftershocks_in_window": int(labels.gk_is_aftershock.sum()),
            "nn_dependent_in_window": int(labels.nn_is_dependent.sum())}
