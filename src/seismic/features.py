"""Part B: target, strictly causal features, and the time-based split.

Target (frozen before any modelling): for a trigger event i (M >= trigger_min_mag),
    y_i = 1  iff  a strictly larger event occurs within `label_horizon_days` after i
                  and within `label_radius_km` of i.
Features of event i use only events that happened BEFORE i (t_j < t_i):
    A  magnitude, depth, location cell
    B  A + local activity (counts in the last 24 h / 7 d, largest recent magnitude, time since last event)
    C  B + cluster-derived: nearest-neighbour log10(eta), has-parent flag, Gardner-Knopoff flag, zone id
The same `build_feature_table` is used offline and in streaming, so there is no train/serve skew.
"""
from __future__ import annotations

from typing import Optional

from pyspark.sql import DataFrame, SparkSession, functions as F

from .blocking import blocked_pairs, refine_pairs
from .common import utc_epoch
from .gk import assign_aftershocks
from .nn_cluster import compute_nearest_neighbors
from .zones import assign_zone

BASE5 = ["event_id", "t_sec", "latitude", "longitude", "magnitude"]


def trigger_events(events: DataFrame, cfg) -> DataFrame:
    return events.filter(F.col("magnitude") >= cfg.trigger_min_mag)


def add_labels(triggers: DataFrame, cfg) -> DataFrame:
    """Adds `y` (0/1 int). A later, strictly larger event nearby marks the earlier event positive."""
    h, r = cfg.label_horizon_days, cfg.label_radius_km
    pairs = blocked_pairs(triggers.select(*BASE5), triggers.select(*BASE5), max_radius_km=r,
                          parent_radius_km=r, lookback_days=h)
    pairs = refine_pairs(pairs, h, r).filter(F.col("c_magnitude") > F.col("p_magnitude"))
    pos = pairs.select(F.col("p_event_id").alias("event_id")).distinct().withColumn("_pos", F.lit(1))
    return (triggers.join(pos, "event_id", "left")
                    .withColumn("y", F.coalesce(F.col("_pos"), F.lit(0)).cast("int")).drop("_pos"))


def activity_features(triggers: DataFrame, pool: DataFrame, cfg) -> DataFrame:
    """Local recent activity from earlier pool events within `activity_radius_km`."""
    lb, r = cfg.activity_lookback_days, cfg.activity_radius_km
    pairs = blocked_pairs(triggers.select(*BASE5), pool.select(*BASE5), max_radius_km=r,
                          parent_radius_km=r, lookback_days=lb)
    pairs = refine_pairs(pairs, lb, r)
    day, week = 86400.0, 7 * 86400.0
    agg = (pairs.groupBy("c_event_id").agg(
        F.sum((F.col("dt_sec") <= day).cast("long")).alias("n_24h"),
        F.sum((F.col("dt_sec") <= week).cast("long")).alias("n_7d"),
        F.max(F.when(F.col("dt_sec") <= week, F.col("p_magnitude"))).alias("maxmag_7d"),
        (F.min("dt_sec") / day).alias("tsl_days"))
        .withColumnRenamed("c_event_id", "event_id"))
    return (triggers.select("event_id").join(agg, "event_id", "left")
            .fillna({"n_24h": 0, "n_7d": 0, "maxmag_7d": 0.0, "tsl_days": float(lb)}))


def eta_features(triggers: DataFrame, pool: DataFrame, cfg) -> DataFrame:
    """log10 nearest-neighbour proximity of each trigger to earlier pool events (+ has_parent flag)."""
    nn = compute_nearest_neighbors(triggers.select(*BASE5), pool.select(*BASE5), cfg)
    return (nn.select("event_id", "log10_eta")
              .withColumn("has_parent", F.col("log10_eta").isNotNull().cast("int"))
              .fillna({"log10_eta": 1.0}))     # sentinel above any real eta; has_parent tells the model it is a fill


def gk_flag_features(triggers: DataFrame, mainshocks: Optional[DataFrame], grid_deg) -> DataFrame:
    """1 if the event lies in the G-K window of an EARLIER, larger independent mainshock."""
    ids = triggers.select("event_id")
    if mainshocks is None:
        return ids.withColumn("gk_flag", F.lit(0))
    after = (assign_aftershocks(triggers, mainshocks, grid_deg=grid_deg)
             .select("event_id").distinct().withColumn("_f", F.lit(1)))
    return ids.join(after, "event_id", "left").withColumn("gk_flag", F.coalesce(F.col("_f"), F.lit(0))).drop("_f")


def build_feature_table(spark: SparkSession, triggers: DataFrame, pool: DataFrame, mainshocks: Optional[DataFrame],
                        grid_deg, cfg, zone_model=None) -> DataFrame:
    """One row per trigger with all A/B/C feature columns (and zone_id when a zone model is given).

    `pool` is every catalog event (M >= mc) that may have happened before a trigger. Causality: every
    feature is computed from pool rows with t_sec < trigger t_sec (see tests/test_features.py).
    """
    base = (triggers.select("event_id", "event_time", "t_sec", "latitude", "longitude", "depth_km", "magnitude")
            .withColumn("cell_id", F.concat_ws("_", F.floor((F.col("latitude") + 90) / cfg.cell_deg).cast("string"),
                                                F.floor((F.col("longitude") + 180) / cfg.cell_deg).cast("string")))
            .fillna({"depth_km": 10.0}))                       # a handful of USGS rows have no depth
    out = (base.join(activity_features(triggers, pool, cfg), "event_id")
               .join(eta_features(triggers, pool, cfg), "event_id")
               .join(gk_flag_features(triggers, mainshocks, grid_deg), "event_id"))
    if zone_model is not None:
        out = assign_zone(zone_model, out)
    return out


def time_split(features: DataFrame, cfg, catalog_end_sec: float):
    """Train / validation / test by time, with a gap so no label window crosses a boundary.

    Events in the `label_horizon_days` before each boundary are dropped (their label looks into the
    next period), and so are events in the last horizon of the catalog (label window incomplete).
    Feature windows look backwards only, so rows right after a boundary need no gap.
    """
    h = cfg.label_horizon_days * 86400.0
    t_tr, t_va = utc_epoch(cfg.train_end), utc_epoch(cfg.val_end)
    train = features.filter(F.col("t_sec") < t_tr - h)
    val = features.filter((F.col("t_sec") >= t_tr) & (F.col("t_sec") < t_va - h))
    test = features.filter((F.col("t_sec") >= t_va) & (F.col("t_sec") <= catalog_end_sec - h))
    return train, val, test
