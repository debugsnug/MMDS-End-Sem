"""Blocked event-to-event join (the scalable core of the project).

Problem: for every 'child' event find earlier 'parent' events that are close in space
and time. A naive join is O(n^2). Blocking turns it into many small joins:

  * space: events are put in cubic cells of the unit-sphere embedding (x, y, z). A cube
    grid has no longitude wrap-around and no polar distortion, unlike a lat/lon grid.
    Cell size >= the largest search radius, so a parent only has to be replicated into
    neighbouring cells it is actually close to (per-axis pruning keeps replication low).
  * time: events are put in buckets of length = look-back window; a parent is replicated
    into its own bucket and the next one, a child looks only in its own bucket.
  * optional salting: a child gets a random salt in [0, S), parents are replicated to all
    S salts, so one hot block (e.g. Ridgecrest) is spread over S tasks.

The blocked join returns *candidate* pairs; `refine_pairs` applies the exact time and
great-circle test, so blocking never changes the result, only the cost.
"""
from __future__ import annotations

from typing import Sequence, Union

from pyspark.sql import Column, DataFrame, functions as F

from .geo import chord_unit, haversine_km_col, EARTH_R_KM

BASE_COLS = ["event_id", "t_sec", "latitude", "longitude", "magnitude"]
Radius = Union[float, Column]


def _prefixed(df: DataFrame, prefix: str, cols: Sequence[str]) -> DataFrame:
    return df.select([F.col(c).alias(f"{prefix}{c}") for c in cols])


def _add_xyz(df: DataFrame, p: str) -> DataFrame:
    la, lo = F.radians(F.col(f"{p}latitude")), F.radians(F.col(f"{p}longitude"))
    return (df.withColumn(f"{p}x", F.cos(la) * F.cos(lo))
              .withColumn(f"{p}y", F.cos(la) * F.sin(lo))
              .withColumn(f"{p}z", F.sin(la)))


def _chord_expr(radius_km: Radius) -> Column:
    if isinstance(radius_km, (int, float)):
        return F.lit(chord_unit(float(radius_km)))
    return 2.0 * F.sin(radius_km / (2.0 * EARTH_R_KM))


def child_keys(children: DataFrame, max_radius_km: float, lookback_days: float,
               salt_buckets: int = 1, extra_cols: Sequence[str] = ()) -> DataFrame:
    """Children with their single block key (k_x, k_y, k_z, k_tb, k_salt)."""
    s = chord_unit(max_radius_km)
    bucket = lookback_days * 86400.0
    c = _add_xyz(_prefixed(children, "c_", list(BASE_COLS) + list(extra_cols)), "c_")
    for axis in "xyz":
        c = c.withColumn(f"k_{axis}", F.floor((F.col(f"c_{axis}") + 1.0) / s).cast("int"))
    return (c.withColumn("k_tb", F.floor(F.col("c_t_sec") / bucket).cast("long"))
             .withColumn("k_salt", F.pmod(F.hash(F.col("c_event_id")), F.lit(int(salt_buckets))).cast("int")))


def parent_keys(parents: DataFrame, max_radius_km: float, parent_radius_km: Radius,
                lookback_days: float, salt_buckets: int = 1, extra_cols: Sequence[str] = ()) -> DataFrame:
    """Parents replicated into every block a child within their radius could live in."""
    s = chord_unit(max_radius_km)
    bucket = lookback_days * 86400.0
    pchord = _chord_expr(parent_radius_km)
    p = _add_xyz(_prefixed(parents, "p_", list(BASE_COLS) + list(extra_cols)), "p_")
    for axis in "xyz":
        coord = F.col(f"p_{axis}") + 1.0
        cell = F.floor(coord / s)
        lo_ok = (coord - cell * s) < pchord           # close to the lower cell face
        hi_ok = ((cell + 1) * s - coord) < pchord     # close to the upper cell face
        offsets = F.filter(F.array(F.lit(0), F.when(lo_ok, F.lit(-1)), F.when(hi_ok, F.lit(1))),
                           lambda v: v.isNotNull())
        p = (p.withColumn(f"_o{axis}", F.explode(offsets))
              .withColumn(f"k_{axis}", (cell + F.col(f"_o{axis}")).cast("int"))
              .drop(f"_o{axis}"))
    p = (p.withColumn("_otb", F.explode(F.array(F.lit(0), F.lit(1))))
          .withColumn("k_tb", (F.floor(F.col("p_t_sec") / bucket) + F.col("_otb")).cast("long"))
          .drop("_otb"))
    p = p.withColumn("k_salt", F.explode(F.sequence(F.lit(0), F.lit(int(salt_buckets) - 1))))
    return p.withColumn("k_salt", F.col("k_salt").cast("int"))


def blocked_pairs(children: DataFrame, parents: DataFrame, *, max_radius_km: float,
                  parent_radius_km: Radius, lookback_days: float, salt_buckets: int = 1,
                  extra_parent_cols: Sequence[str] = (), extra_child_cols: Sequence[str] = ()) -> DataFrame:
    """Candidate (child, parent) pairs sharing a block. Columns are prefixed c_ / p_.

    Inputs must have BASE_COLS (event_id, t_sec, latitude, longitude, magnitude).
    """
    c = child_keys(children, max_radius_km, lookback_days, salt_buckets, extra_child_cols)
    p = parent_keys(parents, max_radius_km, parent_radius_km, lookback_days, salt_buckets, extra_parent_cols)
    return c.join(p, ["k_x", "k_y", "k_z", "k_tb", "k_salt"], "inner")


def exact_condition(lookback_days: float, parent_radius_km: Radius) -> Column:
    """Boolean column of the exact test (parent strictly earlier, within look-back, within the parent's radius)."""
    dt = F.col("c_t_sec") - F.col("p_t_sec")
    radius = F.lit(float(parent_radius_km)) if isinstance(parent_radius_km, (int, float)) else parent_radius_km
    d = haversine_km_col(F.col("p_latitude"), F.col("p_longitude"), F.col("c_latitude"), F.col("c_longitude"))
    return (dt > 0) & (dt <= lookback_days * 86400.0) & (d <= radius)


def refine_pairs(pairs: DataFrame, lookback_days: float, parent_radius_km: Radius) -> DataFrame:
    """Exact test: parent strictly earlier, within look-back, within parent's radius."""
    dt = F.col("c_t_sec") - F.col("p_t_sec")
    out = pairs.withColumn("dt_sec", dt).filter((F.col("dt_sec") > 0) & (F.col("dt_sec") <= lookback_days * 86400.0))
    out = out.withColumn("dist_km", haversine_km_col(F.col("p_latitude"), F.col("p_longitude"),
                                                      F.col("c_latitude"), F.col("c_longitude")))
    radius = F.lit(float(parent_radius_km)) if isinstance(parent_radius_km, (int, float)) else parent_radius_km
    return out.filter(F.col("dist_km") <= radius)


def block_sizes(children: DataFrame, max_radius_km: float, lookback_days: float) -> DataFrame:
    """Number of children per block: used by the skew report (hot blocks = stragglers)."""
    c = child_keys(children, max_radius_km, lookback_days)
    return c.groupBy("k_x", "k_y", "k_z", "k_tb").count()
