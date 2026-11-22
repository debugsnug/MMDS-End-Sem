"""Gardner-Knopoff (G-K) baseline: logic carried over from the Assignment 2 notebook.

Only wrapper functions (`run_gk`, `gk_label_table`) are new; the mainshock selection and the
grid-blocked join are the Assignment 2 algorithms so the baseline numbers stay comparable.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd
from pyspark.sql import DataFrame, SparkSession, Window, functions as F, types as T

from .geo import (choose_grid_deg, gk_radius_km_np, gk_window_days_np, haversine_km_col,
                  haversine_km_np)

MS_SCHEMA = T.StructType([
    T.StructField("mainshock_id", T.StringType()), T.StructField("ms_t", T.DoubleType()),
    T.StructField("ms_lat", T.DoubleType()), T.StructField("ms_lon", T.DoubleType()),
    T.StructField("ms_mag", T.DoubleType()), T.StructField("ms_window_days", T.DoubleType()),
    T.StructField("ms_radius_km", T.DoubleType()), T.StructField("ms_place", T.StringType()),
    T.StructField("ms_time", T.TimestampType())])


def select_mainshocks(cand: pd.DataFrame) -> pd.DataFrame:
    """Independent mainshocks among M>=6 candidates (driver side, ~2k rows).

    Candidates are processed from the largest down; an event inside the space-time window
    of an already accepted larger event is 'dependent'. Whether an event is dependent
    only uses *earlier* events, which is why the resulting labels are causal (needed in Part B).
    """
    c = cand.sort_values(["magnitude", "t_sec"], ascending=[False, True]).reset_index(drop=True)
    t, la, lo, m = c.t_sec.values, c.latitude.values, c.longitude.values, c.magnitude.values
    ids = c.event_id.values
    W, L = gk_window_days_np(m) * 86400.0, gk_radius_km_np(m)
    dependent = np.zeros(len(c), bool)
    for i in range(len(c)):
        if dependent[i]:
            continue
        inside = ((t > t[i]) & (t <= t[i] + W[i]) & (m < m[i])
                  & (haversine_km_np(la[i], lo[i], la, lo) <= L[i]))
        dependent |= inside
    c["is_mainshock"] = ~dependent
    c["window_days"] = W / 86400.0
    c["radius_km"] = L
    return c

def select_mainshocks_causal(cand: pd.DataFrame) -> pd.DataFrame:
    """Select independent G-K mainshocks chronologically.

    An event can only be classified using mainshocks observed earlier.
    Later events never change an earlier event's mainshock status.
    """
    c = cand.sort_values(["t_sec", "event_id"]).reset_index(drop=True).copy()

    active = []
    selected = []

    for r in c.itertuples():
        t = float(r.t_sec)

        # Retain only mainshocks whose G-K time windows are still active.
        active = [a for a in active
                  if t > a["t_sec"]
                  and t <= a["t_sec"] + a["window_days"] * 86400.0]

        dependent = False

        for a in active:
            if (a["magnitude"] > r.magnitude
                    and haversine_km_np(
                        a["latitude"], a["longitude"],
                        r.latitude, r.longitude
                    ) <= a["radius_km"]):
                dependent = True
                break

        if not dependent:
            window_days = float(gk_window_days_np(float(r.magnitude)))
            radius_km = float(gk_radius_km_np(float(r.magnitude)))

            selected.append({
                "event_id": r.event_id,
                "event_time": r.event_time,
                "t_sec": t,
                "latitude": float(r.latitude),
                "longitude": float(r.longitude),
                "magnitude": float(r.magnitude),
                "place": r.place,
                "window_days": window_days,
                "radius_km": radius_km,
                "is_mainshock": True
            })

            active.append({
                "event_id": r.event_id,
                "t_sec": t,
                "latitude": float(r.latitude),
                "longitude": float(r.longitude),
                "magnitude": float(r.magnitude),
                "window_days": window_days,
                "radius_km": radius_km
            })

    return pd.DataFrame(selected)

def mainshocks_to_spark(spark: SparkSession, ms_pdf: pd.DataFrame) -> DataFrame:
    rows = []
    for r in ms_pdf.itertuples():
        ts = pd.Timestamp(r.event_time)
        ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
        rows.append((str(r.event_id), float(r.t_sec), float(r.latitude), float(r.longitude),
                     float(r.magnitude), float(r.window_days), float(r.radius_km),
                     getattr(r, "place", None), ts.to_pydatetime()))
    return spark.createDataFrame(rows, schema=MS_SCHEMA)


def assign_aftershocks(events: DataFrame, mainshocks: DataFrame, grid_deg=None) -> DataFrame:
    """Grid-blocked join of events against mainshocks (Assignment 2, unchanged).

    Each mainshock is replicated into its 3x3 neighbouring lat/lon cells; an event is joined
    only to mainshocks in its own cell, then the exact time/distance test is applied. An
    event matching several mainshocks keeps the largest (earliest on ties).
    """
    ms = mainshocks
    if grid_deg:
        n_lon = int(round(360 / grid_deg))
        ms = (ms.withColumn("_c_lat", F.floor((F.col("ms_lat") + 90) / grid_deg).cast("int"))
                .withColumn("_c_lon", F.floor((F.col("ms_lon") + 180) / grid_deg).cast("int"))
                .withColumn("_dlat", F.explode(F.array(F.lit(-1), F.lit(0), F.lit(1))))
                .withColumn("_dlon", F.explode(F.array(F.lit(-1), F.lit(0), F.lit(1))))
                .withColumn("m_clat", F.col("_c_lat") + F.col("_dlat"))
                .withColumn("m_clon", F.pmod(F.col("_c_lon") + F.col("_dlon"), F.lit(n_lon)))
                .drop("_c_lat", "_c_lon", "_dlat", "_dlon"))
        ev = (events.withColumn("e_clat", F.floor((F.col("latitude") + 90) / grid_deg).cast("int"))
                    .withColumn("e_clon", F.pmod(F.floor((F.col("longitude") + 180) / grid_deg).cast("int"),
                                                 F.lit(n_lon))))
        key = (F.col("e_clat") == F.col("m_clat")) & (F.col("e_clon") == F.col("m_clon"))
    else:
        ev, key = events, F.lit(True)
    cond = (key & (F.col("t_sec") > F.col("ms_t"))
            & (F.col("t_sec") <= F.col("ms_t") + F.col("ms_window_days") * 86400.0)
            & (F.col("magnitude") < F.col("ms_mag")))
    joined = (ev.join(F.broadcast(ms), cond, "inner")
                .withColumn("distance_km", haversine_km_col(F.col("ms_lat"), F.col("ms_lon"),
                                                            F.col("latitude"), F.col("longitude")))
                .filter(F.col("distance_km") <= F.col("ms_radius_km")))
    best = Window.partitionBy("event_id").orderBy(F.desc("ms_mag"), F.asc("ms_t"))
    return (joined.withColumn("_rn", F.row_number().over(best)).filter(F.col("_rn") == 1)
            .select("event_id", "event_time", "latitude", "longitude", "magnitude", "mainshock_id", "ms_mag",
                    ((F.col("t_sec") - F.col("ms_t")) / 86400.0).alias("delay_days"), "distance_km"))


def run_gk(spark: SparkSession, events: DataFrame, mainshock_min_mag: float = 6.0,
           use_grid: bool = True) -> Tuple[DataFrame, DataFrame, int]:
    """Full G-K pass on `events` (all magnitudes recommended, as in Assignment 2).

    Returns (mainshocks_sdf, aftershocks_sdf, grid_deg).
    """
    cand = (events.filter(F.col("magnitude") >= mainshock_min_mag)
                  .select("event_id", "event_time", "t_sec", "latitude", "longitude", "magnitude", "place")
                  .toPandas())
    if cand.empty:
        raise ValueError("No G-K mainshock candidates: lower mainshock_min_mag or check the data.")
    sel = select_mainshocks(cand)
    ms_pdf = sel[sel.is_mainshock]
    ms_sdf = mainshocks_to_spark(spark, ms_pdf)
    grid = choose_grid_deg(ms_pdf.radius_km.values, ms_pdf.latitude.values) if use_grid else None
    after = assign_aftershocks(events, ms_sdf, grid_deg=grid)
    return ms_sdf, after, grid

def run_gk_causal(spark: SparkSession, events: DataFrame,
                  mainshock_min_mag: float = 6.0,
                  use_grid: bool = True):
    """Chronological, causal G-K pass for classification features."""

    cand = (
        events.filter(F.col("magnitude") >= mainshock_min_mag)
        .select("event_id", "event_time", "t_sec", "latitude",
                "longitude", "magnitude", "place")
        .toPandas()
    )

    if cand.empty:
        raise ValueError("No G-K mainshock candidates.")

    sel = select_mainshocks_causal(cand)
    ms_sdf = mainshocks_to_spark(spark, sel)

    grid = (
        choose_grid_deg(sel.radius_km.values, sel.latitude.values)
        if use_grid and not sel.empty else None
    )

    after = assign_aftershocks(events, ms_sdf, grid_deg=grid)

    return ms_sdf, after, grid

def gk_label_table(events: DataFrame, aftershocks: DataFrame) -> DataFrame:
    """One row per event: G-K aftershock flag and G-K cluster id.

    cluster id = mainshock id for aftershocks and the event's own id otherwise, so a
    mainshock and its aftershocks share a cluster id (needed for ARI / Bath comparisons).
    """
    a = aftershocks.select("event_id", F.col("mainshock_id").alias("_ms"))
    return (events.join(a, "event_id", "left")
                  .withColumn("gk_is_aftershock", F.col("_ms").isNotNull())
                  .withColumn("gk_cluster", F.coalesce(F.col("_ms"), F.col("event_id")))
                  .drop("_ms"))
