"""Catalog loading and cleaning. Reproduces the Assignment 2 cleaning rules.

Normalised event schema used everywhere:
    event_id, event_time (timestamp), t_sec (epoch seconds, double), latitude, longitude,
    depth_km, magnitude (double, never null), place
"""
from __future__ import annotations

from typing import Dict, Iterable

import pandas as pd
from pyspark.sql import DataFrame, SparkSession, functions as F, types as T

EVENT_SCHEMA = T.StructType([
    T.StructField("event_id", T.StringType(), False),
    T.StructField("event_time", T.TimestampType(), False),
    T.StructField("latitude", T.DoubleType(), False),
    T.StructField("longitude", T.DoubleType(), False),
    T.StructField("depth_km", T.DoubleType(), True),
    T.StructField("magnitude", T.DoubleType(), False),
    T.StructField("place", T.StringType(), True),
])


def load_catalog_csv(spark: SparkSession, path: str) -> DataFrame:
    """USGS CSV -> cleaned normalised events.

    Same rules as Assignment 2: unique event_id, valid coordinates, event type
    'earthquake'. Events without a magnitude are dropped here because every method in
    this project needs a magnitude (Assignment 2 reported 7,669 such rows).
    """
    raw = spark.read.option("header", True).option("mode", "PERMISSIVE").csv(path)
    required = {"time", "latitude", "longitude", "depth", "mag", "id", "place", "type"}
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(f"CSV is missing USGS columns {sorted(missing)}; found {raw.columns}")
    ev = (raw.select(
            F.col("id").cast("string").alias("event_id"),
            F.to_timestamp("time").alias("event_time"),
            F.col("latitude").cast("double").alias("latitude"),
            F.col("longitude").cast("double").alias("longitude"),
            F.col("depth").cast("double").alias("depth_km"),
            F.col("mag").cast("double").alias("magnitude"),
            F.col("place").cast("string").alias("place"),
            F.col("type").cast("string").alias("event_type"))
          .filter(F.col("event_id").isNotNull())
          .dropDuplicates(["event_id"]))
    valid = F.coalesce(
        F.col("latitude").between(-90, 90) & F.col("longitude").between(-180, 180)
        & F.col("event_time").isNotNull() & (F.col("event_type") == "earthquake")
        & F.col("magnitude").isNotNull(), F.lit(False))
    return with_t_sec(ev.filter(valid).drop("event_type"))


def with_t_sec(df: DataFrame) -> DataFrame:
    return df.withColumn("t_sec", F.col("event_time").cast("double"))


def pandas_to_events(spark: SparkSession, pdf: pd.DataFrame) -> DataFrame:
    """Small pandas frame -> events DataFrame, built from plain Python tuples.

    Going through tuples (not createDataFrame(pandas)) keeps this independent of the
    pandas/pyspark version pairing. Used by tests, the synthetic generator and streaming.
    """
    rows = []
    for r in pdf.itertuples(index=False):
        ts = pd.Timestamp(r.event_time)
        ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
        rows.append((str(r.event_id), ts.to_pydatetime(), float(r.latitude), float(r.longitude),
                     None if pd.isna(r.depth_km) else float(r.depth_km), float(r.magnitude),
                     getattr(r, "place", None)))
    return with_t_sec(spark.createDataFrame(rows, schema=EVENT_SCHEMA))


def write_events(df: DataFrame, path: str) -> None:
    df.write.mode("overwrite").parquet(path)


def read_events(spark: SparkSession, path: str) -> DataFrame:
    return spark.read.parquet(path)


def audit(df: DataFrame, mc: float) -> Dict[str, object]:
    """Row counts and date range used in the report's data section."""
    row = df.agg(F.count("*").alias("rows"),
                 F.min("event_time").alias("first"), F.max("event_time").alias("last"),
                 F.sum((F.col("magnitude") >= mc).cast("long")).alias("rows_ge_mc"),
                 F.sum((F.col("magnitude") < 0).cast("long")).alias("negative_mag"),
                 F.sum((F.col("depth_km") < 0).cast("long")).alias("negative_depth")).first()
    return {k: row[k] for k in row.asDict()} | {"mc": mc}


def select_events(df: DataFrame, min_mag: float) -> DataFrame:
    return df.filter(F.col("magnitude") >= min_mag)


def columns_for_clustering(df: DataFrame) -> DataFrame:
    """The five columns every clustering / join function expects."""
    return df.select("event_id", "t_sec", "latitude", "longitude", "magnitude")


def replicate_catalog(df: DataFrame, factor: int, span_pad_days: float = 30.0) -> DataFrame:
    """Systems stress test ONLY: tile the catalog `factor` times along the time axis.

    Copies get new ids and shifted times, so sequences never mix across copies. The
    result says something about runtime, nothing about accuracy or generalisation.
    """
    if factor <= 1:
        return df
    span = df.agg((F.max("t_sec") - F.min("t_sec")).alias("s")).first()["s"] + span_pad_days * 86400.0
    parts = []
    for k in range(factor):
        parts.append(df.withColumn("event_id", F.concat(F.col("event_id"), F.lit(f"_r{k}")))
                       .withColumn("t_sec", F.col("t_sec") + F.lit(k * span))
                       .withColumn("event_time", F.col("t_sec").cast("timestamp")))
    out = parts[0]
    for p in parts[1:]:
        out = out.unionByName(p)
    return out
