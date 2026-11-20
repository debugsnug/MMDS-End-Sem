"""Spark session factory, stage timer and small helpers shared by all modules."""
from __future__ import annotations

import contextlib
import os
import time
from typing import Dict, Optional

import pandas as pd


def utc_epoch(date_str: str) -> float:
    """'2021-01-01' -> epoch seconds (UTC)."""
    return pd.Timestamp(date_str, tz="UTC").timestamp()


def get_spark(cfg, app_name: str = "seismic", master: Optional[str] = None,
              aqe: bool = True, shuffle_partitions: Optional[int] = None):
    """Create (or reuse) a local SparkSession with UTC time zone.

    aqe=False is used by the scalability experiments so that the shuffle-partition
    setting is not silently re-coalesced by Adaptive Query Execution.
    """
    from pyspark.sql import SparkSession

    os.environ["TZ"] = "UTC"
    if hasattr(time, "tzset"):
        time.tzset()
    spark = (SparkSession.builder.appName(app_name)
             .master(master or cfg.master)
             .config("spark.sql.shuffle.partitions", str(shuffle_partitions or cfg.shuffle_partitions))
             .config("spark.sql.session.timeZone", "UTC")
             .config("spark.driver.memory", cfg.driver_memory)
             .config("spark.ui.enabled", "false")
             .config("spark.sql.adaptive.enabled", str(aqe).lower())
             .config("spark.sql.execution.arrow.pyspark.enabled", str(cfg.use_arrow).lower())
             .getOrCreate())
    spark.sparkContext.setLogLevel("ERROR")
    return spark


class StageTimer:
    """Collects wall-clock seconds per named stage (used for the scalability tables)."""

    def __init__(self) -> None:
        self.seconds: Dict[str, float] = {}

    @contextlib.contextmanager
    def stage(self, name: str):
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.seconds[name] = self.seconds.get(name, 0.0) + (time.perf_counter() - t0)


def banner(text: str) -> None:
    print("\n" + "=" * 8 + f" {text} " + "=" * 8)
