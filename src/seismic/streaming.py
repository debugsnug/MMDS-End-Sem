"""Streaming scoring (plan section 5): Structured Streaming scores arriving events with the saved PipelineModel.

Rolling features need history, so the scorer keeps a bounded buffer of recent events and the list of
active G-K mainshocks (state on the driver, updated in `foreachBatch`). Features are produced by the SAME
`features.build_feature_table` that the batch path uses, so train/serve skew cannot arise from duplicated
logic. The batch-vs-stream comparison is therefore a *consistency* check (state handling, ordering,
micro-batch boundaries), not an independent validation of the model.
"""
from __future__ import annotations

import json
import os
import shutil
import time
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from pyspark.ml import PipelineModel
from pyspark.ml.clustering import KMeansModel
from pyspark.ml.functions import vector_to_array
from pyspark.sql import DataFrame, SparkSession, functions as F, types as T

from . import features as fe
from .data import pandas_to_events
from .geo import choose_grid_deg, gk_radius_km_np, gk_window_days_np, haversine_km_np
from .gk import mainshocks_to_spark, select_mainshocks

EVENT_COLS = ["event_id", "event_time", "t_sec", "latitude", "longitude", "depth_km", "magnitude", "place"]
MS_COLS = ["event_id", "event_time", "t_sec", "latitude", "longitude", "magnitude", "place", "window_days", "radius_km"]


class StreamState:
    """Bounded history + active independent mainshocks (incremental version of gk.select_mainshocks)."""

    def __init__(self, cfg, warm_events: Optional[pd.DataFrame] = None) -> None:
        self.cfg = cfg
        self.keep_sec = max(cfg.nn_max_days, cfg.activity_lookback_days) * 86400.0
        self.buffer = pd.DataFrame(columns=EVENT_COLS)
        self.mainshocks = pd.DataFrame(columns=MS_COLS)
        self.ever_mainshock_ids: set = set()      # diagnostics: every independent mainshock seen so far
        if warm_events is not None and len(warm_events):
            w = warm_events.sort_values("t_sec")
            self.buffer = w[w.t_sec >= w.t_sec.max() - self.keep_sec][EVENT_COLS].reset_index(drop=True)
            cand = w[w.magnitude >= cfg.mainshock_min_mag]
            if len(cand):
                sel = select_mainshocks(cand[["event_id", "event_time", "t_sec", "latitude", "longitude", "magnitude", "place"]])
                self.mainshocks = sel[sel.is_mainshock][MS_COLS].reset_index(drop=True)
                self.ever_mainshock_ids = set(self.mainshocks.event_id)

    def ingest(self, batch: pd.DataFrame) -> None:
        batch = batch.sort_values("t_sec")
        self.buffer = pd.concat([self.buffer, batch[EVENT_COLS]], ignore_index=True).sort_values("t_sec")
        new_ms = []
        for r in batch[batch.magnitude >= self.cfg.mainshock_min_mag].itertuples():
            dependent = False
            for a in list(self.mainshocks.itertuples()) + new_ms:
                if (a.magnitude > r.magnitude and 0 < r.t_sec - a.t_sec <= a.window_days * 86400.0
                        and haversine_km_np(a.latitude, a.longitude, r.latitude, r.longitude) <= a.radius_km):
                    dependent = True
                    break
            if not dependent:
                new_ms.append(pd.Series({"event_id": r.event_id, "event_time": r.event_time, "t_sec": r.t_sec,
                                         "latitude": r.latitude, "longitude": r.longitude, "magnitude": r.magnitude,
                                         "place": r.place, "window_days": float(gk_window_days_np(r.magnitude)),
                                         "radius_km": float(gk_radius_km_np(r.magnitude))}))
        if new_ms:
            self.ever_mainshock_ids |= {m.event_id for m in new_ms}
            self.mainshocks = pd.concat([self.mainshocks, pd.DataFrame(new_ms)[MS_COLS]], ignore_index=True)
        tmax = self.buffer.t_sec.max()
        self.buffer = self.buffer[self.buffer.t_sec >= tmax - self.keep_sec].reset_index(drop=True)
        # mainshocks whose window has expired can no longer label anything
        if len(self.mainshocks):
            alive = self.mainshocks.t_sec + self.mainshocks.window_days * 86400.0 >= tmax - 1.0
            self.mainshocks = self.mainshocks[alive].reset_index(drop=True)


class StreamScorer:
    def __init__(self, spark: SparkSession, cfg, pipeline: PipelineModel, zone_model, state: StreamState) -> None:
        self.spark, self.cfg, self.pipeline, self.zone_model, self.state = spark, cfg, pipeline, zone_model, state
        self.latency: List[Dict[str, float]] = []
        self.scores: List[pd.DataFrame] = []

    def process_batch(self, batch: pd.DataFrame, batch_id: int = 0) -> pd.DataFrame:
        """Score the trigger events (M >= trigger_min_mag) of one micro-batch."""
        t0 = time.perf_counter()
        batch = batch[batch.magnitude >= self.cfg.mc]
        if batch.empty:
            return pd.DataFrame(columns=["event_id", "score", "batch_id"])
        self.state.ingest(batch)
        trig = batch[batch.magnitude >= self.cfg.trigger_min_mag]
        if trig.empty:
            return pd.DataFrame(columns=["event_id", "score", "batch_id"])
        spark = self.spark
        pool = pandas_to_events(spark, self.state.buffer)
        trig_sdf = pandas_to_events(spark, trig)
        ms = self.state.mainshocks
        ms_sdf = mainshocks_to_spark(spark, ms) if len(ms) else None
        grid = choose_grid_deg(ms.radius_km.values, ms.latitude.values) if len(ms) else None
        feats = fe.build_feature_table(spark, trig_sdf, pool, ms_sdf, grid, self.cfg, self.zone_model)
        t1 = time.perf_counter()
        scored = self.pipeline.transform(feats.withColumn("label", F.lit(0.0)))
        out = scored.select("event_id", vector_to_array("probability")[1].alias("score")).toPandas()
        out["batch_id"] = batch_id
        self.latency.append({"batch_id": batch_id, "events": len(batch), "triggers": len(trig),
                             "feature_seconds": t1 - t0, "predict_seconds": time.perf_counter() - t1})
        self.scores.append(out)
        return out


def load_saved_models(models_dir: str):
    return (PipelineModel.load(os.path.join(models_dir, "pipeline_C")),
            KMeansModel.load(os.path.join(models_dir, "zone_kmeans")))


JSON_SCHEMA = T.StructType([
    T.StructField("event_id", T.StringType()), T.StructField("event_time", T.StringType()),
    T.StructField("latitude", T.DoubleType()), T.StructField("longitude", T.DoubleType()),
    T.StructField("depth_km", T.DoubleType()), T.StructField("magnitude", T.DoubleType()),
    T.StructField("place", T.StringType())])


def replay_stream(spark: SparkSession, cfg, scorer: StreamScorer, window_events: pd.DataFrame,
                  work_dir: str, start_ts: pd.Timestamp) -> pd.DataFrame:
    """Replay historical events as a file stream (one JSON file ~ one micro-batch) and score them live.

    window_events needs event_id, event_time (UTC timestamps), latitude, longitude, depth_km, magnitude, place.
    Returns all streamed scores.
    """
    in_dir, ckpt = os.path.join(work_dir, "input"), os.path.join(work_dir, "checkpoint")
    shutil.rmtree(work_dir, ignore_errors=True)
    os.makedirs(in_dir, exist_ok=True)
    stream = spark.readStream.schema(JSON_SCHEMA).option("maxFilesPerTrigger", 1).json(in_dir)

    def handle(batch_df: DataFrame, batch_id: int) -> None:
        pdf = batch_df.toPandas()
        if pdf.empty:
            return
        pdf["event_time"] = pd.to_datetime(pdf["event_time"], utc=True)
        pdf["t_sec"] = (pdf["event_time"] - pd.Timestamp("1970-01-01", tz="UTC")).dt.total_seconds()
        scorer.process_batch(pdf, batch_id)

    query = (stream.writeStream.foreachBatch(handle).option("checkpointLocation", ckpt)
             .trigger(processingTime="1 second").start())
    ev = window_events.sort_values("event_time").copy()
    tick = (ev.event_time - start_ts).dt.total_seconds() // (cfg.tick_sim_hours * 3600)
    for k, chunk in ev.groupby(tick.astype(int)):
        out = chunk.copy()
        out["event_time"] = out.event_time.dt.strftime("%Y-%m-%dT%H:%M:%S.%f").str[:-3] + "Z"
        tmp, fin = os.path.join(in_dir, f".s{k:05d}.tmp"), os.path.join(in_dir, f"slice_{k:05d}.json")
        out[["event_id", "event_time", "latitude", "longitude", "depth_km", "magnitude", "place"]].to_json(tmp, orient="records", lines=True)
        os.replace(tmp, fin)
        time.sleep(cfg.tick_real_seconds)
    query.processAllAvailable()
    query.stop()
    if query.exception():
        raise RuntimeError(f"streaming query failed: {query.exception()}")
    return pd.concat(scorer.scores, ignore_index=True) if scorer.scores else pd.DataFrame(columns=["event_id", "score", "batch_id"])


def compare_stream_to_batch(stream_scores: pd.DataFrame, batch_scores: pd.DataFrame) -> Dict[str, float]:
    """Max / mean absolute difference between streaming and batch probabilities for the same events."""
    m = stream_scores.merge(batch_scores, on="event_id", suffixes=("_stream", "_batch"))
    d = (m.score_stream - m.score_batch).abs()
    return {"n_stream": len(stream_scores), "n_batch": len(batch_scores), "n_compared": len(m),
            "max_abs_diff": float(d.max()) if len(d) else float("nan"),
            "mean_abs_diff": float(d.mean()) if len(d) else float("nan")}


def evaluate_replay(scores: pd.DataFrame, truth: pd.DataFrame, top_mags: int = 5) -> Dict[str, object]:
    """Honest summary of a replay, added in v2.

    scores: event_id, score.  truth: event_id, y (hindsight label from the full catalog), magnitude.
    A replay is a handful of events from ONE sequence, usually with a very high share of positives, so its average
    precision is dominated by the base rate and is NOT a model validation. The summary therefore always reports the base
    rate next to AP, flags the replay as uninformative when the base rate is above 0.5 or there are fewer than 20 events
    of either class, and lists the scores of the largest events (the interesting cases).
    """
    from sklearn.metrics import average_precision_score, roc_auc_score
    d = scores.merge(truth, on="event_id")
    n, pos = len(d), int(d.y.sum())
    out: Dict[str, object] = {"n_events": n, "n_positive": pos, "base_rate": float(d.y.mean()) if n else float("nan")}
    both = d.y.nunique() == 2
    out["average_precision"] = float(average_precision_score(d.y, d.score)) if both else float("nan")
    out["roc_auc"] = float(roc_auc_score(d.y, d.score)) if both else float("nan")
    out["informative"] = bool(both and 0.05 <= out["base_rate"] <= 0.5 and min(pos, n - pos) >= 20)
    out["warning"] = None if out["informative"] else (
        f"base rate {out['base_rate']:.2f} with {pos} positives of {n} events: AP mostly reflects the base rate; "
        "this replay demonstrates the streaming mechanics, not predictive skill")
    out["largest_events"] = (d.sort_values("magnitude", ascending=False).head(top_mags)[["event_id", "magnitude", "score", "y"]]
                              .round(3).to_dict("records"))
    return out
