"""Streaming: replay a historical sequence and score arriving events with the saved set-C PipelineModel.

Needs the models saved by Part B (<work_dir>/part_b/models). Then compares streaming probabilities with batch
probabilities for the same events (a consistency check: both paths share the feature code).

Usage:  python scripts/04_streaming_scoring.py [--preset ridgecrest_2019] [--config ...]
Outputs (<work_dir>/streaming/): stream_scores.csv, batch_scores.csv, latency.csv, comparison.json
"""
import argparse
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from pyspark.ml.functions import vector_to_array                    # noqa: E402
from pyspark.sql import functions as F                              # noqa: E402

from seismic import features as fe                                   # noqa: E402
from seismic import gk, streaming                                    # noqa: E402
from seismic.common import banner, get_spark                         # noqa: E402
from seismic.config import REPLAY_PRESETS, add_common_args, cfg_from_args  # noqa: E402
from seismic.data import read_events                                 # noqa: E402
from seismic.geo import haversine_km_np                              # noqa: E402


def to_pandas_events(sdf) -> pd.DataFrame:
    pdf = sdf.select("event_id", "event_time", "t_sec", "latitude", "longitude", "depth_km", "magnitude", "place").toPandas()
    pdf["event_time"] = pd.to_datetime(pdf["event_time"], utc=True)
    return pdf


def main() -> None:
    ap = add_common_args(argparse.ArgumentParser(description=__doc__))
    ap.add_argument("--preset", default=None, help=f"one of {list(REPLAY_PRESETS)} (default: config.replay_name)")
    args = ap.parse_args()
    cfg = cfg_from_args(args)
    preset = REPLAY_PRESETS[args.preset or cfg.replay_name]
    out = os.path.join(cfg.work_dir, "streaming")
    os.makedirs(out, exist_ok=True)
    spark = get_spark(cfg, "streaming")

    models_dir = os.path.join(cfg.work_dir, "part_b", "models")
    if not os.path.isdir(models_dir):
        sys.exit(f"No saved models in {models_dir}. Run scripts/02_part_b_classification.py first.")
    pipeline, zone_model = streaming.load_saved_models(models_dir)

    events = read_events(spark, os.path.join(cfg.work_dir, "events.parquet")).persist()
    ev_mc = events.filter(F.col("magnitude") >= cfg.mc).persist()
    start = pd.Timestamp(preset["start"], tz="UTC")
    end = start + pd.Timedelta(days=preset["days"])
    lat0, lon0 = preset["center"]

    # warm start: history before the replay window (buffer + active G-K mainshocks), as a live system would have it
    past = to_pandas_events(ev_mc.filter(F.col("t_sec") < start.timestamp()))
    state = streaming.StreamState(cfg, past)
    window_all = to_pandas_events(ev_mc.filter((F.col("t_sec") >= start.timestamp()) & (F.col("t_sec") < end.timestamp())))
    d = haversine_km_np(lat0, lon0, window_all.latitude.values, window_all.longitude.values)
    window = window_all[d <= preset["radius_km"]]
    print(f"replay '{preset['title']}': {len(window):,} events (M>={cfg.mc}) in {preset['days']} days; "
          f"{(window.magnitude >= cfg.trigger_min_mag).sum():,} triggers")

    banner("streaming")
    scorer = streaming.StreamScorer(spark, cfg, pipeline, zone_model, state)
    stream_scores = streaming.replay_stream(spark, cfg, scorer, window, os.path.join(out, "replay_tmp"), start)
    stream_scores.to_csv(os.path.join(out, "stream_scores.csv"), index=False)
    pd.DataFrame(scorer.latency).to_csv(os.path.join(out, "latency.csv"), index=False)

    banner("batch scoring of the same events (consistency check)")
    ms, _, grid = gk.run_gk(spark, events, cfg.mainshock_min_mag)
    trig = fe.trigger_events(ev_mc, cfg).filter((F.col("t_sec") >= start.timestamp()) & (F.col("t_sec") < end.timestamp())
                                                & F.col("event_id").isin(list(window.event_id)))
    bfeat = fe.build_feature_table(spark, trig, ev_mc, ms, grid, cfg, zone_model).withColumn("label", F.lit(0.0))
    batch = (pipeline.transform(bfeat).select("event_id", vector_to_array("probability")[1].alias("score")).toPandas())
    batch.to_csv(os.path.join(out, "batch_scores.csv"), index=False)

    cmp = streaming.compare_stream_to_batch(stream_scores, batch)
    lat = pd.DataFrame(scorer.latency)
    cmp.update({"micro_batches": int(len(lat)), "median_seconds_per_batch": float((lat.feature_seconds + lat.predict_seconds).median()) if len(lat) else None,
                "max_seconds_per_batch": float((lat.feature_seconds + lat.predict_seconds).max()) if len(lat) else None})
    json.dump(cmp, open(os.path.join(out, "comparison.json"), "w"), indent=2)
    print(json.dumps(cmp, indent=2))
    print("Note: stream and batch share the feature code, so agreement shows correct state handling, not model quality.")


if __name__ == "__main__":
    main()
