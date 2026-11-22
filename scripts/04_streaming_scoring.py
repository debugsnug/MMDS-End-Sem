"""Streaming: replay a historical sequence and score arriving events with the saved set-C PipelineModel.

Needs the models saved by Part B (<work_dir>/part_b/models). Then compares streaming probabilities with batch
probabilities for the same events (a consistency check: both paths share the feature code).

Usage:  python scripts/04_streaming_scoring.py [--preset ridgecrest_2019] [--config ...]
Outputs (<work_dir>/streaming/<preset>/): stream_scores.csv, batch_scores.csv, latency.csv, comparison.json
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
    preset_key = args.preset or cfg.replay_name
    preset = REPLAY_PRESETS[preset_key]
    out = os.path.join(cfg.work_dir, "streaming", preset_key)          # v2: one folder per replayed sequence
    os.makedirs(out, exist_ok=True)
    spark = get_spark(cfg, "streaming")

    models_dir = os.path.join(cfg.work_dir, "part_b", "models")
    if not os.path.isdir(models_dir):
        sys.exit(f"No saved models in {models_dir}. Run scripts/02_part_b_classification.py first.")
    pipeline, zone_model = streaming.load_saved_models(models_dir)
    meta_path = os.path.join(models_dir, "meta.json")
    meta = json.load(open(meta_path)) if os.path.exists(meta_path) else {}
    print(f"streaming with the saved set-C pipeline of model '{meta.get('model', 'unknown')}' (zone k = {meta.get('zone_k')})")
    pp = os.path.join(cfg.work_dir, "part_b", "primary_comparison.json")
    if os.path.exists(pp) and meta.get("model") and json.load(open(pp)).get("best_model") != meta["model"]:
        print("WARNING: the saved pipeline is not the best model of the primary comparison; re-run scripts/02_part_b_classification.py")

    events = read_events(spark, os.path.join(cfg.work_dir, "events.parquet")).persist()
    ev_mc = events.filter(F.col("magnitude") >= cfg.mc).persist()
    start = pd.Timestamp(preset["start"], tz="UTC")
    end = start + pd.to_timedelta(preset["days"], unit="D")
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
    # v2 fix: the stream only ever sees the history before the replay plus the replayed events (here: those inside the replay
    # radius), so the batch reference must use exactly that pool and the mainshocks found in it. Using the global catalog made
    # events near the region border differ slightly (neighbours outside the radius), which is a property of the replay region,
    # not of the streaming state handling.
    from seismic.geo import choose_grid_deg
    from seismic.gk import mainshocks_to_spark, select_mainshocks
    window_ids = list(window.event_id)
    visible = ev_mc.filter((F.col("t_sec") < start.timestamp()) | F.col("event_id").isin(window_ids))
    cand = (visible.filter(F.col("magnitude") >= cfg.mainshock_min_mag)
                   .select("event_id", "event_time", "t_sec", "latitude", "longitude", "magnitude", "place").toPandas())
    sel = select_mainshocks(cand)
    ms_pdf = sel[sel.is_mainshock]
    ms = mainshocks_to_spark(spark, ms_pdf)
    grid = choose_grid_deg(ms_pdf.radius_km.values, ms_pdf.latitude.values)
    trig = fe.trigger_events(ev_mc, cfg).filter((F.col("t_sec") >= start.timestamp()) & (F.col("t_sec") < end.timestamp())
                                                & F.col("event_id").isin(window_ids))
    bfeat = fe.build_feature_table(spark, trig, visible, ms, grid, cfg, zone_model).withColumn("label", F.lit(0.0))
    batch = (pipeline.transform(bfeat).select("event_id", vector_to_array("probability")[1].alias("score")).toPandas())
    batch.to_csv(os.path.join(out, "batch_scores.csv"), index=False)

    cmp = streaming.compare_stream_to_batch(stream_scores, batch)
    lat = pd.DataFrame(scorer.latency)
    cmp.update({"micro_batches": int(len(lat)), "median_seconds_per_batch": float((lat.feature_seconds + lat.predict_seconds).median()) if len(lat) else None,
                "max_seconds_per_batch": float((lat.feature_seconds + lat.predict_seconds).max()) if len(lat) else None})
    cmp["model"] = meta.get("model")
    json.dump(cmp, open(os.path.join(out, "comparison.json"), "w"), indent=2)
    print(json.dumps(cmp, indent=2))
    print("Note: stream and batch share the feature code, so agreement shows correct state handling, not model quality.")

    banner("replay evaluation against hindsight labels (v2)")
    truth = (fe.add_labels(fe.trigger_events(ev_mc, cfg), cfg).filter(F.col("event_id").isin(list(window.event_id)))
             .select("event_id", "y", "magnitude").toPandas())
    rep = streaming.evaluate_replay(stream_scores, truth)
    json.dump(rep, open(os.path.join(out, "replay_evaluation.json"), "w"), indent=2, default=float)
    stream_scores.merge(truth, on="event_id").sort_values("magnitude", ascending=False).to_csv(os.path.join(out, "stream_scores_labeled.csv"), index=False)
    print(json.dumps(rep, indent=2, default=float))
    if rep["warning"]:
        print("WARNING:", rep["warning"])


if __name__ == "__main__":
    main()
