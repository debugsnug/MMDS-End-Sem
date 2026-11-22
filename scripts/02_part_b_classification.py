"""Part B: do cluster-derived features improve classification of 'followed by a larger event nearby'?

Steps: G-K mainshocks (for the G-K flag) -> causal feature table + label -> label audit (written BEFORE any
modelling) -> time split with gaps -> zone selection, tuning, final fits, scoring, bootstrap (modeling.run_part_b).

Usage:  python scripts/02_part_b_classification.py [--config ...] [--set models='["lr","rf"]']
Resumable: finished runs are skipped on restart (<work_dir>/part_b/run_log.csv).
This is statistical classification on a catalog, not earthquake prediction.
"""
import argparse
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from pyspark.sql import functions as F                              # noqa: E402

from seismic import features as fe                                   # noqa: E402
from seismic import gk, modeling                                     # noqa: E402
from seismic.common import banner, get_spark                         # noqa: E402
from seismic.config import add_common_args, cfg_from_args            # noqa: E402
from seismic.data import read_events                                 # noqa: E402


def main() -> None:
    ap = add_common_args(argparse.ArgumentParser(description=__doc__))
    cfg = cfg_from_args(ap.parse_args())
    out = os.path.join(cfg.work_dir, "part_b")
    os.makedirs(out, exist_ok=True)
    spark = get_spark(cfg, "part-b")

    events = read_events(spark, os.path.join(cfg.work_dir, "events.parquet")).persist()
    ev_mc = events.filter(F.col("magnitude") >= cfg.mc).persist()
    catalog_end = events.agg(F.max("t_sec")).first()[0]

    feat_path = os.path.join(out, "features.parquet")
    if os.path.exists(feat_path):
        feats = spark.read.parquet(feat_path)
        print("reusing", feat_path)
    else:
        banner("building labels and causal features")
        mainshocks, _, grid = gk.run_gk_causal(spark, events, cfg.mainshock_min_mag)
        trig = fe.trigger_events(ev_mc, cfg)
        feats = fe.build_feature_table(spark, trig, ev_mc, mainshocks, grid, cfg)
        feats = feats.join(fe.add_labels(trig, cfg).select("event_id", "y"), "event_id")
        feats.write.mode("overwrite").parquet(feat_path)
        feats = spark.read.parquet(feat_path)
    feats = feats.persist()

    banner("label audit (frozen before modelling)")
    train, val, test = fe.time_split(feats, cfg, catalog_end)
    audit = {"label_definition": f"strictly larger event within {cfg.label_horizon_days} days and {cfg.label_radius_km} km",
             "trigger_min_mag": cfg.trigger_min_mag, "total_triggers": feats.count(),
             "overall_positive_rate": feats.agg(F.avg("y")).first()[0]}
    for name, d in (("train", train), ("val", val), ("test", test)):
        r = d.agg(F.count("*").alias("n"), F.sum("y").alias("pos"), F.avg("y").alias("rate")).first()
        audit[name] = {"n": r["n"], "positives": int(r["pos"] or 0), "base_rate": r["rate"]}
    print(json.dumps(audit, indent=2, default=str))
    json.dump(audit, open(os.path.join(out, "label_audit.json"), "w"), indent=2, default=str)
    if min(audit[s]["positives"] for s in ("train", "val", "test")) < 20:
        print("WARNING: very few positives in a split; results will be noisy. Consider mc/trigger_min_mag or split dates.")

    # v2: nearest-neighbour cluster ids from Part A (if available) drive the cluster bootstrap: events of one sequence are
    # not independent, so resampling single events makes the interval too narrow. Without them, calendar blocks are used.
    group_map = None
    labels_path = os.path.join(cfg.work_dir, "part_a", "labels.parquet")
    if os.path.exists(labels_path):
        gm = pd.read_parquet(labels_path, columns=["event_id", "nn_cluster"])
        group_map = gm.set_index("event_id").nn_cluster
        print(f"bootstrap groups: NN clusters from {labels_path}")
    else:
        print(f"{labels_path} not found: the bootstrap will resample {cfg.bootstrap_block_days:g}-day blocks instead of clusters "
              "(run scripts/01_part_a_clustering.py first for the cluster bootstrap)")

    banner("training, tuning and evaluating (resumable)")
    result = modeling.run_part_b(spark, train, val, test, cfg, out, group_map=group_map)
    table = modeling.summary_table(os.path.join(out, "run_log.csv"))
    table.to_csv(os.path.join(out, "results_table.csv"), index=False)
    print(table.to_string(index=False))
    print("\nPrimary comparison (set C vs set B, best classifier on validation for set B):")
    print(json.dumps(result.get("primary_C_vs_B"), indent=2))
    print(f"Chance level for PR-AUC on the test set = base rate = {result.get('base_rate_test')}")

    banner("what does the label measure? (diagnostics, v2)")
    from seismic import diagnostics as dg
    from seismic.common import utc_epoch
    fpd = pd.read_parquet(feat_path)
    diag = dg.part_b_diagnostics(fpd, utc_epoch(cfg.train_end), utc_epoch(cfg.val_end), catalog_end, cfg.label_horizon_days, group_map)
    os.makedirs(os.path.join(out, "diagnostics"), exist_ok=True)
    for k, v in diag.items():
        if isinstance(v, pd.DataFrame):
            v.to_csv(os.path.join(out, "diagnostics", f"{k}.csv"), index=False); print(f"\n{k}\n{v.round(4).to_string(index=False)}")
    json.dump({k: v for k, v in diag.items() if not isinstance(v, pd.DataFrame)}, open(os.path.join(out, "diagnostics", "summary.json"), "w"), indent=2, default=float)
    print({k: v for k, v in diag.items() if not isinstance(v, pd.DataFrame)})


if __name__ == "__main__":
    main()
