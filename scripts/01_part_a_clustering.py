"""Part A: clustering-based aftershock identification vs the Gardner-Knopoff baseline.

Outputs (in <work_dir>/part_a/): part_a_summary.json, overlap/bath/omori/dispersion/sensitivity/zone/case-study CSVs,
labels.parquet (per-event G-K and NN labels) and three PNG charts.

Usage:  python scripts/01_part_a_clustering.py [--dbscan] [--config ...] [--set mc=4.5]
There is no ground truth: every number below is agreement or plausibility, never 'accuracy'.
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from pyspark.sql import functions as F                              # noqa: E402

from seismic import evaluation as ev_                                # noqa: E402
from seismic import gk, zones                                        # noqa: E402
from seismic.common import StageTimer, banner, get_spark            # noqa: E402
from seismic.config import REPLAY_PRESETS, add_common_args, cfg_from_args  # noqa: E402
from seismic.data import columns_for_clustering, read_events         # noqa: E402
from seismic.dbscan_ref import st_dbscan_reference                   # noqa: E402
from seismic.geo import haversine_km_np                              # noqa: E402
from seismic.nn_cluster import nn_cluster_events                     # noqa: E402

# Southern California box used for the regional Poisson check (declustering is judged regionally)
SOCAL_BBOX = (32.0, 37.0, -121.0, -114.0)


def main() -> None:
    ap = add_common_args(argparse.ArgumentParser(description=__doc__))
    ap.add_argument("--dbscan", action="store_true", help="also run the single-machine DBSCAN reference")
    args = ap.parse_args()
    cfg = cfg_from_args(args)
    out = os.path.join(cfg.work_dir, "part_a")
    os.makedirs(out, exist_ok=True)
    spark = get_spark(cfg, "part-a")
    timer, summary = StageTimer(), {"config": cfg.to_dict()}

    events = read_events(spark, os.path.join(cfg.work_dir, "events.parquet")).persist()
    ev_mc = events.filter(F.col("magnitude") >= cfg.mc).persist()
    summary["n_events_all"], summary["n_events_mc"] = events.count(), ev_mc.count()
    print(f"events: {summary['n_events_all']:,} total, {summary['n_events_mc']:,} with M >= {cfg.mc}")

    # ---------------------------------------------------------------- G-K baseline (all magnitudes, as in Assignment 2)
    banner("Gardner-Knopoff baseline")
    with timer.stage("gk"):
        mainshocks, after, grid = gk.run_gk(spark, events, cfg.mainshock_min_mag)
        gk_all = gk.gk_label_table(events, after)
        summary["gk_mainshocks"], summary["gk_aftershocks_all_mags"] = mainshocks.count(), after.count()
    gk_mc = gk_all.join(ev_mc.select("event_id"), "event_id")          # restrict to the clustering catalog (M >= mc)

    # ---------------------------------------------------------------- nearest-neighbour clustering
    banner("nearest-neighbour clustering (Spark)")
    with timer.stage("nn_total"):
        nn_lab, nn, eta0, eta_info = nn_cluster_events(spark, columns_for_clustering(ev_mc), cfg)
        nn_lab = nn_lab.persist()
        summary["eta0_log10"], summary["eta0_info"] = eta0, eta_info
        summary["nn_dependent"] = nn_lab.filter("nn_is_dependent").count()
    print(f"eta0 (log10) = {eta0:.2f} via {eta_info.get('method')}; dependent events: {summary['nn_dependent']:,}")

    labels = (ev_mc.select("event_id", "event_time", "t_sec", "latitude", "longitude", "magnitude")
                   .join(gk_mc.select("event_id", "gk_is_aftershock", "gk_cluster"), "event_id")
                   .join(nn_lab, "event_id").persist())
    labels.write.mode("overwrite").parquet(os.path.join(out, "labels.parquet"))

    # ---------------------------------------------------------------- overlap with G-K (NOT accuracy)
    banner("agreement with G-K")
    overlap = ev_.flag_overlap(labels, "gk_is_aftershock", "nn_is_dependent")
    overlap["ari_partitions"] = ev_.adjusted_rand_index(labels, "gk_cluster", "nn_cluster")
    summary["overlap_gk_vs_nn"] = overlap
    pd.DataFrame([overlap]).to_csv(os.path.join(out, "overlap.csv"), index=False)
    print(json.dumps(overlap, indent=2))

    # ---------------------------------------------------------------- physical regularities
    banner("Bath's law and Omori decay")
    bath, omori = {}, {}
    for name, col in (("gk", "gk_cluster"), ("nn", "nn_cluster")):
        tab = ev_.bath_table(labels, col, min_size=5)
        tab.to_csv(os.path.join(out, f"bath_{name}.csv"), index=False)
        bath[name] = ev_.bath_summary(tab)
        om = ev_.omori_table(labels, col, top_k=20)
        om.to_csv(os.path.join(out, f"omori_{name}.csv"), index=False)
        omori[name] = {"median_p": float(om.p.median()), "n_fitted": int(om.p.notna().sum())}
    summary["bath"], summary["omori"] = bath, omori
    print(json.dumps({"bath": bath, "omori": omori}, indent=2))

    banner("Poisson-likeness of the background")
    rows = []
    for region, bbox in (("global", None), ("socal", SOCAL_BBOX)):
        for name, df in (("raw", labels), ("gk_declustered", labels.filter(~F.col("gk_is_aftershock"))),
                         ("nn_declustered", labels.filter(~F.col("nn_is_dependent")))):
            rows.append({"region": region, "catalog": name, **ev_.dispersion_stats(df, bin_days=1.0, bbox=bbox)})
    disp = pd.DataFrame(rows)
    disp.to_csv(os.path.join(out, "dispersion.csv"), index=False)
    summary["dispersion"] = disp.to_dict("records")
    print(disp.to_string(index=False))

    # ---------------------------------------------------------------- sensitivity
    banner("sensitivity to eta0")
    sens = ev_.sensitivity_eta0(labels.join(nn.select("event_id", "log10_eta"), "event_id"), cfg.eta0_sweep)
    sens.to_csv(os.path.join(out, "sensitivity_eta0.csv"), index=False)
    print(sens.to_string(index=False))

    # ---------------------------------------------------------------- MLlib zones (descriptive)
    banner("MLlib zone models (KMeans / BisectingKMeans / GMM)")
    with timer.stage("zones"):
        ztab = zones.compare_zone_models(ev_mc.select("latitude", "longitude"), cfg.zone_kinds, cfg.zone_k_candidates,
                                         seed=1, sample=cfg.silhouette_sample)
    ztab.to_csv(os.path.join(out, "zone_models.csv"), index=False)
    print(ztab.to_string(index=False))

    # ---------------------------------------------------------------- case studies
    banner("case studies")
    cases = []
    for key, p in REPLAY_PRESETS.items():
        t0 = pd.Timestamp(p["start"], tz="UTC").timestamp() - 30 * 86400
        t1 = t0 + (30 + 365) * 86400
        win = (labels.filter((F.col("t_sec") >= t0) & (F.col("t_sec") <= t1)).toPandas())
        if len(win):
            d = haversine_km_np(p["center"][0], p["center"][1], win.latitude.values, win.longitude.values)
            win = win[d <= p["radius_km"]]
        cases.append({"preset": key, **ev_.case_study(win, p)})
    pd.DataFrame(cases).to_csv(os.path.join(out, "case_studies.csv"), index=False)
    print(pd.DataFrame(cases).to_string(index=False))
    summary["case_studies"] = cases

    # ---------------------------------------------------------------- optional DBSCAN reference (single machine)
    if args.dbscan:
        banner("DBSCAN reference (single machine, NOT distributed)")
        with timer.stage("dbscan"):
            pdf = labels.select("event_id", "t_sec", "latitude", "longitude", "magnitude", "gk_is_aftershock", "nn_is_dependent").toPandas()
            db = st_dbscan_reference(pdf, cfg.dbscan_eps_km, cfg.dbscan_eps_days, cfg.dbscan_min_samples)
        a, b = db.db_is_dependent, db.gk_is_aftershock
        c = db.nn_is_dependent
        jac = lambda x, y: float((x & y).sum() / max(1, (x | y).sum()))  # noqa: E731
        summary["dbscan_reference"] = {"dependent": int(a.sum()), "jaccard_vs_gk": jac(a, b), "jaccard_vs_nn": jac(a, c)}
        print(summary["dbscan_reference"])

    summary["timings_seconds"] = timer.seconds
    _charts(out, nn, eta0, labels, sens)
    with open(os.path.join(out, "part_a_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print("\nPart A finished. Results in", out)


def _charts(out, nn, eta0, labels, sens) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    eta = np.array([r[0] for r in nn.filter(F.col("log10_eta").isNotNull()).select("log10_eta").collect()])
    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.hist(eta, bins=60, color="#4c78a8")
    ax.axvline(eta0, color="crimson", ls="--", label=f"eta0 = {eta0:.2f}")
    ax.set(xlabel="log10 nearest-neighbour proximity eta", ylabel="events", title="Two modes: clustered vs background")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out, "eta_histogram.png"), dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.plot(sens.eta0_log10, sens.share_dependent * 100, marker="o", label="share dependent (%)")
    ax2 = ax.twinx()
    ax2.plot(sens.eta0_log10, sens.jaccard_vs_gk, marker="s", color="darkorange", label="Jaccard vs G-K")
    ax.set(xlabel="eta0 (log10)", ylabel="% events labeled dependent", title="Sensitivity to the threshold")
    ax2.set_ylabel("Jaccard vs G-K")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "sensitivity_eta0.png"), dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 3.6))
    for name, col in (("G-K", "gk"), ("NN", "nn")):
        p = os.path.join(out, f"bath_{col}.csv")
        if os.path.exists(p):
            d = pd.read_csv(p)
            if len(d):
                ax.hist(d.delta_m, bins=20, alpha=0.55, label=f"{name} (mean {d.delta_m.mean():.2f})")
    ax.axvline(1.2, color="k", ls=":", label="Bath reference 1.2")
    ax.set(xlabel="mainshock minus largest aftershock (magnitude units)", ylabel="clusters", title="Bath's law")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out, "bath_law.png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
