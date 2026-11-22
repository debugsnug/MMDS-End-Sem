"""Re-analysis from the SAVED outputs of a finished run: no Spark, no model fitting (runs in about a minute).

Answers the review questions that need nothing but the logged files:
  1. Bath's law stratified by mainshock magnitude (G-K vs NN on the same strata)
  2. like-for-like dependent counts (both restricted to M >= mc)
  3. what the Part B label measures: positive rate by recent activity, where the positives come from,
     simple baselines (n_7d only; logistic regression on magnitude and log activity)
  4. C vs B with an EVENT bootstrap and a CLUSTER bootstrap (events of one sequence are not independent)
  5. the streaming replay against hindsight labels (base rate, AP, scores of the largest events)

Usage:  python scripts/05_reanalysis.py [--config ...] [--set work_dir=/path/to/real_run]
Outputs: <work_dir>/reanalysis/*.csv and reanalysis_summary.json
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from seismic import diagnostics as dg                         # noqa: E402
from seismic.common import banner, utc_epoch                    # noqa: E402
from seismic.config import add_common_args, cfg_from_args       # noqa: E402


def main() -> None:
    ap = add_common_args(argparse.ArgumentParser(description=__doc__))
    ap.add_argument("--n-boot", type=int, default=None)
    args = ap.parse_args()
    cfg = cfg_from_args(args)
    n_boot = args.n_boot or cfg.n_boot
    out = os.path.join(cfg.work_dir, "reanalysis")
    os.makedirs(out, exist_ok=True)
    summary = {}

    # ------------------------------------------------------------------ 1-2. Part A labels
    labels_path = os.path.join(cfg.work_dir, "part_a", "labels.parquet")
    lab = pd.read_parquet(labels_path, columns=["event_id", "t_sec", "magnitude", "gk_is_aftershock", "gk_cluster", "nn_is_dependent", "nn_cluster"])
    banner("1. Bath's law by mainshock magnitude")
    strata = {}
    for name, col in (("gk", "gk_cluster"), ("nn", "nn_cluster")):
        t = dg.bath_strata(lab, col)
        t.to_csv(os.path.join(out, f"bath_strata_{name}.csv"), index=False)
        strata[name] = t.to_dict("records")
    both = pd.DataFrame({"mainshock_min_mag": [r["mainshock_min_mag"] for r in strata["gk"]],
                         "G-K clusters": [r["n_clusters"] for r in strata["gk"]], "G-K mean gap": [round(r["mean_delta"], 3) for r in strata["gk"]],
                         "NN clusters": [r["n_clusters"] for r in strata["nn"]], "NN mean gap": [round(r["mean_delta"], 3) for r in strata["nn"]]})
    print(both.to_string(index=False)); print("reference value 1.2")
    both.to_csv(os.path.join(out, "bath_strata_side_by_side.csv"), index=False)
    summary["bath_strata"] = strata

    banner("2. like-for-like dependent counts (both restricted to M >= mc)")
    n = len(lab)
    like = {"events_M_ge_mc": n, "gk_aftershocks_mc": int(lab.gk_is_aftershock.sum()), "nn_dependent_mc": int(lab.nn_is_dependent.sum())}
    like["gk_share"], like["nn_share"] = like["gk_aftershocks_mc"] / n, like["nn_dependent_mc"] / n
    sa = os.path.join(cfg.work_dir, "part_a", "part_a_summary.json")
    if os.path.exists(sa):
        like["gk_aftershocks_all_magnitudes_NOT_comparable"] = json.load(open(sa)).get("gk_aftershocks_all_mags")
    print(json.dumps(like, indent=2)); summary["like_for_like"] = like

    # ------------------------------------------------------------------ 3-4. Part B
    feats = pd.read_parquet(os.path.join(cfg.work_dir, "part_b", "features.parquet"))
    sc_dir = os.path.join(cfg.work_dir, "part_b", "scores")
    prim_path = os.path.join(cfg.work_dir, "part_b", "primary_comparison.json")
    best = json.load(open(prim_path)).get("best_model", "gbt") if os.path.exists(prim_path) else "gbt"
    ref_files = sorted(glob.glob(os.path.join(sc_dir, f"B_B_{best}_seed*_w0_*.csv")))
    test_ids = pd.read_csv(ref_files[0]).event_id.values
    group_map = lab.set_index("event_id").nn_cluster
    diag = dg.part_b_diagnostics(feats, utc_epoch(cfg.train_end), utc_epoch(cfg.val_end), float(feats.t_sec.max()) + 7 * 86400,
                                 cfg.label_horizon_days, group_map, test_ids=test_ids)
    banner("3. what does the Part B label measure?")
    for k, v in diag.items():
        if isinstance(v, pd.DataFrame):
            v.to_csv(os.path.join(out, f"{k}.csv"), index=False); print(f"\n{k}\n{v.round(4).to_string(index=False)}")
    summary["part_b_diagnostics"] = {k: (v.round(5).to_dict("records") if isinstance(v, pd.DataFrame) else v) for k, v in diag.items()}

    banner(f"4. C vs B for {best.upper()}: event bootstrap vs cluster bootstrap")
    seeds = sorted({int(f.split("seed")[1].split("_")[0]) for f in ref_files})
    rd = lambda fs, s: pd.read_csv(glob.glob(os.path.join(sc_dir, f"B_{fs}_{best}_seed{s}_w0_*.csv"))[0])  # noqa: E731
    sB, sC = [rd("B", s) for s in seeds], [rd("C", s) for s in seeds]
    y, ids = sB[0].y.values, sB[0].event_id.values
    newS, baseS = np.vstack([c.score.values for c in sC]), np.vstack([b.score.values for b in sB])
    g, unit = dg.make_groups(ids, group_map, feats.set_index("event_id").t_sec.reindex(ids).values, cfg.bootstrap_block_days)
    ev = dg.paired_bootstrap_diff(y, newS, baseS, n_boot=n_boot, seed=cfg.master_seed)
    cl = dg.paired_bootstrap_diff(y, newS, baseS, n_boot=n_boot, seed=cfg.master_seed, groups=g)
    conc = dg.concentration(g, y, 5)
    tab = pd.DataFrame([{"bootstrap": "event (independent)", **{k: ev[k] for k in ("diff", "ci_low", "ci_high", "includes_zero")}},
                        {"bootstrap": f"{unit} (dependence respected)", **{k: cl[k] for k in ("diff", "ci_low", "ci_high", "includes_zero")}}])
    tab.to_csv(os.path.join(out, "c_vs_b_bootstrap.csv"), index=False)
    print(tab.round(4).to_string(index=False)); print("top-5 clusters:", conc)
    summary["c_vs_b"] = {"model": best, "seeds": seeds, "event": ev, "cluster": cl, "concentration_top5": conc}
    # all models, point estimates only
    rows = []
    for m in ("lr", "dt", "rf", "gbt"):
        a = {fs: np.mean([dg.average_precision_score(pd.read_csv(f).y, pd.read_csv(f).score) for f in glob.glob(os.path.join(sc_dir, f"B_{fs}_{m}_seed*_w0_*.csv"))]) for fs in "ABC"}
        rows.append({"model": m, "A": a["A"], "B": a["B"], "C": a["C"], "C_minus_B": a["C"] - a["B"]})
    ap_tab = pd.DataFrame(rows); ap_tab.to_csv(os.path.join(out, "ap_by_model_and_set.csv"), index=False)
    print(ap_tab.round(4).to_string(index=False))

    # ------------------------------------------------------------------ 5. streaming replay
    from seismic.streaming import evaluate_replay
    paths = sorted(glob.glob(os.path.join(cfg.work_dir, "streaming", "*", "stream_scores.csv")))
    old = os.path.join(cfg.work_dir, "streaming", "stream_scores.csv")           # v1 layout
    if os.path.exists(old):
        paths.append(old)
    summary["replay"] = {}
    for sp in paths:
        name = os.path.basename(os.path.dirname(sp))
        banner(f"5. streaming replay '{name}' against hindsight labels")
        rep = evaluate_replay(pd.read_csv(sp)[["event_id", "score"]], feats[["event_id", "y", "magnitude"]])
        json.dump(rep, open(os.path.join(out, f"replay_evaluation_{name}.json"), "w"), indent=2, default=float)
        print(json.dumps(rep, indent=2, default=float)); summary["replay"][name] = rep
    json.dump(summary, open(os.path.join(out, "reanalysis_summary.json"), "w"), indent=2, default=float)
    print("\nwritten:", out)


if __name__ == "__main__":
    main()
