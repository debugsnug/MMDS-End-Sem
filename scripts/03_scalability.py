"""Scalability experiments: catalog size, cores, shuffle partitions, salting, pair reduction, skew.

Usage:  python scripts/03_scalability.py [--config ...] [--set scale_cores=[1,2] --set stress_factor=1]
Each core count runs in a fresh process (a JVM cannot change its core count), then results are merged
into <work_dir>/scalability/scalability_log.csv (resumable) and scalability_summary.csv / scaling.png.
v2: every configuration is timed cfg.scale_repeats times (median plus min/max), pair counts include the pairs that
survive the exact test, and replicated-catalog (stress) runs are part of the plan (cfg.scale_stress, runtime only).
The scalability runs use the FIXED threshold cfg.nn_eta0_log10, not the estimated one of Part A, so the number of
dependent events differs from Part A by design.
Local mode only: speed-up is capped by the cores of the machine (Colab usually has 2), and runtimes
say nothing about a real cluster. stress_factor > 1 tiles the catalog for a clearly labelled systems
stress test (runtime only; never use it for accuracy results).
"""
import argparse
import os
import subprocess
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))

from seismic import scalability                                       # noqa: E402
from seismic.common import banner, get_spark                          # noqa: E402
from seismic.config import add_common_args, cfg_from_args             # noqa: E402
from seismic.data import read_events, replicate_catalog               # noqa: E402
from seismic.runlog import RunLog                                     # noqa: E402


LOG_NAME = "scalability_log_v2.csv"          # v2 log: repeated timings, exact-pair counts, stress runs (v1 log is left untouched)


def worker(cfg, cores: int) -> None:
    out = os.path.join(cfg.work_dir, "scalability")
    os.makedirs(out, exist_ok=True)
    spark = get_spark(cfg, f"scale-{cores}", master=f"local[{cores}]", aqe=False)   # AQE off: keep partitions meaningful
    base = replicate_catalog(read_events(spark, os.path.join(cfg.work_dir, "events.parquet")), cfg.stress_factor)
    stress_cache = {1: base}
    log = RunLog(os.path.join(out, LOG_NAME))
    plan = [p for p in scalability.experiment_plan(cfg) if p["cores"] == cores]
    todo = [p for p in plan if not log.has(scalability.run_id(p, cfg.stress_factor * p["stress"]))]
    if todo:                    # warm-up with the SAME stages (G-K, NN, roots), not logged, so the first logged run is not slower
        scalability.run_one(spark, base, cfg, min(cfg.scale_fractions), cfg.shuffle_partitions, 1, with_pairs=False, with_gk=True)
    for p in todo:
        k = p["stress"]
        if k not in stress_cache:
            stress_cache[k] = replicate_catalog(base, k)
        rid = scalability.run_id(p, cfg.stress_factor * k)
        # G-K is skipped for replicated catalogs: it is a driver-side O(candidates^2) step that would swamp the NN timings
        row = scalability.run_one(spark, stress_cache[k], cfg, p["fraction"], p["partitions"], p["salt"],
                                  repeats=cfg.scale_repeats, with_gk=(k == 1))
        row.update({"run_id": rid, "exp": p["exp"], "cores": cores, "stress_factor": cfg.stress_factor * k, "master": f"local[{cores}]"})
        log.append(row)
        print(f"[cores={cores}] {p['exp']:<10} frac={p['fraction']:<4} part={p['partitions']:<3} salt={p['salt']} x{k} "
              f"-> NN stages {row['sec_nn_total']:.1f}s; total median {row['sec_total']:.1f}s (min {row['sec_total_min']:.1f}, max {row['sec_total_max']:.1f}), "
              f"blocked {row['pairs_blocked_pct_of_naive']:.3f}% / exact {row['pairs_exact_pct_of_naive']:.4f}% of naive pairs")


def main() -> None:
    ap = add_common_args(argparse.ArgumentParser(description=__doc__))
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--cores", type=int, default=None)
    args = ap.parse_args()
    cfg = cfg_from_args(args)
    if args.worker:
        worker(cfg, args.cores)
        return

    banner("scalability experiments")
    passthrough = (["--config", args.config] if args.config else []) + [x for s in args.sets for x in ("--set", s)]
    for c in cfg.scale_cores:
        subprocess.run([sys.executable, os.path.abspath(__file__), "--worker", "--cores", str(c)] + passthrough, check=True)

    out = os.path.join(cfg.work_dir, "scalability")
    df = pd.read_csv(os.path.join(out, LOG_NAME))
    cols = ["exp", "cores", "fraction", "partitions", "salt", "stress_factor", "repeats", "n_events_mc", "pairs_blocked",
            "pairs_exact", "pairs_naive", "pairs_blocked_pct_of_naive", "pairs_exact_pct_of_naive", "max_over_mean",
            "sec_gk_baseline", "sec_nn_search", "sec_root_finding", "sec_nn_total", "sec_total", "sec_total_min", "sec_total_max", "eta0_log10_used"]
    summary = df[[c for c in cols if c in df.columns]].sort_values(["exp", "cores", "fraction", "partitions", "salt"])
    summary.to_csv(os.path.join(out, "scalability_summary_v2.csv"), index=False)
    print(summary.to_string(index=False))
    _plot(df, out)


def _plot(df: pd.DataFrame, out: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    size = df[(df.exp == "size")]
    # note: the 1.0 fraction of "size" and the stress runs share the 'full catalog' definition; stress runs are plotted separately
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    for c, g in size.groupby("cores"):
        g = g.sort_values("n_events_mc")
        ax[0].errorbar(g.n_events_mc, g.sec_total, yerr=[g.sec_total - g.sec_total_min, g.sec_total_max - g.sec_total], marker="o", capsize=3, label=f"{c} core(s)")
    ax[0].set(xlabel="events (M >= mc)", ylabel="seconds", title="Runtime vs catalog size (time windows)")
    ax[0].legend()
    top = size[size.fraction == size.fraction.max()].sort_values("cores")
    ax[1].bar(top.cores.astype(str), top.sec_total)
    ax[1].set(xlabel="cores", ylabel="seconds", title="Full catalog: runtime vs cores")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "scaling_v2.png"), dpi=150)


if __name__ == "__main__":
    main()
