"""Scalability experiments: catalog size, cores, shuffle partitions, salting, pair reduction, skew.

Usage:  python scripts/03_scalability.py [--config ...] [--set scale_cores=[1,2] --set stress_factor=1]
Each core count runs in a fresh process (a JVM cannot change its core count), then results are merged
into <work_dir>/scalability/scalability_log.csv (resumable) and scalability_summary.csv / scaling.png.
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


def worker(cfg, cores: int) -> None:
    out = os.path.join(cfg.work_dir, "scalability")
    os.makedirs(out, exist_ok=True)
    spark = get_spark(cfg, f"scale-{cores}", master=f"local[{cores}]", aqe=False)   # AQE off: keep partitions meaningful
    events = replicate_catalog(read_events(spark, os.path.join(cfg.work_dir, "events.parquet")), cfg.stress_factor)
    log = RunLog(os.path.join(out, "scalability_log.csv"))
    plan = [p for p in scalability.experiment_plan(cfg) if p["cores"] == cores]
    todo = [p for p in plan if not log.has(scalability.run_id(p, cfg.stress_factor))]
    if todo:                                            # warm-up on the smallest window, not logged
        scalability.run_one(spark, events, cfg, min(cfg.scale_fractions), cfg.shuffle_partitions, 1, False, False)
    for p in todo:
        rid = scalability.run_id(p, cfg.stress_factor)
        row = scalability.run_one(spark, events, cfg, p["fraction"], p["partitions"], p["salt"])
        row.update({"run_id": rid, "exp": p["exp"], "cores": cores, "stress_factor": cfg.stress_factor,
                    "master": f"local[{cores}]"})
        log.append(row)
        print(f"[cores={cores}] {p['exp']:<10} frac={p['fraction']:<4} part={p['partitions']:<3} salt={p['salt']} "
              f"-> {row['sec_total']:.1f}s, pairs {row['pairs_blocked_pct_of_naive']:.3f}% of naive")


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
    df = pd.read_csv(os.path.join(out, "scalability_log.csv"))
    cols = ["exp", "cores", "fraction", "partitions", "salt", "stress_factor", "n_events_mc", "pairs_blocked",
            "pairs_naive", "pairs_blocked_pct_of_naive", "max_over_mean", "sec_gk_baseline", "sec_nn_search",
            "sec_root_finding", "sec_total"]
    summary = df[[c for c in cols if c in df.columns]].sort_values(["exp", "cores", "fraction", "partitions", "salt"])
    summary.to_csv(os.path.join(out, "scalability_summary.csv"), index=False)
    print(summary.to_string(index=False))
    _plot(df, out)


def _plot(df: pd.DataFrame, out: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    size = df[(df.exp == "size")]
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    for c, g in size.groupby("cores"):
        g = g.sort_values("n_events_mc")
        ax[0].plot(g.n_events_mc, g.sec_total, marker="o", label=f"{c} core(s)")
    ax[0].set(xlabel="events (M >= mc)", ylabel="seconds", title="Runtime vs catalog size (time windows)")
    ax[0].legend()
    top = size[size.fraction == size.fraction.max()].sort_values("cores")
    ax[1].bar(top.cores.astype(str), top.sec_total)
    ax[1].set(xlabel="cores", ylabel="seconds", title="Full catalog: runtime vs cores")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "scaling.png"), dpi=150)


if __name__ == "__main__":
    main()
