"""Pandas / NumPy / scikit-learn diagnostics (no Spark needed), added in v2.

They answer the review questions that the saved outputs can answer on their own:
  * Bath's law stratified by mainshock magnitude (the M >= 4.5 cut-off biases the pooled mean low)
  * what the Part B label really measures (positive rate by recent activity, share of positives in sequences)
  * simple reference models that any cluster feature has to beat
  * a bootstrap that resamples whole clusters instead of single events (events in one sequence are not independent)
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score


# ------------------------------------------------------------------ Bath's law (pandas version of evaluation.bath_table)
def bath_table_pandas(df: pd.DataFrame, cluster_col: str, min_size: int = 5, min_main_mag: Optional[float] = None) -> pd.DataFrame:
    """Per cluster: largest event ('mainshock'), largest event after it, and the difference.

    df needs event_id, t_sec, magnitude and `cluster_col`. Ties for the largest magnitude go to the earliest event.
    `min_main_mag` keeps only clusters whose mainshock has at least that magnitude.
    """
    d = df[["event_id", "t_sec", "magnitude", cluster_col]].rename(columns={cluster_col: "cluster"})
    size = d.groupby("cluster").size().rename("size")
    d = d.sort_values(["cluster", "magnitude", "t_sec"], ascending=[True, False, True])
    main = d.groupby("cluster").head(1).rename(columns={"event_id": "main_id", "t_sec": "main_t", "magnitude": "main_mag"}).set_index("cluster")
    j = d.join(main[["main_id", "main_t", "main_mag"]], on="cluster")
    after = j[(j.t_sec > j.main_t) & (j.event_id != j.main_id)].groupby("cluster").magnitude.max().rename("largest_after")
    out = main.join(size).join(after, how="inner")
    out = out[out["size"] >= min_size]
    if min_main_mag is not None:
        out = out[out.main_mag >= min_main_mag]
    out["delta_m"] = out.main_mag - out.largest_after
    return out.reset_index()[["cluster", "size", "main_mag", "largest_after", "delta_m"]]


def bath_strata(df: pd.DataFrame, cluster_col: str, strata: Sequence[Optional[float]] = (None, 5.0, 6.0, 7.0), min_size: int = 5) -> pd.DataFrame:
    """Mean/median Bath gap for clusters whose mainshock magnitude is at least each threshold."""
    rows = []
    for lo in strata:
        t = bath_table_pandas(df, cluster_col, min_size, lo)
        rows.append({"mainshock_min_mag": "all" if lo is None else f">= {lo:g}", "n_clusters": int(len(t)),
                     "mean_delta": float(t.delta_m.mean()) if len(t) else float("nan"),
                     "median_delta": float(t.delta_m.median()) if len(t) else float("nan")})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ what does the Part B label measure?
def activity_bins(df: pd.DataFrame, edges: Sequence[int] = (0, 1, 10, 51), col: str = "n_7d") -> pd.DataFrame:
    """Positive rate of the label by recent activity (default bins: 0, 1-9, 10-50, >50 events in the past 7 days)."""
    rows = []
    bounds = list(edges) + [np.inf]
    for lo, hi in zip(bounds[:-1], bounds[1:]):
        m = (df[col] >= lo) & (df[col] < hi) if hi != np.inf else (df[col] >= lo)
        if lo == 0 and bounds[1] == 1:
            label = "0"
        else:
            label = f"{int(lo)}-{int(hi) - 1}" if hi != np.inf else f">={int(lo)}"
        s = df[m]
        rows.append({"bin": label, "n_events": int(len(s)), "share_of_events": float(len(s) / max(1, len(df))),
                     "positive_rate": float(s.y.mean()) if len(s) else float("nan"), "n_positive": int(s.y.sum())})
    return pd.DataFrame(rows)


def label_decomposition(df: pd.DataFrame) -> pd.DataFrame:
    """Where do the positives come from?

    'in a G-K sequence'      : the event is itself a G-K aftershock (gk_flag = 1)
    'recent activity'        : not a G-K aftershock, but >= 1 earlier M>=mc event within 100 km in the last 7 days
    'isolated'               : no recent activity and not an aftershock - the only group that resembles a real foreshock setting
    """
    cats = {
        "in a G-K sequence (aftershock)": df.gk_flag == 1,
        "recent activity, not a G-K aftershock": (df.gk_flag == 0) & (df.n_7d >= 1),
        "isolated (no recent activity)": (df.gk_flag == 0) & (df.n_7d == 0),
    }
    pos = max(1, int(df.y.sum()))
    rows = []
    for name, m in cats.items():
        s = df[m]
        rows.append({"group": name, "n_events": int(len(s)), "share_of_events": float(len(s) / len(df)),
                     "positive_rate": float(s.y.mean()) if len(s) else float("nan"),
                     "share_of_all_positives": float(s.y.sum() / pos)})
    return pd.DataFrame(rows)


def simple_baselines(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    """Reference scores that any richer feature set has to beat (average precision on the test period)."""
    ap = lambda s: float(average_precision_score(test.y, s))  # noqa: E731
    X = lambda d: np.c_[d.magnitude.values, np.log1p(d.n_7d.values), np.log1p(d.n_24h.values)]  # noqa: E731
    lr = LogisticRegression(max_iter=1000).fit(X(train), train.y)
    rows = [{"model": "chance (test base rate)", "test_ap": float(test.y.mean())},
            {"model": "magnitude only (score = M)", "test_ap": ap(test.magnitude.values)},
            {"model": "n_7d only (score = events in past 7 days)", "test_ap": ap(test.n_7d.values)},
            {"model": "LR on magnitude, log(1+n_7d), log(1+n_24h)", "test_ap": ap(lr.predict_proba(X(test))[:, 1])}]
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ bootstrap (events or whole clusters)
def make_groups(event_ids: Sequence, cluster_map: Optional[pd.Series], t_sec: Optional[np.ndarray] = None,
                block_days: float = 7.0) -> (np.ndarray, str):
    """Group id per event for the bootstrap. NN cluster ids when available, otherwise calendar blocks."""
    if cluster_map is not None:
        g = cluster_map.reindex(event_ids).values
        if not pd.isna(g).any():
            return g, "cluster"
    if t_sec is None:
        raise ValueError("need cluster ids or event times to build bootstrap groups")
    return (np.asarray(t_sec) // (block_days * 86400.0)).astype(np.int64), f"{block_days:g}-day block"


def paired_bootstrap_diff(y: np.ndarray, new: np.ndarray, base: np.ndarray, n_boot: int = 1000, seed: int = 0,
                          alpha: float = 0.05, groups: Optional[np.ndarray] = None) -> Dict[str, object]:
    """Bootstrap CI for PR-AUC(new) - PR-AUC(base) over test events.

    new/base: arrays (n_seeds, n_events). The SAME resampled events are used for both variants and all seeds, and the
    per-seed differences are averaged inside each replicate. With `groups` (one id per event) whole groups are
    resampled with replacement, which respects the dependence between events of one sequence and gives a wider,
    more honest interval. Seed variability = SD of the per-seed differences on the full test set.
    """
    y = np.asarray(y)
    new, base = np.atleast_2d(new), np.atleast_2d(base)
    n = len(y)
    rng = np.random.default_rng(seed)
    per_seed = np.array([average_precision_score(y, a) - average_precision_score(y, b) for a, b in zip(new, base)])
    members = None
    if groups is not None:
        order = np.argsort(groups, kind="stable")
        _, start = np.unique(np.asarray(groups)[order], return_index=True)
        members = np.split(order, start[1:])
    diffs = []
    for _ in range(n_boot):
        if members is None:
            idx = rng.integers(0, n, n)
        else:
            pick = rng.integers(0, len(members), len(members))
            idx = np.concatenate([members[i] for i in pick])
        yy = y[idx]
        if yy.sum() == 0:
            continue
        diffs.append(np.mean([average_precision_score(yy, a[idx]) - average_precision_score(yy, b[idx]) for a, b in zip(new, base)]))
    lo, hi = np.percentile(diffs, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"diff": float(per_seed.mean()), "ci_low": float(lo), "ci_high": float(hi), "includes_zero": bool(lo <= 0 <= hi),
            "seed_sd": float(per_seed.std(ddof=1)) if len(per_seed) > 1 else float("nan"),
            "base_ap": float(np.mean([average_precision_score(y, b) for b in base])),
            "new_ap": float(np.mean([average_precision_score(y, a) for a in new])), "n_boot_used": len(diffs),
            "unit": "event" if members is None else "group", "n_groups": None if members is None else len(members)}


def concentration(groups: np.ndarray, y: np.ndarray, top: int = 5) -> Dict[str, float]:
    """How concentrated are the positives? Share of positives and of events held by the `top` groups with most positives."""
    d = pd.DataFrame({"g": groups, "y": y}).groupby("g").agg(n=("y", "size"), pos=("y", "sum")).sort_values("pos", ascending=False)
    t = d.head(top)
    return {"top": top, "share_of_positives": float(t.pos.sum() / max(1, d.pos.sum())), "share_of_events": float(t.n.sum() / d.n.sum()),
            "n_groups": int(len(d))}


# ------------------------------------------------------------------ Part B diagnostics from the saved feature table
def split_pandas(df: pd.DataFrame, train_end_sec: float, val_end_sec: float, catalog_end_sec: float, horizon_days: float):
    """pandas version of features.time_split (same gaps), for diagnostics without Spark."""
    h = horizon_days * 86400.0
    tr = df[df.t_sec < train_end_sec - h]
    va = df[(df.t_sec >= train_end_sec) & (df.t_sec < val_end_sec - h)]
    te = df[(df.t_sec >= val_end_sec) & (df.t_sec <= catalog_end_sec - h)]
    return tr, va, te


def part_b_diagnostics(features: pd.DataFrame, train_end_sec: float, val_end_sec: float, catalog_end_sec: float,
                       horizon_days: float, group_map: Optional[pd.Series] = None, test_ids: Optional[Sequence] = None) -> Dict[str, object]:
    """Everything needed to explain the Part B label and the strength of the simple baselines.

    test_ids: the exact test events (e.g. from a saved score file); when given they replace the recomputed test split,
    which avoids depending on the catalog end date.
    """
    tr, va, te = split_pandas(features, train_end_sec, val_end_sec, catalog_end_sec, horizon_days)
    if test_ids is not None:
        te = features[features.event_id.isin(set(test_ids))]
    out: Dict[str, object] = {
        "activity_bins_all": activity_bins(features), "activity_bins_test": activity_bins(te),
        "label_decomposition_all": label_decomposition(features), "label_decomposition_test": label_decomposition(te),
        "baselines": simple_baselines(tr, te),
        "split_sizes": {"train": len(tr), "val": len(va), "test": len(te), "test_base_rate": float(te.y.mean())},
    }
    if group_map is not None:
        g = group_map.reindex(te.event_id).values
        ok = ~pd.isna(g)
        if ok.any():
            out["concentration_test"] = concentration(g[ok], te.y.values[ok], 5)
    return out
