"""Part B modelling with Spark MLlib.

Four classifiers (LR, DT, RF, GBT) x three feature sets (A/B/C), identical small grid per
(model, set), tuned on the validation period with a *manual time-aware loop* (CrossValidator
uses random folds, which would leak across time). Final scores are collected (event id, label,
score) so every statistic is computed outside Spark from saved files.
"""
from __future__ import annotations

import json
import os
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.classification import (DecisionTreeClassifier, GBTClassifier, LogisticRegression,
                                        RandomForestClassifier)
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.feature import OneHotEncoder, StandardScaler, StringIndexer, VectorAssembler
from pyspark.ml.functions import vector_to_array
from pyspark.sql import DataFrame, functions as F
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score

from .diagnostics import concentration, make_groups, paired_bootstrap_diff  # noqa: F401  (re-exported)
from .runlog import RunLog
from .zones import assign_zone, fit_zone_model

NUMERIC_A = ["magnitude", "depth_km"]
NUMERIC_B = NUMERIC_A + ["n_24h", "n_7d", "maxmag_7d", "tsl_days"]
NUMERIC_C = NUMERIC_B + ["log10_eta", "has_parent", "gk_flag"]
# "S" is a deliberately simple reference (v2): magnitude and log activity only. Any richer set has to beat it.
FEATURES = {"S": (["magnitude", "log_n7d", "log_n24h"], []),
            "A": (NUMERIC_A, ["cell_id"]), "B": (NUMERIC_B, ["cell_id"]), "C": (NUMERIC_C, ["cell_id", "zone_id"])}
STOCHASTIC = {"rf", "gbt"}          # LR and DT are deterministic: one seed is enough


# ------------------------------------------------------------------ pipelines
def make_pipeline(model: str, fset: str, param, seed: int, cfg, weighted: bool = False) -> Pipeline:
    """Feature encoding + classifier. `param` is the single tuned value (see grids in Config)."""
    numeric, cats = FEATURES[fset]
    idx = [f"{c}_idx" for c in cats]
    ohe = [f"{c}_ohe" for c in cats]
    stages = []
    if cats:
        stages += [StringIndexer(inputCols=cats, outputCols=idx, handleInvalid="keep"),
                   OneHotEncoder(inputCols=idx, outputCols=ohe, handleInvalid="keep")]
    stages.append(VectorAssembler(inputCols=numeric + ohe, outputCol="features_raw", handleInvalid="keep"))
    feat = "features_raw"
    if model == "lr":                      # linear model needs comparable scales; trees do not
        stages.append(StandardScaler(inputCol="features_raw", outputCol="features", withMean=False, withStd=True))
        feat = "features"
    common = dict(labelCol="label", featuresCol=feat)
    if weighted:
        common["weightCol"] = "weight"
    if model == "lr":
        clf = LogisticRegression(regParam=float(param), elasticNetParam=0.0, maxIter=100, **common)
    elif model == "dt":
        clf = DecisionTreeClassifier(maxDepth=int(param), seed=seed, **common)
    elif model == "rf":
        clf = RandomForestClassifier(numTrees=cfg.rf_num_trees, maxDepth=int(param), seed=seed, **common)
    elif model == "gbt":
        clf = GBTClassifier(maxIter=int(param), maxDepth=cfg.gbt_max_depth, seed=seed, **common)
    else:
        raise ValueError(f"unknown model {model}")
    return Pipeline(stages=stages + [clf])


def grid_for(model: str, cfg) -> list:
    return {"lr": cfg.grid_lr, "dt": cfg.grid_dt, "rf": cfg.grid_rf, "gbt": cfg.grid_gbt}[model]


def prepare(df: DataFrame, weighted_ratio: Optional[float] = None) -> DataFrame:
    out = (df.withColumn("label", F.col("y").cast("double") if "y" in df.columns else F.lit(0.0))
             .withColumn("log_n7d", F.log1p(F.col("n_7d").cast("double")))
             .withColumn("log_n24h", F.log1p(F.col("n_24h").cast("double"))))
    if weighted_ratio is not None:
        out = out.withColumn("weight", F.when(F.col("y") == 1, F.lit(float(weighted_ratio))).otherwise(F.lit(1.0)))
    return out


# ------------------------------------------------------------------ scoring and metrics
def score_pdf(model: PipelineModel, df: DataFrame) -> pd.DataFrame:
    """Collect (event_id, label, score) for a (small) evaluation set."""
    pred = model.transform(df)
    return (pred.select("event_id", F.col("label").alias("y"),
                        vector_to_array("probability")[1].alias("score")).toPandas())


def spark_aupr(model: PipelineModel, df: DataFrame) -> float:
    """Spark's areaUnderPR (trapezoidal), reported next to average precision as a cross-check."""
    return float(BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="rawPrediction",
                                               metricName="areaUnderPR").evaluate(model.transform(df)))


def best_f1_threshold(y: np.ndarray, s: np.ndarray) -> float:
    """Threshold maximising positive-class F1 on validation data."""
    p, r, t = precision_recall_curve(y, s)
    f1 = 2 * p[:-1] * r[:-1] / np.maximum(p[:-1] + r[:-1], 1e-12)
    return float(t[int(np.nanargmax(f1))]) if len(t) else 0.5


def metrics_at(y: np.ndarray, s: np.ndarray, thr: float) -> Dict[str, float]:
    pred = (s >= thr).astype(int)
    tp, fp = int(((pred == 1) & (y == 1)).sum()), int(((pred == 1) & (y == 0)).sum())
    fn, tn = int(((pred == 0) & (y == 1)).sum()), int(((pred == 0) & (y == 0)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    return {"precision": prec, "recall": rec, "f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn, "threshold": thr}


def rank_metrics(y: np.ndarray, s: np.ndarray) -> Dict[str, float]:
    if len(np.unique(y)) < 2:
        return {"pr_auc_ap": float("nan"), "roc_auc": float("nan"), "base_rate": float(np.mean(y))}
    return {"pr_auc_ap": float(average_precision_score(y, s)), "roc_auc": float(roc_auc_score(y, s)),
            "base_rate": float(np.mean(y))}


# ------------------------------------------------------------------ tuning
def tune(model: str, fset: str, train: DataFrame, val: DataFrame, cfg, seed: int = 1,
         weighted: bool = False) -> Tuple[object, pd.DataFrame]:
    """Pick the grid value with the best validation PR-AUC (average precision)."""
    rows = []
    for param in grid_for(model, cfg):
        fitted = make_pipeline(model, fset, param, seed, cfg, weighted).fit(train)
        sv = score_pdf(fitted, val)
        rows.append({"param": param, "val_pr_auc": rank_metrics(sv.y.values, sv.score.values)["pr_auc_ap"]})
    tab = pd.DataFrame(rows)
    best = tab.loc[tab.val_pr_auc.fillna(-1).idxmax(), "param"]
    return best, tab


def choose_k(table: pd.DataFrame, tol: float) -> int:
    """Smallest k whose validation PR-AUC is within `tol` of the best (differences of a few 0.001 are noise)."""
    t = table.dropna(subset=["val_pr_auc"])
    best = t.val_pr_auc.max()
    return int(t[t.val_pr_auc >= best - tol].k.min())


def select_zone_k(train: DataFrame, val: DataFrame, cfg, seed: int = 1) -> Tuple[int, pd.DataFrame]:
    """Choose the number of k-means zones on VALIDATION data only (set C, reference model = RF).

    Zones are fitted on training rows; this is the only place k is chosen and it is then fixed. The smallest k within
    `cfg.zone_k_tolerance` of the best validation PR-AUC wins, so the choice is not driven by noise.
    """
    rows = []
    for k in cfg.zone_k_candidates:
        zm = fit_zone_model(train, "kmeans", k, seed)
        tr, va = assign_zone(zm, train), assign_zone(zm, val)
        fitted = make_pipeline("rf", "C", grid_for("rf", cfg)[0], seed, cfg).fit(tr)
        sv = score_pdf(fitted, va)
        rows.append({"k": k, "val_pr_auc": rank_metrics(sv.y.values, sv.score.values)["pr_auc_ap"]})
    tab = pd.DataFrame(rows)
    return choose_k(tab, cfg.zone_k_tolerance), tab


# ------------------------------------------------------------------ statistics
# ------------------------------------------------------------------ orchestration
def _load_or_fit_zone_model(train: DataFrame, zk: int, out_dir: str, seed: int = 1):
    """Zone model is fitted once and persisted: a resumed run must use the SAME zones as the logged scores
    (k-means initialisation can differ between runs with different core counts)."""
    from pyspark.ml.clustering import KMeansModel
    path = os.path.join(out_dir, "models", "zone_kmeans")
    if os.path.isdir(path):
        m = KMeansModel.load(path)
        if len(m.clusterCenters()) == zk:
            return m
    m = fit_zone_model(train, "kmeans", zk, seed=seed)
    os.makedirs(os.path.join(out_dir, "models"), exist_ok=True)
    m.write().overwrite().save(path)
    return m


def run_part_b(spark, train: DataFrame, val: DataFrame, test: DataFrame, cfg, out_dir: str,
               log: Optional[RunLog] = None, save_models: bool = True,
               group_map: Optional[pd.Series] = None) -> Dict[str, object]:
    """Zone selection -> tuning -> final fits -> test scoring -> logs -> bootstrap. Resumable via `log`.

    group_map: optional Series event_id -> cluster id (e.g. the Part A nearest-neighbour clusters). With
    cfg.bootstrap_unit == "cluster" the primary interval resamples whole clusters; otherwise events.
    """
    os.makedirs(os.path.join(out_dir, "scores"), exist_ok=True)
    log = log or RunLog(os.path.join(out_dir, "run_log.csv"))
    t_all = time.perf_counter()

    # ---- zones (fitted on training rows only; k chosen on validation; the model is persisted and reused on resume)
    zk_path = os.path.join(out_dir, "zone_k.json")
    if os.path.exists(zk_path):
        zk = json.load(open(zk_path))["k"]
    else:
        zk, ztab = select_zone_k(prepare(train), prepare(val), cfg)
        json.dump({"k": zk, "table": ztab.to_dict("records")}, open(zk_path, "w"))
    zone_model = _load_or_fit_zone_model(train, zk, out_dir)
    tr, va, te = [prepare(assign_zone(zone_model, d)).persist() for d in (train, val, test)]
    n_pos = tr.filter("y = 1").count()
    ratio = (tr.count() - n_pos) / max(1, n_pos)

    def run_one(fset, model, param, seed, weighted=False, tr_=tr):
        rid = f"B|{fset}|{model}|seed{seed}|w{int(weighted)}|zk{zk}"
        spath = os.path.join(out_dir, "scores", rid.replace("|", "_") + ".csv")
        if log.has(rid) and os.path.exists(spath):
            return pd.read_csv(spath), None
        t0 = time.perf_counter()
        fitted = make_pipeline(model, fset, param, seed, cfg, weighted).fit(tr_)
        fit_s = time.perf_counter() - t0
        t0 = time.perf_counter()
        sv, st = score_pdf(fitted, va), score_pdf(fitted, te)
        pred_s = time.perf_counter() - t0
        thr = best_f1_threshold(sv.y.values, sv.score.values)
        row = {"run_id": rid, "part": "B", "feature_set": fset, "model": model, "param": param, "seed": seed,
               "weighted": weighted, "zone_k": zk, "n_train": int(tr_.count()), "n_val": len(sv), "n_test": len(st),
               "fit_seconds": fit_s, "predict_seconds": pred_s,
               "val_pr_auc": rank_metrics(sv.y.values, sv.score.values)["pr_auc_ap"],
               "spark_test_aupr": spark_aupr(fitted, te)}
        row.update({f"test_{k}": v for k, v in rank_metrics(st.y.values, st.score.values).items()})
        row.update({f"test_{k}": v for k, v in metrics_at(st.y.values, st.score.values, thr).items()})
        st.to_csv(spath, index=False)
        log.append(row)
        return st, fitted

    def tuned(fset, model):
        tkey = f"tune|{fset}|{model}|zk{zk}"
        if log.has(tkey):
            raw = log.get(tkey)["param"]
            return float(raw) if model == "lr" else int(float(raw))
        param, ttab = tune(model, fset, tr, va, cfg, seed=cfg.seeds[0])
        param = float(param) if model == "lr" else int(param)        # plain Python number (JSON / CSV safe)
        log.append({"run_id": tkey, "part": "tune", "feature_set": fset, "model": model, "param": param,
                    "grid": ttab.to_dict("records"), "val_pr_auc": float(ttab.val_pr_auc.max())})
        return param

    best_param: Dict[Tuple[str, str], object] = {}
    scores: Dict[Tuple[str, str, int], pd.DataFrame] = {}
    c_models: Dict[str, object] = {}                       # freshly fitted set-C models (first seed)
    for fset in cfg.feature_sets:
        for model in cfg.models:
            param = tuned(fset, model)
            best_param[(fset, model)] = param
            seeds = cfg.seeds if model in STOCHASTIC else cfg.seeds[:1]
            for seed in seeds:
                st, fitted = run_one(fset, model, param, seed)
                scores[(fset, model, seed)] = st
                if fset == "C" and seed == cfg.seeds[0] and fitted is not None:
                    c_models[model] = fitted

    # ---- simple reference models (v2), e.g. "S:lr" = logistic regression on magnitude and log activity only
    for spec in cfg.baseline_runs:
        fset, model = spec.split(":")
        param = tuned(fset, model)
        for seed in (cfg.seeds if model in STOCHASTIC else cfg.seeds[:1]):
            run_one(fset, model, param, seed)

    # ---- class-weight sensitivity (best classifier by validation on set B), sets B and C
    best_model = pick_best_model(log, cfg, zk)
    if cfg.class_weights_sensitivity:
        trw = prepare(assign_zone(zone_model, train), weighted_ratio=ratio).persist()
        for fset in ("B", "C"):
            if fset in cfg.feature_sets:
                run_one(fset, best_model, best_param[(fset, best_model)], cfg.seeds[0], weighted=True, tr_=trw)
        trw.unpersist()

    # ---- primary comparison: set C vs set B with the classifier that is best on validation for set B
    result: Dict[str, object] = {"zone_k": zk, "best_model": best_model, "base_rate_test": None}
    if "B" in cfg.feature_sets and "C" in cfg.feature_sets:
        seeds = cfg.seeds if best_model in STOCHASTIC else cfg.seeds[:1]
        sB = [scores[("B", best_model, s)] for s in seeds]
        sC = [scores[("C", best_model, s)] for s in seeds]
        y = sB[0].y.values
        for a in sB + sC:   # all runs must be scored on the same events in the same order
            assert (a.event_id.values == sB[0].event_id.values).all()
        newS, baseS = np.vstack([c.score.values for c in sC]), np.vstack([b.score.values for b in sB])
        ev_res = paired_bootstrap_diff(y, newS, baseS, n_boot=cfg.n_boot, seed=cfg.master_seed)
        res = dict(ev_res)
        res["event_bootstrap"] = {k: ev_res[k] for k in ("ci_low", "ci_high", "includes_zero")}
        if cfg.bootstrap_unit == "cluster":
            tsec = test.select("event_id", "t_sec").toPandas().set_index("event_id").t_sec.reindex(sB[0].event_id).values
            groups, unit = make_groups(sB[0].event_id.values, group_map, tsec, cfg.bootstrap_block_days)
            g_res = paired_bootstrap_diff(y, newS, baseS, n_boot=cfg.n_boot, seed=cfg.master_seed, groups=groups)
            res.update({"ci_low": g_res["ci_low"], "ci_high": g_res["ci_high"], "includes_zero": g_res["includes_zero"],
                        "n_groups": g_res["n_groups"], "bootstrap_unit": unit})
            res["concentration_top5"] = concentration(groups, y, 5)
        else:
            res["bootstrap_unit"] = "event"
        res["interpretation"] = ("interval includes zero: no clear difference established (this is not proof of no benefit)"
                                 if res["includes_zero"] else "interval excludes zero")
        result["primary_C_vs_B"] = res
        result["base_rate_test"] = float(np.mean(y))
        json.dump(result, open(os.path.join(out_dir, "primary_comparison.json"), "w"), indent=2, default=float)

    # ---- save the set-C pipeline of the classifier that is actually best (used by the streaming demo)
    if save_models and "C" in cfg.feature_sets:
        mdir = os.path.join(out_dir, "models")
        os.makedirs(mdir, exist_ok=True)
        meta_path = os.path.join(mdir, "meta.json")
        meta = json.load(open(meta_path)) if os.path.exists(meta_path) else {}
        if meta.get("model") != best_model or meta.get("zone_k") != zk or not os.path.isdir(os.path.join(mdir, "pipeline_C")):
            fitted = c_models.get(best_model)
            refit = fitted is None
            if refit:     # resumed run: refit once, and check it reproduces the logged scores
                fitted = make_pipeline(best_model, "C", best_param[("C", best_model)], cfg.seeds[0], cfg).fit(tr)
            check = None
            logged = scores.get(("C", best_model, cfg.seeds[0]))
            if refit and logged is not None:
                now = score_pdf(fitted, te).set_index("event_id").score.reindex(logged.event_id).values
                check = float(np.max(np.abs(now - logged.score.values)))
            fitted.write().overwrite().save(os.path.join(mdir, "pipeline_C"))
            json.dump({"model": best_model, "zone_k": zk, "seed": cfg.seeds[0], "param": best_param[("C", best_model)],
                       "refit_max_abs_score_diff_vs_logged": check}, open(meta_path, "w"), default=float)
    result["total_seconds"] = time.perf_counter() - t_all
    for d in (tr, va, te):
        d.unpersist()
    return result


def pick_best_model(log: RunLog, cfg, zk: int) -> str:
    """Classifier with the best validation PR-AUC on feature set B (so the choice does not favour C)."""
    best, best_v = cfg.models[0], -1.0
    for m in cfg.models:
        row = log.get(f"tune|B|{m}|zk{zk}")
        if row is not None and float(row["val_pr_auc"]) > best_v:
            best, best_v = m, float(row["val_pr_auc"])
    return best


def summary_table(log_path: str) -> pd.DataFrame:
    """Test-set results of every logged Part B run (for the report)."""
    df = pd.read_csv(log_path)
    df = df[df.part == "B"]
    cols = ["feature_set", "model", "seed", "weighted", "test_pr_auc_ap", "spark_test_aupr", "test_roc_auc",
            "test_precision", "test_recall", "test_f1", "test_base_rate", "fit_seconds", "predict_seconds"]
    return df[cols].sort_values(["model", "feature_set", "weighted", "seed"]).reset_index(drop=True)
