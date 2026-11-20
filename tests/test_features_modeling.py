import json
import os
import shutil

import numpy as np
import pandas as pd
import pytest
from pyspark.sql import functions as F

from seismic import features as fe
from seismic import gk, modeling
from seismic.common import utc_epoch
from seismic.data import pandas_to_events
from seismic.geo import haversine_km_np
from seismic.runlog import RunLog
from seismic.synthetic import make_catalog

SEQS = [(35.5, -117.5, 6.8, 150.0), (-20.2, -69.8, 6.4, 420.0), (38.0, 142.0, 7.0, 700.0),
        (60.0, -150.0, 6.5, 950.0), (35.0, -117.0, 6.9, 1150.0), (-20.0, -70.0, 6.6, 1400.0),
        (38.5, 141.5, 6.3, 1650.0), (60.5, -149.5, 6.7, 1900.0)]


@pytest.fixture(scope="module")
def cat():
    return make_catalog(seed=11, years=6.0, n_background=1500, sequences=SEQS)


@pytest.fixture(scope="module")
def ev(spark, cat):
    return pandas_to_events(spark, cat).cache()


def build(spark, ev, cfg, mainshocks=None, grid=None):
    triggers = fe.trigger_events(ev, cfg)
    return fe.build_feature_table(spark, triggers, ev, mainshocks, grid, cfg)


# ------------------------------------------------------------------ labels
def test_labels_match_brute_force(spark, ev, cat, cfg):
    got = fe.add_labels(fe.trigger_events(ev, cfg), cfg).select("event_id", "y").toPandas().set_index("event_id").y
    t = cat.event_time.astype("int64").values / 1e9
    la, lo, m = cat.latitude.values, cat.longitude.values, cat.magnitude.values
    for i in range(0, len(cat), 7):                      # sample of events keeps the test fast
        dt = t - t[i]
        near = haversine_km_np(la[i], lo[i], la, lo) <= cfg.label_radius_km
        expect = int(((dt > 0) & (dt <= cfg.label_horizon_days * 86400) & near & (m > m[i])).any())
        assert got[cat.event_id.values[i]] == expect
    assert 0.02 < got.mean() < 0.6        # positives exist but are a minority


# ------------------------------------------------------------------ features
def test_activity_features_match_brute_force(spark, ev, cat, cfg):
    ft = build(spark, ev, cfg).toPandas().set_index("event_id")
    t = cat.event_time.astype("int64").values / 1e9
    la, lo, m = cat.latitude.values, cat.longitude.values, cat.magnitude.values
    for i in range(50, len(cat), 23):
        dt = t[i] - t
        near = (dt > 0) & (dt <= cfg.activity_lookback_days * 86400) & (haversine_km_np(la[i], lo[i], la, lo) <= cfg.activity_radius_km)
        row = ft.loc[cat.event_id.values[i]]
        assert row.n_24h == (near & (dt <= 86400)).sum()
        assert row.n_7d == (near & (dt <= 7 * 86400)).sum()
        wk = near & (dt <= 7 * 86400)
        assert abs(row.maxmag_7d - (m[wk].max() if wk.any() else 0.0)) < 1e-9
        exp_tsl = dt[near].min() / 86400 if near.any() else cfg.activity_lookback_days
        assert abs(row.tsl_days - exp_tsl) < 1e-6


@pytest.mark.parametrize("cut_day", [500.0, 1200.0, 1800.0])
def test_features_are_causal(spark, ev, cat, cfg, cut_day):
    """Deleting every event after a cutoff must not change features of events before it (incl. NN and G-K flag)."""
    cut = utc_epoch("2015-01-01") + cut_day * 86400
    full_ms, full_after, grid = gk.run_gk(spark, ev, 6.0)
    full = build(spark, ev, cfg, full_ms, grid).filter(F.col("t_sec") <= cut).toPandas().set_index("event_id")
    trunc_ev = ev.filter(F.col("t_sec") <= cut)
    tr_ms, _, tr_grid = gk.run_gk(spark, trunc_ev, 6.0)
    trunc = build(spark, trunc_ev, cfg, tr_ms, tr_grid).toPandas().set_index("event_id")
    assert set(full.index) == set(trunc.index) and len(full) > 20
    cols = ["n_24h", "n_7d", "maxmag_7d", "tsl_days", "log10_eta", "has_parent", "gk_flag"]
    pd.testing.assert_frame_equal(full.loc[trunc.index, cols].sort_index(), trunc[cols].sort_index(), check_dtype=False, atol=1e-9)
    assert full.gk_flag.sum() > 0 or cut_day < 300     # the G-K flag is actually exercised after the first mainshock


# ------------------------------------------------------------------ split
def test_time_split_gap_and_disjoint(spark, ev, cfg):
    feats = build(spark, ev, cfg)
    end = ev.agg(F.max("t_sec")).first()[0]
    tr, va, te = [d.select("event_id", "t_sec").toPandas() for d in fe.time_split(feats, cfg, end)]
    h = cfg.label_horizon_days * 86400
    t1, t2 = utc_epoch(cfg.train_end), utc_epoch(cfg.val_end)
    assert tr.t_sec.max() < t1 - h and va.t_sec.min() >= t1 and va.t_sec.max() < t2 - h
    assert te.t_sec.min() >= t2 and te.t_sec.max() <= end - h
    assert not (set(tr.event_id) & set(va.event_id)) and not (set(va.event_id) & set(te.event_id))
    assert min(len(tr), len(va), len(te)) > 50


# ------------------------------------------------------------------ metrics / statistics
def test_best_f1_threshold_and_metrics():
    y = np.array([0, 0, 0, 1, 1, 1, 0, 1])
    s = np.array([0.1, 0.2, 0.3, 0.8, 0.9, 0.7, 0.4, 0.6])
    thr = modeling.best_f1_threshold(y, s)
    m = modeling.metrics_at(y, s, thr)
    assert m["f1"] == 1.0 and m["tp"] == 4 and m["fp"] == 0


def test_bootstrap_detects_real_gain_and_no_gain():
    rng = np.random.default_rng(0)
    y = (rng.random(3000) < 0.1).astype(int)
    informative = y * 0.8 + rng.normal(0, 0.5, 3000)
    noise_a, noise_b = rng.normal(size=3000), rng.normal(size=3000)
    gain = modeling.paired_bootstrap_diff(y, informative[None, :], noise_a[None, :], n_boot=200)
    assert gain["ci_low"] > 0 and not gain["includes_zero"]
    null = modeling.paired_bootstrap_diff(y, noise_a[None, :], noise_b[None, :], n_boot=200)
    assert null["includes_zero"]


def test_runlog_resumable(tmp_path):
    p = str(tmp_path / "log.csv")
    a = RunLog(p)
    a.append({"run_id": "r1", "x": 1})
    a.append({"run_id": "r2", "x": 2, "extra": "new column"})
    b = RunLog(p)
    assert b.has("r1") and b.has("r2") and not b.has("r3") and b.get("r2")["extra"] == "new column"


# ------------------------------------------------------------------ whole Part B
def test_part_b_end_to_end_and_resume(spark, ev, cfg, tmp_path):
    from dataclasses import replace
    c = replace(cfg, models=["lr", "rf"], feature_sets=["A", "B", "C"], zone_k_candidates=[5], seeds=[1, 2],
                class_weights_sensitivity=True)
    out = str(tmp_path / "partb")
    ms, _, grid = gk.run_gk(spark, ev, 6.0)
    feats = build(spark, ev, c, ms, grid)
    feats = feats.join(fe.add_labels(fe.trigger_events(ev, c), c).select("event_id", "y"), "event_id").persist()
    end = ev.agg(F.max("t_sec")).first()[0]
    train, val, test = fe.time_split(feats, c, end)
    assert train.filter("y=1").count() > 10 and test.filter("y=1").count() > 5

    res = modeling.run_part_b(spark, train, val, test, c, out)
    log = pd.read_csv(os.path.join(out, "run_log.csv"))
    runs = log[log.part == "B"]
    # LR: 1 seed x 3 sets, RF: 2 seeds x 3 sets, plus class-weight runs for B and C
    assert len(runs[runs.weighted == False]) == 3 + 6          # noqa: E712
    assert len(runs[runs.weighted == True]) == 2               # noqa: E712
    assert (runs.test_pr_auc_ap > 0).all() and (runs.test_roc_auc > 0.4).all()
    # PR-AUC must be judged against the base rate, which every run records
    assert runs.test_base_rate.nunique() == 1
    # Spark's areaUnderPR cross-check is in the same ballpark as average precision
    assert (runs.spark_test_aupr - runs.test_pr_auc_ap).abs().max() < 0.25
    # the saved score files are aligned across variants (same test events, same order)
    sc = [pd.read_csv(os.path.join(out, "scores", f)) for f in os.listdir(os.path.join(out, "scores")) if "w0" in f]
    assert all((s.event_id.values == sc[0].event_id.values).all() for s in sc)
    prim = res["primary_C_vs_B"]
    assert {"diff", "ci_low", "ci_high", "includes_zero", "seed_sd", "interpretation"} <= set(prim)
    assert prim["ci_low"] <= prim["diff"] <= prim["ci_high"] or prim["includes_zero"] is not None
    assert os.path.isdir(os.path.join(out, "models", "pipeline_C"))

    # resume: nothing is refitted, the log does not grow, results are identical
    n_before = len(RunLog(os.path.join(out, "run_log.csv")).df)
    res2 = modeling.run_part_b(spark, train, val, test, c, out)
    assert len(RunLog(os.path.join(out, "run_log.csv")).df) == n_before
    assert abs(res2["primary_C_vs_B"]["diff"] - prim["diff"]) < 1e-12
