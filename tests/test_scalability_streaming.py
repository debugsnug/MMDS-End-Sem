import os

import numpy as np
import pandas as pd
import pytest
from pyspark.sql import functions as F

from seismic import features as fe
from seismic import gk, modeling, scalability, streaming
from seismic.data import columns_for_clustering, pandas_to_events, replicate_catalog
from seismic.zones import assign_zone, fit_zone_model
from seismic.synthetic import make_catalog

SEQS = [(35.5, -117.5, 6.8, 150.0), (-20.2, -69.8, 6.4, 420.0), (38.0, 142.0, 7.0, 700.0),
        (60.0, -150.0, 6.5, 950.0), (35.0, -117.0, 6.9, 1150.0), (-18.0, 179.8, 7.3, 1300.0)]


@pytest.fixture(scope="module")
def cat():
    return make_catalog(seed=21, years=6.0, n_background=1200, sequences=SEQS)


@pytest.fixture(scope="module")
def ev(spark, cat):
    return pandas_to_events(spark, cat).cache()


# ------------------------------------------------------------------ scalability
def test_time_fraction_is_nested_and_monotone(spark, ev):
    counts = [scalability.time_fraction(ev, f).count() for f in (0.25, 0.5, 0.75, 1.0)]
    assert counts == sorted(counts) and counts[-1] == ev.count()
    a = set(r[0] for r in scalability.time_fraction(ev, 0.25).select("event_id").collect())
    b = set(r[0] for r in scalability.time_fraction(ev, 0.5).select("event_id").collect())
    assert a <= b


def test_run_one_reports_pair_reduction_and_skew(spark, ev, cfg):
    r = scalability.run_one(spark, ev, cfg, fraction=1.0, partitions=4, salt=1)
    assert r["pairs_blocked"] < r["pairs_naive"] and r["pairs_blocked_pct_of_naive"] < 100
    assert r["block_max"] >= r["block_mean"] and r["max_over_mean"] >= 1
    assert r["sec_nn_search"] > 0 and r["sec_total"] >= r["sec_nn_search"]


def test_salting_does_not_change_results(spark, ev, cfg):
    a = scalability.run_one(spark, ev, cfg, 1.0, 4, 1, with_pairs=False, with_gk=False)
    b = scalability.run_one(spark, ev, cfg, 1.0, 4, 8, with_pairs=False, with_gk=False)
    assert a["n_nn_with_parent"] == b["n_nn_with_parent"] and a["n_nn_dependent"] == b["n_nn_dependent"]


def test_experiment_plan_dedupes_and_covers_sweeps(cfg):
    from dataclasses import replace
    c = replace(cfg, scale_cores=[1, 2], scale_fractions=[0.5, 1.0], scale_partitions=[4, 8], scale_salt=[1, 4],
                shuffle_partitions=8)
    plan = scalability.experiment_plan(c)
    keys = [(p["cores"], p["fraction"], p["partitions"], p["salt"]) for p in plan]
    assert len(keys) == len(set(keys))
    assert {p["exp"] for p in plan} == {"size", "partitions", "salt"}
    assert any(p["salt"] == 4 for p in plan) and any(p["partitions"] == 4 for p in plan)


def test_replicated_catalog_for_stress_test(spark, ev):
    big = replicate_catalog(ev, 3)
    assert big.count() == 3 * ev.count()
    assert big.select("event_id").distinct().count() == big.count()
    # copies do not overlap in time, so sequences never mix across copies
    r = big.withColumn("copy", F.regexp_extract("event_id", r"_r(\d+)$", 1)).groupBy("copy").agg(F.min("t_sec").alias("a"), F.max("t_sec").alias("b")).orderBy("copy").collect()
    assert r[0]["b"] < r[1]["a"] < r[1]["b"] < r[2]["a"]


# ------------------------------------------------------------------ streaming
def _arr_state(cat, cfg, start):
    pdf = cat.assign(t_sec=(cat.event_time - pd.Timestamp("1970-01-01", tz="UTC")).dt.total_seconds())
    return pdf, streaming.StreamState(cfg, pdf[pdf.event_time < start])


def test_stream_state_mainshocks_match_batch(cat, cfg):
    pdf = cat.assign(t_sec=(cat.event_time - pd.Timestamp("1970-01-01", tz="UTC")).dt.total_seconds())
    state = streaming.StreamState(cfg)
    for _, chunk in pdf.groupby(pdf.event_time.dt.floor("5D")):          # many small micro-batches
        state.ingest(chunk)
    sel = gk.select_mainshocks(pdf[pdf.magnitude >= cfg.mainshock_min_mag])
    assert state.ever_mainshock_ids == set(sel[sel.is_mainshock].event_id)
    assert len(state.ever_mainshock_ids) >= 3


def test_stream_scores_equal_batch_scores(spark, ev, cat, cfg):
    """Same model, same events: micro-batch scoring must reproduce batch scoring (consistency check)."""
    from dataclasses import replace
    c = replace(cfg, zone_k_candidates=[5])
    start = pd.Timestamp("2018-07-20", tz="UTC")
    end = pd.Timestamp("2018-08-10", tz="UTC")
    # --- train a small set-C model on events before the window (zones fitted on the same training rows)
    ms, _, grid = gk.run_gk(spark, ev, 6.0)
    trig = fe.trigger_events(ev, c)
    feats = fe.build_feature_table(spark, trig, ev, ms, grid, c)
    feats = feats.join(fe.add_labels(trig, c).select("event_id", "y"), "event_id")
    train = feats.filter(F.col("event_time") < F.lit(start.to_pydatetime()))
    zm = fit_zone_model(train, "kmeans", 5, seed=1)
    tr = modeling.prepare(assign_zone(zm, train))
    pipe = modeling.make_pipeline("lr", "C", 0.01, 1, c).fit(tr)
    # --- batch scoring of the window
    win = trig.filter((F.col("event_time") >= F.lit(start.to_pydatetime())) & (F.col("event_time") < F.lit(end.to_pydatetime())))
    bfeat = fe.build_feature_table(spark, win, ev, ms, grid, c, zm).withColumn("label", F.lit(0.0))
    from pyspark.ml.functions import vector_to_array
    batch = pipe.transform(bfeat).select("event_id", vector_to_array("probability")[1].alias("score")).toPandas()
    assert len(batch) > 50
    # --- streaming scoring in 1-day micro-batches with warm-started state
    pdf, state = _arr_state(cat, c, start)
    scorer = streaming.StreamScorer(spark, c, pipe, zm, state)
    w = pdf[(pdf.event_time >= start) & (pdf.event_time < end)]
    for i, (_, chunk) in enumerate(w.groupby(w.event_time.dt.floor("1D"))):
        scorer.process_batch(chunk, i)
    stream = pd.concat(scorer.scores, ignore_index=True)
    cmp = streaming.compare_stream_to_batch(stream, batch)
    assert cmp["n_compared"] == len(batch) == cmp["n_stream"]
    assert cmp["max_abs_diff"] < 1e-9, cmp
    assert all(l["feature_seconds"] > 0 for l in scorer.latency)


@pytest.mark.slow
def test_real_file_stream_replay(spark, ev, cat, cfg, tmp_path):
    """Smoke test of the Structured Streaming path (file source + foreachBatch) on a short replay."""
    from dataclasses import replace
    c = replace(cfg, zone_k_candidates=[5], tick_real_seconds=0.2, tick_sim_hours=24.0)
    start, end = pd.Timestamp("2018-07-24", tz="UTC"), pd.Timestamp("2018-08-03", tz="UTC")
    ms, _, grid = gk.run_gk(spark, ev, 6.0)
    trig = fe.trigger_events(ev, c)
    feats = fe.build_feature_table(spark, trig, ev, ms, grid, c).join(fe.add_labels(trig, c).select("event_id", "y"), "event_id")
    zm = fit_zone_model(feats, "kmeans", 5, seed=1)
    pipe = modeling.make_pipeline("lr", "C", 0.01, 1, c).fit(modeling.prepare(assign_zone(zm, feats)))
    pdf, state = _arr_state(cat, c, start)
    scorer = streaming.StreamScorer(spark, c, pipe, zm, state)
    w = pdf[(pdf.event_time >= start) & (pdf.event_time < end)]
    out = streaming.replay_stream(spark, c, scorer, w, str(tmp_path / "replay"), start)
    expected = (w.magnitude >= c.trigger_min_mag).sum()
    assert len(out) == expected and out.score.between(0, 1).all()
    assert out.batch_id.nunique() >= 3
