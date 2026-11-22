"""Tests for the v2 changes: stratified Bath, like-for-like counts, label diagnostics, simple baselines, cluster bootstrap,
zone-k rule, exact pair counts, repeated timings and honest replay evaluation."""
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import average_precision_score

from seismic import diagnostics as dg
from seismic import evaluation as ev_
from seismic import modeling, scalability, streaming
from seismic.data import columns_for_clustering, pandas_to_events
from seismic.geo import gk_radius_km_np, haversine_km_np
from seismic.nn_cluster import count_pairs


# ------------------------------------------------------------------ Bath by mainshock magnitude
@pytest.fixture(scope="module")
def clustered(catalog_pdf):
    d = catalog_pdf.copy()
    d["t_sec"] = (d.event_time - pd.Timestamp("1970-01-01", tz="UTC")).dt.total_seconds()
    d["cl"] = np.where(d.true_cluster >= 0, d.true_cluster.astype(str), d.event_id)      # background events are singletons
    return d


def test_bath_pandas_equals_spark(spark, clustered):
    sdf = pandas_to_events(spark, clustered).join(spark.createDataFrame(clustered[["event_id", "cl"]]), "event_id")
    for lo in (None, 6.0, 7.0):
        a = ev_.bath_table(sdf, "cl", min_size=5, min_main_mag=lo).sort_values("cluster").reset_index(drop=True)
        b = dg.bath_table_pandas(clustered, "cl", min_size=5, min_main_mag=lo).sort_values("cluster").reset_index(drop=True)
        assert len(a) == len(b) and len(a) > 0
        np.testing.assert_allclose(a.delta_m.values, b.delta_m.values, atol=1e-9)


def test_bath_strata_filter_and_ordering(clustered):
    t = dg.bath_strata(clustered, "cl", strata=(None, 6.5, 7.0))
    assert t.n_clusters.is_monotonic_decreasing
    # hand-made case: mainshock 7.0, largest later event 5.8 -> gap 1.2 (the foreshock 6.5 before it is ignored)
    d = pd.DataFrame({"event_id": list("abcdef"), "t_sec": [0, 1, 2, 3, 4, 5], "magnitude": [6.5, 7.0, 5.8, 5.0, 4.9, 4.6], "c": "x"})
    assert abs(dg.bath_table_pandas(d, "c", 5).delta_m.iloc[0] - 1.2) < 1e-9
    assert dg.bath_table_pandas(d, "c", 5, min_main_mag=7.5).empty


# ------------------------------------------------------------------ what does the label measure?
def _toy(n=4000, seed=0):
    rng = np.random.default_rng(seed)
    n7 = rng.choice([0, 3, 20, 80], n, p=[0.5, 0.3, 0.12, 0.08])
    y = (rng.random(n) < np.select([n7 == 0, n7 == 3, n7 == 20, n7 == 80], [0.07, 0.27, 0.8, 0.92])).astype(int)
    return pd.DataFrame({"event_id": [f"e{i}" for i in range(n)], "n_7d": n7, "n_24h": (n7 * 0.4).astype(int), "magnitude": 4.5 + rng.exponential(0.4, n),
                         "gk_flag": (n7 >= 20).astype(int), "y": y, "t_sec": np.arange(n, dtype=float)})


def test_activity_bins_and_decomposition():
    d = _toy()
    b = dg.activity_bins(d)
    assert list(b.bin) == ["0", "1-9", "10-50", ">=51"] and b.n_events.sum() == len(d)
    assert b.positive_rate.is_monotonic_increasing and b.positive_rate.iloc[0] < 0.12 and b.positive_rate.iloc[-1] > 0.85
    dec = dg.label_decomposition(d)
    assert abs(dec.share_of_all_positives.sum() - 1.0) < 1e-12 and abs(dec.share_of_events.sum() - 1.0) < 1e-12
    iso = dec[dec.group.str.startswith("isolated")].iloc[0]
    assert iso.positive_rate < 0.1 and iso.share_of_all_positives < 0.3           # isolated events carry few positives


def test_simple_baselines_beat_chance_and_match_sklearn():
    d = _toy(6000)
    tr, te = d.iloc[:4000], d.iloc[4000:]
    t = dg.simple_baselines(tr, te).set_index("model")
    assert abs(t.loc["n_7d only (score = events in past 7 days)", "test_ap"] - average_precision_score(te.y, te.n_7d)) < 1e-12
    assert t.loc["LR on magnitude, log(1+n_7d), log(1+n_24h)", "test_ap"] > t.loc["chance (test base rate)", "test_ap"] + 0.2


# ------------------------------------------------------------------ bootstrap
def test_cluster_bootstrap_is_wider_when_events_are_dependent():
    """Positives and score gains concentrated in few sequences: the event bootstrap is over-confident."""
    rng = np.random.default_rng(3)
    n_cl, per = 60, 40
    g = np.repeat(np.arange(n_cl), per)
    cl_pos = rng.random(n_cl) < 0.25                     # whole sequences are positive or not
    y = np.where(np.repeat(cl_pos, per), rng.random(n_cl * per) < 0.9, rng.random(n_cl * per) < 0.05).astype(int)
    shift = np.repeat(rng.normal(0, 0.6, n_cl), per)     # the new model gains or loses per sequence
    base = rng.normal(size=n_cl * per) + 1.2 * y
    new = base + shift
    e = dg.paired_bootstrap_diff(y, new[None], base[None], n_boot=300, seed=1)
    c = dg.paired_bootstrap_diff(y, new[None], base[None], n_boot=300, seed=1, groups=g)
    assert (c["ci_high"] - c["ci_low"]) > 1.3 * (e["ci_high"] - e["ci_low"])
    assert c["unit"] == "group" and c["n_groups"] == n_cl and e["unit"] == "event"


def test_make_groups_prefers_clusters_and_falls_back_to_blocks():
    ids = np.array(["a", "b", "c"])
    cmap = pd.Series({"a": 1, "b": 1, "c": 2})
    g, unit = dg.make_groups(ids, cmap, np.array([0.0, 1.0, 2.0]))
    assert unit == "cluster" and list(g) == [1, 1, 2]
    g, unit = dg.make_groups(ids, None, np.array([0.0, 6 * 86400.0, 8 * 86400.0]), block_days=7)
    assert "block" in unit and list(g) == [0, 0, 1]
    g, unit = dg.make_groups(ids, pd.Series({"a": 1}), np.array([0.0, 6 * 86400.0, 8 * 86400.0]))   # incomplete map -> blocks
    assert "block" in unit


def test_concentration():
    g = np.array([1] * 10 + [2] * 10 + [3] * 80)
    y = np.array([1] * 9 + [0] + [1] * 9 + [0] + [0] * 80)
    c = dg.concentration(g, y, top=2)
    assert abs(c["share_of_positives"] - 1.0) < 1e-12 and abs(c["share_of_events"] - 0.2) < 1e-12


# ------------------------------------------------------------------ zone k rule
def test_choose_k_prefers_smaller_k_within_tolerance():
    t = pd.DataFrame({"k": [10, 20, 40, 80], "val_pr_auc": [0.7870, 0.7831, 0.7892, 0.7893]})
    assert modeling.choose_k(t, 0.002) == 40          # 40 and 80 tie within tolerance -> smaller wins; 10 is 0.0023 below
    assert modeling.choose_k(t, 0.0) == 80
    assert modeling.choose_k(t, 0.01) == 10


# ------------------------------------------------------------------ pair counts and repeated timings
def test_count_pairs_matches_brute_force(spark, events, catalog_pdf, cfg):
    r = count_pairs(columns_for_clustering(events), None, cfg)
    t = (catalog_pdf.event_time - pd.Timestamp("1970-01-01", tz="UTC")).dt.total_seconds().values
    la, lo, m = catalog_pdf.latitude.values, catalog_pdf.longitude.values, catalog_pdf.magnitude.values
    exact = 0
    for j in range(len(t)):
        dt = t[j] - t
        d = haversine_km_np(la, lo, la[j], lo[j])
        exact += int(((dt > 0) & (dt <= cfg.nn_max_days * 86400) & (d <= cfg.nn_radius_scale * gk_radius_km_np(m))).sum())
    assert r["exact"] == exact
    assert r["exact"] <= r["blocked"] <= r["naive"] and r["naive"] == len(t) ** 2


def test_run_one_repeats_and_records_threshold(spark, events, cfg):
    r = scalability.run_one(spark, events, cfg, 1.0, 4, 1, repeats=2)
    assert r["repeats"] == 2 and len(r["sec_total_runs"]) == 2
    assert r["sec_total_min"] <= r["sec_total"] <= r["sec_total_max"]
    assert r["sec_nn_total"] == pytest.approx(r["sec_nn_search"] + r["sec_root_finding"])
    assert r["eta0_log10_used"] == cfg.nn_eta0_log10                    # fixed threshold, recorded
    assert r["pairs_exact"] <= r["pairs_blocked"] < r["pairs_naive"]
    assert 0 < r["pairs_exact_pct_of_naive"] <= r["pairs_blocked_pct_of_naive"]


# ------------------------------------------------------------------ honest replay evaluation
def test_replay_evaluation_flags_uninformative_replays():
    n = 31
    truth = pd.DataFrame({"event_id": [f"e{i}" for i in range(n)], "y": [1] * 27 + [0] * 4, "magnitude": np.linspace(4.5, 7.1, n)})
    scores = pd.DataFrame({"event_id": truth.event_id, "score": np.linspace(0.1, 0.9, n)})
    r = streaming.evaluate_replay(scores, truth)
    assert not r["informative"] and "base rate" in r["warning"] and abs(r["base_rate"] - 27 / 31) < 1e-12
    assert r["largest_events"][0]["magnitude"] == pytest.approx(7.1)
    rng = np.random.default_rng(0)
    truth2 = pd.DataFrame({"event_id": [f"e{i}" for i in range(200)], "y": (rng.random(200) < 0.25).astype(int), "magnitude": 5.0})
    scores2 = pd.DataFrame({"event_id": truth2.event_id, "score": rng.random(200)})
    r2 = streaming.evaluate_replay(scores2, truth2)
    assert r2["informative"] and r2["warning"] is None
