"""Core clustering tests: blocking correctness, G-K behaviour, nearest-neighbour recovery."""
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import adjusted_rand_score

from seismic import gk
from seismic.data import columns_for_clustering
from seismic.geo import gk_radius_km_np, haversine_km_np
from seismic.nn_cluster import (compute_nearest_neighbors, count_blocked_pairs, estimate_eta0_log10,
                                link_and_find_roots, nn_cluster_events)


def brute_force_nn(pdf, cfg):
    """Reference O(n^2) implementation of the same truncated NN search."""
    t = pdf.event_time.astype("int64").values / 1e9
    la, lo, m = pdf.latitude.values, pdf.longitude.values, pdf.magnitude.values
    out = {}
    for j in range(len(pdf)):
        dt = t[j] - t
        d = haversine_km_np(la, lo, la[j], lo[j])
        ok = (dt > 0) & (dt <= cfg.nn_max_days * 86400) & (d <= cfg.nn_radius_scale * gk_radius_km_np(m))
        if not ok.any():
            out[pdf.event_id.values[j]] = (None, np.nan)
            continue
        le = (np.log10(dt[ok] / (365.25 * 86400)) + cfg.nn_df * np.log10(np.maximum(d[ok], cfg.nn_min_km))
              - cfg.nn_b * m[ok])
        k = int(np.argmin(le))
        out[pdf.event_id.values[j]] = (pdf.event_id.values[np.where(ok)[0][k]], float(le[k]))
    return out


@pytest.mark.parametrize("salt", [1, 4])
def test_blocked_nn_matches_brute_force(spark, events, catalog_pdf, cfg, salt):
    """Blocking (incl. antimeridian / polar events, with and without salting) must not change the NN result."""
    got = compute_nearest_neighbors(columns_for_clustering(events), None, cfg, salt_buckets=salt).toPandas()
    ref = brute_force_nn(catalog_pdf, cfg)
    assert len(got) == len(catalog_pdf)
    n_with_parent = 0
    for r in got.itertuples():
        pid, eta = ref[r.event_id]
        if pid is None:
            assert r.parent_id is None or pd.isna(r.parent_id)
        else:
            n_with_parent += 1
            assert r.parent_id == pid
            assert abs(r.log10_eta - eta) < 1e-6
    assert n_with_parent > 300       # the test is not vacuous


def test_blocking_prunes_pairs(spark, events, cfg):
    blocked, naive = count_blocked_pairs(columns_for_clustering(events), None, cfg)
    assert blocked < naive


def test_gk_blocked_equals_unblocked(spark, events, cfg):
    ev = events
    ms, after_grid, grid = gk.run_gk(spark, ev, 6.0, use_grid=True)
    _, after_all, _ = gk.run_gk(spark, ev, 6.0, use_grid=False)
    a = set(r[0] for r in after_grid.select("event_id").collect())
    b = set(r[0] for r in after_all.select("event_id").collect())
    assert a == b and len(a) > 50
    assert ms.count() >= 3


def test_gk_planted_sequence_labels(spark, events, catalog_pdf):
    _, after, _ = gk.run_gk(spark, events, 6.0)
    flagged = set(r[0] for r in after.select("event_id").collect())
    big = catalog_pdf[catalog_pdf.true_cluster == 0]            # M7.0 sequence
    mainshock = big.loc[big.magnitude.idxmax(), "event_id"]
    assert mainshock not in flagged                               # a mainshock is not its own aftershock
    share = big.event_id.isin(flagged).mean()
    assert share > 0.8                                            # most planted aftershocks fall in the G-K window


def test_nn_recovers_planted_sequences(spark, events, catalog_pdf, cfg):
    labels, nn, eta0, info = nn_cluster_events(spark, columns_for_clustering(events), cfg, eta0_log10=-5.0)
    pdf = labels.toPandas().merge(catalog_pdf[["event_id", "true_cluster"]], on="event_id")
    planted = pdf[pdf.true_cluster >= 0]
    # events of one planted sequence should (mostly) end in one NN cluster
    ari = adjusted_rand_score(planted.true_cluster, planted.nn_cluster)
    assert ari > 0.7, ari
    # background events should mostly stay independent
    bg = pdf[pdf.true_cluster < 0]
    assert bg.nn_is_dependent.mean() < 0.25


def test_root_finding_long_chain(spark):
    """Pointer jumping must resolve a 200-event chain (depth 200) to a single root."""
    rows = [(f"e{i}", f"e{i-1}" if i else None, -9.0 if i else None) for i in range(200)]
    nn = spark.createDataFrame(rows, "event_id string, parent_id string, log10_eta double")
    out = link_and_find_roots(spark, nn, -5.0).toPandas()
    assert out.nn_cluster.nunique() == 1 and out.nn_cluster.iloc[0] == "e0"
    assert out.nn_is_dependent.sum() == 199


def test_root_finding_threshold_cuts_links(spark):
    rows = [("a", None, None), ("b", "a", -8.0), ("c", "b", -2.0), ("d", "c", -8.0)]
    nn = spark.createDataFrame(rows, "event_id string, parent_id string, log10_eta double")
    out = link_and_find_roots(spark, nn, -5.0).toPandas().set_index("event_id")
    assert out.loc["b", "nn_cluster"] == "a"
    assert out.loc["c", "nn_cluster"] == "c" and out.loc["d", "nn_cluster"] == "c"


def test_eta0_estimation_separates_two_modes(spark):
    rng = np.random.default_rng(0)
    vals = np.concatenate([rng.normal(-8, 0.8, 1500), rng.normal(-3, 0.8, 1500)])
    nn = spark.createDataFrame([(f"e{i}", None, float(v)) for i, v in enumerate(vals)],
                               "event_id string, parent_id string, log10_eta double")
    eta0, info = estimate_eta0_log10(nn, default=-5.0)
    assert info["method"] == "gmm" and -6.5 < eta0 < -4.5
