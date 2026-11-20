import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

from seismic import evaluation as ev
from seismic import gk, zones
from seismic.data import columns_for_clustering
from seismic.dbscan_ref import st_dbscan_reference
from seismic.nn_cluster import nn_cluster_events


# ------------------------------------------------------------------ zones
def test_zones_unit_sphere_handles_antimeridian(spark):
    """Two tight blobs on either side of lon=180 are one zone on the sphere (would be split in raw lon)."""
    rng = np.random.default_rng(1)
    la = np.concatenate([rng.normal(0, 0.5, 100), rng.normal(0, 0.5, 100), rng.normal(50, 0.5, 100)])
    lo = np.concatenate([179.5 + rng.normal(0, 0.2, 100), -179.5 + rng.normal(0, 0.2, 100),
                         rng.normal(10, 0.5, 100)])
    df = spark.createDataFrame([(float(a), float(((b + 180) % 360) - 180)) for a, b in zip(la, lo)],
                               "latitude double, longitude double")
    model = zones.fit_zone_model(df, "kmeans", 2, seed=1)
    z = zones.assign_zone(model, df).toPandas()
    assert z.iloc[:200].zone_id.nunique() == 1          # both antimeridian blobs share one zone
    assert z.iloc[200:].zone_id.iloc[0] != z.iloc[0].zone_id


def test_zone_models_all_kinds(spark, events):
    ev_df = events.filter("magnitude >= 4.5")
    table = zones.compare_zone_models(ev_df, ["kmeans", "bisecting", "gmm"], [4], sample=1.0)
    assert set(table.kind) == {"kmeans", "bisecting", "gmm"}
    ok = table[table.error == ""]
    assert len(ok) >= 2 and (ok.silhouette > 0).all()


def test_zone_assignment_uses_fitted_model_only(spark, events):
    d = events.filter("magnitude >= 4.5")
    train, other = d.filter("t_sec < 1.45e9"), d.filter("t_sec >= 1.45e9")
    m = zones.fit_zone_model(train, "kmeans", 5, seed=2)
    a = zones.assign_zone(m, other).select("event_id", "zone_id").toPandas()
    b = zones.assign_zone(m, d).select("event_id", "zone_id").toPandas().set_index("event_id")
    # same event, same zone regardless of which rows are scored alongside it
    assert (a.zone_id.values == b.loc[a.event_id, "zone_id"].values).all()


# ------------------------------------------------------------------ metrics
def test_ari_matches_sklearn(spark):
    rng = np.random.default_rng(0)
    a = rng.integers(0, 6, 400)
    b = np.where(rng.random(400) < 0.7, a, rng.integers(0, 6, 400))
    df = spark.createDataFrame([(int(x), int(y)) for x, y in zip(a, b)], "a int, b int")
    assert abs(ev.adjusted_rand_index(df, "a", "b") - adjusted_rand_score(a, b)) < 1e-9


def test_flag_overlap_counts(spark):
    df = spark.createDataFrame([(True, True), (True, False), (False, True), (False, False), (True, True)], "a boolean, b boolean")
    r = ev.flag_overlap(df, "a", "b")
    assert (r["both"], r["a_only"], r["b_only"], r["neither"]) == (2, 1, 1, 1)
    assert abs(r["jaccard"] - 0.5) < 1e-12


def test_bath_table_known_cluster(spark):
    rows = [("a", 0.0, 7.0, "c1"), ("b", 10.0, 5.8, "c1"), ("c", 20.0, 5.0, "c1"), ("d", 30.0, 4.6, "c1"),
            ("e", 40.0, 4.5, "c1"), ("x", 5.0, 5.0, "c2")]          # c2 has a single event -> dropped
    df = spark.createDataFrame(rows, "event_id string, t_sec double, magnitude double, cl string")
    tab = ev.bath_table(df, "cl", min_size=5)
    assert len(tab) == 1 and abs(tab.delta_m.iloc[0] - 1.2) < 1e-9


def test_bath_ignores_foreshocks(spark):
    """A larger event *before* the biggest event is not an aftershock."""
    rows = [("f", 0.0, 6.5, "c"), ("m", 10.0, 7.0, "c"), ("a1", 20.0, 5.0, "c"), ("a2", 30.0, 4.8, "c"), ("a3", 40.0, 4.6, "c")]
    df = spark.createDataFrame(rows, "event_id string, t_sec double, magnitude double, c string").withColumnRenamed("c", "cl")
    tab = ev.bath_table(df, "cl", min_size=5)
    assert abs(tab.delta_m.iloc[0] - 2.0) < 1e-9


def test_omori_mle_recovers_p():
    rng = np.random.default_rng(5)
    for p_true in (0.8, 1.1, 1.4):
        a, b = 0.05, 200.0
        u = rng.uniform(size=3000)
        t = (a ** (1 - p_true) + u * (b ** (1 - p_true) - a ** (1 - p_true))) ** (1 / (1 - p_true))
        assert abs(ev.omori_mle_p(t, t_min=a, t_max=b) - p_true) < 0.08


def test_omori_nan_for_tiny_samples():
    assert np.isnan(ev.omori_mle_p(np.array([0.1, 0.2, 0.5])))


def test_dispersion_poisson_vs_clustered(spark):
    rng = np.random.default_rng(2)
    poisson = np.sort(rng.uniform(0, 2000 * 86400, 4000))
    clustered = np.sort(np.concatenate([rng.uniform(0, 2000 * 86400, 1000),
                                        np.concatenate([c + rng.exponential(86400 * 0.3, 60) for c in rng.uniform(0, 1990 * 86400, 50)])]))
    mk = lambda t: spark.createDataFrame([(float(x), 0.0, 0.0) for x in t], "t_sec double, latitude double, longitude double")  # noqa: E731
    p, c = ev.dispersion_stats(mk(poisson)), ev.dispersion_stats(mk(clustered))
    assert abs(p["dispersion_index"] - 1) < 0.25 and abs(p["cv_interevent"] - 1) < 0.1
    assert c["dispersion_index"] > 2 and c["cv_interevent"] > 1.3


# ------------------------------------------------------------------ declustering behaves as the plan says
def test_declustering_moves_background_towards_poisson(spark, events, cfg):
    """Raw catalog is clustered; after removing G-K aftershocks and NN dependents it is closer to Poisson."""
    base = events.filter("magnitude >= 4.5")
    _, after, _ = gk.run_gk(spark, base, 6.0)
    gk_lab = gk.gk_label_table(base, after)
    nn_lab, _, _, _ = nn_cluster_events(spark, columns_for_clustering(base), cfg, eta0_log10=-5.0)
    both = gk_lab.join(nn_lab, "event_id")
    raw = ev.dispersion_stats(base)
    gk_bg = ev.dispersion_stats(both.filter(~F_col("gk_is_aftershock")))
    nn_bg = ev.dispersion_stats(both.filter(~F_col("nn_is_dependent")))
    assert raw["dispersion_index"] > gk_bg["dispersion_index"]
    assert raw["dispersion_index"] > nn_bg["dispersion_index"]
    assert nn_bg["dispersion_index"] < 2.0


def F_col(name):
    from pyspark.sql import functions as F
    return F.col(name)


def test_sensitivity_monotone_in_eta0(spark, events, cfg):
    base = events.filter("magnitude >= 4.5")
    _, after, _ = gk.run_gk(spark, base, 6.0)
    gk_lab = gk.gk_label_table(base, after)
    nn_lab, nn, _, _ = nn_cluster_events(spark, columns_for_clustering(base), cfg, eta0_log10=-5.0)
    tab = ev.sensitivity_eta0(gk_lab.join(nn, "event_id"), [-8, -6, -5, -4, -2])
    assert tab.n_dependent.is_monotonic_increasing and tab.n_dependent.iloc[-1] > tab.n_dependent.iloc[0]


def test_case_study_summary():
    lab = pd.DataFrame({"event_id": list("abcdef"), "magnitude": [7, 5, 5, 4.8, 4.6, 4.5],
                        "gk_cluster": ["a"] * 4 + ["e", "f"], "nn_cluster": ["a"] * 3 + ["d", "e", "f"],
                        "gk_is_aftershock": [False, True, True, True, False, False],
                        "nn_is_dependent": [False, True, True, False, False, False]})
    r = ev.case_study(lab, {"title": "t"})
    assert r["gk_cluster_size"] == 4 and r["nn_cluster_size"] == 3 and abs(r["member_jaccard"] - 0.75) < 1e-12


# ------------------------------------------------------------------ DBSCAN reference
def test_dbscan_reference_finds_planted_sequences(catalog_pdf):
    pdf = catalog_pdf.assign(t_sec=catalog_pdf.event_time.astype("int64") / 1e9)
    out = st_dbscan_reference(pdf, eps_km=60, eps_days=7, min_samples=5)
    planted = out[out.true_cluster >= 0]
    assert adjusted_rand_score(planted.true_cluster, planted.db_cluster) > 0.6
    assert out[out.true_cluster < 0].db_is_dependent.mean() < 0.3
