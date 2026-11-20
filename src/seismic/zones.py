"""Seismic zones with MLlib clustering (KMeans, BisectingKMeans, GaussianMixture).

Clustering is done on unit-sphere coordinates (x, y, z), never on raw latitude/longitude,
because longitude wraps at +-180 degrees and degrees of longitude shrink towards the poles.
Zones are descriptive (k-means assumes compact blobs, fault zones are elongated) and serve as
a categorical feature in Part B. They are always *fitted on training rows only*.
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
from pyspark.ml.clustering import BisectingKMeans, GaussianMixture, KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.ml.feature import VectorAssembler
from pyspark.sql import DataFrame, functions as F

KINDS = {"kmeans": KMeans, "bisecting": BisectingKMeans, "gmm": GaussianMixture}


def add_unit_xyz(df: DataFrame) -> DataFrame:
    la, lo = F.radians("latitude"), F.radians("longitude")
    # names are prefixed so they can never collide with a label column called "y"
    return (df.withColumn("_ux", F.cos(la) * F.cos(lo)).withColumn("_uy", F.cos(la) * F.sin(lo))
              .withColumn("_uz", F.sin(la)))


def _vectorise(df: DataFrame) -> DataFrame:
    base = df if {"_ux", "_uy", "_uz"} <= set(df.columns) else add_unit_xyz(df)
    return VectorAssembler(inputCols=["_ux", "_uy", "_uz"], outputCol="xyz").transform(base)


def fit_zone_model(train: DataFrame, kind: str, k: int, seed: int = 1):
    """Fit one zone model on `train` (rows with latitude/longitude)."""
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {list(KINDS)}")
    est = KINDS[kind](k=k, seed=seed, featuresCol="xyz", predictionCol="zone_id")
    return est.fit(_vectorise(train))


def assign_zone(model, df: DataFrame) -> DataFrame:
    """Add integer column zone_id using an already-fitted model (never re-fits)."""
    out = model.transform(_vectorise(df.drop("zone_id") if "zone_id" in df.columns else df)).drop("xyz")
    return out.withColumn("zone_id", F.col("zone_id").cast("int"))


def zone_metrics(model, df: DataFrame, sample: float = 0.3, seed: int = 1) -> Dict[str, float]:
    """Silhouette (squared Euclidean, on a sample), cluster-size spread and the number of non-empty zones."""
    vec = _vectorise(df)
    pred = model.transform(vec)
    sizes = [r["count"] for r in pred.groupBy("zone_id").count().collect()]
    s = pred.sample(False, min(1.0, sample), seed) if sample < 1.0 else pred
    try:
        sil = ClusteringEvaluator(predictionCol="zone_id", featuresCol="xyz",
                                  metricName="silhouette", distanceMeasure="squaredEuclidean").evaluate(s)
    except Exception:  # fewer than 2 populated clusters in the sample
        sil = float("nan")
    return {"silhouette": float(sil), "n_zones_nonempty": len(sizes), "size_min": int(min(sizes)),
            "size_median": float(np.median(sizes)), "size_max": int(max(sizes))}


def compare_zone_models(train: DataFrame, kinds, ks, seed: int = 1, sample: float = 0.3):
    """Fit every (kind, k) on training rows and report metrics (feeds the 'MLlib clustering' comparison)."""
    import pandas as pd
    rows = []
    for kind in kinds:
        for k in ks:
            try:
                m = fit_zone_model(train, kind, k, seed)
                rows.append({"kind": kind, "k": k, **zone_metrics(m, train, sample, seed), "error": ""})
            except Exception as exc:  # e.g. singular GMM covariance on very tight clusters
                rows.append({"kind": kind, "k": k, "error": str(exc)[:120]})
    return pd.DataFrame(rows)
