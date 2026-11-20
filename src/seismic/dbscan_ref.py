"""Optional single-machine space-time DBSCAN reference (NOT distributed; MLlib has no DBSCAN).

Space and time are combined into one Euclidean metric by scaling time so that
`eps_days` of time counts like `eps_km` of distance:  d^2 = chord_km^2 + (dt_days * eps_km/eps_days)^2.
Chord distance is used for space (indistinguishable from great-circle distance at these scales).
This is an approximation of a space-time cylinder (it is a space-time ball), stated in the report.
Intended for the M>=4.5 subset only (~1e5 events); memory grows with neighbourhood sizes.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN

from .geo import EARTH_R_KM, unit_xyz_np


def st_dbscan_reference(pdf: pd.DataFrame, eps_km: float = 50.0, eps_days: float = 7.0,
                        min_samples: int = 5) -> pd.DataFrame:
    """pdf needs event_id, t_sec, latitude, longitude, magnitude.

    Returns pdf with `db_cluster` (-1 = noise) and `db_is_dependent`: a clustered event that is
    not the largest event of its cluster (same notion of 'dependent' as the NN method).
    """
    x, y, z = unit_xyz_np(pdf.latitude.values, pdf.longitude.values)
    feats = np.column_stack([x * EARTH_R_KM, y * EARTH_R_KM, z * EARTH_R_KM,
                             (pdf.t_sec.values / 86400.0) * (eps_km / eps_days)])
    labels = DBSCAN(eps=eps_km, min_samples=min_samples, algorithm="ball_tree", n_jobs=-1).fit_predict(feats)
    out = pdf.copy()
    out["db_cluster"] = labels
    out["db_is_dependent"] = False
    for cl, idx in out[out.db_cluster >= 0].groupby("db_cluster").groups.items():
        sub = out.loc[idx]
        biggest = sub.magnitude.idxmax()
        out.loc[[i for i in idx if i != biggest], "db_is_dependent"] = True
    return out
