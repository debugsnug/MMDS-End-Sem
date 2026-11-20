"""Geometry helpers. numpy versions (driver side) and Spark Column versions (executor side).

The Gardner-Knopoff and haversine formulas are carried over unchanged from Assignment 2.
"""
from __future__ import annotations

import math

import numpy as np
from pyspark.sql import functions as F

EARTH_R_KM = 6371.0088


# ---------------------------- numpy ----------------------------
def gk_radius_km_np(m):
    return 10 ** (0.1238 * np.asarray(m, float) + 0.983)


def gk_window_days_np(m):
    m = np.asarray(m, float)
    return np.where(m >= 6.5, 10 ** (0.032 * m + 2.7389), 10 ** (0.5409 * m - 0.547))


def haversine_km_np(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = p2 - p1
    dl = np.radians(lon2) - np.radians(lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * EARTH_R_KM * np.arcsin(np.sqrt(np.minimum(1.0, a)))


def unit_xyz_np(lat, lon):
    la, lo = np.radians(np.asarray(lat, float)), np.radians(np.asarray(lon, float))
    return np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)


def chord_unit(km: float) -> float:
    """Straight-line (chord) length on the unit sphere for a great-circle distance in km."""
    return 2.0 * math.sin(km / (2.0 * EARTH_R_KM))


# ---------------------------- Spark columns ----------------------------
def gk_radius_km_col(m):
    return F.pow(F.lit(10.0), 0.1238 * m + 0.983)


def gk_window_days_col(m):
    return F.when(m >= 6.5, F.pow(F.lit(10.0), 0.032 * m + 2.7389)).otherwise(
        F.pow(F.lit(10.0), 0.5409 * m - 0.547))


def haversine_km_col(lat1, lon1, lat2, lon2):
    dphi = F.radians(lat2 - lat1)
    dl = F.radians(lon2 - lon1)
    a = F.sin(dphi / 2) ** 2 + F.cos(F.radians(lat1)) * F.cos(F.radians(lat2)) * F.sin(dl / 2) ** 2
    return 2 * EARTH_R_KM * F.asin(F.sqrt(F.least(F.lit(1.0), a)))


def choose_grid_deg(radius_km, lat):
    """Smallest 'nice' lat/lon cell (degrees) that is >= the largest search radius.

    Used by the Gardner-Knopoff join (unchanged from Assignment 2). Wrap-around in
    longitude is handled by the caller with pmod.
    """
    r = np.asarray(radius_km, float)
    lat_edge = np.minimum(np.abs(np.asarray(lat, float)) + r / 111.195, 89.0)
    need = float(np.max(np.maximum(r / 111.195, r / (111.195 * np.cos(np.radians(lat_edge))))))
    for g in (1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 15, 18, 20, 24, 30, 36, 40, 45, 60, 90, 120, 180, 360):
        if g >= need:
            return g
    return 360
