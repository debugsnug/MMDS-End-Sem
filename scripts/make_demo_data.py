"""Write a small synthetic catalog in the USGS CSV format (for smoke-testing the whole pipeline offline).

Usage:  python scripts/make_demo_data.py --out /tmp/seismic_demo/demo_catalog.csv
        python scripts/00_prepare_data.py --config config/demo.json   # then run steps 01..04 with the same config
The synthetic catalog contains planted sequences (including a Ridgecrest-like M7.1 on 2019-07-06 so the
streaming replay preset has something to replay). It is NOT real data and its metrics mean nothing.
"""
import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from seismic.synthetic import make_catalog  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/tmp/seismic_demo/demo_catalog.csv")
    ap.add_argument("--seed", type=int, default=5)
    args = ap.parse_args()
    start = pd.Timestamp("2010-01-01", tz="UTC")
    ridge = (pd.Timestamp("2019-07-06 03:19", tz="UTC") - start).days + 0.1
    seqs = [(35.5, -117.5, 6.8, 300.0), (-20.2, -69.8, 6.4, 700.0), (38.0, 142.0, 7.4, 1060.0),
            (60.0, -150.0, 6.5, 1500.0), (-35.9, -72.7, 7.3, 1900.0), (35.77, -117.60, 7.1, ridge),
            (-18.0, 179.8, 6.9, 3300.0), (35.0, -117.0, 6.9, 3900.0), (38.5, 141.5, 6.6, 4400.0),
            (-20.0, -70.0, 6.7, 4900.0), (60.5, -149.5, 6.8, 5200.0), (35.2, -117.2, 7.0, 5350.0)]
    cat = make_catalog(seed=args.seed, years=15.0, n_background=6000, sequences=seqs, start="2010-01-01")
    out = pd.DataFrame({"time": cat.event_time.dt.strftime("%Y-%m-%dT%H:%M:%S.%f").str[:-3] + "Z",
                        "latitude": cat.latitude, "longitude": cat.longitude, "depth": cat.depth_km,
                        "mag": cat.magnitude.round(2), "magType": "mw", "net": "demo", "id": cat.event_id,
                        "place": cat.place, "type": "earthquake"})
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"Wrote {len(out):,} synthetic events to {args.out}")


if __name__ == "__main__":
    main()
