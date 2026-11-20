"""Step 0: USGS CSV -> cleaned Parquet + audit (rows, date range, M>=mc count).

Usage:  python scripts/00_prepare_data.py [--config config/default.json] [--set key=value ...]
Optional catalog extension (e.g. back to 2000): re-download with Assignment 2's code using a new
START_DATE, point `catalog_csv` at the new file, and re-run this script AND Part A (the G-K baseline
must be recomputed on the larger catalog).
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from seismic.common import banner, get_spark                    # noqa: E402
from seismic.config import add_common_args, cfg_from_args        # noqa: E402
from seismic.data import audit, load_catalog_csv, read_events, write_events  # noqa: E402


def main() -> None:
    ap = add_common_args(argparse.ArgumentParser(description=__doc__))
    cfg = cfg_from_args(ap.parse_args())
    spark = get_spark(cfg, "prepare")
    out = os.path.join(cfg.work_dir, "events.parquet")
    os.makedirs(cfg.work_dir, exist_ok=True)

    banner("loading and cleaning the catalog")
    ev = load_catalog_csv(spark, cfg.catalog_csv)
    write_events(ev, out)
    ev = read_events(spark, out)
    info = audit(ev, cfg.mc)
    info = {k: str(v) if k in ("first", "last") else v for k, v in info.items()}
    print(json.dumps(info, indent=2))
    with open(os.path.join(cfg.work_dir, "data_audit.json"), "w") as f:
        json.dump(info, f, indent=2)
    print("Saved:", out)


if __name__ == "__main__":
    main()
