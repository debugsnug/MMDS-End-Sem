import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from seismic.config import Config          # noqa: E402
from seismic.data import pandas_to_events  # noqa: E402
from seismic.synthetic import make_catalog  # noqa: E402


@pytest.fixture(scope="session")
def cfg():
    return Config(master="local[2]", shuffle_partitions=4, driver_memory="2g",
                  work_dir="/tmp/seismic_test_work", seeds=[1], n_boot=100,
                  grid_lr=[0.01], grid_dt=[6], grid_rf=[6], rf_num_trees=20, grid_gbt=[10],
                  zone_k_candidates=[5, 8], models=["lr", "dt", "rf", "gbt"],
                  train_end="2017-07-01", val_end="2018-10-01")


@pytest.fixture(scope="session")
def spark(cfg):
    from seismic.common import get_spark
    s = get_spark(cfg, "seismic-tests")
    yield s
    s.stop()


@pytest.fixture(scope="session")
def catalog_pdf():
    return make_catalog(seed=3, years=6.0, n_background=1000)


@pytest.fixture(scope="session")
def events(spark, catalog_pdf):
    return pandas_to_events(spark, catalog_pdf).cache()
