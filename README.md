# Scalable Clustering and Classification of Seismic Sequences with Spark

Implementation of the end-semester plan (MMDS). Same USGS catalog as Assignment 2 (2,145,525 events, 2010–2024).

* **Part A (clustering):** Gardner–Knopoff (G-K) baseline vs nearest-neighbour clustering (Zaliapin–Ben-Zion) in Spark, MLlib zone models (KMeans / BisectingKMeans / GMM), optional single-machine DBSCAN reference. No ground truth exists, so evaluation = overlap with G-K, ARI, Båth's law, Omori decay, Poisson-likeness of the background, η₀ sensitivity, case studies.
* **Part B (classification, MLlib):** does adding cluster-derived features help predict "a strictly larger event within 7 days and 50 km"? Four classifiers (LR, DT, RF, GBT) × feature sets A/B/C, time-based split, validation-only tuning, paired bootstrap.
* **Scalability and streaming:** catalog-size / cores / partitions / salting / skew experiments; Structured Streaming scoring of a replayed sequence with the saved `PipelineModel`.

This is statistical classification on a catalog. It makes **no earthquake-forecasting claim**.

## Assumptions (correct these first)

| # | Assumption | Where to change |
|---|---|---|
| A1 | Input is the Assignment 2 USGS CSV with columns `time, latitude, longitude, depth, mag, id, place, type` | `catalog_csv` |
| A2 | Runs in Colab/local mode with Drive for storage; Java 8/11/17/21, PySpark 3.5.1, pandas < 3 | `requirements.txt` |
| A3 | Clustering catalog and Part B triggers use **M ≥ 4.5** (global completeness) | `mc`, `trigger_min_mag` |
| A4 | Label: strictly larger event within **7 days and 50 km** (frozen after the label audit, before modelling) | `label_horizon_days`, `label_radius_km` |
| A5 | Split: train < 2021-01-01, validation 2021–2022, test 2023-01-01 → end; the 7 days before each boundary and the last 7 days are dropped | `train_end`, `val_end` |
| A6 | NN search box: parents ≤ 365 days earlier and within 3 × the G-K radius of the parent; events with no parent in the box are background | `nn_max_days`, `nn_radius_scale` |
| A7 | NN parameters b = 1.0, d_f = 1.6; η₀ estimated with a 2-component MLlib GMM on log₁₀η, fallback −5.0 | `nn_b`, `nn_df`, `nn_eta0_log10` |
| A8 | "Dependent event" (NN) = event with a parent below η₀; compared with G-K "aftershock", both restricted to M ≥ mc | `evaluation.py` |
| A9 | Local activity features count earlier M ≥ mc events within 100 km over 24 h / 7 d / 30 d | `activity_*` |
| A10 | Primary metric: PR-AUC as average precision on collected test scores; Spark `areaUnderPR` is logged as a cross-check; chance level = test base rate | `modeling.py` |
| A11 | Primary comparison: set C vs set B with the classifier that is best on validation for set B; bootstrap over test events (1000 replicates) | `n_boot` |
| A12 | Seeds 1–3 for the stochastic models (RF, GBT); LR/DT are deterministic and run once | `seeds` |
| A13 | Arrow transfer is **off** (it breaks on Java 21); turn on with `use_arrow=true` on Java 8/11/17 for faster `toPandas` | `use_arrow` |

## Layout

```
MMDS/
├── README.md, requirements.txt, pytest.ini
├── config/default.json        real run (Colab/Drive paths)
├── config/demo.json           synthetic smoke run (/tmp paths, tiny grids)
├── src/seismic/
│   ├── config.py              every tunable (Config dataclass, JSON + --set overrides, replay presets)
│   ├── common.py              Spark session, StageTimer
│   ├── geo.py                 haversine, G-K formulas, grid chooser (numpy + Spark column versions)
│   ├── data.py                CSV cleaning (Assignment 2 rules), parquet IO, audit, stress-test replication
│   ├── gk.py                  Gardner–Knopoff baseline (Assignment 2 logic) + label table
│   ├── blocking.py            3-D cell + time-bucket blocked join, salting, skew helper
│   ├── nn_cluster.py          nearest-neighbour search, η₀ estimation, pointer-jumping cluster roots
│   ├── zones.py               MLlib KMeans/Bisecting/GMM on the unit sphere
│   ├── evaluation.py          overlap, ARI, Båth, Omori MLE, dispersion, η₀ sensitivity, case studies
│   ├── dbscan_ref.py          optional single-machine space-time DBSCAN
│   ├── features.py            label, causal features A/B/C, time split
│   ├── modeling.py            pipelines, tuning, scoring, bootstrap, resumable orchestration
│   ├── runlog.py              resumable CSV run log
│   ├── scalability.py         timed runs, experiment plan, skew report
│   ├── streaming.py           stream state, scorer, file-stream replay, batch comparison
│   └── synthetic.py           synthetic catalog with planted sequences (tests / demo only)
├── scripts/
│   ├── 00_prepare_data.py     CSV → parquet + audit
│   ├── 01_part_a_clustering.py
│   ├── 02_part_b_classification.py
│   ├── 03_scalability.py
│   ├── 04_streaming_scoring.py
│   └── make_demo_data.py      synthetic USGS-format CSV
└── tests/                     pytest suite (see "Verification")
```

## Run order (Colab)

```python
from google.colab import drive; drive.mount('/content/drive')
!pip -q install -r seismic_spark/requirements.txt      # Colab already has Java; restart runtime if pandas changes
%cd seismic_spark
!python scripts/00_prepare_data.py          --config config/default.json
!python scripts/01_part_a_clustering.py     --config config/default.json --dbscan
!python scripts/02_part_b_classification.py --config config/default.json
!python scripts/03_scalability.py           --config config/default.json
!python scripts/04_streaming_scoring.py     --config config/default.json
```

Override anything without editing files, e.g. `--set mc=4.0 --set 'models=["lr","rf"]' --set seeds=[1]`.
All outputs go to `<work_dir>/{part_a,part_b,scalability,streaming}/`. Part B and scalability are **resumable**: re-running skips logged runs.
If you change `mc`, `trigger_min_mag`, label or feature settings, delete `<work_dir>/part_b/` first (the feature table is cached).

**Smoke run without the real data (≈ 10 min on 2 cores):**

```bash
python scripts/make_demo_data.py
for s in 00_prepare_data 01_part_a_clustering 02_part_b_classification 03_scalability 04_streaming_scoring; do
  python scripts/$s.py --config config/demo.json; done
```
Demo numbers come from synthetic data and mean nothing scientifically; the run only proves the code path.

## How each plan item is implemented

| Plan item | Code |
|---|---|
| G-K baseline reused | `gk.py` (mainshock selection and grid-blocked join unchanged from Assignment 2) |
| Nearest-neighbour clustering reusing a blocked join | `blocking.py` + `nn_cluster.py` — **same idea, different blocking**: event-to-event search needs time as well as space, and a lat/lon grid distorts near the poles, so cells are cubes of the unit-sphere embedding plus time buckets. The G-K join keeps the Assignment 2 lat/lon grid |
| MLlib KMeans / Bisecting / GMM zones (unit-sphere coords) | `zones.py`, `compare_zone_models` |
| DBSCAN single-machine reference, labelled "not distributed" | `dbscan_ref.py`, `--dbscan` |
| Overlap (Jaccard, ARI), Båth, Omori, Poisson check, sensitivity, case studies | `evaluation.py`, `01_part_a_clustering.py` |
| Label frozen after audit; time split with gaps | `features.py`, `02_part_b_classification.py` (`label_audit.json`) |
| Causal features A/B/C | `features.build_feature_table` |
| Zone KMeans fitted on training rows only, k chosen on validation | `modeling.select_zone_k`, `run_part_b` |
| 4 classifiers, same grid per set, manual time-aware tuning | `modeling.py` (`tune`) |
| PR-AUC vs base rate, ROC-AUC, P/R/F1/confusion at validation threshold | `modeling.rank_metrics`, `metrics_at` |
| Paired bootstrap C vs B; "interval includes zero ⇒ no clear difference" | `modeling.paired_bootstrap_diff` |
| Class-weight sensitivity | `run_part_b` (`weightCol`) |
| Size / cores / partitions / skew / salting / pair reduction | `scalability.py`, `03_scalability.py` |
| Replicated-data stress test, clearly labelled | `data.replicate_catalog`, `--set stress_factor=N` |
| Streaming scoring with saved `PipelineModel`; stream vs batch | `streaming.py`, `04_streaming_scoring.py` |
| Resumable CSV log of every run | `runlog.py` |

## Verification

```bash
pip install -r requirements.txt
python -m pytest                     # whole suite (≈ 10–15 min on 1–2 cores; Spark start-up dominates)
python -m pytest -m "not slow"       # skips the real file-stream replay
```

| Test file | What it proves |
|---|---|
| `test_clustering_core.py` | Blocked NN equals an O(n²) brute-force reference (with and without salting, incl. antimeridian and polar events); blocking prunes pairs; G-K blocked = unblocked; planted sequences are recovered (ARI > 0.7); pointer jumping resolves a 200-deep chain; η₀ estimate separates two modes |
| `test_zones_eval.py` | Zones treat longitude ±180 as one place; zone assignment never re-fits; ARI equals scikit-learn; Båth ignores foreshocks; Omori MLE recovers p; dispersion ≈ 1 for Poisson and > 1 for clustered; **declustering lowers the dispersion index**; η₀ sensitivity is monotone; DBSCAN reference finds planted sequences |
| `test_features_modeling.py` | Labels and activity features equal brute force; **features are causal** (deleting future events changes nothing, incl. NN and G-K flag); split has gaps and is disjoint; bootstrap detects a real gain and reports "includes zero" for noise; full Part B run, class-weight runs, aligned score files, and **resume produces identical results without refitting** |
| `test_scalability_streaming.py` | Time-window fractions are nested; pair reduction and skew are reported; salting does not change results; stress replication does not mix copies; incremental stream state finds the same mainshocks as batch G-K; **streaming scores equal batch scores (max diff < 1e-9)**; real Structured Streaming file replay |

## Known limitations (state these in the report)

* **No ground truth** for aftershock labels. Agreement with G-K is not accuracy; the physical checks show plausibility only.
* **Search-box truncation** in the NN method (A6) is the one approximation; blocking itself is exact (tested against brute force). Parents further than the box are not found, so those events count as background.
* **η₀** matters: the sensitivity sweep and the GMM estimate are reported; a fixed value is a choice, not a fact.
* **G-K window sizes** (up to ~925 days for the largest events) are generous, which motivated this project; do not over-claim that clustering is "right".
* **Part B is hard**: foreshock-style identification is known to be difficult. A small gain or no gain is a valid finding. Judge PR-AUC against the base rate printed with every result.
* **Streaming vs batch agreement is a consistency check** (shared feature code), not independent validation of the model.
* **Local mode** only: speed-up is capped by the machine's cores, and nothing here says how a real cluster would behave. ~2M rows is large but not huge; the optional stress test is runtime-only.
* Zones from k-means are descriptive (fault zones are elongated).
