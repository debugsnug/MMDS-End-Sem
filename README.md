# Scalable Clustering and Classification of Seismic Sequences on the USGS Earthquake Catalog Using Apache Spark

## 1. Project Overview

This project investigates large-scale earthquake sequence analysis using Apache Spark, statistical methods, clustering algorithms, and machine learning.

Using the historical USGS earthquake catalog, the project implements earthquake sequence identification, declustering, subsequent-event classification, scalability benchmarking, streaming evaluation, and historical case-study reanalysis.

The dataset contains **2,145,569 raw earthquake records** covering 2010–2024. After removing 7,669 records without magnitude values, **2,137,900 events** remain for analysis.

### Project Components

- **Part A – Clustering and Declustering:** Comparison of the Gardner–Knopoff (G–K) method and nearest-neighbour earthquake declustering.
- **Part B – Classification:** Machine-learning models for classifying subsequent larger-event occurrence.
- **Part C – Scalability:** Performance evaluation across different core counts, dataset sizes, partition configurations, and synthetic workloads.
- **Part D – Streaming:** Micro-batch streaming evaluation and comparison with batch scoring.
- **Part E – Reanalysis:** Historical earthquake sequence case studies.

## 2. Technology Stack

| Technology | Purpose |
|---|---|
| Python | Core implementation |
| Apache Spark / PySpark | Distributed data processing |
| NumPy | Numerical computation |
| Pandas | Data manipulation |
| SciPy | Scientific computing |
| Scikit-learn | Machine learning and clustering |
| Matplotlib | Visualization |
| Pytest | Automated testing |

### Validated Environment

The implementation was tested using:

| Component | Version |
|---|---|
| Operating System | Ubuntu |
| Python | 3.10.12 |
| Java | OpenJDK 17.0.19 |
| PySpark | 3.5.1 |
| NumPy | 2.2.6 |
| Pandas | 2.3.3 |
| Scikit-learn | 1.7.2 |
| SciPy | 1.15.3 |
| Pytest | 9.2.0 |

Dependency specifications are available in `requirements.txt`.

## 3. Project Structure

```text
MMDS Project 2/
│
├── config/
│   ├── default.json
│   ├── causal.json
│   ├── k40.json
│   └── demo.json
│
├── dataset/
│   └── usgs_earthquakes_historical.csv
│
├── scripts/
│   ├── 00_prepare_data.py
│   ├── 01_part_a_clustering.py
│   ├── 02_part_b_classification.py
│   ├── 03_scalability.py
│   ├── 04_streaming_scoring.py
│   └── 05_reanalysis.py
│
├── src/
│   └── seismic/
│       └── Core implementation modules
│
├── tests/
│   └── Automated test suite
│
├── reference_results/
│   └── Retained reference experiment outputs
│
├── main_experiment/
│   └── Main experiment outputs
│
├── causal_experiment/
│   └── Causal G–K experiment outputs
│
├── cluster_sensitivity_k40/
│   └── Controlled k=40 experiment outputs
│
├── CHANGES.md
├── README.md
├── requirements.txt
└── pytest.ini
```

The experiment directories contain saved outputs from the corresponding runs. `reference_results/` retains reference outputs and should not be confused with the latest main experiment results.

## 4. Environment Setup

Run all commands from the project root directory.

### Step 1 – Create a Virtual Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Step 2 – Install Dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Step 3 – Verify the Environment

```bash
python --version
java -version
python -c "import pyspark; print(pyspark.__version__)"
```

Ensure Java is available before executing the Spark pipeline.

## 5. Dataset

The project uses the historical USGS earthquake catalog.

Expected dataset location:

```text
dataset/usgs_earthquakes_historical.csv
```

Dataset statistics:

| Description | Count |
|---|---:|
| Raw records | 2,145,569 |
| Records without magnitude | 7,669 |
| Processed earthquake events | 2,137,900 |
| Time period | 2010–2024 |

The configuration files use paths relative to the project root rather than machine-specific paths.

### Configuration Files

| Configuration | Description |
|---|---|
| `config/default.json` | Main experiment configuration |
| `config/causal.json` | Causal G–K experiment configuration |
| `config/k40.json` | Controlled k=40 sensitivity experiment |
| `config/demo.json` | Demonstration configuration |

## 6. Execution Instructions

Activate the virtual environment and execute the following commands from the project root.

### Step 1 – Data Preparation

```bash
python scripts/00_prepare_data.py --config config/default.json
```

### Step 2 – Part A: Clustering and Declustering

```bash
python scripts/01_part_a_clustering.py --config config/default.json
```

### Step 3 – Part B: Classification

```bash
python scripts/02_part_b_classification.py --config config/default.json
```

The current implementation uses chronological, causal G–K mainshock selection.

### Step 4 – Part C: Scalability Evaluation

```bash
python scripts/03_scalability.py --config config/default.json
```

### Step 5 – Part D: Streaming Evaluation

```bash
python scripts/04_streaming_scoring.py --config config/default.json --preset tohoku_2011
```

### Step 6 – Part E: Historical Reanalysis

```bash
python scripts/05_reanalysis.py --config config/default.json
```

### Optional – Controlled k=40 Experiment

```bash
python scripts/02_part_b_classification.py --config config/k40.json
```

Experiment outputs are saved under the working directory specified in the selected configuration.

## 7. Experimental Results

### Part A – Clustering and Declustering

For the earthquake subset with magnitude **M ≥ 4.5**, the following results were obtained:

| Metric | G–K | Nearest Neighbour |
|---|---:|---:|
| Dependent events | 36,450 | 39,663 |
| Percentage of catalog subset | 32.40% | 35.26% |
| Dispersion statistic | 2.5288 | 1.0729 |

The raw catalog dispersion statistic was **10.6441**.

Additional analyses include:

- Event overlap and Jaccard similarity
- Adjusted Rand Index (ARI)
- Båth's law analysis
- Omori-law analysis
- Nearest-neighbour eta sensitivity
- Spatial clustering
- Ridgecrest (2019), Tohoku (2011), and Maule (2010) case studies

The two declustering approaches use different definitions of dependent events. Their results should therefore be interpreted with these methodological differences and magnitude thresholds in mind.

### Part B – Subsequent-Event Classification

The classification target identifies whether a strictly larger earthquake occurs within **7 days and 50 km** of an event.

This is a subsequent-event classification task, not a validated earthquake foreshock-prediction system.

The main experiment produced the following average precision scores:

| Feature Configuration | Average Precision |
|---|---:|
| A | 0.3301 |
| B | 0.7939 |
| C | 0.7972 |

Local activity information accounts for the largest improvement in classification performance.

The incremental improvement from configuration B to C was approximately **0.00335**. Its cluster-bootstrap 95% confidence interval was approximately **[0.00047, 0.00631]** in the main experiment.

The controlled k=40 experiment showed that this incremental improvement was sensitive to the spatial clustering configuration.

### Part C – Scalability Evaluation

For the 112,484-event scalability workload:

| Number of Cores | Median Runtime |
|---:|---:|
| 1 | 13.44 seconds |
| 2 | 10.62 seconds |
| 4 | 9.19 seconds |

Larger synthetic workloads were also evaluated:

| Workload | Events | Runtime |
|---|---:|---:|
| 4× | 449,936 | 18.81 seconds |
| 8× | 899,872 | 35.36 seconds |

Additional experiments examined catalog-size scaling, partition counts, and data salting.

Runtime measurements are environment-dependent and were calculated using repeated runs.

### Part D – Streaming Evaluation

The Tohoku 2011 replay processed **1,322 events** across **61 micro-batches**.

| Metric | Result |
|---|---:|
| Events processed | 1,322 |
| Micro-batches | 61 |
| Median micro-batch runtime | 2.326 seconds |
| Maximum micro-batch runtime | 4.899 seconds |
| Maximum absolute batch-versus-stream score difference | 0 |

Batch and streaming event counts matched, and the maximum absolute score difference was zero.

The replay had a high positive-label rate. Therefore, this experiment demonstrates streaming execution and batch-versus-stream consistency rather than general predictive quality.

### Part E – Historical Reanalysis

Historical earthquake sequences were reanalysed using the implemented clustering and classification workflow.

The case studies include:

- Ridgecrest, 2019
- Tohoku, 2011
- Maule, 2010

These case studies provide additional context for examining earthquake sequence behaviour and the differences between declustering methods.

## 8. Causal G–K Implementation

The G–K implementation was extended with chronological, causal mainshock selection.

The causal procedure processes events chronologically and considers active earlier mainshocks instead of selecting mainshocks retrospectively based on later events.

The Part B classification script was updated to use this implementation.

Validation compared the causal and earlier implementations on the evaluated catalog:

- G–K event flags were identical.
- Complete feature tables were identical.
- Classification metrics remained unchanged.

Further implementation details are documented in `CHANGES.md`.

## 9. Testing and Validation

The project includes an automated test suite.

Run all tests using:

```bash
python -m pytest -v
```

**Validation result: 52 tests passed.**

Additional validation covered:

- Spark execution and environment checks
- Dataset preparation
- Feature generation
- Clustering and classification
- Causal G–K output comparison
- Streaming-versus-batch consistency

## 10. Reproducibility Notes

- Execute commands from the project root directory.
- Ensure the dataset is available at the configured path.
- Use the validated Python, Java, and PySpark environment where possible.
- Configuration files control experiment parameters and output directories.
- Experimental outputs from different configurations should not be treated as directly interchangeable.
- Runtime measurements may vary with hardware and system load.
- The classification target represents subsequent larger-event occurrence within a defined space-time window; it does not establish earthquake predictability.

---

**Project:** Scalable Clustering and Classification of Seismic Sequences on the USGS Earthquake Catalog Using Apache Spark
