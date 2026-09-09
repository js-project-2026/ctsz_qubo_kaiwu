# QUBO feature selection for scRNA-seq

Python reimplementation of the QUBO (quadratic unconstrained binary optimization) feature-selection method in:

> Romero, Gupta, Gatlin, Chapkin, Cai. *Quantum annealing for enhanced feature selection in single-cell RNA sequencing data analysis.* Quantum Machine Intelligence (2025) 7:114.

This repo applies the same **Eq. (4)** cost to **GEO GSE308682** 10x CRISPR scRNA-seq (not the paper’s hESC→EC or PC9 datasets).

## Method

$$
\min_{F} \left[ -\alpha \sum_{i} I_{i} F_{i} + (1-\alpha)\sum_{i,j} R_{ij} F_{i} F_{j} \right]
$$

with $Q = (1-\alpha)R - \alpha\,\mathrm{diag}(I)$, discrete mutual information $I$ and $R$ from quantile bins, and $\alpha$ searched so the unconstrained solution has about $k$ ones (authors’ $R/(k-1)$ scaling).

Real-data processing follows the paper’s Methods as closely as this matrix allows: library-size / mito / detection QC, analytic Pearson residuals ([Lause et al. 2021](https://doi.org/10.1038/s41592-021-01346-6)), a highly variable gene pool, and a continuous target $T$ (scanpy diffusion pseudotime or a held-out gene residual).

Local solvers: D-Wave Ocean `TabuSampler` / `SimulatedAnnealingSampler` (**classical**, from `dwave-ocean-sdk`). Optional quantum backends: D-Wave **Leap hybrid** (`SOLVER="leap"`) and QBoson **Kaiwu CIM** (`SOLVER="kaiwu_cim"`).

## Repository layout

| Path | Role |
| --- | --- |
| `qubo.ipynb` | Notebook (thin wrapper; `%autoreload` the `.py` modules) |
| `qubo_v1.py` | Same experiment as a script |
| `qubo_model.py` | Synthetic data, MI, $Q$, solvers, $\alpha$ search |
| `data_loader.py` | 10x load, QC, residuals, HVG, DPT |
| `tenx_parser.py` | Cell Ranger MTX/TSV(+gz) parser |
| `qubo_experiment.py` | LASSO path / random forest compare, MSE |
| `paths.py` | Resolves `QUBO_REPO_DIR` / `QUBO_DATA_DIR` |
| `docs/compare-romero-2025.html` | Latest notebook vs paper (English) |
| `docs/compare-romero-2025.zh.html` | Same report in Chinese, including GSE308682 train/test split |
| `docs/briefing-qubo-effort.html` | Bilingual progress report: k=50 at 5,000 genes, charts, optional CIM compare |
| `docs/quantum-ultra-early-markers.html` | Strategy essay (bilingual); §09 is this repo’s gene-panel instance |
| `docs/quantum-ultra-early-markers-v2.html` | Investor recast of the same science |

## Data

10x files are **not** in git. Set **`QUBO_DATA_DIR`** to the directory that contains GEO **GSE308682**:

```text
GSE308682_filtered_matrix.mtx.gz
GSE308682_filtered_features.tsv.gz
GSE308682_filtered_barcodes.tsv.gz
GSE308682_feature_reference.csv.gz
```

```bash
export QUBO_DATA_DIR=/path/to/gse308682_dir
# Optional: Ocean tabu budget in milliseconds per read (default scales with p; Ocean's 20 ms is not used).
# export QUBO_TABU_TIMEOUT_MS=8000
```

Or assign `os.environ["QUBO_DATA_DIR"]` at the top of `qubo_v1.py` / `qubo.ipynb` (see those files). Optional: `QUBO_REPO_DIR` if you launch Python from a directory that does not contain `qubo_model.py`. A template is in `.env.example`.

You can also pass `data_dir=` to `load_experiment(...)`. There is no machine-specific default path.

## Setup

This repo **exits** unless the interpreter is **Python 3.10.x** (Kaiwu’s official wheel is 3.10-only).

**Kaiwu** (`kaiwu_tabu`, `kaiwu_sa`, `kaiwu_cim`): use `.venv-py310` and that kernel:

```bash
# macOS Homebrew example
brew install python@3.10
python3.10 -m venv .venv-py310
source .venv-py310/bin/activate
pip install -r requirements.txt
pip install kaiwu==1.3.1 ipykernel
python -m ipykernel install --user --name qubo-py310 --display-name "Python 3.10 (Kaiwu)"
```

Then in the notebook: kernel picker → **Python 3.10 (Kaiwu)** → run the pip cell. A non-3.10 kernel raises `SystemExit`.

## Run

Set `QUBO_DATA_DIR` (above), then edit the flags at the top of `qubo.ipynb` / `qubo_v1.py`:

```python
USE_REAL_DATA = True
N_TOP_GENES = 5000
TARGET_MODE = "pseudotime"
SOLVER = "tabu"  # see solver table below
```

| `SOLVER` | Stack | Hardware | Env |
| --- | --- | --- | --- |
| `tabu` | `dwave-ocean-sdk` | classical CPU | `QUBO_TABU_TIMEOUT_MS` (ms/read; Ocean default 20 is too small at $p=5000$) |
| `sa` | `dwave-ocean-sdk` | classical CPU | — |
| `leap` | `dwave-ocean-sdk` | D-Wave Leap hybrid | `DWAVE_API_TOKEN` |
| `custom_sa` | this repo | classical CPU | — |
| `kaiwu_sa` | [kaiwu SDK](https://kaiwu-sdk-docs.qboson.com/) | classical CPU | none for CIM; optional local license |
| `kaiwu_tabu` | kaiwu SDK | classical CPU | none for CIM; optional local license |
| `kaiwu_cim` | kaiwu CIM + [kaiwu-pytorch-plugin](https://github.com/qboson/kaiwu-pytorch-plugin) | QBoson photonic CIM | `KAIWU_USER_ID`, `KAIWU_SDK_CODE` |

`kaiwu_tabu` / `kaiwu_sa` run on the local CPU. They do **not** need a CIM platform key. Set `COMPARE_KAIWU_TABU = True` in `qubo_v1.py` / `qubo.ipynb` to solve the **same** $Q$ with Ocean tabu and Kaiwu tabu. Official Kaiwu wheels target **Python 3.10**.

The last notebook run used **`tabu`**: Ocean is installed, but that sampler is not a QPU. To use D-Wave’s cloud hybrid solver, set `SOLVER = "leap"`. To use QBoson’s coherent Ising machine, install Kaiwu and set `SOLVER = "kaiwu_cim"` (that path **does** need keys).

```bash
pip install kaiwu==1.3.1 torch
pip install git+https://github.com/qboson/kaiwu-pytorch-plugin.git
export KAIWU_USER_ID=...
export KAIWU_SDK_CODE=...
```

Register at [platform.qboson.com](https://platform.qboson.com/) for CIM quota. The PyTorch plugin is an RBM/BM training layer on top of the same Kaiwu samplers; this repo uses those samplers on the feature-selection $Q$ matrix (Ising conversion via `kaiwu.conversion.qubo_matrix_to_ising_matrix`).

Do not commit API tokens:

```bash
export DWAVE_API_TOKEN=...
export KAIWU_USER_ID=...
export KAIWU_SDK_CODE=...
```

Then:

```bash
python qubo_v1.py
```

or run `qubo.ipynb` from the first cell. Pairwise MI at $p=5000$ takes several minutes; tqdm bars show progress.

## Results vs the paper

Open [`docs/compare-romero-2025.html`](docs/compare-romero-2025.html) or the Chinese version [`docs/compare-romero-2025.zh.html`](docs/compare-romero-2025.zh.html).

**This is not a drop-in replication** of Table 1 or the hESC/PC9 figures:

- Different tissue and $T$ (CRISPR hematopoiesis DPT vs endothelial / TKI trajectories).
- The corrected 5,000-gene tabu run returned **|F\*|=50** at α\*=0.3945, energy **−1.7771** (10 s/read, zero init). A prior 20 ms / random-init run with |F\*|=2,456 is discarded. Linear test MSE (leaky 70/30 after selection): QUBO 0.0063, LASSO 0.0048, RF 0.0064, all genes 0.0092. Overlaps at k=50: Q∩L 24/50, Q∩RF 11/50.

Synthetic §2.3 in `qubo_model.generate_synthetic_data` is the setting for Table 1–style source recall (`USE_REAL_DATA = False`).

