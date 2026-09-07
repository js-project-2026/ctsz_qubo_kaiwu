# QUBO feature selection for scRNA-seq

Python reimplementation of the QUBO (quadratic unconstrained binary optimization) feature-selection method in:

> Romero, Gupta, Gatlin, Chapkin, Cai. *Quantum annealing for enhanced feature selection in single-cell RNA sequencing data analysis.* Quantum Machine Intelligence (2025) 7:114.

Authors’ code: [cailab-tamu/QUBO_feature_selection](https://github.com/cailab-tamu/QUBO_feature_selection).

This repo applies the same **Eq. (4)** cost to **GEO GSE308682** 10x CRISPR scRNA-seq (not the paper’s hESC→EC or PC9 datasets).

## Method

\[
\min_F -\alpha \sum_i I_i F_i + (1-\alpha)\sum_{i,j} R_{ij} F_i F_j
\]

with \(Q = (1-\alpha)R - \alpha\,\mathrm{diag}(I)\), discrete mutual information \(I\) and \(R\) from quantile bins, and \(\alpha\) searched so the unconstrained solution has about \(k\) ones (authors’ \(R/(k-1)\) scaling).

Real-data processing follows the paper’s Methods as closely as this matrix allows: library-size / mito / detection QC, analytic Pearson residuals ([Lause et al. 2021](https://doi.org/10.1038/s41592-021-01346-6)), a highly variable gene pool, and a continuous target \(T\) (scanpy diffusion pseudotime or a held-out gene residual).

Local solvers: D-Wave Ocean `TabuSampler` / `SimulatedAnnealingSampler` (**classical**, from `dwave-ocean-sdk`). Optional quantum backends: D-Wave **Leap hybrid** (`SOLVER="leap"`) and QBoson **Kaiwu CIM** (`SOLVER="kaiwu_cim"`).

## Repository layout

| Path | Role |
| --- | --- |
| `qubo.ipynb` | Notebook (thin wrapper; `%autoreload` the `.py` modules) |
| `qubo_v1.py` | Same experiment as a script |
| `qubo_model.py` | Synthetic data, MI, \(Q\), solvers, \(\alpha\) search |
| `data_loader.py` | 10x load, QC, residuals, HVG, DPT |
| `tenx_parser.py` | Cell Ranger MTX/TSV(+gz) parser |
| `qubo_experiment.py` | LASSO path / random forest compare, MSE |
| `docs/compare-romero-2025.html` | Latest notebook vs paper (open in a browser) |

## Data

10x files are **not** in git. Point the loader at a Cell Ranger directory (matrix + features + barcodes), default:

```text
/Users/<you>/Projects/sample_data
```

or set `data_dir` in `load_experiment(...)`. Expected study: **GSE308682**.

## Setup

Python 3.10+ (developed on 3.14).

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

Edit the flags at the top of `qubo.ipynb` / `qubo_v1.py`:

```python
USE_REAL_DATA = True
N_TOP_GENES = 5000
TARGET_MODE = "pseudotime"
SOLVER = "tabu"  # see solver table below
```

| `SOLVER` | Stack | Hardware | Env |
| --- | --- | --- | --- |
| `tabu` | `dwave-ocean-sdk` | classical CPU | — |
| `sa` | `dwave-ocean-sdk` | classical CPU | — |
| `leap` | `dwave-ocean-sdk` | D-Wave Leap hybrid | `DWAVE_API_TOKEN` |
| `custom_sa` | this repo | classical CPU | — |
| `kaiwu_sa` | [kaiwu SDK](https://kaiwu-sdk-docs.qboson.com/) | classical CPU | optional license |
| `kaiwu_tabu` | kaiwu SDK | classical CPU | optional license |
| `kaiwu_cim` | kaiwu CIM + [kaiwu-pytorch-plugin](https://github.com/qboson/kaiwu-pytorch-plugin) | QBoson photonic CIM | `KAIWU_USER_ID`, `KAIWU_SDK_CODE` |

The last notebook run used **`tabu`**: Ocean is installed, but that sampler is not a QPU. To use D-Wave’s cloud hybrid solver, set `SOLVER = "leap"`. To use QBoson’s coherent Ising machine, install Kaiwu and set `SOLVER = "kaiwu_cim"`.

```bash
pip install kaiwu==1.3.1 torch
pip install git+https://github.com/qboson/kaiwu-pytorch-plugin.git
export KAIWU_USER_ID=...
export KAIWU_SDK_CODE=...
```

Register at [platform.qboson.com](https://platform.qboson.com/) for CIM quota. The PyTorch plugin is an RBM/BM training layer on top of the same Kaiwu samplers; this repo uses those samplers on the feature-selection \(Q\) matrix (Ising conversion via `kaiwu.conversion.qubo_matrix_to_ising_matrix`).

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

or run `qubo.ipynb` from the first cell. Pairwise MI at \(p=5000\) takes several minutes; tqdm bars show progress.

## Results vs the paper

Open [`docs/compare-romero-2025.html`](docs/compare-romero-2025.html).

**This is not a drop-in replication** of Table 1 or the hESC/PC9 figures:

- Different tissue and \(T\) (CRISPR hematopoiesis DPT vs endothelial / TKI trajectories).
- The 5,000-gene + tabu notebook run returned **|F\*|=2,456** instead of **k=50** (energy large and positive). Treat that gene list as an incomplete solve, not a paper-style top-50 set.

Synthetic §2.3 in `qubo_model.generate_synthetic_data` is the setting for Table 1–style source recall (`USE_REAL_DATA = False`).

## License

MIT. The QUBO formulation is from Romero et al. (2025); cite that paper if you use this method.
