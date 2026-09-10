# QUBO v2 — Ranzoni / Cvejic fetal hematopoiesis (Smart-seq2)

Fetal liver + bone marrow scRNA-seq from Ranzoni et al., *Cell Stem Cell* (2021).
Data: local clone of
[cvejic-group/integrative-scrna-scatac-human-foetal](https://gitlab.com/cvejic-group/integrative-scrna-scatac-human-foetal).

The v1 GSE308682 10x pipeline (`qubo.ipynb`, `data_loader.py` at repo root) is unchanged.
This folder reuses `qubo_model.py` (MI, Eq. (4), Tabu, α-search) on a different matrix.
v2 files are named `foetal_loader.py` / `foetal_paths.py` so they do not shadow v1 `data_loader` / `paths` when the notebook cwd is `v2/`.

## What is loaded

| Object | Role |
| --- | --- |
| `MergedAllSamples.h5ad` | post-QC STAR counts (4,504 cells) |
| `MergedAllSamples_PAGA.h5ad` | hematopoietic cells + published `dpt_pseudotime` |

Endothelial cells are already absent from the PAGA object. Default **X** is **log1p(CPM)** (Ranzoni-like), not Pearson residuals. Default **T** is the authors’ PAGA dpt (HSC/MPP root), not min-HBE1 DPT.

FASTQ (`E-MTAB-9067`) and `scATAC_CSV_file_for_Scanpy` are not used.

## Paths

```bash
export QUBO_V2_DATA_DIR=/path/to/integrative-scrna-scatac-human-foetal
# or the Data/ScanpyObjets folder inside that clone
```

## Smoke test vs benchmark

| | Smoke | Benchmark |
| --- | --- | --- |
| Flag | `SMOKE = True` | `SMOKE = False` (default) |
| Cells | 200 subsample | all PAGA hematopoietic cells (~4,463) |
| Genes | 80 HVGs | 5,000 HVGs |
| k | 10 | 50 |
| MI / tabu | seconds | minutes (pairwise MI at p=5000) |

Smoke still uses the **real h5ad**, just a cell/gene subset. It is not synthetic data.

## Run

From this folder or the repo root (the script adds both to `sys.path`):

```bash
# smoke
python v2/qubo_v2.py

# benchmark: edit SMOKE = False in v2/qubo_v2.py (or the notebook), then
python v2/qubo_v2.py
```

Notebook: `v2/qubo_v2.ipynb`. Set `QUBO_V2_DATA_DIR` in the setup cell. Same kernel notes as v1 if you want Kaiwu (Python 3.10).

## Knobs

- `TRANSFORM = "log"` (default) or `"pearson"` (Lause residuals; UMI-oriented, sensitivity only)
- `TARGET_MODE = "paga_dpt"` (default), `"dpt"` (recompute DPT, iroot = max MLLT3 in HSC/MPP), or `"gene"`
- `ROOT_GENE = "MLLT3"`

## Results vs Ranzoni / Cvejic

Open [`docs_v2/compare-ranzoni-2021.html`](../docs_v2/compare-ranzoni-2021.html) (English) or [中文](../docs_v2/compare-ranzoni-2021.zh.html). Bilingual briefing: [`docs_v2/briefing-qubo-v2.html`](../docs_v2/briefing-qubo-v2.html).

Notebook run (4,463 × 5,000, published PAGA dpt): Ocean **|F\*|=51**, E=−0.7813, PASS; Kaiwu **|F\*|=45**, overlap 31/50. Q∩L 15/50. Ridge MSE: QUBO 0.0060, LASSO 0.0046, RF 0.0049, all 0.0180. Stem genes MLLT3/HOPX/SPINK2/NPR3 recovered without cluster labels.
