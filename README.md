# Cell-resolved mapping of complement transcriptional programs in acute and chronic kidney injury

This repository contains the computational workflow for a secondary analysis of publicly available human kidney single-nucleus RNA-sequencing (snRNA-seq) data examining complement-associated transcriptional programs in acute kidney injury (AKI), chronic kidney disease (CKD), and normal/reference kidney.

The analysis uses individual nuclei to characterize transcriptional heterogeneity and cell-state structure, while treating the donor as the biological replicate for disease-level statistical inference. Complement-associated RNA expression is interpreted as transcriptional remodeling and not as direct evidence of biochemical complement activation, pathway flux, protein cleavage, or therapeutic responsiveness.

---

## Environment

For reproducibility, the computational environment is managed using [Pixi](https://pixi.prefix.dev/latest/). The repository includes `pixi.toml` and `pixi.lock` defining the analysis environment.

Install [Pixi](https://pixi.prefix.dev/latest/) and create the environment from the configuration files included in this repository.

R dependencies required for Slingshot and tradeSeq can be installed with:

> `pixi run setup-r`

and checked with:

> `pixi run check-r`

Python is used for data acquisition, preprocessing, scVI modeling, complement scoring, pseudobulk differential expression, metacell construction, DECIPHER modeling, and most trajectory analyses. R is used for Slingshot and tradeSeq.

## Hardware

The analysis was performed on a 16" M1 Max MacBook Pro with 64 GB RAM. Scripts have not been validated for use on other hardware configurations.

## Analysis overview

The workflow is organized into sequential stages:

1. Data acquisition and quality-control diagnostics
2. Doublet removal, ambient-RNA correction, and scVI latent representation
3. Complement transcriptional scoring and score-sensitivity analyses
4. Donor-level pseudobulk differential expression
5. Donor-preserving metacell construction and DECIPHER representation
6. Cell-state trajectory inference and robustness analyses
7. Donor-level trajectory summaries and disease association
8. Trajectory-associated gene-expression and complement-focused analyses
9. Supplementary-table compilation

## Script organization

* **01 — Quality Control & Data Acquisition** (`01a_download.py`, `01b_metrics.py`)
  * Downloads source data from CELLxGENE Census and runs diagnostic checks without filtering nuclei.

* **02 — Preprocessing, Scoring & Validation** (`02a_scvi-processing.py`, `02b_scvi-plot.py`, `02c_scvi-validation.py`)
  * Removes doublets, performs ambient-RNA correction, builds the scVI latent space, scores complement modules, and runs robustness checks.

* **03 — Pseudobulk Differential Expression** (`03a_pseudobulk-de.py`, `03b_pydeseq2-plot.py`)
  * Aggregates data to the donor level and performs the prespecified disease contrasts using PyDESeq2.

* **04 — Metacells & Trajectory Analysis** (`04a_metacell.py`, `04b_decipher.py`, `04c01_trajectory-primary.py` through `04c08_complement-trajectory.py`)
  * Constructs donor-preserving metacells, fits DECIPHER representations, performs Slingshot trajectory inference and robustness analyses, summarizes donor-level trajectory position, tests disease associations, and performs tradeSeq and complement-focused trajectory analyses.

* **05 — Supplementary Table Compilation** (`05_compile_supplementary_tables.py`)
  * Compiles existing analysis outputs into the S1--S6 supplementary data files without recomputing statistical analyses.
