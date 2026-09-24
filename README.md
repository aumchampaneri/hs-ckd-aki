# Cell-resolved mapping of complement transcriptional programs in acute and chronic kidney injury

This repository contains the computational workflow for a secondary analysis of publicly available human kidney single-nucleus RNA-sequencing (snRNA-seq) data examining complement-associated transcriptional programs in acute kidney injury (AKI), chronic kidney disease (CKD), and normal/reference kidney.

The analysis uses individual nuclei to characterize transcriptional heterogeneity and cell-state structure, while treating the donor as the biological replicate for disease-level statistical inference. Complement-associated RNA expression is interpreted as transcriptional remodeling and not as direct evidence of biochemical complement activation, pathway flux, protein cleavage, or therapeutic responsiveness.

## Abstract
[Insert]

---

## Environment

For reproducibility, the computational environment is managed using [Pixi](https://pixi.prefix.dev/latest/).

Install [Pixi](https://pixi.prefix.dev/latest/) and create the environment from the configuration files included in this repository.

R dependencies required for trajectory-associated gene-expression analysis can be installed with:

> `pixi run setup-r`

Python is used for data acquisition, preprocessing, scVI modeling, complement scoring, pseudobulk differential expression, metacell construction, DECIPHER modeling, and most trajectory analyses. R is used where required for tradeSeq.

## Hardware
The analysis was performed on a 16" M1 Max MacBook Pro with 64GB RAM. Scripts have not been validated for use on any other hardware.

## Analysis Overview
The workflow is organized into sequential stages:
1. Data acquisition and quality-control diagnostics
2. Doublet removal, ambient-RNA correction, and scVI latent representation
3. Complement transcriptional scoring and score-sensitivity analyses
4. Donor-level pseudobulk differential expression
5. Donor-preserving metacell construction and DECIPHER representation
6. Cell-state trajectory inference and robustness analyses
7. Donor-level trajectory summaries and disease association
8. Trajectory-associated gene-expression analysis

## Script organization

* **01 — Quality Control & Data Acquisition** (`01a_download.py`, `01b_metrics.py`)
  * Downloads source data from CELLxGENE Census and runs diagnostic checks (e.g., QC thresholds, doublets, mitochondrial fractions) without filtering out nuclei.


* **02 — Preprocessing, Scoring & Validation** (`02a_scvi-processing.py`, `02b_scvi-plot.py`, `02c_scvi-validation.py`)
  * Cleans ambient RNA/doublets, builds the scVI latent space, scores complement modules, and runs statistical robustness checks.


* **03 — Pseudobulk Differential Expression** (`03a_pseudobulk-de.py`, `03b_pydeseq2-plot.py`)
  * Aggregates data to the donor level to perform primary disease contrasts (AKI vs. CKD vs. Normal) using PyDESeq2, avoiding cell-level pseudoreplication.


* **04 — Metacells & Trajectory Analysis** (`04a_metacell.py` through `04c06_trajectory-tradeseq.py`)
  * Constructs donor-preserving SEACells, fits DECIPHER spaces, maps pseudotime trajectories via Slingshot/tradeSeq, and tests for disease associations.
