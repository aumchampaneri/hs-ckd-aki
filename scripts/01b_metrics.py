# %% Look at metrics
# > Looks at data characteristics
# > Determine what QC needed for data cleaning
# > Evaluate thresholds for MT, doublets, and empty droplets

# %% PATH SETUP
from pathlib import Path
import gc

SCRIPT_DIR = Path.cwd()
PROJECT_DIR = SCRIPT_DIR.parent

OUTPUT_DIR = PROJECT_DIR / "outputs/"
DATA_DIR = PROJECT_DIR / "data/"

QC_DIR = OUTPUT_DIR / "qc/"
QC_DIR.mkdir(parents=True, exist_ok=True)

# %% IMPORTS
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp

plt.ioff()

# %% LOAD DATA
adata = sc.read_h5ad(DATA_DIR / "7ff0197b-d175-49bf-b4fa-150fe0995d93.h5ad")
adata.obs_names_make_unique()
adata.var_names_make_unique()

# %% COUNT MATRIX
####
# Reference assignment to avoid copying a massive matrix multiple times
if "counts" in adata.layers:
    counts_ref = adata.layers["counts"]
elif adata.raw is not None:
    counts_ref = adata.raw.X
else:
    counts_ref = adata.X

# Force CSR format for memory efficiency, but avoid copying if already CSR
if not sp.issparse(counts_ref):
    counts = sp.csr_matrix(counts_ref)
elif not sp.isspmatrix_csr(counts_ref):
    counts = counts_ref.tocsr()
else:
    counts = counts_ref

values = counts.data
assert np.all(np.isfinite(values)), "Count matrix contains non-finite values."
assert np.all(values >= 0), "Count matrix contains negative values."
assert np.allclose(values, np.round(values)), "Count matrix is not integer-valued."

# Assign references instead of .copy()
adata.X = counts
adata.layers["counts"] = counts

# Free the temporary reference
del counts_ref
gc.collect()

# %% RAW COUNT SUMMARY
####
total_counts = np.asarray(counts.sum(axis=1)).ravel()
# Use getnnz() instead of (counts > 0) to avoid creating a dense boolean matrix in RAM
detected_genes = counts.getnnz(axis=1)
gene_counts = np.asarray(counts.sum(axis=0)).ravel()

count_summary = pd.DataFrame({
    "metric": [
        "cells_nuclei", "genes", "total_counts_mean", "total_counts_median",
        "total_counts_sd", "total_counts_min", "total_counts_max",
        "detected_genes_mean", "detected_genes_median", "detected_genes_sd",
        "detected_genes_min", "detected_genes_max", "matrix_sparsity"
    ],
    "value": [
        adata.n_obs, adata.n_vars, total_counts.mean(), np.median(total_counts),
        total_counts.std(), total_counts.min(), total_counts.max(),
        detected_genes.mean(), np.median(detected_genes), detected_genes.std(),
        detected_genes.min(), detected_genes.max(), 1 - counts.nnz / (adata.n_obs * adata.n_vars)
    ]
})
count_summary.to_csv(QC_DIR / "raw_count_summary.csv", index=False)

del total_counts, detected_genes, gene_counts, counts
gc.collect()

# %% CALCULATE MAD THRESHOLDS (MATHEMATICAL JUSTIFICATION)
####
def calculate_mad_thresholds(metric_array, nmads=5):
    median = np.median(metric_array)
    mad = np.median(np.abs(metric_array - median))
    lower = max(0, median - (nmads * mad))
    upper = median + (nmads * mad)
    return lower, upper

thresholds = []
for col in ["nCount_RNA", "nFeature_RNA", "percent.mt"]:
    if col in adata.obs:
        vals = adata.obs[col].dropna().values
        lower, upper = calculate_mad_thresholds(vals, nmads=5)
        thresholds.append({"metric": col, "median": np.median(vals), "mad_lower_5": lower, "mad_upper_5": upper})

pd.DataFrame(thresholds).to_csv(QC_DIR / "mad_threshold_suggestions.csv", index=False)

# %% JOINT QC METRICS (Identify Doublets, Empty Drops, Dying Cells)
####
qc_cols = [c for c in ["nFeature_RNA", "nCount_RNA", "percent.mt"] if c in adata.obs]

if "nCount_RNA" in adata.obs and "nFeature_RNA" in adata.obs:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Using ax.scatter with rasterized=True prevents matplotlib from crashing on 1.3M vector points
    axes[0].scatter(
        adata.obs["nCount_RNA"], adata.obs["nFeature_RNA"],
        alpha=0.1, s=1, edgecolor="none", rasterized=True
    )
    axes[0].set_title("Features vs. Total Counts\n(Look for extreme right/top tails = doublets)")
    axes[0].set_xlabel("nCount_RNA")
    axes[0].set_ylabel("nFeature_RNA")

    if "percent.mt" in adata.obs:
        axes[1].scatter(
            adata.obs["nCount_RNA"], adata.obs["percent.mt"],
            alpha=0.1, s=1, edgecolor="none", rasterized=True
        )
        axes[1].set_title("MT% vs. Total Counts\n(Look for high MT / low counts = dying)")
        axes[1].set_xlabel("nCount_RNA")
        axes[1].set_ylabel("percent.mt")

    fig.tight_layout()
    fig.savefig(QC_DIR / "qc_joint_distributions.png", dpi=300)
    plt.close(fig)

# %% AMBIENT RNA PROFILING
####
ambient_genes = [g for g in adata.var_names if g.startswith("HBA") or g.startswith("HBB")]
if ambient_genes:
    # Use raw=False to use normalized data if available, but since we reset X to counts,
    # we must normalize temporarily to score genes safely without leaking memory.
    adata_tmp = adata[:, ambient_genes].copy()
    sc.pp.normalize_total(adata_tmp, target_sum=1e4)
    sc.pp.log1p(adata_tmp)

    sc.tl.score_genes(adata_tmp, gene_list=ambient_genes, score_name="hemoglobin_score")
    adata.obs["hemoglobin_score"] = adata_tmp.obs["hemoglobin_score"]

    del adata_tmp
    gc.collect()

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(
        adata.obs["nCount_RNA"], adata.obs["hemoglobin_score"],
        alpha=0.1, s=1, edgecolor="none", rasterized=True
    )
    ax.set_title("Hemoglobin Score vs Total Counts\n(High HB in non-RBCs = Ambient RNA)")
    ax.set_xlabel("nCount_RNA")
    ax.set_ylabel("Hemoglobin Score")
    fig.savefig(QC_DIR / "ambient_rna_check.png", dpi=300)
    plt.close(fig)

# %% REVIEWER REQUESTED METRICS (ADDITIONAL REPORTING)
####
print("\n=== RAW COUNTS VALIDATION ===")
# Avoid pulling a new reference if we already established X is raw counts
v = adata.X.data
print(f"dtype: {adata.X.dtype} | min: {v.min()} | max: {v.max()} | mean: {v.mean():.2f} | sd: {v.std():.2f}")
print(f"integer-valued: {np.allclose(v, np.round(v))} | nonzero: {adata.X.nnz} | sparsity: {1 - adata.X.nnz / (adata.X.shape[0] * adata.X.shape[1]):.4f}")

if "disease" in adata.obs:
    print("\n=== ASSAY & SUSPENSION CROSSTABS ===")
    for col in ["assay", "suspension_type", "modality"]:
        if col in adata.obs:
            print(f"\nDisease x {col}:\n", pd.crosstab(adata.obs["disease"], adata.obs[col]))

    if "donor_id" in adata.obs:
        print("\n=== DONOR DISTRIBUTION ===")
        print(f"Donors per disease state:\n{adata.obs.groupby('disease')['donor_id'].nunique()}")

# Keep memory clean at the end of execution
gc.collect()
