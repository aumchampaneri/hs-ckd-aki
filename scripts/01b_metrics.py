# %% Look at metrics
# > Looks at data characteristics
# > Determine what QC needed for data cleaning
# > Evaluate thresholds for MT, doublets, and empty droplets

# %% PATH SETUP
from pathlib import Path

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
if "counts" in adata.layers:
    counts = adata.layers["counts"].copy()
elif adata.raw is not None:
    counts = adata.raw.X.copy()
else:
    counts = adata.X.copy()

if not sp.issparse(counts):
    counts = sp.csr_matrix(counts)

values = counts.data
assert np.all(np.isfinite(values)), "Count matrix contains non-finite values."
assert np.all(values >= 0), "Count matrix contains negative values."
assert np.allclose(values, np.round(values)), "Count matrix is not integer-valued."

adata.X = counts.copy()
adata.layers["counts"] = counts.copy()

# %% RAW COUNT SUMMARY
####
total_counts = np.asarray(counts.sum(axis=1)).ravel()
detected_genes = np.asarray((counts > 0).sum(axis=1)).ravel()
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

# %% CALCULATE MAD THRESHOLDS (MATHEMATICAL JUSTIFICATION)
####
# Reviewers prefer MAD (Median Absolute Deviation) over arbitrary cutoffs.
# Typically, a cell > 3-5 MADs from the median is considered an outlier.

def calculate_mad_thresholds(metric_array, nmads=5):
    median = np.median(metric_array)
    mad = np.median(np.abs(metric_array - median))
    lower = max(0, median - (nmads * mad)) # Don't go below 0
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
    
    # 1. Doublets / Empty Droplets
    sns.scatterplot(
        data=adata.obs, x="nCount_RNA", y="nFeature_RNA", 
        alpha=0.1, s=5, ax=axes[0], edgecolor="none"
    )
    axes[0].set_title("Features vs. Total Counts\n(Look for extreme right/top tails = doublets)")
    
    # 2. Dying Cells / Stripped Nuclei
    if "percent.mt" in adata.obs:
        sns.scatterplot(
            data=adata.obs, x="nCount_RNA", y="percent.mt", 
            alpha=0.1, s=5, ax=axes[1], edgecolor="none"
        )
        axes[1].set_title("MT% vs. Total Counts\n(Look for high MT / low counts = dying)")
        
    fig.tight_layout()
    fig.savefig(QC_DIR / "qc_joint_distributions.png", dpi=300)
    plt.close(fig)

# %% AMBIENT RNA PROFILING
####
# Highly expressed ubiquitous genes (like Hemoglobin in highly vascularized tissue)
# that shouldn't be everywhere can indicate ambient RNA soup.

ambient_genes = [g for g in adata.var_names if g.startswith("HBA") or g.startswith("HBB")]
if ambient_genes:
    sc.tl.score_genes(adata, gene_list=ambient_genes, score_name="hemoglobin_score", use_raw=False)
    
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.scatterplot(
        data=adata.obs, x="nCount_RNA", y="hemoglobin_score", 
        alpha=0.1, s=5, edgecolor="none"
    )
    ax.set_title("Hemoglobin Score vs Total Counts\n(High HB in non-RBCs = Ambient RNA)")
    fig.savefig(QC_DIR / "ambient_rna_check.png", dpi=300)
    plt.close(fig)

# %% REVIEWER REQUESTED METRICS (ADDITIONAL REPORTING)
####
print("\n=== RAW COUNTS VALIDATION ===")
x = adata.raw.X if adata.raw is not None else adata.X
x = x if sp.issparse(x) else sp.csr_matrix(x)
v = x.data
print(f"dtype: {x.dtype} | min: {v.min()} | max: {v.max()} | mean: {v.mean():.2f} | sd: {v.std():.2f}")
print(f"integer-valued: {np.allclose(v, np.round(v))} | nonzero: {x.nnz} | sparsity: {1 - x.nnz / (x.shape[0] * x.shape[1]):.4f}")

if "disease" in adata.obs:
    print("\n=== ASSAY & SUSPENSION CROSSTABS ===")
    for col in ["assay", "suspension_type", "modality"]:
        if col in adata.obs:
            print(f"\nDisease x {col}:\n", pd.crosstab(adata.obs["disease"], adata.obs[col]))

    if "donor_id" in adata.obs:
        print("\n=== DONOR DISTRIBUTION ===")
        print(f"Donors per disease state:\n{adata.obs.groupby('disease')['donor_id'].nunique()}")
