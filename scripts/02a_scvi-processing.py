# %% scVI Processing
# > Filter doublets via Scrublet
# > Train a scVI model on the CKD-AKI dataset
#
# %% PATH SETUP
from pathlib import Path
import gc

SCRIPT_DIR = Path.cwd()
PROJECT_DIR = SCRIPT_DIR.parent

OUTPUT_DIR = PROJECT_DIR / "outputs/"
DATA_DIR = PROJECT_DIR / "data/"

SCVI_DIR = DATA_DIR / "scvi_model/"
SCVI_DIR.mkdir(parents=True, exist_ok=True)

# %% IMPORTS
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
import scvi
import torch

plt.ioff()
scvi.settings.seed = 67

sc.set_figure_params(
    dpi=150,
    dpi_save=600,
    frameon=False,
    fontsize=8,
    vector_friendly=True,
    transparent=True
)
sc.settings.figdir = OUTPUT_DIR

# %% LOAD DATA
adata = sc.read_h5ad(DATA_DIR / "7ff0197b-d175-49bf-b4fa-150fe0995d93.h5ad")

# %% DOUBLET REMOVAL (SCRUBLET)
# Reviewer justification: Wolock et al., Cell Systems 2019
# Must be run per batch (experiment_id) as doublets form during individual sequencing runs

# sc.pp.scrublet natively handles batching
sc.pp.scrublet(adata, batch_key="experiment_id")

# Plot the doublet score histograms for the supplementary materials
fig, ax = plt.subplots(figsize=(8, 6))
sc.pl.scrublet_score_distribution(adata, show=False, ax=ax)
fig.savefig(OUTPUT_DIR / "qc" / "scrublet_distribution.png", dpi=300)
plt.close(fig)

# Filter out predicted doublets
initial_cells = adata.n_obs
adata = adata[~adata.obs["predicted_doublet"]].copy()
print(f"Removed {initial_cells - adata.n_obs} predicted doublets.")

gc.collect()

# %% LAYER PREP
adata.X = adata.raw.X.copy()  # Copy raw counts into X for SCVI

# %% HVG: SELECTION-SUBSET-STORE
scvi.data.poisson_gene_selection(adata, n_top_genes=4000, inplace=True)

adata = adata[:, adata.var["highly_variable"]].copy()

# %% SCVI: SETUP-INITIALIZE-TRAIN
scvi.model.SCVI.setup_anndata(
    adata,
    layer="counts",
    batch_key="experiment_id",  # Primary technical noise
    categorical_covariate_keys=["assay"],  # Batch effect (assay)
)

model = scvi.model.SCVI(
    adata,
    n_layers=2,
    n_hidden=256,
    n_latent=30,
    dropout_rate=0.1,
    gene_likelihood="nb",
)

model.train(
    max_epochs=500,
    batch_size=2048,
    early_stopping=True,
    early_stopping_patience=20,
    early_stopping_monitor="elbo_validation",
    accelerator="mps", # Hardware acceleration
)

# Ensure convergence
train_test_results = model.history["elbo_train"]
train_test_results["elbo_validation"] = model.history["elbo_validation"]
train_test_results.iloc[10:].plot(logy=True)
plt.savefig(SCVI_DIR / "elbo_convergence.png")
plt.close()

# %% EXTRACTION AND EMBEDDING
adata.obsm["X_scVI"] = model.get_latent_representation()

sc.pp.neighbors(adata, use_rep="X_scVI", n_neighbors=15)
sc.tl.umap(adata, min_dist=0.3)

# %% SAVE RESULTS
adata.write_h5ad(DATA_DIR / "adata_scvi.h5ad")
model.save(SCVI_DIR, overwrite=True)

# %% PLOTS
sc.tl.leiden(adata, resolution=0.5)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

sc.pl.umap(adata, color="library", ax=axes[0], show=False, title="Integration by Library")
sc.pl.umap(adata, color="donor_id", ax=axes[1], show=False, title="Distribution by Donor")
sc.pl.umap(adata, color="leiden", ax=axes[2], show=False, title="Clusters (Leiden)")
fig.savefig(SCVI_DIR / "scvi_umaps.png", dpi=300)
plt.close(fig)

gc.collect()
