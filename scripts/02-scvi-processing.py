# %% scVI Processing
# > Train a scVI model on the CKD-AKI dataset
#
# %% PATH SETUP
from pathlib import Path

SCRIPT_DIR = Path.cwd()
PROJECT_DIR = SCRIPT_DIR.parent

OUTPUT_DIR = PROJECT_DIR / "outputs/"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DATA_DIR = PROJECT_DIR / "data/"
DATA_DIR.mkdir(parents=True, exist_ok=True)

SCVI_DIR = DATA_DIR / "scvi_model/"
SCVI_DIR.mkdir(parents=True, exist_ok=True)

# %% IMPORTS
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
import scvi
import torch

# %%
adata = sc.read_h5ad(DATA_DIR / "7ff0197b-d175-49bf-b4fa-150fe0995d93.h5ad")
# %%
# INSPECTION
####
adata.obs[["nFeature_RNA", "nCount_RNA", "percent.mt"]].describe()

adata.raw
adata.raw.X
adata.raw.X.shape
adata.raw.X[:5, :5]

adata.X = adata.raw.X.copy()

# %%
# HVG: SELECTION-SUBSET-STORE
####
scvi.data.poisson_gene_selection(adata)

adata = adata[:, adata.var["highly_variable"]].copy()
adata.layers["counts"] = adata.X.copy()

if not sp.issparse(adata.layers["counts"]):
    adata.layers["counts"] = sp.csr_matrix(adata.layers["counts"])

adata.obs["batch"] = (
    adata.obs["donor_id"].astype(str) + "_" + adata.obs["library"].astype(str)
)

adata.obs["batch"] = adata.obs["batch"].astype("category")

# %%
# SCVI: SETUP-INITIALIZE-TRAIN
####
scvi.model.SCVI.setup_anndata(adata, layer="counts", batch_key="batch")

model = scvi.model.SCVI(
    adata,
    n_layers=2,
    n_hidden=256,
    n_latent=30,
    dropout_rate=0.1,
    gene_likelihood="nb",  # We use Negative Binomial count likelihoods, following Boyeau et al., 2023.
)

model.train(
    max_epochs=300,
    batch_size=2048,
    early_stopping=True,
    early_stopping_patience=20,
    early_stopping_monitor="elbo_validation",
    accelerator="mps",
    plan_kwargs={"lr": 1e-3},
)

# Ensure convergence
train_test_results = model.history["elbo_train"]
train_test_results["elbo_validation"] = model.history["elbo_validation"]
train_test_results.iloc[10:].plot(logy=True)  # exclude first 10 epochs
plt.show()

# %%
# EXTRACTION AND EMBEDDING
####
adata.obsm["X_scVI"] = model.get_latent_representation()

sc.pp.neighbors(adata, use_rep="X_scVI", n_neighbors=15)
sc.tl.umap(adata, min_dist=0.3)

# %%
# SAVE RESULTS
####
adata.write_h5ad(DATA_DIR / "adata_scvi.h5ad")
model.save(SCVI_DIR, overwrite=True)
