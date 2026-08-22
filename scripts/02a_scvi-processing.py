# %% THREAD LIMITING -- must happen before numpy/scipy/sklearn/numba are imported,
# since BLAS/numba thread pools are sized at import time. Prevents oversubscription
# when running N_JOBS parallel worker processes below.
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMBA_NUM_THREADS", "1")

# %% PATH SETUP
from pathlib import Path
import gc
import shutil

SCRIPT_DIR = Path.cwd()
PROJECT_DIR = SCRIPT_DIR.parent

OUTPUT_DIR = PROJECT_DIR / "outputs/"
DATA_DIR = PROJECT_DIR / "data/"

QC_DIR = OUTPUT_DIR / "qc/"
QC_DIR.mkdir(parents=True, exist_ok=True)

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
import decontx
from joblib import Parallel, delayed

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
sc.pl.scrublet_score_distribution(adata, show=False)
plt.savefig(QC_DIR / "scrublet_distribution.png", dpi=300, bbox_inches="tight")
plt.close()
initial_cells = adata.n_obs
adata = adata[~adata.obs["predicted_doublet"]].copy()
print(f"Removed {initial_cells - adata.n_obs} predicted doublets.")

gc.collect()

# %% LAYER PREP
adata.layers["counts"] = adata.raw.X.copy()
adata.X = adata.layers["counts"].copy()

# %% AMBIENT RNA REMOVAL (decontX) -- checkpointed, parallelized across batches
# Reviewer justification: demonstrate ambient contamination is addressed rather
# than assumed. decontX (Yang et al., Genome Biology 2020); `decontx-python` is
# a third-party pure-Python reimplementation (self-reported >0.999 correlation
# with the canonical Bioconductor R version, not independently verified here).
#
# Ambient RNA composition is specific to each sequencing run, so this runs per
# experiment_id. Structured in three resumable stages so a killed/crashed run
# (e.g. closing the laptop, or an OOM kill) only loses in-progress work, not
# completed batches:
#   1. Split each batch to its own file on disk (cheap, idempotent)
#   2. Process batches in parallel; each writes its own checkpoint file
#   3. Reassemble checkpoints into the full adata, in original cell order
#
# Re-running this whole section after any kill will skip every batch that
# already has a checkpoint file and only compute what's missing.

BATCH_COL = "experiment_id"
N_JOBS = 4  # M1 Max has 10 cores; leave headroom since each worker's numba/BLAS calls aren't fully single-threaded even with the env vars above. Raise/lower based on observed memory and CPU use.

DECONTX_DIR = QC_DIR / "decontx_batches"
SPLIT_DIR = DECONTX_DIR / "input_batches"
OUT_DIR = DECONTX_DIR / "output_batches"
SPLIT_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)

# split into per-batch files
batch_ids = adata.obs[BATCH_COL].astype(str).unique().tolist()
for batch_id in batch_ids:
    split_path = SPLIT_DIR / f"{batch_id}.h5ad"
    if split_path.exists():
        continue
    sub = adata[adata.obs[BATCH_COL].astype(str) == batch_id].copy()
    sub.write_h5ad(split_path)
    del sub
print(f"Stage 1 complete: {len(batch_ids)} batch input files in {SPLIT_DIR}")

# per-batch decontX, parallelized, checkpointed
def process_batch(batch_id, split_dir, out_dir):
    out_counts_path = out_dir / f"{batch_id}_counts.npz"
    out_meta_path = out_dir / f"{batch_id}_meta.csv"
    if out_counts_path.exists() and out_meta_path.exists():
        return batch_id, "skipped (checkpoint found)"

    sub_raw = sc.read_h5ad(split_dir / f"{batch_id}.h5ad")

    if sub_raw.n_obs < 50:
        counts = sub_raw.layers["counts"]
        if not sp.issparse(counts):
            counts = sp.csr_matrix(counts)
        sp.save_npz(out_counts_path, counts.tocsr())
        pd.DataFrame({
            "obs_name": sub_raw.obs_names,
            "decontX_contamination": np.nan,
        }).to_csv(out_meta_path, index=False)
        return batch_id, f"too few cells ({sub_raw.n_obs}), copied raw counts unchanged"

    sub_clust = sub_raw.copy()
    sc.pp.normalize_total(sub_clust)
    sc.pp.log1p(sub_clust)
    sc.pp.highly_variable_genes(sub_clust, n_top_genes=min(2000, sub_clust.n_vars - 1))
    sc.pp.pca(sub_clust, mask_var="highly_variable")
    sc.pp.neighbors(sub_clust)
    sc.tl.leiden(sub_clust, resolution=1.0, key_added="decontx_cluster", flavor="igraph", n_iterations=2)

    sub_raw.obs["decontx_cluster"] = sub_clust.obs["decontx_cluster"].values
    decontx.decontx(sub_raw, cluster_key="decontx_cluster")

    counts = sub_raw.layers["decontX_counts"]
    if not sp.issparse(counts):
        counts = sp.csr_matrix(counts)
    counts.data = np.round(counts.data)
    sp.save_npz(out_counts_path, counts.tocsr())

    pd.DataFrame({
        "obs_name": sub_raw.obs_names,
        "decontX_contamination": sub_raw.obs["decontX_contamination"].values,
    }).to_csv(out_meta_path, index=False)

    return batch_id, f"done, mean_contam={sub_raw.obs['decontX_contamination'].mean():.3f}"


results = Parallel(n_jobs=N_JOBS, backend="loky", verbose=10)(
    delayed(process_batch)(batch_id, SPLIT_DIR, OUT_DIR) for batch_id in batch_ids
)
for batch_id, status in results:
    print(f"{batch_id}: {status}")

# reassemble (memory-conscious -- avoids holding multiple full
# copies of the count matrix at once, which caused an earlier OOM kill here)
missing = [b for b in batch_ids if not (OUT_DIR / f"{b}_counts.npz").exists()]
if missing:
    raise RuntimeError(
        f"{len(missing)} batch(es) missing output -- re-run this section to fill "
        f"them in before proceeding. First few: {missing[:5]}"
    )

counts_chunks, meta_chunks = [], []
for batch_id in batch_ids:
    counts_chunks.append(sp.load_npz(OUT_DIR / f"{batch_id}_counts.npz"))
    meta_chunks.append(pd.read_csv(OUT_DIR / f"{batch_id}_meta.csv"))

decontx_counts_unordered = sp.vstack(counts_chunks, format="csr")
del counts_chunks
gc.collect()

combined_meta = pd.concat(meta_chunks, ignore_index=True).set_index("obs_name")
del meta_chunks
gc.collect()

row_pos = pd.Series(range(len(combined_meta)), index=combined_meta.index)
reorder_idx = row_pos.loc[adata.obs_names].values
decontx_counts = decontx_counts_unordered[reorder_idx]
del decontx_counts_unordered
gc.collect()

adata.obs["decontX_contamination"] = combined_meta.loc[adata.obs_names, "decontX_contamination"].values
del combined_meta
gc.collect()

print("decontX contamination summary (all batches):")
print(adata.obs["decontX_contamination"].describe())
print(f"Cells >50% estimated contamination: {(adata.obs['decontX_contamination'] > 0.5).sum()} "
      f"({(adata.obs['decontX_contamination'] > 0.5).mean():.2%})")

fig, ax = plt.subplots(figsize=(5, 4))
adata.obs["decontX_contamination"].hist(bins=100, ax=ax)
ax.set_xlabel("decontX estimated contamination fraction")
ax.set_title("Per-cell ambient RNA contamination")
plt.savefig(QC_DIR / "decontx_contamination_distribution.png", dpi=300, bbox_inches="tight")
plt.close(fig)

# Drop the pre-decontX counts layer entirely rather than holding it alongside the
# decontaminated matrix -- if a before/after comparison is ever needed, the
# original counts are still recoverable from input_batches/*.h5ad.
del adata.layers["counts"]
gc.collect()

# Assign by reference, not .copy() -- layers["counts"] and X point at the same
# object rather than duplicating it.
adata.layers["counts"] = decontx_counts
adata.X = decontx_counts
adata.raw = adata  # snapshot at full gene space, decontaminated -- 02b picks this up automatically
gc.collect()

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
    accelerator="mps",
)
train_test_results = model.history["elbo_train"]
train_test_results["elbo_validation"] = model.history["elbo_validation"]
train_test_results.iloc[10:].plot(logy=True)
plt.savefig(QC_DIR / "elbo_convergence.png", bbox_inches="tight")
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
fig.savefig(SCVI_DIR / "scvi_umaps.png", dpi=300, bbox_inches="tight")

plt.close(fig)

gc.collect()
