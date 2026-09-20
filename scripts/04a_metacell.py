# %% Metacell Generation
#
# %% SETUP
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
import contextlib
import gc
import io

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp

# %% PATHS / CONFIG
PROJECT_DIR = Path.cwd().parent
DATA_DIR = PROJECT_DIR / "data"
OUTPUT_DIR = DATA_DIR / "metacells"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

H5AD = DATA_DIR / "adata_scvi.h5ad"
COMPARTMENTS = ["PT", "FIB", "Immune", "Glomerular"]

EMBEDDING_KEY = "X_scVI"
CELLS_PER_METACELL = 50
MIN_DONOR_CELLS = 50
MAX_ITER = 200
N_WAYPOINT_EIGS = 10
PROGRESS_EVERY = 20

# 0 = cpu_count - 1; 1 = sequential
N_WORKERS = 1
STALL_TIMEOUT = 300

DISEASE_COL = "ConditionCategory"
DONOR_COL = "donor_id"
STATE_COL = "CellStateLevel1"
SUBCLASS_COL = "SubclassLevel3"
ASSAY_COL = "assay"

COVID_AKI_DONORS = {
    "COV-02-0010", "COV-02-0002", "COV-02-0094", "COV-02-0096"
}

# Empty = full run; set donor IDs for targeted reruns.
ONLY_DONORS = set()

# Existing *_metacells.h5ad to replace ONLY_DONORS in.
MERGE_INTO = None

COMPARTMENT_MAP = {
    "PT": ("SubclassLevel1", ["PT"]),
    "FIB": ("SubclassLevel1", ["FIB"]),
    "Immune": ("Class", ["immune cells"]),
    "Glomerular": (
        "SubclassLevel3",
        ["POD", "dPOD", "PEC", "MC",
         "EC-GC", "EC-GC-FILIP1+", "dEC-GC", "aEC-GC"],
    ),
}

# %% HELPERS
def compartment_mask(obs, compartment):
    col, values = COMPARTMENT_MAP[compartment]
    if col not in obs:
        raise ValueError(
            f"Column '{col}' not found in .obs for '{compartment}'. "
            f"Available columns: {list(obs.columns)}"
        )
    return obs[col].isin(values)


def dominant_and_purity(series):
    if not len(series):
        return np.nan, np.nan
    counts = series.value_counts()
    return counts.index[0], counts.iloc[0] / len(series)


def run_seacells_for_donor(donor_adata, donor_id):
    n_cells = donor_adata.n_obs
    if n_cells < MIN_DONOR_CELLS:
        return np.array([f"{donor_id}::mc0"] * n_cells)

    n_metacells = max(1, round(n_cells / CELLS_PER_METACELL))
    n_metacells = min(n_metacells, max(1, n_cells - 1))
    n_waypoint_eigs = max(1, min(N_WAYPOINT_EIGS, n_cells - 2))

    embedding = donor_adata.obsm[EMBEDDING_KEY]
    n_duplicated = n_cells - len(np.unique(embedding, axis=0))

    # Keep the original embedding untouched; jitter only the copy used by
    # SEACells when exact duplicates would make its adaptive bandwidth zero.
    if n_duplicated:
        print(
            f"    [{donor_id}] {n_duplicated}/{n_cells} duplicate scVI "
            "embeddings -- jittering before kernel construction."
        )
        rng = np.random.default_rng(hash(donor_id) % (2**32))
        scale = np.std(embedding, axis=0).mean()
        donor_adata = donor_adata.copy()
        donor_adata.obsm[EMBEDDING_KEY] = embedding + rng.normal(
            scale=max(scale * 1e-4, 1e-8), size=embedding.shape
        )

    try:
        import SEACells
        import warnings

        conv_notice = None
        with contextlib.redirect_stdout(io.StringIO()):
            model = SEACells.core.SEACells(
                donor_adata,
                build_kernel_on=EMBEDDING_KEY,
                n_SEACells=n_metacells,
                n_waypoint_eigs=n_waypoint_eigs,
                convergence_epsilon=1e-5,
            )
            model.construct_kernel_matrix()
            model.initialize_archetypes()

            with warnings.catch_warnings():
                warnings.simplefilter("always")
                try:
                    model.fit(min_iter=10, max_iter=MAX_ITER)
                except RuntimeWarning as e:
                    conv_notice = str(e)

            labels = model.get_hard_assignments()["SEACell"].astype(str).to_numpy()
        del model

        if conv_notice:
            print(f"    [{donor_id}] non-convergence notice: {conv_notice}")

        if n_metacells > 1:
            top_frac = pd.Series(labels).value_counts(normalize=True).iloc[0]
            threshold = max(0.5, min(1.0, 4 / n_metacells))
            if top_frac > threshold:
                print(
                    f"    [{donor_id}] WARNING: dominant metacell holds "
                    f"{top_frac:.0%} of {n_cells} cells; threshold {threshold:.0%}."
                )

        return np.array([f"{donor_id}::{label}" for label in labels])

    except Exception as e:
        print(
            f"    [{donor_id}] SEACells fit failed "
            f"({type(e).__name__}: {e}); falling back to one metacell."
        )
        return np.array([f"{donor_id}::mc0"] * n_cells)


def run_donors_parallel(donor_tasks, n_workers):
    labels = {}
    fallback = 0

    if n_workers == 1:
        for i, (donor_id, donor) in enumerate(donor_tasks, 1):
            result = run_seacells_for_donor(donor, donor_id)
            labels[donor_id] = result
            if result[0].endswith("::mc0") and len(result) >= MIN_DONOR_CELLS:
                fallback += 1
            if i % PROGRESS_EVERY == 0 or i == len(donor_tasks):
                print(f"  Processed {i}/{len(donor_tasks)} donors")
        return labels, fallback

    executor = ProcessPoolExecutor(max_workers=n_workers)
    futures = {
        executor.submit(run_seacells_for_donor, donor, donor_id): donor_id
        for donor_id, donor in donor_tasks
    }
    pending = set(futures)
    completed = 0

    while pending:
        done, pending = wait(
            pending, timeout=STALL_TIMEOUT, return_when=FIRST_COMPLETED
        )

        if not done:
            stuck = [futures[f] for f in pending]
            print(
                f"  WARNING: no donor completed in {STALL_TIMEOUT}s -- "
                f"falling back for {len(pending)} pending donors: "
                f"{stuck[:8]}{' ...' if len(stuck) > 8 else ''}"
            )
            for f in pending:
                donor_id = futures[f]
                labels[donor_id] = np.array(
                    [f"{donor_id}::mc0"] * next(
                        len(d) for d_id, d in donor_tasks if d_id == donor_id
                    )
                )
                fallback += 1
                completed += 1
            break

        for f in done:
            donor_id = futures[f]
            try:
                result = f.result()
            except Exception as e:
                donor = next(d for d_id, d in donor_tasks if d_id == donor_id)
                print(
                    f"    [{donor_id}] worker raised "
                    f"{type(e).__name__}: {e} -- falling back."
                )
                result = np.array([f"{donor_id}::mc0"] * donor.n_obs)

            labels[donor_id] = result
            if result[0].endswith("::mc0") and len(result) >= MIN_DONOR_CELLS:
                fallback += 1
            completed += 1

            if completed % PROGRESS_EVERY == 0 or completed == len(donor_tasks):
                print(f"  Completed {completed}/{len(donor_tasks)} donors")

    executor.shutdown(wait=False)
    return labels, fallback


# %% LOAD
print(f"Loading {H5AD}...")
full = sc.read_h5ad(H5AD)

n_raw_genes = full.raw.n_vars if full.raw is not None else None
print(
    f"Full atlas: {full.n_obs:,} cells x {full.n_vars:,} HVGs, "
    f"{n_raw_genes:,} raw genes"
    if n_raw_genes
    else f"Full atlas: {full.n_obs:,} cells x {full.n_vars:,} HVGs -- .raw missing"
)

# %% PROCESS
results = {}

for compartment in COMPARTMENTS:
    print(f"\n{'=' * 70}\n{compartment}\n{'=' * 70}")

    adata = full[compartment_mask(full.obs, compartment)].copy()

    if ONLY_DONORS:
        donor_ids = adata.obs[DONOR_COL].astype(str)
        found = set(donor_ids[donor_ids.isin(ONLY_DONORS)].unique())
        missing = ONLY_DONORS - found
        if missing:
            print(f"WARNING: donors not found: {sorted(missing)}")
        adata = adata[donor_ids.isin(ONLY_DONORS)].copy()
        print(f"ONLY_DONORS: {sorted(found)}")

    n_donors = adata.obs[DONOR_COL].nunique()
    print(
        f"{adata.n_obs:,} cells across {n_donors} donors "
        f"(~{adata.n_obs / n_donors:.0f}/donor)"
    )

    if EMBEDDING_KEY not in adata.obsm:
        raise ValueError(f"'{EMBEDDING_KEY}' not found in .obsm: {list(adata.obsm)}")
    if adata.raw is None:
        raise ValueError("adata.raw is None; raw counts are required.")

    print("\nCells by disease:")
    print(adata.obs[DISEASE_COL].value_counts())
    print("\nDonors by disease:")
    print(adata.obs.groupby(DISEASE_COL, observed=True)[DONOR_COL].nunique())

    # %% SEACELLS
    n_workers = N_WORKERS or max(1, (os.cpu_count() or 4) - 1)
    print(f"\nSEACells: {n_donors} donors, {n_workers} workers")

    donor_indices = adata.obs.groupby(DONOR_COL, observed=True).indices
    donor_tasks = []

    for donor_id, idx in donor_indices.items():
        donor = adata[idx].copy()
        donor.raw = None
        donor_tasks.append((str(donor_id), donor))

    # This function is called from the notebook cell, while the worker itself
    # is top-level/pickleable. This keeps the REPL-style layout without
    # submitting nested/local functions to ProcessPoolExecutor.
    donor_labels, n_fallback = run_donors_parallel(donor_tasks, n_workers)

    seacell_labels = np.empty(adata.n_obs, dtype=object)
    for donor_id, idx in donor_indices.items():
        seacell_labels[idx] = donor_labels[str(donor_id)]

    adata.obs["SEACell"] = seacell_labels
    n_metacells = adata.obs["SEACell"].nunique()

    print(
        f"\nDone: {n_metacells} metacells across {n_donors} donors "
        f"(~{adata.n_obs / n_metacells:.0f} cells/metacell)"
    )
    if n_fallback:
        print(f"NOTE: {n_fallback} donor(s) fell back to one metacell.")

    # %% RAW COUNTS
    print("\nAggregating raw counts...")
    seacell_order = sorted(adata.obs["SEACell"].astype(str).unique())
    seacell_index = {s: i for i, s in enumerate(seacell_order)}
    codes = adata.obs["SEACell"].astype(str).map(seacell_index).to_numpy()

    raw_X = adata.raw.X
    if not sp.issparse(raw_X):
        raw_X = sp.csr_matrix(raw_X)

    indicator = sp.csr_matrix(
        (np.ones(len(codes)), (np.arange(len(codes)), codes)),
        shape=(len(codes), len(seacell_order)),
    )
    mc_counts_mat = (indicator.T @ raw_X).tocsr()

    raw_var = adata.raw.var.copy()
    gene_symbol_col = next(
        (c for c in ["feature_name", "gene_name", "symbol"] if c in raw_var),
        None,
    )
    raw_var["gene_symbol"] = (
        raw_var[gene_symbol_col].astype(str)
        if gene_symbol_col else raw_var.index.astype(str)
    )

    print(f"{mc_counts_mat.shape[0]:,} metacells x {mc_counts_mat.shape[1]:,} genes")

    # %% SCVI
    embedding_df = pd.DataFrame(
        adata.obsm[EMBEDDING_KEY], index=adata.obs_names
    )
    embedding_df["SEACell"] = adata.obs["SEACell"].astype(str).values
    mc_embedding = embedding_df.groupby("SEACell").mean().reindex(seacell_order)

    # %% METACELL METADATA
    has_assay = ASSAY_COL in adata.obs.columns
    if not has_assay:
        print(f"WARNING: '{ASSAY_COL}' not found; assay metadata will be skipped.")

    records = []
    for seacell_id, group in adata.obs.groupby("SEACell", observed=True):
        dom_disease, disease_purity = dominant_and_purity(group[DISEASE_COL])
        dom_state, state_purity = dominant_and_purity(group[STATE_COL])
        dom_subclass, subclass_purity = dominant_and_purity(group[SUBCLASS_COL])
        donor_id = group[DONOR_COL].iloc[0]
        n_covid = group[DONOR_COL].isin(COVID_AKI_DONORS).sum()

        record = {
            "SEACell": seacell_id,
            "n_cells": len(group),
            "donor_id": donor_id,
            "dominant_disease": dom_disease,
            "disease_purity": disease_purity,
            "dominant_state": dom_state,
            "state_purity": state_purity,
            "dominant_subclass": dom_subclass,
            "subclass_purity": subclass_purity,
            "n_covid_aki_cells": n_covid,
            "is_covid_aki_metacell": n_covid > 0,
            "compartment": compartment,
        }

        if has_assay:
            dom_assay, assay_purity = dominant_and_purity(group[ASSAY_COL])
            record.update(
                dominant_assay=dom_assay,
                assay_purity=assay_purity,
            )
        records.append(record)

    mc_meta = pd.DataFrame(records).set_index("SEACell").loc[seacell_order]

    # %% BUILD
    mc_adata = ad.AnnData(X=mc_counts_mat.copy(), obs=mc_meta, var=raw_var)
    mc_adata.layers["counts"] = mc_counts_mat.copy()
    mc_adata.obsm["X_scVI"] = mc_embedding.values

    # %% MERGE
    if MERGE_INTO:
        print(f"\nMerging into {MERGE_INTO}...")
        existing = ad.read_h5ad(MERGE_INTO)
        target_donors = ONLY_DONORS or set(mc_adata.obs[DONOR_COL].astype(str))
        keep = ~existing.obs[DONOR_COL].astype(str).isin(target_donors)
        existing_kept = existing[keep].copy()

        if not (existing_kept.var_names == mc_adata.var_names).all():
            raise ValueError("Existing and new metacell gene sets do not match.")

        mc_adata = ad.concat(
            [existing_kept, mc_adata], join="outer", merge="same"
        )
        if "counts" not in mc_adata.layers:
            mc_adata.layers["counts"] = mc_adata.X.copy()
        mc_meta = mc_adata.obs.copy()

    # %% SAVE / QC
    out_path = OUTPUT_DIR / f"{compartment}_metacells.h5ad"
    qc_path = OUTPUT_DIR / f"{compartment}_metacell_qc.csv"

    mc_adata.write_h5ad(out_path)
    mc_meta.to_csv(qc_path)

    purity_cols = ["disease_purity", "state_purity", "subclass_purity", "n_cells"]
    if "assay_purity" in mc_meta:
        purity_cols.insert(3, "assay_purity")

    print(f"\nWrote {mc_adata.shape[0]} metacells x {mc_adata.shape[1]} genes")
    print(mc_meta[purity_cols].describe().round(2))
    print("\nMetacells by dominant disease:")
    print(mc_meta["dominant_disease"].value_counts())

    if "dominant_assay" in mc_meta:
        print("\nMetacells by dominant assay:")
        print(mc_meta["dominant_assay"].value_counts())

    print(
        "\nMetacells/donor:",
        mc_meta.groupby(DONOR_COL).size().agg(["min", "median", "max"]).to_dict(),
    )

    low_purity = mc_meta[mc_meta["disease_purity"] < 0.6]
    print(f"\n{len(low_purity)} / {len(mc_meta)} metacells have disease_purity < 0.6")

    if "assay_purity" in mc_meta:
        low_assay = mc_meta[mc_meta["assay_purity"] < 0.6]
        print(f"{len(low_assay)} / {len(mc_meta)} metacells have assay_purity < 0.6")

    results[compartment] = {"metacells": out_path, "qc": qc_path}

    del adata, mc_counts_mat, mc_adata, embedding_df, mc_embedding, indicator, raw_X
    gc.collect()

# %% RESULTS
print(f"\n{'=' * 70}\nAll compartments complete\n{'=' * 70}")
for compartment, paths in results.items():
    print(f"{compartment}: {paths['metacells']}")
