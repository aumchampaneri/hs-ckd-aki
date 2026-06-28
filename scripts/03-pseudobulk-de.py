# %% Pseudobulk DE
#
#
# %% PATH SETUP
from pathlib import Path

SCRIPT_DIR = Path.cwd()
PROJECT_DIR = SCRIPT_DIR.parent

OUTPUT_DIR = PROJECT_DIR / "outputs/"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DATA_DIR = PROJECT_DIR / "data/"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# %% IMPORTS
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
from adpbulk import ADPBulk
from pydeseq2.dds import DeseqDataSet
from pydeseq2.ds import DeseqStats
from scipy import stats

# %%
# CONFIGURATION & CONTROLS
# > Define keys, thresholds, and comparison specs
####
adata_path = DATA_DIR / "adata_scvi.h5ad"
disease_key = "disease"
donor_key = "donor_id"
normal_label = "normal"
min_donors_per_group = 5
min_cells_per_pseudobulk = 25
min_total_count = 20
min_detected_donors = 3

# Keep covariate adjustment conservative
adjustment_covariates = []

comparison_specs = [
    {
        "name": "ckd_vs_normal",
        "group1": "chronic kidney disease",
        "group2": normal_label,
    },
    {"name": "aki_vs_normal", "group1": "acute kidney failure", "group2": normal_label},
    {
        "name": "aki_vs_ckd",
        "group1": "acute kidney failure",
        "group2": "chronic kidney disease",
    },
]

# %% LOAD DATA
adata = sc.read_h5ad(adata_path)
print(f"Loaded Atlas: {adata.n_obs} cells, {adata.n_vars} genes.")
print(f"Using {adata.raw.n_vars} raw genes for DE.")


# %%
# UTILITIES & PSEUDOBULK GENERATION
# > ADPBulk utilities to generate pseudobulks
####
def raw_gene_symbols(adata):
    """Fallback utility to map gene names from raw data."""
    raw_var = adata.raw.var.copy()
    for col in ["feature_name", "gene_name", "symbol"]:
        if col in raw_var.columns:
            return raw_var[col].astype(str).to_numpy()
    return adata.raw.var_names.astype(str).to_numpy()


def benjamini_hochberg(pvals):
    """Standard FDR correction."""
    pvals = np.asarray(pvals, dtype=float)
    qvals = np.full_like(pvals, np.nan, dtype=float)
    ok = np.isfinite(pvals)
    if ok.sum() == 0:
        return qvals
    p = pvals[ok]
    order = np.argsort(p)
    ranked = p[order]
    n = len(ranked)
    adjusted = ranked * n / np.arange(1, n + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    qvals[ok] = np.clip(adjusted, 0, 1)[np.argsort(order)]
    return qvals


def filter_genes_for_de(counts_mat):
    """Filter genes by total expression and detection rate across donors."""
    total_count = np.asarray(counts_mat.sum(axis=0)).ravel()
    detected_donors = np.asarray((counts_mat > 0).sum(axis=0)).ravel()
    keep = (total_count >= min_total_count) & (detected_donors >= min_detected_donors)
    return keep, total_count, detected_donors


def build_donor_pseudobulk(adata, query=None):
    """
    Refactored to use adpbulk for aggregation.
    Generates pseudobulked counts, metadata, and normalized matrices.
    """
    # 1. Optional subsetting
    sub = adata[adata.obs.query(query).index].copy() if query else adata
    categories = [donor_key, disease_key] + [
        c for c in adjustment_covariates if c in sub.obs
    ]

    # 2. Aggregation using adpbulk
    adpb = ADPBulk(sub, categories, use_raw=True)
    counts_df = adpb.fit_transform()
    meta = adpb.get_meta()

    # 3. Retrieve cell counts per group for filtering
    cell_counts = (
        sub.obs.groupby(categories, observed=True).size().reset_index(name="n_cells")
    )
    meta = meta.merge(cell_counts, on=categories, how="left")

    # 4. Filter out pseudobulks with too few cells
    valid_mask = meta["n_cells"] >= min_cells_per_pseudobulk
    meta = meta[valid_mask].reset_index(drop=True)
    counts_df = counts_df.loc[valid_mask]

    # 5. Generate matrices for sensitivity stats (CPM, Log1p)
    counts_mat = sp.csr_matrix(counts_df.values)
    library_size = np.asarray(counts_mat.sum(axis=1)).ravel()
    cpm = counts_mat.multiply(1e6 / np.maximum(library_size, 1)[:, None]).tocsr()

    log1p_cpm = cpm.copy()
    log1p_cpm.data = np.log1p(log1p_cpm.data)

    return {
        "counts_df": counts_df,  # Retained for cleaner PyDESeq2 ingestion
        "counts": counts_mat,
        "meta": meta,
        "library_size": library_size,
        "cpm": cpm,
        "log1p_cpm": log1p_cpm,
    }


# %%
# DE MODELS & STATS
# > PyDESeq2 model fitting and result summarization
####
def subset_pseudobulk_to_groups(pb, group1, group2):
    """Subsets the adpbulk output to the explicit contrast groups."""
    keep = pb["meta"][disease_key].astype(str).isin([group1, group2])
    keep_idx = keep.to_numpy()

    return {
        "counts_df": pb["counts_df"].iloc[keep_idx].copy(),
        "counts": pb["counts"][keep_idx, :].tocsr(),
        "meta": pb["meta"].loc[keep].reset_index(drop=True),
        "library_size": pb["library_size"][keep_idx],
        "cpm": pb["cpm"][keep_idx, :].tocsr(),
        "log1p_cpm": pb["log1p_cpm"][keep_idx, :].tocsr(),
    }


def run_pydeseq2(pb, keep_genes, group1, group2, name):
    """Runs PyDESeq2 using the adpbulk DataFrame."""
    # Subset to passing genes and cast to int for PyDESeq2
    counts_df = pb["counts_df"].iloc[:, keep_genes].copy()
    counts_df = np.rint(counts_df).astype(np.int64)

    # Align metadata and structure categories
    metadata = pb["meta"][
        [disease_key] + [c for c in adjustment_covariates if c in pb["meta"]]
    ].copy()
    metadata.index = counts_df.index
    metadata[disease_key] = metadata[disease_key].astype("category")

    for cov in adjustment_covariates:
        if cov in metadata:
            metadata[cov] = metadata[cov].astype("category")

    design_terms = [c for c in adjustment_covariates if c in metadata.columns] + [
        disease_key
    ]
    design_formula = "~ " + " + ".join(design_terms)

    dds = DeseqDataSet(
        counts=counts_df,
        metadata=metadata,
        design=design_formula,
        ref_level=[disease_key, group2],
        refit_cooks=True,
        min_replicates=7,
        n_cpus=4,
        quiet=True,
    )

    dds.deseq2()
    stat_res = DeseqStats(
        dds,
        contrast=[disease_key, group1, group2],
        alpha=0.05,
        cooks_filter=True,
        independent_filter=True,
        n_cpus=4,
        quiet=True,
    )

    stat_res.summary()
    res = stat_res.results_df.reset_index().rename(columns={"index": "gene_id"})
    res["gene_id"] = res["gene_id"].astype(str)

    res = res.rename(
        columns={
            "baseMean": "deseq2_base_mean",
            "log2FoldChange": "deseq2_log2fc",
            "lfcSE": "deseq2_lfc_se",
            "stat": "deseq2_wald_stat",
            "pvalue": "deseq2_p_value",
            "padj": "deseq2_q_value",
        }
    )

    res["deseq2_design"] = design_formula
    res["deseq2_n_genes_tested"] = int(keep_genes.sum())

    return res


def run_pseudobulk_de(spec):
    """Pipeline executor for a given contrast spec."""
    name = spec["name"]
    group1 = spec["group1"]
    group2 = spec["group2"]
    query = spec.get("query")

    pb = build_donor_pseudobulk(adata, query=query)
    pb = subset_pseudobulk_to_groups(pb, group1, group2)
    meta = pb["meta"]

    # Validation
    mask1 = (meta[disease_key].astype(str) == group1).to_numpy()
    mask2 = (meta[disease_key].astype(str) == group2).to_numpy()
    if mask1.sum() < min_donors_per_group or mask2.sum() < min_donors_per_group:
        raise ValueError(f"{name}: Insufficient donors for contrast.")

    # Gene Filtering
    keep_genes, total_count, detected_donors = filter_genes_for_de(pb["counts"])

    gene_info = pd.DataFrame(
        {
            "gene_id": adata.raw.var_names.astype(str),
            "gene_symbol": raw_gene_symbols(adata),
            "total_pseudobulk_count": total_count,
            "detected_pseudobulk_donors": detected_donors,
            "passes_expression_filter": keep_genes,
        }
    )

    # Run primary models
    deseq = run_pydeseq2(pb, keep_genes, group1, group2, name)
    out = gene_info.merge(deseq, on="gene_id", how="left")

    # Metadata bindings
    out["comparison_id"] = name
    out["comparison"] = f"{group1} vs {group2}"
    out["is_de_primary_q_0_05"] = out["deseq2_q_value"] <= 0.05

    # Save Outputs
    out_path = OUTPUT_DIR / f"pseudobulk_de_{name}.csv"
    out.to_csv(out_path, index=False)

    print(
        f"{name}: Wrote to {out_path.name} | DE Genes (q < 0.05): {out['is_de_primary_q_0_05'].sum()}"
    )
    return out


# %%
# EXECUTE PIPELINE
# > Iterate over comparison specs and run the DE pipeline
####
pseudobulk_results = {}
for spec in comparison_specs:
    pseudobulk_results[spec["name"]] = run_pseudobulk_de(spec)

print("Pseudobulk DE pipeline completed.")
