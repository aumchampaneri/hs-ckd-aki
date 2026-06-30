# %% Pseudobulk DE
#
#
# %% PATH SETUP
from pathlib import Path

SCRIPT_DIR = Path.cwd()
PROJECT_DIR = SCRIPT_DIR.parent

OUTPUT_DIR = PROJECT_DIR / "outputs/" / "pseudobulk_de"
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
    {"name": "aki_vs_normal", "group1": "acute kidney injury", "group2": normal_label},
    {
        "name": "aki_vs_ckd",
        "group1": "acute kidney injury",
        "group2": "chronic kidney disease",
    },
]

# %% LOAD DATA
adata = sc.read_h5ad(adata_path)
print(f"Loaded Atlas: {adata.n_obs} cells, {adata.n_vars} genes.")
print(f"Using {adata.raw.n_vars} raw genes for DE.")


# %%
# VALIDATE COMPARISON SPECS
# > Catch label typos early instead of failing downstream with confusing errors
####
_observed_disease_labels = set(adata.obs[disease_key].astype(str).unique())
for _spec in comparison_specs:
    for _label in (_spec["group1"], _spec["group2"]):
        if _label not in _observed_disease_labels:
            raise ValueError(
                f"comparison_specs['{_spec['name']}']: label '{_label}' not found in "
                f"adata.obs['{disease_key}']. Available labels: {sorted(_observed_disease_labels)}"
            )


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
    Uses adpbulk for aggregation (method="sum", identical to summing raw
    counts within each donor x disease group via a manual design matrix).
    Mirrors the notebook's manual implementation: pseudobulk *grouping* is
    strictly donor + disease; adjustment_covariates are NOT part of the
    grouping key and are joined on afterward as donor-level metadata, so
    populating adjustment_covariates later won't change which cells get
    pooled into each pseudobulk sample.
    """
    # 1. Optional subsetting
    sub = adata[adata.obs.query(query).index].copy() if query else adata
    group_cols = [donor_key, disease_key]

    # 2. Aggregation using adpbulk. method="sum" is the library default but
    # is set explicitly here so the equivalence to the notebook's manual
    # `design @ raw_counts` summation doesn't silently depend on a default
    # that could change in a future adpbulk release.
    adpb = ADPBulk(sub, group_cols, method="sum", use_raw=True)
    counts_df = adpb.fit_transform()
    meta = adpb.get_meta()

    # ADPBulk pulls columns directly from adata.raw.var.index when
    # use_raw=True, so gene order should already match adata.raw.var_names.
    # Assert this explicitly rather than relying on that implementation
    # detail silently holding - keep_genes (a boolean array aligned to
    # adata.raw.var_names) is later used to positionally index counts_df
    # columns in run_pydeseq2, so a mismatch here would silently corrupt
    # every downstream DE result.
    expected_genes = adata.raw.var_names.astype(str)
    assert list(counts_df.columns.astype(str)) == list(expected_genes), (
        "ADPBulk gene column order does not match adata.raw.var_names; "
        "downstream gene_id alignment (keep_genes indexing) would be wrong."
    )

    # 3. Attach donor-level covariates after aggregation (not part of the
    # grouping key) - matches the notebook's `donor_cov` join.
    cov_cols = [c for c in adjustment_covariates if c in sub.obs]
    if cov_cols:
        donor_cov = (
            sub.obs[[donor_key] + cov_cols]
            .astype(str)
            .drop_duplicates(subset=[donor_key])
            .set_index(donor_key)
        )
        meta = meta.join(donor_cov, on=donor_key)

    # 4. Retrieve cell counts per group for filtering
    cell_counts = (
        sub.obs.groupby(group_cols, observed=True).size().reset_index(name="n_cells")
    )
    meta = meta.merge(cell_counts, on=group_cols, how="left")
    # Left-merge should preserve meta's original row order/length (matching
    # counts_df), but guard against silent misalignment if that ever breaks
    # (e.g. duplicate category combinations producing extra rows).
    assert len(meta) == counts_df.shape[0], (
        "meta and counts_df row counts diverged after merge - "
        "check for duplicate category combinations in cell_counts."
    )

    # 5. Filter out pseudobulks with too few cells
    # NOTE: counts_df (from adpb.fit_transform()) is indexed by pseudobulk
    # group labels, not by meta's row-number index, so a boolean *Series*
    # mask (which aligns by label) will fail or silently misalign. Convert
    # to a numpy array so indexing is purely positional.
    valid_mask = (meta["n_cells"] >= min_cells_per_pseudobulk).to_numpy()
    meta = meta[valid_mask].reset_index(drop=True)
    counts_df = counts_df.iloc[valid_mask]

    # 6. Generate matrices for sensitivity stats (CPM, Log1p)
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
