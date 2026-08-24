# %% scVI Plotting
# > Plot scVI results
# > Calculate complement-related transcriptional program scores
#
# %% PATH SETUP
from pathlib import Path

PROJECT_DIR = Path.cwd().parent

OUTPUT_DIR = PROJECT_DIR / "outputs/scvi/"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DATA_DIR = PROJECT_DIR / "data/"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# %% IMPORTS

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import decoupler as dc

print(f"scanpy {sc.__version__}, decoupler {dc.__version__}")  # log versions for methods reporting

plt.ioff()
# %%
adata = sc.read_h5ad(DATA_DIR / "adata_scvi.h5ad")
adata

# %% PLOT SETUP
sc.set_figure_params(
    dpi=150,
    dpi_save=600,
    frameon=False,
    fontsize=8,
    vector_friendly=True,
    transparent=True
)
sc.settings.figdir = OUTPUT_DIR

# %% complement module definitions (non-overlapping)
# Each gene is assigned to exactly one module: receptors (C3AR1, C5AR1, C5AR2)
# live only in `receptor`, never in a production module (previously also in
# alternative/terminal, causing double counting). `cfhr` is scored but held
# out of every composite equation below, since CFHR1-5 competitively
# antagonize CFH and do not share a uniform inhibitory direction with the
# other `regulator` genes.
complement_gene_sets = {
    "classical": [
        "C1QA", "C1QB", "C1QC", "C1R", "C1S", "C2", "C4A", "C4B", "C4BPA", "C4BPB"
    ],
    "lectin": [
        "MBL2", "FCN1", "FCN2", "FCN3", "MASP1", "MASP2", "MASP3"
    ],
    "alternative": [
        "C3", "CFB", "CFD", "CFP"
    ],
    "terminal": [
        "C5", "C6", "C7", "C8A", "C8B", "C8G", "C9"
    ],
    "receptor": [
        "C3AR1", "C5AR1", "C5AR2", "CR1", "CR2", "ITGAM", "ITGAX", "VSIG4"
    ],
    "regulator": [
        "CFH", "CFI", "CD46", "CD55", "CD59", "SERPING1"
    ],
}

# Reported descriptively only; excluded from production/response/activation/load equations.
descriptive_gene_sets = {
    "cfhr": ["CFHR1", "CFHR2", "CFHR3", "CFHR4", "CFHR5"],
}

MIN_GENES_PER_MODULE = 2  # modules resolving fewer genes than this are not scored


def build_raw_gene_symbol_map(adata):
    raw_var = adata.raw.var
    if "feature_name" in raw_var.columns:
        return dict(zip(raw_var["feature_name"].astype(str), raw_var.index.astype(str)))
    return dict(zip(raw_var.index.astype(str), raw_var.index.astype(str)))


def build_net(adata, gene_sets, min_genes=MIN_GENES_PER_MODULE):
    gene_map = build_raw_gene_symbol_map(adata)

    net_rows = []
    resolved_genes = {}
    missing_genes = {}
    for module, genes in gene_sets.items():
        genes_resolved = [gene_map[g] for g in genes if g in gene_map]
        missing = [g for g in genes if g not in gene_map]

        print(f"{module}: {len(genes_resolved)}/{len(genes)} genes resolved")
        if missing:
            print(f"  missing: {missing}")
        if len(genes_resolved) < min_genes:
            raise ValueError(f"{module} has too few resolved genes: {genes_resolved}")

        resolved_genes[module] = genes_resolved
        missing_genes[module] = missing
        for gene in genes_resolved:
            net_rows.append({"source": module, "target": gene, "weight": 1.0})

    return pd.DataFrame(net_rows), resolved_genes, missing_genes


def score_modules_ulm(adata, net):
    # ULM is a deterministic closed-form linear regression per cell -- no
    # random seed applies. Scored against adata.raw (raw counts), matching
    # the original use_raw=True convention.
    #
    # bsize=50000 sized for a 64GB machine -- decoupler's default (250,000)
    # causes a multi-GB-per-batch dense conversion that can appear to hang.
    #
    # tmin=MIN_GENES_PER_MODULE (2) overrides decoupler's own default of 5,
    # which would otherwise silently drop any module with fewer than 5
    # matched genes -- e.g. `alternative` (4 genes) -- before scoring even
    # starts, and without raising an error. We already enforce our own
    # min_genes threshold in build_net, so this just stops decoupler's
    # stricter default from re-filtering on top of that.
    adata_raw = adata.raw.to_adata()
    adata_raw.obs = adata.obs

    result = dc.mt.ulm(data=adata_raw, net=net, verbose=True, bsize=50000, tmin=MIN_GENES_PER_MODULE)
    if result is not None:
        # decoupler drops cells with zero expression across every matched
        # gene in this module and returns a new, smaller AnnData rather
        # than mutating adata_raw in place -- use that returned object.
        adata_raw = result

    scores = dc.pp.get_obsm(adata=adata_raw, key="score_ulm").to_df()
    scores.columns = [f"{c}_score" for c in scores.columns]

    # Reindex onto the full cell set: any cells decoupler dropped as "empty"
    # get NaN here (they had literally zero expression of every gene in this
    # module, so there's no real score to assign). At 783/1,367,561 cells
    # (~0.06%), this is negligible but worth noting in your methods --
    # dominant_pathway/composite scores will be NaN for these specific cells.
    scores = scores.reindex(adata.obs_names)
    adata.obs[scores.columns] = scores.values
    return scores.columns.tolist()


def zscore_columns(adata, cols, suffix="_z"):
    # Places modules built from gene sets of different sizes (n=4 to n=10)
    # on a comparable scale before they are combined in any composite metric.
    #
    # Uses nan-aware mean/std: ~783 cells with zero expression across every
    # matched gene in a module come back as NaN from decoupler (see
    # score_modules_ulm), and plain np.mean/np.std would propagate that NaN
    # into every cell's z-score, not just the missing ones.
    z_cols = []
    for col in cols:
        z_col = col.replace("_score", suffix)
        vals = adata.obs[col].to_numpy(dtype=float)
        mean = np.nanmean(vals)
        std = np.nanstd(vals, ddof=0)
        adata.obs[z_col] = (vals - mean) / std
        z_cols.append(z_col)
    return z_cols

complement_net, resolved_complement_genes, missing_complement_genes = build_net(
    adata, complement_gene_sets
)
program_score_cols = score_modules_ulm(adata, complement_net)
program_z_cols = zscore_columns(adata, program_score_cols)

descriptive_net, resolved_descriptive_genes, missing_descriptive_genes = build_net(
    adata, descriptive_gene_sets, min_genes=2
)
descriptive_score_cols = score_modules_ulm(adata, descriptive_net)
descriptive_z_cols = zscore_columns(adata, descriptive_score_cols)  # cfhr_z: descriptive only

# %% complement-related transcriptional program scores (composite metrics)
# Renamed from "complement activation scores": these are transcript-level
# enrichment proxies, not validated measures of protein-level complement
# activation (cleavage, deposition, or secretion).
#
# z_p = z-scored ULM score for module p in {classical, lectin, alternative, terminal, receptor, regulator}
#
#   production_score    = mean(z_classical, z_lectin, z_alternative, z_terminal)
#   response_score       = z_receptor
#   activation_index      = mean(z_classical, z_lectin, z_alternative, z_terminal, z_receptor) - z_regulator
#   net_complement_load   = sum(z_classical, z_lectin, z_alternative, z_terminal, z_receptor, z_regulator)
#   dominant_pathway      = argmax_p(z_p)   (over the six non-overlapping modules; cfhr excluded)

production_cols = ["classical_z", "lectin_z", "alternative_z", "terminal_z"]
activating_cols = production_cols + ["receptor_z"]

adata.obs["production_score"] = adata.obs[production_cols].mean(axis=1)
adata.obs["response_score"] = adata.obs["receptor_z"]
adata.obs["activation_index"] = adata.obs[activating_cols].mean(axis=1) - adata.obs["regulator_z"]
adata.obs["net_complement_load"] = adata.obs[program_z_cols].sum(axis=1)
adata.obs["dominant_pathway"] = (
    adata.obs[program_z_cols]
    .idxmax(axis=1)
    .str.replace("_z", "", regex=False)
    .astype("category")
)

composite_cols = [
    "production_score",
    "response_score",
    "activation_index",
    "net_complement_load",
    "dominant_pathway",
]
adata.obs[program_z_cols + descriptive_z_cols + composite_cols].head()

# %% PLOT COMPLEMENT-RELATED TRANSCRIPTIONAL PROGRAM SCORES
sc.pl.umap(adata, color="Class", save="_full_dataset_Class.svg", show=False)
sc.pl.umap(adata, color="SubclassLevel1", save="_full_dataset_SubclassLevel1.svg", show=False)
sc.pl.umap(adata, color="SubclassLevel2", save="_full_dataset_SubclassLevel2.svg", show=False)
sc.pl.umap(adata, color="cell_type", save="_full_dataset_cell_type.svg", show=False)
sc.pl.umap(adata, color="tissue", save="_full_dataset_tissue.svg", show=False)
sc.pl.umap(adata, color="disease", save="_full_dataset_disease.svg", show=False)

sc.pl.umap(
    adata,
    color=program_z_cols + descriptive_z_cols + ["activation_index"],
    cmap="inferno",
    vmin="p1",
    vmax="p99",
    save="_full_dataset_program_scores.svg",
    show=False
)

sc.pl.umap(
    adata,
    color="dominant_pathway",
    save="_full_dataset_dominant_pathway.svg",
)

sc.pl.violin(
    adata,
    keys=program_z_cols + ["activation_index"],
    groupby="disease",
    rotation=20,
    save="_full_dataset_program_scores_by_disease.svg",
)

# %% save resolved/missing gene log for supplementary reporting
gene_resolution_log = pd.DataFrame([
    {"module": m, "n_input": len(complement_gene_sets.get(m, descriptive_gene_sets.get(m, []))),
     "n_resolved": len(resolved_complement_genes.get(m, resolved_descriptive_genes.get(m, []))),
     "missing": ", ".join(missing_complement_genes.get(m, missing_descriptive_genes.get(m, [])))}
    for m in list(complement_gene_sets) + list(descriptive_gene_sets)
])
gene_resolution_log.to_csv(OUTPUT_DIR / "complement_module_gene_resolution.csv", index=False)
gene_resolution_log

# %% persist scored obs for downstream analyses (avoids re-running ~15min ULM fits)
adata.obs.to_parquet(OUTPUT_DIR / "complement_scores_obs.parquet")
print(f"Saved scored obs to {OUTPUT_DIR / 'complement_scores_obs.parquet'}")
