# %% scVI Processing
# > Train a scVI model on the CKD-AKI dataset
#
# %% PATH SETUP
from pathlib import Path

PROJECT_DIR = Path.cwd().parent

OUTPUT_DIR = PROJECT_DIR / "outputs/plots/scvi/"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DATA_DIR = PROJECT_DIR / "data/"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# %% IMPORTS
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc

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

# %% complement scoring
complement_gene_sets = {
    "classical": [
        "C1QA", "C1QB", "C1QC", "C1R", "C1S", "C2", "C4A", "C4B", "C4BPA", "C4BPB"
    ],
    "lectin": [
        "MBL2", "FCN1", "FCN2", "FCN3", "MASP1", "MASP2", "MASP3"
    ],
    "alternative": [
        "C3", "CFB", "CFD", "CFP", "C3AR1"
    ],
    "terminal": [
        "C5", "C5AR1", "C5AR2", "C6", "C7", "C8A", "C8B", "C8G", "C9"
    ],
    "receptor": [
        "C3AR1", "C5AR1", "C5AR2", "CR1", "CR2", "ITGAM", "ITGAX", "VSIG4"
    ],
    "regulator": [
        "CFH", "CFHR1", "CFHR2", "CFHR3", "CFHR4", "CFHR5", "CFI", "CD46", "CD55", "CD59", "SERPING1"
    ],
}

def build_raw_gene_symbol_map(adata):
    raw_var = adata.raw.var
    if "feature_name" in raw_var.columns:
        return dict(zip(raw_var["feature_name"].astype(str), raw_var.index.astype(str)))
    return dict(zip(raw_var.index.astype(str), raw_var.index.astype(str)))


def score_gene_program_from_raw(adata, gene_symbols, score_name, min_genes=2):
    gene_map = build_raw_gene_symbol_map(adata)
    genes_resolved = [gene_map[g] for g in gene_symbols if g in gene_map]
    missing = [g for g in gene_symbols if g not in gene_map]

    print(f"{score_name}: {len(genes_resolved)}/{len(gene_symbols)} genes resolved")
    if missing:
        print(f"  missing: {missing}")
    if len(genes_resolved) < min_genes:
        raise ValueError(f"{score_name} has too few resolved genes: {genes_resolved}")

    sc.tl.score_genes(
        adata,
        gene_list=genes_resolved,
        use_raw=True,
        score_name=score_name,
    )
    return genes_resolved, missing

resolved_complement_genes = {}
missing_complement_genes = {}
for program, genes in complement_gene_sets.items():
    resolved, missing = score_gene_program_from_raw(
        adata,
        genes,
        f"{program}_score",
    )
    resolved_complement_genes[program] = resolved
    missing_complement_genes[program] = missing

# %% derived complement scores
program_score_cols = [f"{program}_score" for program in complement_gene_sets]
producer_score_cols = [
    "classical_score",
    "lectin_score",
    "alternative_score",
    "terminal_score",
    "regulator_score",
]
activation_score_cols = [
    "classical_score",
    "lectin_score",
    "alternative_score",
    "terminal_score",
    "receptor_score",
]

adata.obs["production_score"] = adata.obs[producer_score_cols].mean(axis=1)
adata.obs["response_score"] = adata.obs["receptor_score"]
adata.obs["activation_index"] = adata.obs[activation_score_cols].mean(axis=1) - adata.obs["regulator_score"]
adata.obs["net_complement_load"] = adata.obs[program_score_cols].sum(axis=1)
adata.obs["dominant_pathway"] = (
    adata.obs[program_score_cols]
    .idxmax(axis=1)
    .str.replace("_score", "", regex=False)
    .astype("category")
)

adata.obs[
    program_score_cols + [
        "production_score",
        "response_score",
        "activation_index",
        "net_complement_load",
        "dominant_pathway",
    ]
].head()

# %% PLOT COMPLEMENT SCORES
sc.pl.umap(adata, color="Class", layer="X_scVI", save="_full_dataset_Class.svg", show=False)
sc.pl.umap(adata, color="SubclassLevel1", layer="X_scVI", save="_full_dataset_SubclassLevel1.svg", show=False)
sc.pl.umap(adata, color="SubclassLevel2", layer="X_scVI", save="_full_dataset_SubclassLevel2.svg", show=False)
sc.pl.umap(adata, color="cell_type", layer="X_scVI", save="_full_dataset_cell_type.svg", show=False)
sc.pl.umap(adata, color="tissue", layer="X_scVI", save="_full_dataset_tissue.svg", show=False)
sc.pl.umap(adata, color="disease", layer="X_scVI", save="_full_dataset_disease.svg", show=False)

sc.pl.umap(
    adata,
    color=program_score_cols + ["activation_index"],
    cmap="inferno",
    vmin="p1",
    vmax="p99",
    layer="X_scVI",
    save="_full_dataset_program_scores.svg",
    show=False
)

sc.pl.umap(
    adata,
    color="dominant_pathway",
    layer="X_scVI",
    save="_full_dataset_dominant_pathway.svg",
)

sc.pl.violin(
    adata,
    keys=program_score_cols + ["activation_index"],
    groupby="disease",
    rotation=20,
    save="_full_dataset_program_scores_by_disease.svg",
)
