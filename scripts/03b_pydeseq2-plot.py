# %% Plotting pyDeSeq2 results
#
#
# %% PATH SETUP
from pathlib import Path

SCRIPT_DIR = Path.cwd()
PROJECT_DIR = SCRIPT_DIR.parent

OUTPUT_DIR = PROJECT_DIR / "outputs/"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PLOT_DIR = OUTPUT_DIR / "plots/" / "pseudobulk_de"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

DATA_DIR = PROJECT_DIR / "data/"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# %% IMPORTS
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

# %%
# CONFIGURATION
# > Keys and comparison IDs must match 03-pseudobulk-de.py exactly, since
# this script reads the CSVs that script writes to OUTPUT_DIR.
####
global_comparison_ids = ["ckd_vs_normal", "aki_vs_normal", "aki_vs_ckd"]

comparison_title_map = {
    "ckd_vs_normal": "CKD vs normal",
    "aki_vs_normal": "AKI vs normal",
    "aki_vs_ckd": "AKI vs CKD",
}

complement_volcano_dir = PLOT_DIR / "complement_aware"
complement_volcano_dir.mkdir(parents=True, exist_ok=True)

inflammasome_volcano_dir = PLOT_DIR / "inflammasome_aware"
inflammasome_volcano_dir.mkdir(parents=True, exist_ok=True)

background_color = "#d1d5db"

# %%
# GENE PROGRAM DEFINITIONS
# > Complement + inflammasome gene sets used to annotate volcano plots
####
complement_programs = {
    "classical": [
        "C1QA",
        "C1QB",
        "C1QC",
        "C1R",
        "C1S",
        "C2",
        "C4A",
        "C4B",
        "C4BPA",
        "C4BPB",
    ],
    "lectin": ["MBL2", "FCN1", "FCN2", "FCN3", "MASP1", "MASP2", "MASP3"],
    "alternative": ["C3", "CFB", "CFD", "CFP", "C3AR1"],
    "terminal": ["C5", "C5AR1", "C5AR2", "C6", "C7", "C8A", "C8B", "C8G", "C9"],
    "receptor": ["C3AR1", "C5AR1", "C5AR2", "CR1", "CR2", "ITGAM", "ITGAX", "VSIG4"],
    "regulator": [
        "CFH",
        "CFHR1",
        "CFHR2",
        "CFHR3",
        "CFHR4",
        "CFHR5",
        "CFI",
        "CD46",
        "CD55",
        "CD59",
        "SERPING1",
    ],
}

complement_pathway_palette = {
    "classical": "#2563eb",
    "lectin": "#0f766e",
    "alternative": "#f97316",
    "terminal": "#dc2626",
    "receptor": "#7c3aed",
    "regulator": "#16a34a",
    "complement_other": "#111827",
}

# A gene can appear in more than one program; this priority makes the visual role stable.
complement_pathway_priority = [
    "regulator",
    "receptor",
    "terminal",
    "alternative",
    "lectin",
    "classical",
]

inflammasome_programs = {
    "sensors": [
        "NLRP3",
        "NLRP1",
        "NLRP6",
        "NLRP7",
        "NLRC4",
        "AIM2",
        "MEFV",
        "IFI16",
        "PYHIN1",
        "NOD1",
        "NOD2",
    ],
    "adapters_caspases": ["PYCARD", "CASP1", "CASP4", "CASP5", "CASP8", "CARD8"],
    "gasdermin_pyroptosis": ["GSDMD", "GSDME", "DFNA5", "GSDMB", "GSDMC", "GSDMA"],
    "cytokines_il1_il18": [
        "IL1B",
        "IL1A",
        "IL18",
        "IL18R1",
        "IL18RAP",
        "IL1R1",
        "IL1RAP",
        "IL1RN",
    ],
    "priming_nfkb_tlr": [
        "TLR2",
        "TLR4",
        "MYD88",
        "TICAM1",
        "NFKB1",
        "NFKB2",
        "RELA",
        "RELB",
        "CHUK",
        "IKBKB",
        "TNF",
        "TNFRSF1A",
    ],
    "nlrp3_mito_stress": [
        "TXNIP",
        "NEK7",
        "P2RX7",
        "GBP5",
        "PINK1",
        "PRKN",
        "PARK7",
        "SQSTM1",
        "SOD2",
        "HMOX1",
        "GPX4",
        "VDAC1",
        "VDAC2",
        "MAVS",
    ],
}

inflammasome_palette = {
    "sensors": "#b91c1c",
    "adapters_caspases": "#7c2d12",
    "gasdermin_pyroptosis": "#ea580c",
    "cytokines_il1_il18": "#c026d3",
    "priming_nfkb_tlr": "#2563eb",
    "nlrp3_mito_stress": "#059669",
    "inflammasome_other": "#111827",
}

inflammasome_program_priority = [
    "sensors",
    "adapters_caspases",
    "gasdermin_pyroptosis",
    "cytokines_il1_il18",
    "nlrp3_mito_stress",
    "priming_nfkb_tlr",
]


def build_gene_set_program_table(programs, priority):
    """Maps each gene to all programs it belongs to, plus one stable 'primary' program."""
    rows = []
    for program, genes in programs.items():
        for gene in genes:
            rows.append({"gene_symbol": gene, "program": program})
    program_df = pd.DataFrame(rows).drop_duplicates()
    gene_to_programs = (
        program_df.groupby("gene_symbol")["program"]
        .apply(lambda x: ";".join(sorted(set(x))))
        .to_dict()
    )
    primary = {}
    for gene, programs_joined in gene_to_programs.items():
        program_set = set(programs_joined.split(";"))
        primary[gene] = next(
            (p for p in priority if p in program_set), sorted(program_set)[0]
        )
    return gene_to_programs, primary


complement_gene_to_pathways, complement_gene_to_primary_pathway = (
    build_gene_set_program_table(complement_programs, complement_pathway_priority)
)
complement_symbol_set = set(complement_gene_to_pathways)

inflammasome_gene_to_programs, inflammasome_gene_to_primary_program = (
    build_gene_set_program_table(inflammasome_programs, inflammasome_program_priority)
)
inflammasome_symbol_set = set(inflammasome_gene_to_programs)

# %%
# SUBCLASSLEVEL1 REFERENCE
# > Readable labels for SubclassLevel1 acronyms used in triptych titles
####
subclasslevel1_label_map = {
    "PT": "Proximal tubule",
    "TAL": "Thick ascending limb",
    "PC": "Collecting duct principal cells",
    "EC": "Endothelial cells",
    "FIB": "Fibroblasts / stromal fibroblasts",
    "IC": "Intercalated cells",
    "DTL": "Descending thin limb",
    "DCT": "Distal convoluted tubule",
    "Myeloid": "Myeloid immune cells",
    "CNT": "Connecting tubule",
    "Lymphoid": "Lymphoid immune cells",
    "VSM/P": "Vascular smooth muscle / pericytes",
    "VSM_P": "Vascular smooth muscle / pericytes",
    "POD": "Podocytes",
    "ATL": "Ascending thin limb",
    "PEC": "Parietal epithelial cells",
    "PapE": "Papillary epithelium",
    "NEU": "Neural cells",
    "Ad": "Adventitial/adipose-associated rare cells",
}


def normalize_subclasslevel1_key(value):
    return str(value).replace("/", "_")


def pretty_subclasslevel1_label(value):
    key = str(value)
    if key in subclasslevel1_label_map:
        return f"{key} - {subclasslevel1_label_map[key]}"
    normalized = normalize_subclasslevel1_key(key)
    if normalized in subclasslevel1_label_map:
        original = key.replace("_", "/") if "_" in key else key
        return f"{original} - {subclasslevel1_label_map[normalized]}"
    return key.replace("_", " ")


# %%
# SHARED LABEL/HELPER UTILITIES
####
def pretty_disease_label(value):
    label = str(value)
    replacements = {
        "acute kidney injury": "AKI",
        "chronic kidney disease": "CKD",
        "normal": "normal",
    }
    return replacements.get(label, label)


def label_ranked_genes(
    ax,
    df,
    label_col="gene_symbol",
    x_col="primary_log2fc",
    y_col="plot_y",
    max_labels=14,
):
    """Annotates the top-ranked rows of df (already sorted) with leader-line gene labels."""
    label_df = df.dropna(subset=[x_col, y_col]).copy().head(max_labels)
    if label_df.empty:
        return
    x_span = max(df[x_col].max() - df[x_col].min(), 1)
    y_span = max(df[y_col].max() - df[y_col].min(), 1)
    for i, (_, row) in enumerate(label_df.iterrows()):
        dx = 0.012 * x_span * (1 if row[x_col] >= 0 else -1)
        dy = 0.025 * y_span * ((i % 3) + 1)
        ax.annotate(
            str(row[label_col]),
            xy=(row[x_col], row[y_col]),
            xytext=(row[x_col] + dx, row[y_col] + dy),
            fontsize=7,
            ha="left" if row[x_col] >= 0 else "right",
            va="bottom",
            arrowprops={
                "arrowstyle": "-",
                "lw": 0.35,
                "color": "#6b7280",
                "alpha": 0.7,
            },
        )


def get_contrast_labels(df, comparison_id):
    if {"group1", "group2"}.issubset(df.columns) and df["group1"].notna().any():
        group1 = pretty_disease_label(df["group1"].dropna().astype(str).iloc[0])
        group2 = pretty_disease_label(df["group2"].dropna().astype(str).iloc[0])
    else:
        fallback = {
            "ckd_vs_normal": ("CKD", "normal"),
            "aki_vs_normal": ("AKI", "normal"),
            "aki_vs_ckd": ("AKI", "CKD"),
        }
        group1, group2 = fallback.get(comparison_id, ("group1", "group2"))
    return group1, group2


def get_direction_labels(df, comparison_id):
    group1, group2 = get_contrast_labels(df, comparison_id)
    return f"Higher in {group2}", f"Higher in {group1}"


def get_xaxis_label(df, comparison_id):
    group1, group2 = get_contrast_labels(df, comparison_id)
    return f"primary log2 fold change ({group1} / {group2})"


# %%
# LOAD DE TABLES
# > Reads every pseudobulk_de_*.csv written by 03-pseudobulk-de.py and
# normalizes columns to the "primary_*" naming used throughout the plots
# below. 03-pseudobulk-de.py only runs PyDESeq2 (no Welch sensitivity /
# method-selection step), so primary_* here is always the DESeq2 result;
# group1/group2 are recovered by splitting the "comparison" column
# ("group1 vs group2") since they aren't stored as separate fields.
####
required_de_columns = {
    "gene_id",
    "gene_symbol",
    "deseq2_log2fc",
    "deseq2_q_value",
    "passes_expression_filter",
    "comparison",
}


def load_pseudobulk_de_tables(output_dir, prefix="pseudobulk_de_"):
    tables = {}
    skipped = []
    for path in sorted(output_dir.glob(f"{prefix}*.csv")):
        df = pd.read_csv(path)
        missing = required_de_columns.difference(df.columns)
        if missing:
            skipped.append({"file": path.name, "reason": f"missing {sorted(missing)}"})
            continue

        df["primary_log2fc"] = df["deseq2_log2fc"]
        df["primary_p_value"] = df["deseq2_p_value"]
        df["primary_q_value"] = df["deseq2_q_value"]
        df["neg_log10_primary_q"] = -np.log10(df["primary_q_value"].clip(lower=1e-300))
        if "is_de_primary_q_0_05" not in df.columns:
            df["is_de_primary_q_0_05"] = df["primary_q_value"] <= 0.05

        split_groups = df["comparison"].astype(str).str.split(" vs ", n=1, expand=True)
        df["group1"] = split_groups[0]
        df["group2"] = split_groups[1] if split_groups.shape[1] > 1 else np.nan

        name = path.stem.replace(prefix, "")
        tables[name] = df

    if skipped:
        pd.DataFrame(skipped).to_csv(
            complement_volcano_dir / "skipped_non_de_tables.csv", index=False
        )
        print("Skipped non-gene-level tables:", [s["file"] for s in skipped])
    return tables


all_pseudobulk_de_tables = load_pseudobulk_de_tables(OUTPUT_DIR / "pseudobulk_de")
print("Loaded gene-level DE tables:", sorted(all_pseudobulk_de_tables))


# %%
# COMPLEMENT-AWARE VOLCANO HELPERS
####
def prepare_complement_volcano_df(df):
    plot_df = df.copy()
    plot_df = plot_df[plot_df["passes_expression_filter"].fillna(False)].copy()
    plot_df["gene_symbol"] = plot_df["gene_symbol"].astype(str)
    plot_df["is_complement"] = plot_df["gene_symbol"].isin(complement_symbol_set)
    plot_df["complement_pathways"] = (
        plot_df["gene_symbol"].map(complement_gene_to_pathways).fillna("")
    )
    plot_df["primary_complement_pathway"] = (
        plot_df["gene_symbol"]
        .map(complement_gene_to_primary_pathway)
        .fillna("background")
    )
    plot_df["plot_q"] = plot_df["primary_q_value"].clip(lower=1e-300)
    plot_df["plot_y"] = -np.log10(plot_df["plot_q"])
    plot_df["abs_lfc"] = plot_df["primary_log2fc"].abs()
    return plot_df.replace([np.inf, -np.inf], np.nan)


def build_complement_legend_handles(include_background=True):
    """Handle list shared by both per-axes legends and the triptych's single bottom legend."""
    handles = []
    if include_background:
        handles.append(
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor=background_color,
                markersize=5,
                label="Other genes",
            )
        )
    for pathway, color in complement_pathway_palette.items():
        if pathway == "complement_other":
            continue
        marker = "s" if pathway == "regulator" else "o"
        handles.append(
            Line2D(
                [0],
                [0],
                marker=marker,
                color="none",
                markerfacecolor=color,
                markeredgecolor="black",
                markeredgewidth=0.4,
                markersize=6,
                label=pathway,
            )
        )
    return handles


def add_complement_legend(ax, include_background=True):
    handles = build_complement_legend_handles(include_background)
    ax.legend(handles=handles, frameon=False, fontsize=7, loc="upper right", ncols=1)


def plot_complement_aware_volcano(
    df,
    comparison_id,
    out_prefix,
    title=None,
    q_threshold=0.05,
    lfc_threshold=0.5,
    label_top_complement=14,
    label_top_background=4,
    complement_only=False,
    ax=None,
    show_legend=True,
):
    plot_df = prepare_complement_volcano_df(df)
    if complement_only:
        plot_df = plot_df[plot_df["is_complement"]].copy()
    if plot_df.empty:
        print("No plottable genes for", out_prefix)
        return None

    made_fig = ax is None
    if ax is None:
        fig, ax = plt.subplots(figsize=(7.2, 5.4))
    else:
        fig = ax.figure

    bg = plot_df[~plot_df["is_complement"]]
    comp = plot_df[plot_df["is_complement"]]

    if not complement_only and not bg.empty:
        ax.scatter(
            bg["primary_log2fc"],
            bg["plot_y"],
            s=8,
            c=background_color,
            alpha=0.35,
            linewidths=0,
            rasterized=True,
        )
        bg_sig = bg[bg["is_de_primary_q_0_05"].fillna(False)]
        if not bg_sig.empty:
            ax.scatter(
                bg_sig["primary_log2fc"],
                bg_sig["plot_y"],
                s=8,
                c="#9ca3af",
                alpha=0.55,
                linewidths=0,
                rasterized=True,
            )

    for pathway, color in complement_pathway_palette.items():
        sub = comp[comp["primary_complement_pathway"] == pathway]
        if sub.empty:
            continue
        marker = "s" if pathway == "regulator" else "o"
        ax.scatter(
            sub["primary_log2fc"],
            sub["plot_y"],
            s=42 if not complement_only else 58,
            c=color,
            marker=marker,
            edgecolors="black",
            linewidths=0.45,
            alpha=0.92,
            label=pathway,
        )

    ax.axhline(-np.log10(q_threshold), color="#111827", lw=0.8, ls="--", alpha=0.55)
    ax.axvline(-lfc_threshold, color="#111827", lw=0.8, ls=":", alpha=0.45)
    ax.axvline(lfc_threshold, color="#111827", lw=0.8, ls=":", alpha=0.45)
    ax.axvline(0, color="#111827", lw=0.8, alpha=0.55)

    left_label, right_label = get_direction_labels(plot_df, comparison_id)
    y_top = plot_df["plot_y"].quantile(0.995)
    x_min, x_max = np.nanpercentile(plot_df["primary_log2fc"], [0.5, 99.5])
    ax.text(
        x_min, y_top, left_label, ha="left", va="bottom", fontsize=8, color="#374151"
    )
    ax.text(
        x_max, y_top, right_label, ha="right", va="bottom", fontsize=8, color="#374151"
    )

    comp_to_label = comp[
        (comp["primary_q_value"] <= 0.10) | (comp["abs_lfc"] >= lfc_threshold)
    ].sort_values(["primary_q_value", "abs_lfc"], ascending=[True, False])
    label_ranked_genes(ax, comp_to_label, max_labels=label_top_complement)

    if not complement_only and label_top_background:
        bg_to_label = bg.sort_values(
            ["is_de_primary_q_0_05", "primary_q_value", "abs_lfc"],
            ascending=[False, True, False],
        )
        label_ranked_genes(ax, bg_to_label, max_labels=label_top_background)

    n_comp_sig = int((comp["primary_q_value"] <= q_threshold).sum())
    n_comp = int(len(comp))
    title = title or comparison_title_map.get(comparison_id, comparison_id)
    suffix = (
        "complement genes only"
        if complement_only
        else "all genes with complement overlay"
    )
    ax.set_title(
        f"{title}\n{suffix} | complement q<{q_threshold}: {n_comp_sig}/{n_comp}",
        fontsize=11,
    )
    ax.set_xlabel(get_xaxis_label(plot_df, comparison_id))
    ax.set_ylabel("-log10 primary BH q-value")
    ax.grid(True, alpha=0.14)
    if show_legend:
        add_complement_legend(ax, include_background=not complement_only)

    if made_fig:
        fig.tight_layout()
        fig.savefig(
            complement_volcano_dir / f"{out_prefix}.png", dpi=300, bbox_inches="tight"
        )
        fig.savefig(complement_volcano_dir / f"{out_prefix}.pdf", bbox_inches="tight")
        plt.show()
    return ax


# %%
# COMPLEMENT VOLCANO TRIPTYCH (GLOBAL + REUSABLE FOR SUBCLASSLEVEL1)
# > Per-panel legends are suppressed; a single shared legend covering all
# complement pathway colors is drawn once, centered below the three panels.
####
def plot_complement_volcano_triptych(
    tables,
    comparison_ids=global_comparison_ids,
    table_id_map=None,
    figure_title=None,
    output_stem="global_complement_aware_volcano_triptych",
    label_top_complement=10,
    label_top_background=2,
    figsize_per_panel=6.2,
):
    table_id_map = table_id_map or {
        comparison_id: comparison_id for comparison_id in comparison_ids
    }
    available = [
        comparison_id
        for comparison_id in comparison_ids
        if table_id_map.get(comparison_id) in tables
    ]
    if not available:
        print(f"No comparison tables available for {output_stem}.")
        return None

    fig, axes = plt.subplots(
        1,
        len(available),
        figsize=(figsize_per_panel * len(available), 5.2),
        sharey=False,
    )
    if len(available) == 1:
        axes = [axes]

    for ax, comparison_id in zip(axes, available):
        table_id = table_id_map[comparison_id]
        plot_complement_aware_volcano(
            tables[table_id],
            comparison_id=comparison_id,
            out_prefix=f"unused_{table_id}",
            title=comparison_title_map.get(comparison_id, comparison_id),
            complement_only=False,
            label_top_complement=label_top_complement,
            label_top_background=label_top_background,
            ax=ax,
            show_legend=False,
        )

    if figure_title:
        fig.suptitle(figure_title, y=1.03, fontsize=14)

    # Reserve space at the bottom for one shared legend instead of a
    # per-panel legend (all three panels share the same pathway/color key).
    fig.tight_layout(rect=[0, 0.1, 1, 1])
    legend_handles = build_complement_legend_handles(include_background=True)
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=len(legend_handles),
        frameon=False,
        fontsize=8,
        bbox_to_anchor=(0.5, 0.0),
    )

    fig.savefig(
        complement_volcano_dir / f"{output_stem}.png", dpi=300, bbox_inches="tight"
    )
    fig.savefig(complement_volcano_dir / f"{output_stem}.pdf", bbox_inches="tight")
    plt.show()
    return fig


def plot_global_complement_volcano_triptych(
    tables, comparison_ids=global_comparison_ids
):
    return plot_complement_volcano_triptych(
        tables,
        comparison_ids=comparison_ids,
        figure_title="Global pseudobulk DE",
        output_stem="global_complement_aware_volcano_triptych",
    )


plot_global_complement_volcano_triptych(all_pseudobulk_de_tables)


# %%
# SUBCLASSLEVEL1 COMPLEMENT TRIPTYCHS
# > One triptych per SubclassLevel1 population, IF its three targeted
# pseudobulk_de_targeted_SubclassLevel1_{population}_{comparison_id}.csv
# files already exist in OUTPUT_DIR. 03-pseudobulk-de.py does not currently
# generate these targeted runs (it only runs the 3 global comparisons), so
# this section will print "No comparison tables available" and no-op until
# that targeted-DE generation step is added.
####
def discover_subclasslevel1_triptych_maps(tables):
    groups = {}
    prefix = "targeted_SubclassLevel1_"
    for table_id in tables:
        if not table_id.startswith(prefix):
            continue
        matched_comparison = None
        for comparison_id in global_comparison_ids:
            suffix = f"_{comparison_id}"
            if table_id.endswith(suffix):
                matched_comparison = comparison_id
                pop_key = table_id[len(prefix) : -len(suffix)]
                break
        if matched_comparison is None:
            continue
        groups.setdefault(pop_key, {})[matched_comparison] = table_id
    return groups


def plot_subclasslevel1_complement_triptychs(
    tables, require_all_three=True, max_triptychs=None
):
    groups = discover_subclasslevel1_triptych_maps(tables)
    manifest = []
    for pop_key, table_id_map in sorted(groups.items()):
        available = [c for c in global_comparison_ids if c in table_id_map]
        if require_all_three and len(available) < len(global_comparison_ids):
            manifest.append(
                {
                    "subclasslevel1": pop_key,
                    "display_label": pretty_subclasslevel1_label(pop_key),
                    "status": "missing_comparisons",
                    "available_comparisons": ";".join(available),
                    "missing_comparisons": ";".join(
                        [c for c in global_comparison_ids if c not in table_id_map]
                    ),
                    "output_stem": "",
                }
            )
            continue
        output_stem = f"subclasslevel1_{pop_key}_complement_aware_volcano_triptych"
        plot_complement_volcano_triptych(
            tables,
            comparison_ids=global_comparison_ids,
            table_id_map=table_id_map,
            figure_title=f"SubclassLevel1: {pretty_subclasslevel1_label(pop_key)}",
            output_stem=output_stem,
            label_top_complement=9,
            label_top_background=1,
            figsize_per_panel=5.8,
        )
        manifest.append(
            {
                "subclasslevel1": pop_key,
                "display_label": pretty_subclasslevel1_label(pop_key),
                "status": "plotted",
                "available_comparisons": ";".join(available),
                "missing_comparisons": "",
                "output_stem": output_stem,
            }
        )
        if (
            max_triptychs is not None
            and sum(m["status"] == "plotted" for m in manifest) >= max_triptychs
        ):
            break
    manifest_df = pd.DataFrame(manifest)
    manifest_df.to_csv(
        complement_volcano_dir / "subclasslevel1_triptych_manifest.csv", index=False
    )
    return manifest_df


subclasslevel1_triptych_manifest = plot_subclasslevel1_complement_triptychs(
    all_pseudobulk_de_tables,
    require_all_three=True,
    max_triptychs=None,
)
if subclasslevel1_triptych_manifest.empty:
    print(
        "No SubclassLevel1 targeted DE tables found yet - run targeted DE generation first if you want these triptychs."
    )


# %%
# INFLAMMASOME-AWARE VOLCANO HELPERS
# > Same structure as the complement-aware plots, highlighting inflammasome
# / pyroptosis / IL-1-IL-18 / NLRP3-mito-stress genes instead
####
def prepare_inflammasome_volcano_df(df):
    plot_df = df.copy()
    plot_df = plot_df[plot_df["passes_expression_filter"].fillna(False)].copy()
    plot_df["gene_symbol"] = plot_df["gene_symbol"].astype(str)
    plot_df["is_inflammasome"] = plot_df["gene_symbol"].isin(inflammasome_symbol_set)
    plot_df["inflammasome_programs"] = (
        plot_df["gene_symbol"].map(inflammasome_gene_to_programs).fillna("")
    )
    plot_df["primary_inflammasome_program"] = (
        plot_df["gene_symbol"]
        .map(inflammasome_gene_to_primary_program)
        .fillna("background")
    )
    plot_df["plot_q"] = plot_df["primary_q_value"].clip(lower=1e-300)
    plot_df["plot_y"] = -np.log10(plot_df["plot_q"])
    plot_df["abs_lfc"] = plot_df["primary_log2fc"].abs()
    return plot_df.replace([np.inf, -np.inf], np.nan)


def build_inflammasome_legend_handles(include_background=True):
    """Handle list shared by both per-axes legends and the triptych's single bottom legend."""
    handles = []
    if include_background:
        handles.append(
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor=background_color,
                markersize=5,
                label="Other genes",
            )
        )
    for program, color in inflammasome_palette.items():
        if program == "inflammasome_other":
            continue
        marker = "s" if program in {"sensors", "nlrp3_mito_stress"} else "o"
        handles.append(
            Line2D(
                [0],
                [0],
                marker=marker,
                color="none",
                markerfacecolor=color,
                markeredgecolor="black",
                markeredgewidth=0.4,
                markersize=6,
                label=program,
            )
        )
    return handles


def add_inflammasome_legend(ax, include_background=True):
    handles = build_inflammasome_legend_handles(include_background)
    ax.legend(handles=handles, frameon=False, fontsize=7, loc="upper right", ncols=1)


def plot_inflammasome_aware_volcano(
    df,
    comparison_id,
    out_prefix,
    title=None,
    q_threshold=0.05,
    lfc_threshold=0.5,
    label_top_inflammasome=14,
    label_top_background=4,
    inflammasome_only=False,
    ax=None,
    show_legend=True,
):
    plot_df = prepare_inflammasome_volcano_df(df)
    if inflammasome_only:
        plot_df = plot_df[plot_df["is_inflammasome"]].copy()
    if plot_df.empty:
        print("No plottable genes for", out_prefix)
        return None

    made_fig = ax is None
    if ax is None:
        fig, ax = plt.subplots(figsize=(7.2, 5.4))
    else:
        fig = ax.figure

    bg = plot_df[~plot_df["is_inflammasome"]]
    infl = plot_df[plot_df["is_inflammasome"]]

    if not inflammasome_only and not bg.empty:
        ax.scatter(
            bg["primary_log2fc"],
            bg["plot_y"],
            s=8,
            c=background_color,
            alpha=0.35,
            linewidths=0,
            rasterized=True,
        )
        bg_sig = bg[bg["is_de_primary_q_0_05"].fillna(False)]
        if not bg_sig.empty:
            ax.scatter(
                bg_sig["primary_log2fc"],
                bg_sig["plot_y"],
                s=8,
                c="#9ca3af",
                alpha=0.55,
                linewidths=0,
                rasterized=True,
            )

    for program, color in inflammasome_palette.items():
        sub = infl[infl["primary_inflammasome_program"] == program]
        if sub.empty:
            continue
        marker = "s" if program in {"sensors", "nlrp3_mito_stress"} else "o"
        ax.scatter(
            sub["primary_log2fc"],
            sub["plot_y"],
            s=44 if not inflammasome_only else 60,
            c=color,
            marker=marker,
            edgecolors="black",
            linewidths=0.45,
            alpha=0.92,
            label=program,
        )

    ax.axhline(-np.log10(q_threshold), color="#111827", lw=0.8, ls="--", alpha=0.55)
    ax.axvline(-lfc_threshold, color="#111827", lw=0.8, ls=":", alpha=0.45)
    ax.axvline(lfc_threshold, color="#111827", lw=0.8, ls=":", alpha=0.45)
    ax.axvline(0, color="#111827", lw=0.8, alpha=0.55)

    left_label, right_label = get_direction_labels(plot_df, comparison_id)
    y_top = plot_df["plot_y"].quantile(0.995)
    x_min, x_max = np.nanpercentile(plot_df["primary_log2fc"], [0.5, 99.5])
    ax.text(
        x_min, y_top, left_label, ha="left", va="bottom", fontsize=8, color="#374151"
    )
    ax.text(
        x_max, y_top, right_label, ha="right", va="bottom", fontsize=8, color="#374151"
    )

    infl_to_label = infl[
        (infl["primary_q_value"] <= 0.10)
        | (infl["abs_lfc"] >= lfc_threshold)
        | (infl["gene_symbol"] == "NLRP3")
    ].sort_values(["primary_q_value", "abs_lfc"], ascending=[True, False])
    label_ranked_genes(ax, infl_to_label, max_labels=label_top_inflammasome)

    if not inflammasome_only and label_top_background:
        bg_to_label = bg.sort_values(
            ["is_de_primary_q_0_05", "primary_q_value", "abs_lfc"],
            ascending=[False, True, False],
        )
        label_ranked_genes(ax, bg_to_label, max_labels=label_top_background)

    n_infl_sig = int((infl["primary_q_value"] <= q_threshold).sum())
    n_infl = int(len(infl))
    title = title or comparison_title_map.get(comparison_id, comparison_id)
    suffix = (
        "inflammasome genes only"
        if inflammasome_only
        else "all genes with inflammasome overlay"
    )
    ax.set_title(
        f"{title}\n{suffix} | inflammasome q<{q_threshold}: {n_infl_sig}/{n_infl}",
        fontsize=11,
    )
    ax.set_xlabel(get_xaxis_label(plot_df, comparison_id))
    ax.set_ylabel("-log10 primary BH q-value")
    ax.grid(True, alpha=0.14)
    if show_legend:
        add_inflammasome_legend(ax, include_background=not inflammasome_only)

    if made_fig:
        fig.tight_layout()
        fig.savefig(
            inflammasome_volcano_dir / f"{out_prefix}.png", dpi=300, bbox_inches="tight"
        )
        fig.savefig(inflammasome_volcano_dir / f"{out_prefix}.pdf", bbox_inches="tight")
        plt.show()
    return ax


# %%
# GLOBAL INFLAMMASOME-AWARE TRIPTYCH
# > Per-panel legends are suppressed; a single shared legend covering all
# inflammasome program colors is drawn once, centered below the three panels.
####
def plot_inflammasome_volcano_triptych(
    tables,
    comparison_ids=global_comparison_ids,
    table_id_map=None,
    figure_title=None,
    output_stem="global_inflammasome_aware_volcano_triptych",
    label_top_inflammasome=10,
    label_top_background=2,
    figsize_per_panel=6.2,
):
    table_id_map = table_id_map or {
        comparison_id: comparison_id for comparison_id in comparison_ids
    }
    available = [
        comparison_id
        for comparison_id in comparison_ids
        if table_id_map.get(comparison_id) in tables
    ]
    if not available:
        print(f"No comparison tables available for {output_stem}.")
        return None

    fig, axes = plt.subplots(
        1,
        len(available),
        figsize=(figsize_per_panel * len(available), 5.2),
        sharey=False,
    )
    if len(available) == 1:
        axes = [axes]

    for ax, comparison_id in zip(axes, available):
        table_id = table_id_map[comparison_id]
        plot_inflammasome_aware_volcano(
            tables[table_id],
            comparison_id=comparison_id,
            out_prefix=f"unused_{table_id}",
            title=comparison_title_map.get(comparison_id, comparison_id),
            inflammasome_only=False,
            label_top_inflammasome=label_top_inflammasome,
            label_top_background=label_top_background,
            ax=ax,
            show_legend=False,
        )

    if figure_title:
        fig.suptitle(figure_title, y=1.03, fontsize=14)

    fig.tight_layout(rect=[0, 0.1, 1, 1])
    legend_handles = build_inflammasome_legend_handles(include_background=True)
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=len(legend_handles),
        frameon=False,
        fontsize=8,
        bbox_to_anchor=(0.5, 0.0),
    )

    fig.savefig(
        inflammasome_volcano_dir / f"{output_stem}.png", dpi=300, bbox_inches="tight"
    )
    fig.savefig(inflammasome_volcano_dir / f"{output_stem}.pdf", bbox_inches="tight")
    plt.show()
    return fig


plot_inflammasome_volcano_triptych(
    all_pseudobulk_de_tables,
    comparison_ids=global_comparison_ids,
    figure_title="Global pseudobulk DE - inflammasome overlay",
    output_stem="global_inflammasome_aware_volcano_triptych",
)


# %%
# SUBCLASSLEVEL1 INFLAMMASOME TRIPTYCHS
# > Same caveat as the complement version: requires targeted SubclassLevel1
# DE CSVs to already exist; no-ops gracefully otherwise.
####
def plot_subclasslevel1_inflammasome_triptychs(
    tables, require_all_three=True, max_triptychs=None
):
    groups = discover_subclasslevel1_triptych_maps(tables)
    manifest = []
    for pop_key, table_id_map in sorted(groups.items()):
        available = [c for c in global_comparison_ids if c in table_id_map]
        if require_all_three and len(available) < len(global_comparison_ids):
            manifest.append(
                {
                    "subclasslevel1": pop_key,
                    "display_label": pretty_subclasslevel1_label(pop_key),
                    "status": "missing_comparisons",
                    "available_comparisons": ";".join(available),
                    "missing_comparisons": ";".join(
                        [c for c in global_comparison_ids if c not in table_id_map]
                    ),
                    "output_stem": "",
                }
            )
            continue
        output_stem = f"subclasslevel1_{pop_key}_inflammasome_aware_volcano_triptych"
        plot_inflammasome_volcano_triptych(
            tables,
            comparison_ids=global_comparison_ids,
            table_id_map=table_id_map,
            figure_title=f"SubclassLevel1: {pretty_subclasslevel1_label(pop_key)}",
            output_stem=output_stem,
            label_top_inflammasome=9,
            label_top_background=1,
            figsize_per_panel=5.8,
        )
        manifest.append(
            {
                "subclasslevel1": pop_key,
                "display_label": pretty_subclasslevel1_label(pop_key),
                "status": "plotted",
                "available_comparisons": ";".join(available),
                "missing_comparisons": "",
                "output_stem": output_stem,
            }
        )
        if (
            max_triptychs is not None
            and sum(m["status"] == "plotted" for m in manifest) >= max_triptychs
        ):
            break
    manifest_df = pd.DataFrame(manifest)
    manifest_df.to_csv(
        inflammasome_volcano_dir / "subclasslevel1_inflammasome_triptych_manifest.csv",
        index=False,
    )
    return manifest_df


subclasslevel1_inflammasome_triptych_manifest = (
    plot_subclasslevel1_inflammasome_triptychs(
        all_pseudobulk_de_tables,
        require_all_three=True,
        max_triptychs=None,
    )
)

print("Triptych plotting pipeline completed.")
