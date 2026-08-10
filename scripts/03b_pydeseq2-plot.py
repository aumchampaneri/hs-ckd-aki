# %% Plotting pyDeSeq2 results
#
#
# %% PATH SETUP
from pathlib import Path

SCRIPT_DIR = Path.cwd()
PROJECT_DIR = SCRIPT_DIR.parent

OUTPUT_DIR = PROJECT_DIR / "outputs/" / "pseudobulk_de"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PLOT_DIR = PROJECT_DIR / "outputs/" / "plots/" / "pseudobulk_de"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

DATA_DIR = PROJECT_DIR / "data/"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# %% IMPORTS
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import pandas as pd
from adjustText import adjust_text
from matplotlib.lines import Line2D

# %%
# GLOBAL FONT SETTINGS FOR PUBLICATION-QUALITY, LEGIBLE PLOTS
# > PLOS figure requirements (journals.plos.org/plosone/s/figures, applies
# across the PLOS family): only Arial, Times, or Symbol; 8-12 pt AT FINAL
# PRINT DIMENSIONS. Since FIGURE_SIZES below targets PLOS's actual print
# widths (column width or full page width), the point sizes here ARE the
# final print sizes - no extra scaling needed.
# Arial is pinned (no fallback list) to match the font used downstream in
# Inkscape - one less thing to reconcile after export. matplotlib normally
# fails SILENTLY and substitutes DejaVu Sans if the requested font isn't in
# its cache, so this is checked explicitly below rather than left to chance.
####
FONT_SIZES = {
    "annotation": 4,  # gene-symbol point labels (leader-line annotations)
    "direction_text": 8,  # "Higher in X" corner labels on volcano plots
    "legend": 8,
    "axis_label": 10,
    "tick_label": 9,
    "panel_title": 11,
    "figure_title": 12,  # PLOS max; consider dropping suptitles entirely -
    # PLOS asks that figure titles/captions live in the manuscript, not the
    # image file, so "Global pseudobulk DE" as a suptitle is borderline.
}

FIGURE_SIZES = {
    "single_volcano": (5.2, 4.6),  # PLOS column width (13.2cm)
    "triptych_panel_width": 2.45,  # 3 panels -> 7.35in, under 7.5in page max
    "triptych_panel_height": 4.2,
    "subclasslevel1_panel_width": 2.45,
    "triptych_legend_margin": 0.15,
    "concordance": (5.2, 4.6),  # PLOS column width
    "pathway_summary_width": 5.2,  # PLOS column width
    "pathway_summary_row_height": 0.35,
    "pathway_summary_base_height": 1.6,
    "pathway_summary_min_height": 4.0,
}

MARKER_SIZES = {
    "background": 8,  # unlabeled/background gene scatter points (rasterized)
    "background_significant": 8,  # significant-but-not-in-pathway background points
    "concordance_background": 7,
    "complement_pathway_overlay": 42,  # pathway markers, drawn over the full background
    "complement_pathway_only": 58,  # pathway markers, complement_only mode (fewer points on screen)
    "inflammasome_pathway_overlay": 44,
    "inflammasome_pathway_only": 60,
    "concordance_pathway": 48,
}
PATHWAY_MARKER_SCALE = 0.25  # multiplies every "*_pathway*" entry above - tune
# this single number to size the complement/inflammasome pathway markers up
# or down without touching each scatter() call individually.

# EXPORT_DPI governs the resolution of every rasterized element embedded in
# the saved PDF/PNG - in these plots that's only the gray background gene
# cloud (rasterized=True); the pathway-colored markers, text, and lines stay
# vector and are unaffected by this value, so bumping it up only sharpens
# the background scatter when zoomed in downstream (e.g. in Inkscape).
EXPORT_DPI = 300
PLOT_FONT = "Arial"

_available_font_names = {f.name for f in fm.fontManager.ttflist}
if PLOT_FONT not in _available_font_names:
    raise RuntimeError(
        f"'{PLOT_FONT}' was not found in matplotlib's font cache, so plots "
        "would silently render in a substitute font instead (usually DejaVu "
        "Sans) - which would then mismatch the Arial used in Inkscape. "
        "Confirm Arial is installed system-wide, then clear matplotlib's "
        "cache (delete the 'fontlist-*.json' file inside the directory "
        "printed by `import matplotlib; print(matplotlib.get_cachedir())`) "
        "and re-run."
    )

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": [PLOT_FONT],
        "font.size": FONT_SIZES["tick_label"],
        "axes.titlesize": FONT_SIZES["panel_title"],
        "axes.labelsize": FONT_SIZES["axis_label"],
        "xtick.labelsize": FONT_SIZES["tick_label"],
        "ytick.labelsize": FONT_SIZES["tick_label"],
        "legend.fontsize": FONT_SIZES["legend"],
        "figure.titlesize": FONT_SIZES["figure_title"],
    }
)

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

concordance_dir = PLOT_DIR / "concordance"
concordance_dir.mkdir(parents=True, exist_ok=True)

pathway_level_dir = PLOT_DIR / "pathway_level"
pathway_level_dir.mkdir(parents=True, exist_ok=True)

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
    """Places gene-symbol text objects at the top-ranked rows of df (already
    sorted). Positions are NOT finalized here - the fixed dx/dy offset
    heuristic this used to use falls apart once many labels are packed into
    a small PLOS-width panel. Instead, call finalize_gene_labels() once per
    axis after ALL label_ranked_genes() calls for that axis, so the
    repulsion algorithm sees every label at once (across complement AND
    background label sets) and can spread them apart without overlap.
    Returns the list of Text objects for the caller to accumulate.
    """
    label_df = df.dropna(subset=[x_col, y_col]).copy().head(max_labels)
    if label_df.empty:
        return []
    return [
        ax.text(
            row[x_col],
            row[y_col],
            str(row[label_col]),
            fontsize=FONT_SIZES["annotation"],
            ha="center",
            va="center",
        )
        for _, row in label_df.iterrows()
    ]


def finalize_gene_labels(ax, texts, avoid_x=None, avoid_y=None):
    """Repels overlapping gene-symbol labels apart (via adjustText) and
    draws a thin leader line back to each label's original point.
    avoid_x/avoid_y (optional, e.g. the full scatter cloud for the panel)
    let labels also dodge unlabeled points, not just each other, which
    matters once panels are narrow enough that labels land on top of
    background dots. NaN/inf entries (untested genes, clipped q-values,
    etc.) are dropped first - adjustText builds a KDTree internally, which
    raises on any non-finite coordinate.
    """
    texts = [t for t in texts if t is not None]
    if not texts:
        return
    avoid_kwargs = {}
    if avoid_x is not None and avoid_y is not None:
        avoid_x = np.asarray(avoid_x, dtype=float)
        avoid_y = np.asarray(avoid_y, dtype=float)
        finite_mask = np.isfinite(avoid_x) & np.isfinite(avoid_y)
        if finite_mask.any():
            avoid_kwargs["x"] = avoid_x[finite_mask]
            avoid_kwargs["y"] = avoid_y[finite_mask]
    adjust_text(
        texts,
        ax=ax,
        arrowprops={"arrowstyle": "-", "lw": 0.35, "color": "#6b7280", "alpha": 0.7},
        expand=(1.35, 1.6),
        force_text=(0.4, 0.6),
        **avoid_kwargs,
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


all_pseudobulk_de_tables = load_pseudobulk_de_tables(OUTPUT_DIR)
print("Loaded gene-level DE tables:", sorted(all_pseudobulk_de_tables))


def report_expression_filter_impact(tables):
    """Every volcano plot in this script only shows genes where
    passes_expression_filter == True (see prepare_complement_volcano_df /
    prepare_inflammasome_volcano_df) AND have a non-null q-value (points
    with a NaN x/y silently don't render). There are two independent
    reasons a gene can be missing from a plot, from 03a_pseudobulk-de.py:
      1. passes_expression_filter == False: excluded from PyDESeq2 entirely
         by the min_total_count/min_detected_donors pre-filter.
      2. passes_expression_filter == True but primary_q_value is NaN:
         DESeq2 itself excluded it via independent_filter (low-mean-count
         FDR optimization) or cooks_filter (outlier-donor detection) -
         both legitimate, but distinct from #1.
    This reports both, per table, so what actually reaches each figure is
    visible rather than assumed.
    """
    rows = []
    for table_id, df in sorted(tables.items()):
        n_total = len(df)
        passes = df["passes_expression_filter"].fillna(False)
        n_pass = int(passes.sum())
        n_pass_with_q = int((passes & df["primary_q_value"].notna()).sum())
        n_pass_nan_q = n_pass - n_pass_with_q
        rows.append(
            {
                "table_id": table_id,
                "n_total_genes": n_total,
                "n_passing_expression_filter": n_pass,
                "n_excluded_pre_filter": n_total - n_pass,
                "n_passing_with_valid_q": n_pass_with_q,
                "n_passing_but_nan_q_deseq2_internal": n_pass_nan_q,
                "pct_actually_plottable": (
                    round(100 * n_pass_with_q / n_total, 1) if n_total else np.nan
                ),
            }
        )
    report_df = pd.DataFrame(rows)
    print(report_df.to_string(index=False))
    return report_df


expression_filter_report = report_expression_filter_impact(all_pseudobulk_de_tables)


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
    ax.legend(
        handles=handles,
        frameon=False,
        fontsize=FONT_SIZES["legend"],
        loc="upper right",
        ncols=1,
    )


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
        fig, ax = plt.subplots(figsize=FIGURE_SIZES["single_volcano"])
    else:
        fig = ax.figure

    bg = plot_df[~plot_df["is_complement"]]
    comp = plot_df[plot_df["is_complement"]]

    if not complement_only and not bg.empty:
        ax.scatter(
            bg["primary_log2fc"],
            bg["plot_y"],
            s=MARKER_SIZES["background"],
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
                s=MARKER_SIZES["background_significant"],
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
            s=(
                MARKER_SIZES["complement_pathway_overlay"]
                if not complement_only
                else MARKER_SIZES["complement_pathway_only"]
            )
            * PATHWAY_MARKER_SCALE,
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
        x_min,
        y_top,
        left_label,
        ha="left",
        va="bottom",
        fontsize=FONT_SIZES["direction_text"],
        color="#374151",
    )
    ax.text(
        x_max,
        y_top,
        right_label,
        ha="right",
        va="bottom",
        fontsize=FONT_SIZES["direction_text"],
        color="#374151",
    )

    comp_to_label = comp[
        (comp["primary_q_value"] <= 0.10) | (comp["abs_lfc"] >= lfc_threshold)
    ].sort_values(["primary_q_value", "abs_lfc"], ascending=[True, False])
    gene_labels = label_ranked_genes(ax, comp_to_label, max_labels=label_top_complement)

    if not complement_only and label_top_background:
        bg_to_label = bg.sort_values(
            ["is_de_primary_q_0_05", "primary_q_value", "abs_lfc"],
            ascending=[False, True, False],
        )
        gene_labels += label_ranked_genes(
            ax, bg_to_label, max_labels=label_top_background
        )

    finalize_gene_labels(
        ax, gene_labels, avoid_x=plot_df["primary_log2fc"], avoid_y=plot_df["plot_y"]
    )

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
        fontsize=FONT_SIZES["panel_title"],
    )
    ax.set_xlabel(get_xaxis_label(plot_df, comparison_id))
    ax.set_ylabel("-log10 primary BH q-value")
    ax.grid(True, alpha=0.14)
    if show_legend:
        add_complement_legend(ax, include_background=not complement_only)

    if made_fig:
        fig.tight_layout()
        fig.savefig(
            complement_volcano_dir / f"{out_prefix}.png", dpi=EXPORT_DPI, bbox_inches="tight"
        )
        fig.savefig(
            complement_volcano_dir / f"{out_prefix}.pdf", dpi=EXPORT_DPI, bbox_inches="tight"
        )
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
    figsize_per_panel=FIGURE_SIZES["triptych_panel_width"],
    pathway_only=False,
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
        figsize=(figsize_per_panel * len(available), FIGURE_SIZES["triptych_panel_height"]),
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
            complement_only=pathway_only,
            label_top_complement=label_top_complement,
            label_top_background=label_top_background,
            ax=ax,
            show_legend=False,
        )

    if figure_title:
        fig.suptitle(figure_title, y=1.03, fontsize=FONT_SIZES["figure_title"])

    # Reserve space at the bottom for one shared legend instead of a
    # per-panel legend (all three panels share the same pathway/color key).
    fig.tight_layout(rect=[0, FIGURE_SIZES["triptych_legend_margin"], 1, 1])
    legend_handles = build_complement_legend_handles(include_background=not pathway_only)
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=len(legend_handles),
        frameon=False,
        fontsize=FONT_SIZES["legend"],
        bbox_to_anchor=(0.5, 0.0),
    )

    fig.savefig(
        complement_volcano_dir / f"{output_stem}.png", dpi=EXPORT_DPI, bbox_inches="tight"
    )
    fig.savefig(
        complement_volcano_dir / f"{output_stem}.pdf", dpi=EXPORT_DPI, bbox_inches="tight"
    )
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


def plot_global_complement_pathway_only_volcano_triptych(
    tables, comparison_ids=global_comparison_ids
):
    """Same triptych, but with every non-complement gene dropped entirely
    (not just de-emphasized) - so the panel shows only how the complement
    genes move relative to EACH OTHER across comparisons, without the
    background cloud competing for attention. More genes are labeled by
    default since there's no clutter left to avoid."""
    return plot_complement_volcano_triptych(
        tables,
        comparison_ids=comparison_ids,
        figure_title="Global pseudobulk DE - complement genes only",
        output_stem="global_complement_only_volcano_triptych",
        label_top_complement=30,
        label_top_background=0,
        pathway_only=True,
    )


plot_global_complement_volcano_triptych(all_pseudobulk_de_tables)
plot_global_complement_pathway_only_volcano_triptych(all_pseudobulk_de_tables)


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
    tables, require_all_three=True, max_triptychs=None, pathway_only=False
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
        stem_suffix = "_complement_only" if pathway_only else "_complement_aware"
        output_stem = f"subclasslevel1_{pop_key}{stem_suffix}_volcano_triptych"
        plot_complement_volcano_triptych(
            tables,
            comparison_ids=global_comparison_ids,
            table_id_map=table_id_map,
            figure_title=f"SubclassLevel1: {pretty_subclasslevel1_label(pop_key)}",
            output_stem=output_stem,
            label_top_complement=9 if not pathway_only else 20,
            label_top_background=1 if not pathway_only else 0,
            figsize_per_panel=FIGURE_SIZES["subclasslevel1_panel_width"],
            pathway_only=pathway_only,
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
    ax.legend(
        handles=handles,
        frameon=False,
        fontsize=FONT_SIZES["legend"],
        loc="upper right",
        ncols=1,
    )


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
        fig, ax = plt.subplots(figsize=FIGURE_SIZES["single_volcano"])
    else:
        fig = ax.figure

    bg = plot_df[~plot_df["is_inflammasome"]]
    infl = plot_df[plot_df["is_inflammasome"]]

    if not inflammasome_only and not bg.empty:
        ax.scatter(
            bg["primary_log2fc"],
            bg["plot_y"],
            s=MARKER_SIZES["background"],
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
                s=MARKER_SIZES["background_significant"],
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
            s=(
                MARKER_SIZES["inflammasome_pathway_overlay"]
                if not inflammasome_only
                else MARKER_SIZES["inflammasome_pathway_only"]
            )
            * PATHWAY_MARKER_SCALE,
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
        x_min,
        y_top,
        left_label,
        ha="left",
        va="bottom",
        fontsize=FONT_SIZES["direction_text"],
        color="#374151",
    )
    ax.text(
        x_max,
        y_top,
        right_label,
        ha="right",
        va="bottom",
        fontsize=FONT_SIZES["direction_text"],
        color="#374151",
    )

    infl_to_label = infl[
        (infl["primary_q_value"] <= 0.10)
        | (infl["abs_lfc"] >= lfc_threshold)
        | (infl["gene_symbol"] == "NLRP3")
    ].sort_values(["primary_q_value", "abs_lfc"], ascending=[True, False])
    gene_labels = label_ranked_genes(
        ax, infl_to_label, max_labels=label_top_inflammasome
    )

    if not inflammasome_only and label_top_background:
        bg_to_label = bg.sort_values(
            ["is_de_primary_q_0_05", "primary_q_value", "abs_lfc"],
            ascending=[False, True, False],
        )
        gene_labels += label_ranked_genes(
            ax, bg_to_label, max_labels=label_top_background
        )

    finalize_gene_labels(
        ax, gene_labels, avoid_x=plot_df["primary_log2fc"], avoid_y=plot_df["plot_y"]
    )

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
        fontsize=FONT_SIZES["panel_title"],
    )
    ax.set_xlabel(get_xaxis_label(plot_df, comparison_id))
    ax.set_ylabel("-log10 primary BH q-value")
    ax.grid(True, alpha=0.14)
    if show_legend:
        add_inflammasome_legend(ax, include_background=not inflammasome_only)

    if made_fig:
        fig.tight_layout()
        fig.savefig(
            inflammasome_volcano_dir / f"{out_prefix}.png", dpi=EXPORT_DPI, bbox_inches="tight"
        )
        fig.savefig(
            inflammasome_volcano_dir / f"{out_prefix}.pdf", dpi=EXPORT_DPI, bbox_inches="tight"
        )
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
    figsize_per_panel=FIGURE_SIZES["triptych_panel_width"],
    pathway_only=False,
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
        figsize=(figsize_per_panel * len(available), FIGURE_SIZES["triptych_panel_height"]),
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
            inflammasome_only=pathway_only,
            label_top_inflammasome=label_top_inflammasome,
            label_top_background=label_top_background,
            ax=ax,
            show_legend=False,
        )

    if figure_title:
        fig.suptitle(figure_title, y=1.03, fontsize=FONT_SIZES["figure_title"])

    fig.tight_layout(rect=[0, FIGURE_SIZES["triptych_legend_margin"], 1, 1])
    legend_handles = build_inflammasome_legend_handles(include_background=not pathway_only)
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=len(legend_handles),
        frameon=False,
        fontsize=FONT_SIZES["legend"],
        bbox_to_anchor=(0.5, 0.0),
    )

    fig.savefig(
        inflammasome_volcano_dir / f"{output_stem}.png", dpi=EXPORT_DPI, bbox_inches="tight"
    )
    fig.savefig(
        inflammasome_volcano_dir / f"{output_stem}.pdf", dpi=EXPORT_DPI, bbox_inches="tight"
    )
    plt.show()
    return fig


plot_inflammasome_volcano_triptych(
    all_pseudobulk_de_tables,
    comparison_ids=global_comparison_ids,
    figure_title="Global pseudobulk DE - inflammasome overlay",
    output_stem="global_inflammasome_aware_volcano_triptych",
)

plot_inflammasome_volcano_triptych(
    all_pseudobulk_de_tables,
    comparison_ids=global_comparison_ids,
    figure_title="Global pseudobulk DE - inflammasome genes only",
    output_stem="global_inflammasome_only_volcano_triptych",
    label_top_inflammasome=30,
    label_top_background=0,
    pathway_only=True,
)


# %%
# SUBCLASSLEVEL1 INFLAMMASOME TRIPTYCHS
# > Same caveat as the complement version: requires targeted SubclassLevel1
# DE CSVs to already exist; no-ops gracefully otherwise.
####
def plot_subclasslevel1_inflammasome_triptychs(
    tables, require_all_three=True, max_triptychs=None, pathway_only=False
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
        stem_suffix = "_inflammasome_only" if pathway_only else "_inflammasome_aware"
        output_stem = f"subclasslevel1_{pop_key}{stem_suffix}_volcano_triptych"
        plot_inflammasome_volcano_triptych(
            tables,
            comparison_ids=global_comparison_ids,
            table_id_map=table_id_map,
            figure_title=f"SubclassLevel1: {pretty_subclasslevel1_label(pop_key)}",
            output_stem=output_stem,
            label_top_inflammasome=9 if not pathway_only else 20,
            label_top_background=1 if not pathway_only else 0,
            figsize_per_panel=FIGURE_SIZES["subclasslevel1_panel_width"],
            pathway_only=pathway_only,
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


# %%
# CROSS-CONTRAST CONCORDANCE SCATTER
# > Reads concordance_{x}_vs_{y}.csv written by 03-pseudobulk-de.py (one row
# per gene tested in both contrasts) and plots log2FC(x) vs log2FC(y),
# colored by complement/inflammasome pathway membership. Answers a question
# no single volcano plot can: is gene dysregulation shared across disease
# states relative to the same baseline, or state/contrast-specific? Points
# in the upper-right/lower-left quadrants are concordant; off-diagonal
# points are discordant (state-specific).
####
def load_concordance_table(output_dir, x_id, y_id):
    path = output_dir / f"concordance_{x_id}_vs_{y_id}.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


def annotate_concordance_pathways(df):
    df = df.copy()
    df["is_complement"] = df["gene_symbol"].isin(complement_symbol_set)
    df["is_inflammasome"] = df["gene_symbol"].isin(inflammasome_symbol_set)
    df["primary_complement_pathway"] = df["gene_symbol"].map(
        complement_gene_to_primary_pathway
    )
    df["primary_inflammasome_program"] = df["gene_symbol"].map(
        inflammasome_gene_to_primary_program
    )
    return df


def plot_concordance_scatter(
    df, x_id, y_id, gene_set="complement", q_threshold=0.05, label_top_n=16
):
    """gene_set: 'complement' or 'inflammasome' - which panel/palette to overlay."""
    df = annotate_concordance_pathways(df)
    is_col = "is_complement" if gene_set == "complement" else "is_inflammasome"
    pathway_col = (
        "primary_complement_pathway"
        if gene_set == "complement"
        else "primary_inflammasome_program"
    )
    palette = (
        complement_pathway_palette if gene_set == "complement" else inflammasome_palette
    )
    other_key = "complement_other" if gene_set == "complement" else "inflammasome_other"

    bg = df[~df[is_col]]
    panel = df[df[is_col]]
    if panel.empty:
        print(f"No {gene_set} genes with data for concordance plot {x_id} vs {y_id}.")
        return None

    fig, ax = plt.subplots(figsize=FIGURE_SIZES["concordance"])
    if not bg.empty:
        ax.scatter(
            bg["log2fc_x"],
            bg["log2fc_y"],
            s=MARKER_SIZES["concordance_background"],
            c=background_color,
            alpha=0.3,
            linewidths=0,
            rasterized=True,
        )

    for pathway, color in palette.items():
        if pathway == other_key:
            continue
        sub = panel[panel[pathway_col] == pathway]
        if sub.empty:
            continue
        marker = (
            "s" if pathway in {"regulator", "sensors", "nlrp3_mito_stress"} else "o"
        )
        edge_widths = np.where(sub["significance_class"] == "both_sig", 0.9, 0.35)
        ax.scatter(
            sub["log2fc_x"],
            sub["log2fc_y"],
            s=MARKER_SIZES["concordance_pathway"] * PATHWAY_MARKER_SCALE,
            c=color,
            marker=marker,
            edgecolors="black",
            linewidths=edge_widths,
            alpha=0.9,
            label=pathway,
        )

    axis_lim = (
        np.nanmax(
            np.abs(np.concatenate([df["log2fc_x"].dropna(), df["log2fc_y"].dropna()]))
        )
        * 1.1
    )
    axis_lim = max(axis_lim, 1)
    ax.set_xlim(-axis_lim, axis_lim)
    ax.set_ylim(-axis_lim, axis_lim)
    ax.axhline(0, color="#111827", lw=0.8, alpha=0.5)
    ax.axvline(0, color="#111827", lw=0.8, alpha=0.5)
    ax.plot(
        [-axis_lim, axis_lim],
        [-axis_lim, axis_lim],
        color="#9ca3af",
        lw=0.8,
        ls="--",
        alpha=0.6,
    )

    label_pool = panel[panel["significance_class"] == "both_sig"].copy()
    if len(label_pool) < label_top_n:
        label_pool = panel.sort_values(
            ["significance_class", "q_value_x", "q_value_y"],
            key=lambda s: (
                s
                if s.name != "significance_class"
                else s.map(
                    {"both_sig": 0, "x_only_sig": 1, "y_only_sig": 1, "neither_sig": 2}
                )
            ),
        )
    gene_labels = [
        ax.text(
            row["log2fc_x"],
            row["log2fc_y"],
            str(row["gene_symbol"]),
            fontsize=FONT_SIZES["annotation"],
            ha="center",
            va="center",
            color="#374151",
        )
        for _, row in label_pool.head(label_top_n).iterrows()
    ]
    finalize_gene_labels(
        ax, gene_labels, avoid_x=df["log2fc_x"], avoid_y=df["log2fc_y"]
    )

    x_title = comparison_title_map.get(x_id, x_id)
    y_title = comparison_title_map.get(y_id, y_id)
    n_concordant = int((panel["sign_concordant"].astype("boolean").fillna(False)).sum())
    ax.set_title(
        f"{gene_set.capitalize()} gene concordance: {x_title} vs {y_title}\n"
        f"{n_concordant}/{len(panel)} {gene_set} genes concordant in direction",
        fontsize=FONT_SIZES["panel_title"],
    )
    ax.set_xlabel(f"log2FC, {x_title}")
    ax.set_ylabel(f"log2FC, {y_title}")
    ax.grid(True, alpha=0.15)
    ax.legend(frameon=False, fontsize=FONT_SIZES["legend"], loc="upper left")

    fig.tight_layout()
    out_stem = f"concordance_{gene_set}_{x_id}_vs_{y_id}"
    fig.savefig(concordance_dir / f"{out_stem}.png", dpi=EXPORT_DPI, bbox_inches="tight")
    fig.savefig(concordance_dir / f"{out_stem}.pdf", dpi=EXPORT_DPI, bbox_inches="tight")
    plt.show()
    return fig


concordance_pairs = [("ckd_vs_normal", "aki_vs_normal")]
for x_id, y_id in concordance_pairs:
    concordance_table = load_concordance_table(OUTPUT_DIR, x_id, y_id)
    if concordance_table is None:
        print(
            f"No concordance table found for {x_id} vs {y_id} - run 03-pseudobulk-de.py first."
        )
        continue
    plot_concordance_scatter(concordance_table, x_id, y_id, gene_set="complement")
    plot_concordance_scatter(concordance_table, x_id, y_id, gene_set="inflammasome")


# %%
# PATHWAY-LEVEL GROUP-DIFFERENCE SUMMARY PLOT
# > Reads pathway_level_group_tests.csv (donor-level module score,
# Mann-Whitney U per program per comparison) written by 03-pseudobulk-de.py.
# One dot per program x comparison: x-axis = effect size (median score
# difference), color = signed -log10(q), size = -log10(q). This is the
# formal pathway-level counterpart to eyeballing how many individual genes
# clear q<0.05 on a volcano plot.
####
def load_pathway_level_tests(output_dir):
    path = output_dir / "pathway_level_group_tests.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


def plot_pathway_level_summary(
    df, gene_set, comparison_ids=global_comparison_ids, q_threshold=0.05
):
    sub = df[(df["gene_set"] == gene_set) & (df["status"] == "tested")].copy()
    if sub.empty:
        print(f"No tested {gene_set} pathway-level rows to plot.")
        return None

    sub["comparison_label"] = sub["comparison_id"].map(
        lambda c: comparison_title_map.get(c, c)
    )
    sub = sub[sub["comparison_id"].isin(comparison_ids)]
    sub["neg_log10_q"] = -np.log10(sub["q_value"].clip(lower=1e-300))
    sub["is_significant"] = sub["q_value"] <= q_threshold

    program_order = (
        sub.groupby("program")["neg_log10_q"]
        .max()
        .sort_values(ascending=True)
        .index.tolist()
    )
    comparison_order = [
        comparison_title_map.get(c, c)
        for c in comparison_ids
        if c in sub["comparison_id"].unique()
    ]

    y_lookup = {p: i for i, p in enumerate(program_order)}
    x_lookup = {c: i for i, c in enumerate(comparison_order)}

    fig, ax = plt.subplots(
        figsize=(
            FIGURE_SIZES["pathway_summary_width"],
            max(
                FIGURE_SIZES["pathway_summary_min_height"],
                FIGURE_SIZES["pathway_summary_row_height"] * len(program_order)
                + FIGURE_SIZES["pathway_summary_base_height"],
            ),
        )
    )
    max_abs_diff = max(sub["median_diff_group1_minus_group2"].abs().max(), 0.1)
    sca = ax.scatter(
        sub["comparison_label"].map(x_lookup),
        sub["program"].map(y_lookup),
        s=40 + 220 * (sub["neg_log10_q"].clip(upper=10) / 10),
        c=sub["median_diff_group1_minus_group2"],
        cmap="coolwarm",
        vmin=-max_abs_diff,
        vmax=max_abs_diff,
        edgecolors=np.where(sub["is_significant"], "black", "#9ca3af"),
        linewidths=np.where(sub["is_significant"], 1.0, 0.3),
    )
    ax.set_xticks(range(len(comparison_order)))
    ax.set_xticklabels(comparison_order, rotation=20, ha="right")
    ax.set_yticks(range(len(program_order)))
    ax.set_yticklabels(program_order)
    ax.set_title(
        f"{gene_set.capitalize()} pathway-level group differences\n"
        "donor module score (Mann-Whitney U), dot size/outline = significance",
        fontsize=FONT_SIZES["panel_title"],
    )
    ax.grid(True, axis="x", alpha=0.15)
    cbar = fig.colorbar(sca, ax=ax, shrink=0.75)
    cbar.set_label("median score diff (group1 - group2)")

    fig.tight_layout()
    out_stem = f"pathway_level_summary_{gene_set}"
    fig.savefig(pathway_level_dir / f"{out_stem}.png", dpi=EXPORT_DPI, bbox_inches="tight")
    fig.savefig(pathway_level_dir / f"{out_stem}.pdf", dpi=EXPORT_DPI, bbox_inches="tight")
    plt.show()
    return fig


pathway_level_tests_table = load_pathway_level_tests(OUTPUT_DIR)
if pathway_level_tests_table is None:
    print("No pathway_level_group_tests.csv found - run 03-pseudobulk-de.py first.")
else:
    plot_pathway_level_summary(pathway_level_tests_table, gene_set="complement")
    plot_pathway_level_summary(pathway_level_tests_table, gene_set="inflammasome")

print("Triptych plotting pipeline completed.")
