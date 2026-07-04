# %% eGFR Continuous Correlation (CKD donors)
#
# Donor-level Spearman correlation between complement/inflammasome pathway
# module scores and baseline eGFR severity, restricted to CKD donors.
# Complements the categorical CKD-vs-normal contrast in 03-pseudobulk-de.py
# with a within-CKD "dose-response" question: does pathway activation scale
# with disease severity, not just presence/absence of disease.
#
# %% PATH SETUP
from pathlib import Path

SCRIPT_DIR = Path.cwd()
PROJECT_DIR = SCRIPT_DIR.parent

OUTPUT_DIR = PROJECT_DIR / "outputs/" / "eGFR_correlation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DATA_DIR = PROJECT_DIR / "data/"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# %% IMPORTS
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
from adpbulk import ADPBulk
from scipy import stats

plt.ioff()

# %%
# CONFIGURATION
# > Keys must match 03-pseudobulk-de.py exactly (this script is
# self-contained but mirrors that script's conventions).
####
adata_path = DATA_DIR / "adata_scvi.h5ad"
disease_key = "disease"
donor_key = "donor_id"
ckd_label = "chronic kidney disease"
egfr_col = "Baseline eGFR (ml/min/1.73m2) (Binned)"

min_donors_for_correlation = 5  # minimum CKD donors with a usable eGFR bin

# %% LOAD DATA
adata = sc.read_h5ad(adata_path)
print(f"Loaded Atlas: {adata.n_obs} cells, {adata.n_vars} genes.")

# %%
# EGFR BIN LABEL DIAGNOSTIC
# > Confirms (a) the bin labels actually present, (b) that eGFR is truly a
# per-donor constant (not varying across a donor's cells), and (c) how many
# CKD donors have a usable (non-null) eGFR bin. Run this and check the
# printed output before trusting the parser below - if the assertion in the
# next cell fails, the bin label format needs a tweak to parse_bin_bounds().
####
if egfr_col not in adata.obs.columns:
    raise KeyError(
        f"'{egfr_col}' not found in adata.obs. Available columns: {list(adata.obs.columns)}"
    )

ckd_mask = adata.obs[disease_key].astype(str) == ckd_label

print("eGFR bin value counts (all cells, all disease states):")
print(adata.obs[egfr_col].value_counts(dropna=False))

per_donor_nunique = (
    adata.obs.loc[ckd_mask].groupby(donor_key, observed=True)[egfr_col].nunique()
)
n_donors_multi_bin = int((per_donor_nunique > 1).sum())
print(
    f"\nCKD donors with >1 distinct eGFR bin value (should be 0): {n_donors_multi_bin} of {len(per_donor_nunique)}"
)

ckd_donor_egfr = (
    adata.obs.loc[ckd_mask, [donor_key, egfr_col]]
    .drop_duplicates(subset=[donor_key])
    .reset_index(drop=True)
)
n_ckd_donors_total = ckd_donor_egfr[donor_key].nunique()
n_ckd_donors_with_egfr = int(ckd_donor_egfr[egfr_col].notna().sum())
print(
    f"CKD donors with non-null eGFR bin: {n_ckd_donors_with_egfr} of {n_ckd_donors_total}"
)

print("\nCKD donor counts per raw eGFR bin label:")
print(ckd_donor_egfr[egfr_col].value_counts(dropna=False))

# %%
# EGFR BIN LABEL PARSER
# > Bin labels observed: fine 10-unit ranges ("90-99 ml/min/1.73m2"), an
# ambiguous coarse catch-all (">60 ml/min/1.73m2") that overlaps every fine
# bin from 60-69 up through 170-179, and a missing-data sentinel
# ("Not available"). The catch-all can't be given an unambiguous rank
# relative to donors in the fine-grained bins, so by default it's excluded
# from the correlation (bin_kind="ambiguous_open_ended") rather than guessed
# at - flip include_ambiguous_open_ended_bins to True for a sensitivity run
# that anchors it to its lower bound (60) instead.
####
include_ambiguous_open_ended_bins = False

_fine_bin_pattern = re.compile(r"^(\d+)\s*-\s*(\d+)\s*ml/min")
_open_ended_pattern = re.compile(r"^[><]=?\s*(\d+)\s*ml/min")


def parse_bin_bounds(label):
    """Returns (lower, upper, midpoint, bin_kind) for a single eGFR bin label."""
    if pd.isna(label) or str(label).strip().lower() == "not available":
        return np.nan, np.nan, np.nan, "missing"

    label = str(label).strip()
    fine_match = _fine_bin_pattern.match(label)
    if fine_match:
        lower, upper = float(fine_match.group(1)), float(fine_match.group(2))
        return lower, upper, (lower + upper) / 2, "fine"

    open_match = _open_ended_pattern.match(label)
    if open_match:
        lower = float(open_match.group(1))
        return lower, np.inf, np.nan, "ambiguous_open_ended"

    return np.nan, np.nan, np.nan, "unparsed"


observed_bin_labels = sorted(adata.obs[egfr_col].dropna().unique())
bin_reference = pd.DataFrame(
    [
        {
            "bin_label": label,
            **dict(
                zip(["lower", "upper", "midpoint", "bin_kind"], parse_bin_bounds(label))
            ),
        }
        for label in observed_bin_labels
    ]
)
bin_reference = bin_reference.sort_values("lower", na_position="last").reset_index(
    drop=True
)
print("Parsed eGFR bin reference table:")
print(bin_reference)

n_unparsed = int((bin_reference["bin_kind"] == "unparsed").sum())
if n_unparsed:
    print(
        f"\nWARNING: {n_unparsed} bin label(s) did not match either regex pattern "
        "and will be treated as missing - check bin_reference above."
    )

n_ckd_ambiguous = int(
    ckd_donor_egfr[egfr_col]
    .astype(str)
    .map(lambda v: parse_bin_bounds(v)[3])
    .eq("ambiguous_open_ended")
    .sum()
)
print(
    f"\nCKD donors falling in the ambiguous '>60' catch-all bin: {n_ckd_ambiguous} of {n_ckd_donors_total}"
)
if n_ckd_ambiguous and not include_ambiguous_open_ended_bins:
    print(
        "These will be EXCLUDED from the correlation (include_ambiguous_open_ended_bins=False)."
    )

# %%
# GENE PROGRAM DEFINITIONS
# > Duplicated from 03-pseudobulk-de.py / 04-plotting.py (kept in sync
# manually) so this script can independently compute pathway module scores.
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

complement_programs_with_composite = dict(complement_programs)
complement_programs_with_composite["composite_all_complement"] = sorted(
    {g for genes in complement_programs.values() for g in genes}
)
inflammasome_programs_with_composite = dict(inflammasome_programs)
inflammasome_programs_with_composite["composite_all_inflammasome"] = sorted(
    {g for genes in inflammasome_programs.values() for g in genes}
)


# %%
# UTILITIES
# > raw_gene_symbols and benjamini_hochberg duplicated from
# 03-pseudobulk-de.py for self-containment.
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


# %%
# CKD DONOR PSEUDOBULK (RAW LAYER)
# > Restricts to CKD cells only, then aggregates per donor via ADPBulk
# using adata.raw (use_raw=True) - the full ~35k-gene feature set, not the
# 4000-gene HVG matrix in adata.X. This matters here specifically because
# most complement/inflammasome panel genes are not highly variable and
# would otherwise be silently dropped from the module scores.
####
ckd_adata = adata[adata.obs[disease_key].astype(str) == ckd_label].copy()
print(
    f"CKD subset: {ckd_adata.n_obs} cells, {ckd_adata.obs[donor_key].nunique()} donors."
)

group_cols = [donor_key, disease_key]
adpb = ADPBulk(ckd_adata, group_cols, method="sum", use_raw=True)
ckd_counts_df = adpb.fit_transform()
ckd_meta = adpb.get_meta()

expected_genes = ckd_adata.raw.var_names.astype(str)
assert list(ckd_counts_df.columns.astype(str)) == list(expected_genes), (
    "ADPBulk gene column order does not match adata.raw.var_names - "
    "gene_symbol alignment for module scoring would be wrong."
)

cell_counts = (
    ckd_adata.obs.groupby(group_cols, observed=True).size().reset_index(name="n_cells")
)
ckd_meta = ckd_meta.merge(cell_counts, on=group_cols, how="left")
assert len(ckd_meta) == ckd_counts_df.shape[0], (
    "meta and counts_df row counts diverged after merge - check for duplicate "
    "category combinations in cell_counts."
)

ckd_counts_mat = sp.csr_matrix(ckd_counts_df.values)
ckd_library_size = np.asarray(ckd_counts_mat.sum(axis=1)).ravel()
ckd_cpm = ckd_counts_mat.multiply(
    1e6 / np.maximum(ckd_library_size, 1)[:, None]
).tocsr()
ckd_log1p_cpm = ckd_cpm.copy()
ckd_log1p_cpm.data = np.log1p(ckd_log1p_cpm.data)

print(
    f"CKD donor pseudobulk: {ckd_meta.shape[0]} donor pseudobulks, {ckd_counts_df.shape[1]} raw genes."
)

# %%
# MERGE EGFR ONTO DONOR PSEUDOBULK
# > Keeps only donors with a "fine" (unambiguous) eGFR bin, per the
# exclusion decision above. include_ambiguous_open_ended_bins=True would
# additionally keep the ">60" catch-all anchored to its lower bound (60) -
# off by default since it's not a real sensitivity-tested option here.
####
ckd_donor_egfr_parsed = ckd_donor_egfr.copy()
ckd_donor_egfr_parsed[
    ["egfr_lower", "egfr_upper", "egfr_midpoint", "egfr_bin_kind"]
] = (
    ckd_donor_egfr_parsed[egfr_col]
    .astype(str)
    .apply(lambda v: pd.Series(parse_bin_bounds(v)))
)

allowed_bin_kinds = ["fine"] + (
    ["ambiguous_open_ended"] if include_ambiguous_open_ended_bins else []
)
if include_ambiguous_open_ended_bins:
    # Anchor the open-ended bin to its lower bound so it at least has *a*
    # numeric value for Spearman ranking, understanding this likely
    # understates its true eGFR (only relevant if the flag above is flipped).
    open_mask = ckd_donor_egfr_parsed["egfr_bin_kind"] == "ambiguous_open_ended"
    ckd_donor_egfr_parsed.loc[open_mask, "egfr_midpoint"] = ckd_donor_egfr_parsed.loc[
        open_mask, "egfr_lower"
    ]

ckd_donor_egfr_usable = ckd_donor_egfr_parsed[
    ckd_donor_egfr_parsed["egfr_bin_kind"].isin(allowed_bin_kinds)
].copy()

ckd_meta_with_egfr = ckd_meta.merge(
    ckd_donor_egfr_usable[[donor_key, egfr_col, "egfr_midpoint", "egfr_bin_kind"]],
    on=donor_key,
    how="inner",
)
n_donors_for_correlation = ckd_meta_with_egfr[donor_key].nunique()
print(f"CKD donors with usable eGFR bin + pseudobulk: {n_donors_for_correlation}")
if n_donors_for_correlation < min_donors_for_correlation:
    raise ValueError(
        f"Only {n_donors_for_correlation} CKD donors have a usable eGFR bin; "
        f"need at least {min_donors_for_correlation} for a meaningful correlation."
    )


# %%
# DONOR-LEVEL PATHWAY MODULE SCORES
# > Same z-score-per-gene-then-average-across-program approach as
# 03-pseudobulk-de.py's compute_donor_pathway_scores, applied here to the
# CKD-only raw-layer pseudobulk.
####
def compute_module_scores(log1p_cpm_dense, gene_symbols, programs):
    symbol_to_cols = {}
    for idx, sym in enumerate(gene_symbols):
        symbol_to_cols.setdefault(sym, []).append(idx)

    scores = {}
    detected_gene_counts = {}
    for program_name, genes in programs.items():
        cols = sorted({idx for g in genes for idx in symbol_to_cols.get(g, [])})
        detected_gene_counts[program_name] = len(cols)
        if not cols:
            scores[program_name] = np.full(log1p_cpm_dense.shape[0], np.nan)
            continue
        sub = log1p_cpm_dense[:, cols]
        gene_mean = sub.mean(axis=0)
        gene_std = sub.std(axis=0, ddof=0)
        safe_std = np.where(gene_std == 0, 1.0, gene_std)
        z = (sub - gene_mean) / safe_std
        scores[program_name] = z.mean(axis=1)
    return pd.DataFrame(scores), detected_gene_counts


ckd_gene_symbols = raw_gene_symbols(ckd_adata)
ckd_log1p_cpm_dense = np.asarray(ckd_log1p_cpm.toarray())

complement_scores_df, complement_gene_counts = compute_module_scores(
    ckd_log1p_cpm_dense, ckd_gene_symbols, complement_programs_with_composite
)
inflammasome_scores_df, inflammasome_gene_counts = compute_module_scores(
    ckd_log1p_cpm_dense, ckd_gene_symbols, inflammasome_programs_with_composite
)

# ckd_meta (and therefore ckd_counts_df / ckd_log1p_cpm rows) share the same
# row order, so scores can be attached positionally before filtering to the
# eGFR-usable donor subset.
ckd_meta_scored = pd.concat(
    [ckd_meta.reset_index(drop=True), complement_scores_df, inflammasome_scores_df],
    axis=1,
)
ckd_donor_table = ckd_meta_scored.merge(
    ckd_donor_egfr_usable[[donor_key, egfr_col, "egfr_midpoint", "egfr_bin_kind"]],
    on=donor_key,
    how="inner",
)

donor_table_path = OUTPUT_DIR / "egfr_ckd_donor_pathway_scores.csv"
ckd_donor_table.to_csv(donor_table_path, index=False)
print(f"Wrote {donor_table_path.name} ({len(ckd_donor_table)} donors).")


# %%
# EGFR x PATHWAY SPEARMAN CORRELATION
# > One row per program: Spearman rho + p-value between donor module score
# and eGFR midpoint (Spearman uses ranks internally, so tied fine-bin
# midpoints are handled correctly - no separate rank column needed).
# BH-corrected jointly across complement + inflammasome programs.
####
def run_egfr_correlations(donor_table, program_cols, gene_set_name, gene_counts):
    rows = []
    valid = donor_table.dropna(subset=["egfr_midpoint"])
    for program in program_cols:
        scores = valid[program]
        egfr = valid["egfr_midpoint"]
        mask = scores.notna() & egfr.notna()
        n = int(mask.sum())
        if n < min_donors_for_correlation or gene_counts.get(program, 0) == 0:
            rows.append(
                {
                    "gene_set": gene_set_name,
                    "program": program,
                    "n_donors": n,
                    "n_genes_detected_in_program": int(gene_counts.get(program, 0)),
                    "spearman_rho": np.nan,
                    "p_value": np.nan,
                    "status": "skipped_insufficient_data",
                }
            )
            continue
        rho, p_value = stats.spearmanr(scores[mask], egfr[mask])
        rows.append(
            {
                "gene_set": gene_set_name,
                "program": program,
                "n_donors": n,
                "n_genes_detected_in_program": int(gene_counts.get(program, 0)),
                "spearman_rho": float(rho),
                "p_value": float(p_value),
                "status": "tested",
            }
        )
    return pd.DataFrame(rows)


complement_program_cols = list(complement_programs_with_composite.keys())
inflammasome_program_cols = list(inflammasome_programs_with_composite.keys())

complement_egfr_corr = run_egfr_correlations(
    ckd_donor_table, complement_program_cols, "complement", complement_gene_counts
)
inflammasome_egfr_corr = run_egfr_correlations(
    ckd_donor_table, inflammasome_program_cols, "inflammasome", inflammasome_gene_counts
)

egfr_correlation_results = pd.concat(
    [complement_egfr_corr, inflammasome_egfr_corr], ignore_index=True
)
egfr_correlation_results["q_value"] = np.nan
tested_mask = egfr_correlation_results["status"] == "tested"
egfr_correlation_results.loc[tested_mask, "q_value"] = benjamini_hochberg(
    egfr_correlation_results.loc[tested_mask, "p_value"].to_numpy()
)

egfr_correlation_path = OUTPUT_DIR / "egfr_ckd_pathway_correlation.csv"
egfr_correlation_results.to_csv(egfr_correlation_path, index=False)

n_tested = int(tested_mask.sum())
n_sig = int((egfr_correlation_results.loc[tested_mask, "q_value"] <= 0.05).sum())
print(
    f"Wrote {egfr_correlation_path.name} | {n_tested} tested, {n_sig} significant at q<0.05"
)
print(egfr_correlation_results.sort_values("p_value"))

print("eGFR correlation pipeline completed.")
