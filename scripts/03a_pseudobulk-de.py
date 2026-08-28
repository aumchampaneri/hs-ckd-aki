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
min_compartment_cells = 500  # minimum raw cells in a SubclassLevel1 population before attempting targeted DE

# Candidate donor-level covariates for the DE design (Reviewer 2, point #3:
# disease group may be confounded with assay modality, sex, race,
# diabetes/hypertension, age, region, tissue source). Maps a short name ->
# the corresponding adata.obs column. Each candidate is only added to the
# actual DE design (`adjustment_covariates`, populated after the audit step
# below) if it passes a crosstab-degeneracy check against disease_key -
# a covariate with an empty or near-empty cell for some disease group would
# make the design matrix singular or produce unstable coefficient estimates,
# so it is reported and excluded rather than silently included.
adjustment_covariate_candidates = {
    "assay_platform": "assay",  # 10x 3' v3 vs 10x multiome; suspension_type
                                 # was checked and found invariant (this whole
                                 # cohort is single-nucleus - no sc/sn split)
    "sex": "sex",
    "diabetes_history": "diabetes_history_clean",
    "hypertension": "hypertension_clean",
    "race": "Race_grouped",
    "age_group": "Age_grouped",
    "egfr_group": "egfr_grouped",
    "region": "region_grouped",
    "tissue_collection": "TissueCollection_grouped",
}

# Minimum number of donors required in every (covariate level x disease
# group) cell for a candidate covariate to be considered non-degenerate.
# Below this, PyDESeq2's design matrix can become rank-deficient or produce
# coefficients driven by a single donor.
min_covariate_cell_donors = 3

# Populated below by run_covariate_audit(); left as [] here so the script
# still runs top-to-bottom if that cell is skipped, but every downstream
# design formula depends on this being set by the audit before use.
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

# Whether to run the targeted per-SubclassLevel1 DE contrasts (feeds the
# SubclassLevel1 triptych plots in 04-plotting.py). Set False to skip and
# only produce the 3 global comparisons above.
run_subclasslevel1_targeted_de = True
subclasslevel1_key = "SubclassLevel1"

# Apply apeGLM-style LFC shrinkage (PyDESeq2's lfc_shrink) on top of the raw
# Wald MLE log2FoldChange. Shrinkage pulls noisy, low-count-driven fold
# changes toward zero without changing p-values, which is the standard fix
# for over-inflated log2FC on genes with few informative donors. The raw MLE
# estimate is retained alongside the shrunk value for every gene.
apply_lfc_shrinkage = True

# Comparison-ID pairs to run cross-contrast concordance analysis on. Pairs
# should share a common reference group (group2) so "concordant vs
# discordant" has a clean interpretation (e.g. does CKD and AKI dysregulate
# the same genes relative to the same normal baseline).
concordance_pairs = [("ckd_vs_normal", "aki_vs_normal")]

# Donor-level pathway module scoring + group-difference testing. Programs
# duplicated here (kept in sync manually with 04-plotting.py) so this script
# can independently compute pathway-level statistics without depending on
# the plotting script.
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
# COVARIATE COARSENING
# > Reviewer 2 (#3): several candidate covariates (Race, Age, eGFR bin,
# region, tissue collection method) have too many levels for the donor
# pool, producing empty/near-empty cells once crossed with disease group.
# Coarsen to clinically defensible, more balanced groupings BEFORE the
# audit, rather than lowering the audit's minimum-cell-donors threshold to
# force sparse categories through. Boundaries below are derived from the
# observed level counts in this cohort (see donor_cohort_summary output)
# and should be reviewed, not assumed correct for a different cohort.
####
def add_coarsened_covariates(adata):
    obs = adata.obs

    # NOTE: adata.obs columns are pandas Categorical (AnnData default),
    # which raises TypeError if you try to assign a value that isn't
    # already a defined category (e.g. via .where()/.mask()). Cast each
    # source column to plain object dtype before deriving a grouped
    # column - this also preserves true missing values as NaN, unlike
    # .astype(str), which would turn NaN into the literal string "nan".
    def _as_object(col):
        return obs[col].astype(object) if col in obs else None

    # diabetes_history / hypertension: "Not available" -> missing, so those
    # donors are excluded (complete-case) from any model using the
    # covariate, rather than being forced into a sparse third category.
    for col in ["diabetes_history", "hypertension"]:
        s = _as_object(col)
        if s is not None:
            obs[f"{col}_clean"] = s.replace({"Not available": np.nan})

    # Race: collapsed further to White vs Non-White after the first-pass
    # 3-level grouping (White/Black/Other) still had an undersized cell in
    # the covariate audit (min 2 donors). Binary is the coarsest defensible
    # split available in this cohort - if it still fails the audit, Race
    # cannot be jointly modeled with disease group here and that is a
    # reportable limitation, not something to force further.
    s = _as_object("Race")
    if s is not None:
        obs["Race_grouped"] = np.where(s == "White", "White", "Non_White")
        obs["Race_grouped"] = pd.Series(obs["Race_grouped"], index=obs.index).where(
            s.notna(), np.nan
        )

    # Age: collapse 9 bins to 3. Boundaries chosen from the observed
    # per-bin counts in this cohort (see the value_counts print), merging
    # the sparse 10-19/20-29 and 80-89/90-99 tails into their neighbors.
    s = _as_object("Age")
    if s is not None:
        def _age_group(val):
            if pd.isna(val):
                return np.nan
            lo = int(str(val).split("-")[0])
            if lo < 40:
                return "<40"
            elif lo < 60:
                return "40-59"
            else:
                return "60+"
        obs["Age_grouped"] = s.apply(_age_group)

    # eGFR (binned): collapsed to the standard clinical binary cutoff
    # (<60 vs >=60 ml/min/1.73m2, the threshold for CKD stage 3+) after the
    # 3-bucket grouping still had an undersized cell (min 2 donors). "Not
    # available" -> missing. ">60 ml/min/1.73m2" is an ambiguous/
    # unspecified bin (likely used for donors recorded only as "normal"
    # range) - placed in the >=60 bucket; confirm this assumption against
    # your source-dataset documentation.
    egfr_col = "Baseline eGFR (ml/min/1.73m2) (Binned)"
    s = _as_object(egfr_col)
    if s is not None:
        def _egfr_group(val):
            if pd.isna(val) or val == "Not available":
                return np.nan
            if val == ">60 ml/min/1.73m2":
                return ">=60"  # ASSUMPTION - see comment above
            lo = int(str(val).split("-")[0])
            return "<60" if lo < 60 else ">=60"
        obs["egfr_grouped"] = s.apply(_egfr_group)

    # region: collapsed further to Cortex-containing vs not, after the
    # first-pass 4-level grouping had a near-empty cell (min 1 donor).
    # Merges Medulla with Other_or_unknown (Papilla + unknown region).
    s = _as_object("region")
    if s is not None:
        cortex_containing = {"Cortex", "Cortex/Medulla"}
        obs["region_grouped"] = np.where(
            s.isin(cortex_containing), "Cortex_containing", "Medulla_or_other"
        )
        obs["region_grouped"] = pd.Series(obs["region_grouped"], index=obs.index).where(
            s.notna(), np.nan
        )

    # TissueCollection: collapsed further to Biopsy vs Surgical (merging
    # Nephrectomy and Transplant Pre-perfusion Biopsy, both organ-level
    # procurement rather than needle sampling), after the first-pass
    # 3-level grouping had an undersized cell (min 2 donors).
    s = _as_object("TissueCollection")
    if s is not None:
        biopsy_procedures = {"Percutaneous Needle Biopsy", "Intra-operative Biopsy"}
        obs["TissueCollection_grouped"] = np.where(
            s.isin(biopsy_procedures), "Biopsy", "Surgical"
        )
        obs["TissueCollection_grouped"] = pd.Series(
            obs["TissueCollection_grouped"], index=obs.index
        ).where(s.notna(), np.nan)

    return adata


adata = add_coarsened_covariates(adata)


# %%
# DONOR-LEVEL COHORT TABLE + COVARIATE AUDIT
# > Reviewer 1 & Reviewer 2 (#1, #3): report exact donor/cell counts per
# disease group with demographics and missingness, and explicitly audit
# which candidate covariates are usable in the DE design before any are
# added - rather than asserting "covariates were audited" without showing
# the audit. Both outputs are written to disk so they can go directly into
# the response letter / revised Methods & Supplementary Materials.
####
def build_donor_cohort_table(adata, covariate_candidates):
    """One row per donor_id. Assumes covariate columns are (or should be)
    donor-invariant; explicitly checks that assumption rather than silently
    taking the first observed value per donor."""
    cols_present = {
        short: col for short, col in covariate_candidates.items() if col in adata.obs
    }
    missing_cols = sorted(set(covariate_candidates) - set(cols_present))
    if missing_cols:
        print(
            f"Covariate audit: candidates not found in adata.obs, skipping: {missing_cols}"
        )

    obs_cols = [donor_key, disease_key] + list(cols_present.values())
    obs = adata.obs[obs_cols].copy()

    # Donor-invariance check: flag any donor where a nominally donor-level
    # column takes >1 distinct non-null value across its cells/nuclei. This
    # would indicate the column is actually library- or cell-level, not
    # donor-level, and should not be joined onto the donor pseudobulk as-is.
    non_invariant = {}
    for short, col in cols_present.items():
        n_unique = obs.groupby(donor_key, observed=True)[col].nunique(dropna=True)
        offenders = n_unique[n_unique > 1]
        if len(offenders) > 0:
            non_invariant[short] = offenders.index.tolist()
    if non_invariant:
        print(
            "Covariate audit: WARNING - these covariates vary WITHIN at least one "
            f"donor_id (not donor-invariant), affected donors: {non_invariant}"
        )

    donor_cell_counts = obs.groupby(donor_key, observed=True).size().rename("n_cells_total")
    donor_level = (
        obs.groupby(donor_key, observed=True)
        .first()  # documented above as an approximation where non-invariance was flagged
        .join(donor_cell_counts)
        .reset_index()
    )

    cohort_table_path = OUTPUT_DIR / "donor_cohort_table.csv"
    donor_level.to_csv(cohort_table_path, index=False)

    # Per-disease-group summary: donor counts, cell counts, missingness per
    # covariate - the minimum table R1/R2 ask for.
    summary_rows = []
    for group, sub in donor_level.groupby(disease_key, observed=True):
        row = {
            "disease_group": group,
            "n_donors": len(sub),
            "n_cells_total": int(sub["n_cells_total"].sum()),
        }
        for short, col in cols_present.items():
            row[f"{short}_n_missing"] = int(sub[col].isna().sum())
        summary_rows.append(row)
    cohort_summary = pd.DataFrame(summary_rows)
    cohort_summary_path = OUTPUT_DIR / "donor_cohort_summary_by_disease.csv"
    cohort_summary.to_csv(cohort_summary_path, index=False)

    print(
        f"Cohort table: {len(donor_level)} donors -> {cohort_table_path.name}; "
        f"per-group summary -> {cohort_summary_path.name}"
    )
    return donor_level, cols_present, non_invariant


def run_covariate_audit(donor_level, cols_present, min_cell_donors=min_covariate_cell_donors):
    """For each candidate covariate, crosstab against disease_key and flag
    it as usable only if every observed (covariate level x disease group)
    cell has >= min_cell_donors donors. A covariate that is empty or
    near-empty in some disease group cannot be estimated jointly with the
    disease effect and would otherwise silently produce a rank-deficient or
    unstable design matrix."""
    audit_rows = []
    usable = []
    for short, col in cols_present.items():
        # Rows with a missing covariate value (e.g. diabetes_history_clean
        # NaN from "Not available") are excluded from the crosstab entirely
        # rather than counted as a "nan" level - those donors are simply
        # complete-case-excluded from any model using this covariate.
        non_missing = donor_level.dropna(subset=[col])
        n_missing = len(donor_level) - len(non_missing)
        ct = pd.crosstab(non_missing[col].astype(str), non_missing[disease_key].astype(str))
        # Only check cells for the disease groups actually used in
        # comparison_specs, since unused disease labels shouldn't block a
        # covariate from being usable.
        used_groups = sorted(
            {g for spec in comparison_specs for g in (spec["group1"], spec["group2"])}
        )
        used_groups = [g for g in used_groups if g in ct.columns]
        ct_used = ct[used_groups] if used_groups else ct
        min_cell = int(ct_used.replace(0, np.nan).min().min()) if ct_used.size else 0
        n_levels = int((donor_level[col].notna()).astype(str).nunique())
        is_usable = bool(ct_used.size) and (ct_used.min().min() >= min_cell_donors)
        audit_rows.append(
            {
                "covariate": short,
                "column": col,
                "n_levels": non_missing[col].astype(str).nunique(),
                "n_donors_missing_excluded": n_missing,
                "min_cell_donors_in_used_groups": min_cell,
                "usable": is_usable,
                "reason": "ok" if is_usable else "empty_or_undersized_cell_in_crosstab",
            }
        )
        if is_usable:
            usable.append(short)

    audit_df = pd.DataFrame(audit_rows)
    audit_path = OUTPUT_DIR / "covariate_audit.csv"
    audit_df.to_csv(audit_path, index=False)
    print(f"Covariate audit -> {audit_path.name}: usable covariates = {usable}")
    return usable, audit_df


_donor_cohort_table, _covariate_cols_present, _covariate_non_invariant = (
    build_donor_cohort_table(adata, adjustment_covariate_candidates)
)
_usable_covariates, _covariate_audit_df = run_covariate_audit(
    _donor_cohort_table, _covariate_cols_present
)

# Populate the design-facing list from the audit result. Covariates that
# vary within a donor_id (see WARNING above) are additionally excluded here
# even if they passed the crosstab check, since a per-donor "first value"
# join would be attaching a value that isn't stably true of that donor.
adjustment_covariates = [
    adjustment_covariate_candidates[short]
    for short in _usable_covariates
    if short not in _covariate_non_invariant
]
print(f"adjustment_covariates set to: {adjustment_covariates}")


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
            .drop_duplicates(subset=[donor_key])
            .set_index(donor_key)
        )
        # Cast non-null values to str for consistent categorical handling,
        # but preserve real missingness as NaN rather than the literal
        # string "nan" - a donor with an unknown covariate value should be
        # excluded (complete-case) from models using that covariate in
        # run_pydeseq2 below, not silently given its own "nan" category.
        for c in cov_cols:
            donor_cov[c] = donor_cov[c].astype(object)
            notna = donor_cov[c].notna()
            donor_cov.loc[notna, c] = donor_cov.loc[notna, c].astype(str)
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

    # Complete-case filtering: a covariate with a missing value (NaN,
    # preserved as such by build_donor_pseudobulk) excludes that pseudobulk
    # sample from THIS contrast's design, since PyDESeq2/patsy cannot fit a
    # NaN into a categorical design term. Different contrasts may drop
    # different samples depending on which covariates were audited as
    # usable and which donors have missing values for them.
    design_terms = [c for c in adjustment_covariates if c in metadata.columns]
    if design_terms:
        complete_mask = metadata[design_terms].notna().all(axis=1)
        n_dropped = int((~complete_mask).sum())
        if n_dropped:
            print(
                f"[{name}] dropping {n_dropped} pseudobulk sample(s) with missing "
                f"covariate value(s) in {design_terms}: "
                f"{metadata.loc[~complete_mask].index.tolist()}"
            )
            metadata = metadata.loc[complete_mask]
            counts_df = counts_df.loc[metadata.index]

        # Guard: if complete-case filtering pushed either contrast group
        # below min_donors_per_group, the adjusted design is no longer
        # supportable. Fall back to the unadjusted (~ disease only) design
        # for this contrast and say so explicitly, rather than silently
        # fitting an underpowered adjusted model.
        group_counts = metadata[disease_key].value_counts()
        if group_counts.get(group1, 0) < min_donors_per_group or group_counts.get(
            group2, 0
        ) < min_donors_per_group:
            print(
                f"[{name}] WARNING: complete-case filtering left "
                f"{dict(group_counts)} (< min_donors_per_group={min_donors_per_group} "
                f"in at least one group). Falling back to unadjusted design "
                f"(~ {disease_key}) for this contrast; original covariate-complete "
                f"metadata is discarded for THIS contrast only."
            )
            metadata = pb["meta"][[disease_key]].copy()
            metadata.index = pb["counts_df"].index
            counts_df = pb["counts_df"].iloc[:, keep_genes].copy()
            counts_df = np.rint(counts_df).astype(np.int64)
            design_terms = []

    # Explicitly set category order so group2 is always the reference
    # (dropped) level. DeseqDataSet's `ref_level` argument is deprecated and
    # a no-op in the installed pydeseq2 version, so relying on it silently
    # leaves the reference to whatever pandas picks (alphabetical by
    # default) - this also guarantees the LFC-shrinkage coefficient below
    # ("{disease_key}[T.{group1}]") always represents "group1 vs group2",
    # matching the direction of the `contrast` used by DeseqStats.
    metadata[disease_key] = pd.Categorical(
        metadata[disease_key].astype(str), categories=[group2, group1]
    )

    for cov in design_terms:
        metadata[cov] = metadata[cov].astype("category")

    design_terms = design_terms + [disease_key]
    design_formula = "~ " + " + ".join(design_terms)

    dds = DeseqDataSet(
        counts=counts_df,
        metadata=metadata,
        design=design_formula,
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
            "log2FoldChange": "deseq2_log2fc_mle",
            "lfcSE": "deseq2_lfc_se_mle",
            "stat": "deseq2_wald_stat",
            "pvalue": "deseq2_p_value",
            "padj": "deseq2_q_value",
        }
    )

    # deseq2_log2fc / deseq2_lfc_se are the "primary" columns downstream
    # plotting/analysis code consumes. Default to the raw MLE; overwritten
    # below with the shrunk estimate wherever shrinkage converges.
    res["deseq2_log2fc"] = res["deseq2_log2fc_mle"]
    res["deseq2_lfc_se"] = res["deseq2_lfc_se_mle"]
    res["deseq2_log2fc_shrunk"] = np.nan
    res["deseq2_lfc_se_shrunk"] = np.nan
    res["deseq2_lfc_shrunk_applied"] = False
    res["deseq2_shrinkage_coeff"] = np.nan

    if apply_lfc_shrinkage:
        shrink_coeff = f"{disease_key}[T.{group1}]"
        if shrink_coeff not in stat_res.LFC.columns:
            print(
                f"{name}: shrink coeff '{shrink_coeff}' not found in LFC columns "
                f"{list(stat_res.LFC.columns)}; skipping shrinkage for this contrast "
                "(raw MLE log2fc will be used as deseq2_log2fc)."
            )
        else:
            # lfc_shrink() overwrites stat_res.results_df's log2FoldChange/lfcSE
            # in place for genes where shrinkage converged; p-values are left
            # unchanged (shrinkage only affects effect-size estimates).
            stat_res.lfc_shrink(coeff=shrink_coeff)
            shrunk = stat_res.results_df.reset_index().rename(
                columns={"index": "gene_id"}
            )
            shrunk["gene_id"] = shrunk["gene_id"].astype(str)
            shrunk = shrunk.rename(
                columns={
                    "log2FoldChange": "deseq2_log2fc_shrunk",
                    "lfcSE": "deseq2_lfc_se_shrunk",
                }
            )
            res = res.drop(
                columns=["deseq2_log2fc_shrunk", "deseq2_lfc_se_shrunk"]
            ).merge(
                shrunk[["gene_id", "deseq2_log2fc_shrunk", "deseq2_lfc_se_shrunk"]],
                on="gene_id",
                how="left",
            )

            convergence = stat_res._LFC_shrink_converged.astype(object).to_dict()
            converged_flag = res["gene_id"].map(convergence)
            # .eq(True).fillna(False) avoids errors/ambiguity from pandas'
            # nullable-boolean NA values when used directly as a boolean mask.
            converged_mask = converged_flag.eq(True).fillna(False).to_numpy()

            res.loc[converged_mask, "deseq2_log2fc"] = res.loc[
                converged_mask, "deseq2_log2fc_shrunk"
            ]
            res.loc[converged_mask, "deseq2_lfc_se"] = res.loc[
                converged_mask, "deseq2_lfc_se_shrunk"
            ]
            res.loc[converged_mask, "deseq2_lfc_shrunk_applied"] = True
            res["deseq2_shrinkage_coeff"] = shrink_coeff

            n_converged = int(converged_mask.sum())
            print(
                f"{name}: LFC shrinkage converged for {n_converged}/{len(res)} genes."
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

    # Donor/cell counts actually contributing to THIS pseudobulk comparison
    # (post compartment-subsetting and post min_cells_per_pseudobulk
    # filtering, i.e. the same donors/cells the DESeq2 fit above was run
    # on) - Reviewer 1 #11 asks that every figure/caption state the exact
    # donor and cell/nucleus counts per panel; without these columns 04's
    # plots have no way to display them and every count has to be
    # separately, and inconsistently, recomputed downstream.
    # NOTE: this is the count BEFORE any complete-case covariate exclusion
    # inside run_pydeseq2 (which is design/contrast-specific and can drop a
    # few additional donors per covariate missingness) - i.e. it is the
    # denominator for "donors available for this contrast," not
    # necessarily the exact N in the final fitted model when covariates
    # were adjusted for. Worth a footnote in the manuscript if the two
    # ever diverge materially for a given contrast.
    out["n_donors_group1"] = int(mask1.sum())
    out["n_donors_group2"] = int(mask2.sum())
    out["n_cells_group1"] = int(meta.loc[mask1, "n_cells"].sum())
    out["n_cells_group2"] = int(meta.loc[mask2, "n_cells"].sum())

    # Save Outputs
    out_path = OUTPUT_DIR / f"pseudobulk_de_{name}.csv"
    out.to_csv(out_path, index=False)

    print(
        f"{name}: Wrote to {out_path.name} | DE Genes (q < 0.05): {out['is_de_primary_q_0_05'].sum()} "
        f"| donors {out['n_donors_group1'].iloc[0]} vs {out['n_donors_group2'].iloc[0]} "
        f"| cells {out['n_cells_group1'].iloc[0]} vs {out['n_cells_group2'].iloc[0]}"
    )
    return out


# %%
# EXECUTE PIPELINE
# > Iterate over comparison specs and run the DE pipeline
####
pseudobulk_results = {}
for spec in comparison_specs:
    pseudobulk_results[spec["name"]] = run_pseudobulk_de(spec)

print("Global pseudobulk DE pipeline completed.")


# %%
# CROSS-CONTRAST CONCORDANCE ANALYSIS
# > For comparison pairs sharing a common reference group (e.g. CKD vs
# normal and AKI vs normal), builds a gene-level table of both contrasts'
# shrunk log2FC/q-value side by side, plus summary correlation statistics.
# Answers: is gene dysregulation shared across disease states relative to
# the same baseline, or state-specific? Feeds the concordance scatter plot
# in 04-plotting.py.
####
def build_concordance_table(df_x, df_y, x_id, y_id):
    """Inner-joins two DE result tables on gene_id, keeping only genes that
    passed the expression filter in both contrasts."""
    left = df_x[
        [
            "gene_id",
            "gene_symbol",
            "deseq2_log2fc",
            "deseq2_q_value",
            "is_de_primary_q_0_05",
            "passes_expression_filter",
        ]
    ].rename(
        columns={
            "deseq2_log2fc": "log2fc_x",
            "deseq2_q_value": "q_value_x",
            "is_de_primary_q_0_05": "is_de_x",
            "passes_expression_filter": "passes_filter_x",
        }
    )
    right = df_y[
        [
            "gene_id",
            "deseq2_log2fc",
            "deseq2_q_value",
            "is_de_primary_q_0_05",
            "passes_expression_filter",
        ]
    ].rename(
        columns={
            "deseq2_log2fc": "log2fc_y",
            "deseq2_q_value": "q_value_y",
            "is_de_primary_q_0_05": "is_de_y",
            "passes_expression_filter": "passes_filter_y",
        }
    )

    merged = left.merge(right, on="gene_id", how="inner")
    merged = merged[
        merged["passes_filter_x"].fillna(False)
        & merged["passes_filter_y"].fillna(False)
    ].copy()
    merged["comparison_id_x"] = x_id
    merged["comparison_id_y"] = y_id

    valid_signs = merged["log2fc_x"].notna() & merged["log2fc_y"].notna()
    merged["sign_concordant"] = pd.NA
    merged.loc[valid_signs, "sign_concordant"] = np.sign(
        merged.loc[valid_signs, "log2fc_x"]
    ) == np.sign(merged.loc[valid_signs, "log2fc_y"])

    def _sig_class(row):
        if bool(row["is_de_x"]) and bool(row["is_de_y"]):
            return "both_sig"
        if bool(row["is_de_x"]):
            return "x_only_sig"
        if bool(row["is_de_y"]):
            return "y_only_sig"
        return "neither_sig"

    merged["significance_class"] = merged.apply(_sig_class, axis=1)
    return merged.drop(columns=["passes_filter_x", "passes_filter_y"])


def summarize_concordance(merged, x_id, y_id):
    valid = merged.dropna(subset=["log2fc_x", "log2fc_y"])
    if len(valid) < 3:
        return pd.DataFrame(
            [
                {
                    "comparison_id_x": x_id,
                    "comparison_id_y": y_id,
                    "n_genes_compared": int(len(valid)),
                    "pearson_r": np.nan,
                    "pearson_p": np.nan,
                    "spearman_rho": np.nan,
                    "spearman_p": np.nan,
                    "n_both_sig": 0,
                    "n_concordant_both_sig": 0,
                    "n_discordant_both_sig": 0,
                    "frac_concordant_among_both_sig": np.nan,
                }
            ]
        )

    pearson_r, pearson_p = stats.pearsonr(valid["log2fc_x"], valid["log2fc_y"])
    spearman_rho, spearman_p = stats.spearmanr(valid["log2fc_x"], valid["log2fc_y"])
    both_sig = valid[valid["significance_class"] == "both_sig"]
    n_concordant = int(both_sig["sign_concordant"].fillna(False).sum())
    n_discordant = int(len(both_sig) - n_concordant)
    frac_concordant = n_concordant / len(both_sig) if len(both_sig) else np.nan

    return pd.DataFrame(
        [
            {
                "comparison_id_x": x_id,
                "comparison_id_y": y_id,
                "n_genes_compared": int(len(valid)),
                "pearson_r": pearson_r,
                "pearson_p": pearson_p,
                "spearman_rho": spearman_rho,
                "spearman_p": spearman_p,
                "n_both_sig": int(len(both_sig)),
                "n_concordant_both_sig": n_concordant,
                "n_discordant_both_sig": n_discordant,
                "frac_concordant_among_both_sig": frac_concordant,
            }
        ]
    )


for x_id, y_id in concordance_pairs:
    if x_id not in pseudobulk_results or y_id not in pseudobulk_results:
        print(f"Skipping concordance pair ({x_id}, {y_id}): missing comparison table.")
        continue

    concordance_table = build_concordance_table(
        pseudobulk_results[x_id], pseudobulk_results[y_id], x_id, y_id
    )
    concordance_summary = summarize_concordance(concordance_table, x_id, y_id)

    concordance_table.to_csv(
        OUTPUT_DIR / f"concordance_{x_id}_vs_{y_id}.csv", index=False
    )
    concordance_summary.to_csv(
        OUTPUT_DIR / f"concordance_summary_{x_id}_vs_{y_id}.csv", index=False
    )

    r = concordance_summary["pearson_r"].iloc[0]
    n = concordance_summary["n_genes_compared"].iloc[0]
    print(
        f"Concordance {x_id} vs {y_id}: n={n} genes, Pearson r={r:.3f}"
        if pd.notna(r)
        else f"Concordance {x_id} vs {y_id}: insufficient overlapping genes for correlation."
    )


# %%
# DONOR-LEVEL PATHWAY MODULE SCORING + GROUP-DIFFERENCE TESTS
# > Rather than eyeballing how many individual pathway genes clear q<0.05 on
# a volcano plot, this computes one donor-level module score per pathway
# (mean of per-gene z-scored log1p CPM across the pathway's detected genes,
# from the *global*, unfiltered-by-contrast donor pseudobulk) and tests
# group1 vs group2 with a two-sided Mann-Whitney U test - giving an actual
# pathway-level p-value rather than a gene-count heuristic.
####
def compute_donor_pathway_scores(adata, programs):
    """Returns (score_df, detected_gene_counts) where score_df has one row
    per donor pseudobulk sample, one column per program (plus donor_key,
    disease_key, n_cells)."""
    pb = build_donor_pseudobulk(adata)  # all donors x disease groups, no query
    gene_symbols = raw_gene_symbols(adata)

    symbol_to_cols = {}
    for idx, sym in enumerate(gene_symbols):
        symbol_to_cols.setdefault(sym, []).append(idx)

    log1p_cpm = np.asarray(pb["log1p_cpm"].toarray())

    scores = {}
    detected_gene_counts = {}
    for program_name, genes in programs.items():
        cols = sorted({idx for g in genes for idx in symbol_to_cols.get(g, [])})
        detected_gene_counts[program_name] = len(cols)
        if not cols:
            scores[program_name] = np.full(log1p_cpm.shape[0], np.nan)
            continue
        sub = log1p_cpm[:, cols]
        gene_mean = sub.mean(axis=0)
        gene_std = sub.std(axis=0, ddof=0)
        safe_std = np.where(gene_std == 0, 1.0, gene_std)
        z = (sub - gene_mean) / safe_std
        scores[program_name] = z.mean(axis=1)

    score_df = pd.DataFrame(scores)
    score_df[donor_key] = pb["meta"][donor_key].to_numpy()
    score_df[disease_key] = pb["meta"][disease_key].to_numpy()
    score_df["n_cells"] = pb["meta"]["n_cells"].to_numpy()
    return score_df, detected_gene_counts


def run_pathway_level_tests(
    score_df, detected_gene_counts, gene_set_name, min_donors=min_donors_per_group
):
    program_cols = [
        c for c in score_df.columns if c not in {donor_key, disease_key, "n_cells"}
    ]
    rows = []
    for comp in comparison_specs:
        s1_all = score_df.loc[score_df[disease_key].astype(str) == comp["group1"]]
        s2_all = score_df.loc[score_df[disease_key].astype(str) == comp["group2"]]
        for program in program_cols:
            n_genes = detected_gene_counts.get(program, 0)
            s1 = s1_all[program].dropna()
            s2 = s2_all[program].dropna()
            base_row = {
                "comparison_id": comp["name"],
                "group1": comp["group1"],
                "group2": comp["group2"],
                "gene_set": gene_set_name,
                "program": program,
                "n_donors_group1": int(len(s1)),
                "n_donors_group2": int(len(s2)),
                "n_genes_detected_in_program": int(n_genes),
            }
            if n_genes == 0 or len(s1) < min_donors or len(s2) < min_donors:
                rows.append(
                    {
                        **base_row,
                        "status": "skipped_insufficient_data",
                        "median_score_group1": np.nan,
                        "median_score_group2": np.nan,
                        "median_diff_group1_minus_group2": np.nan,
                        "mannwhitney_u_stat": np.nan,
                        "p_value": np.nan,
                    }
                )
                continue
            u_stat, p_value = stats.mannwhitneyu(s1, s2, alternative="two-sided")
            rows.append(
                {
                    **base_row,
                    "status": "tested",
                    "median_score_group1": float(s1.median()),
                    "median_score_group2": float(s2.median()),
                    "median_diff_group1_minus_group2": float(s1.median() - s2.median()),
                    "mannwhitney_u_stat": float(u_stat),
                    "p_value": float(p_value),
                }
            )
    return pd.DataFrame(rows)


complement_programs_with_composite = dict(complement_programs)
complement_programs_with_composite["composite_all_complement"] = sorted(
    {g for genes in complement_programs.values() for g in genes}
)
inflammasome_programs_with_composite = dict(inflammasome_programs)
inflammasome_programs_with_composite["composite_all_inflammasome"] = sorted(
    {g for genes in inflammasome_programs.values() for g in genes}
)

complement_score_df, complement_gene_counts = compute_donor_pathway_scores(
    adata, complement_programs_with_composite
)
inflammasome_score_df, inflammasome_gene_counts = compute_donor_pathway_scores(
    adata, inflammasome_programs_with_composite
)

complement_pathway_tests = run_pathway_level_tests(
    complement_score_df, complement_gene_counts, "complement"
)
inflammasome_pathway_tests = run_pathway_level_tests(
    inflammasome_score_df, inflammasome_gene_counts, "inflammasome"
)

pathway_level_tests = pd.concat(
    [complement_pathway_tests, inflammasome_pathway_tests], ignore_index=True
)
pathway_level_tests["q_value"] = np.nan
tested_mask = pathway_level_tests["status"] == "tested"
pathway_level_tests.loc[tested_mask, "q_value"] = benjamini_hochberg(
    pathway_level_tests.loc[tested_mask, "p_value"].to_numpy()
)

pathway_level_tests_path = OUTPUT_DIR / "pathway_level_group_tests.csv"
pathway_level_tests.to_csv(pathway_level_tests_path, index=False)

n_tested = int(tested_mask.sum())
n_sig = int((pathway_level_tests.loc[tested_mask, "q_value"] <= 0.05).sum())
print(
    f"Pathway-level tests: wrote {pathway_level_tests_path.name} | {n_tested} tested, {n_sig} significant at q<0.05"
)


# %%
# TARGETED SUBCLASSLEVEL1 DE - SPEC BUILDER
# > For each SubclassLevel1 population with enough raw cells, preflight
# each of the 3 global comparisons by building that population's donor
# pseudobulk and counting donors per group post-filtering. Only specs that
# clear both min_compartment_cells and min_donors_per_group are queued for
# an actual PyDESeq2 run - this avoids paying for a full DESeq2 fit just to
# discover a contrast is underpowered.
####
def safe_name(value):
    return str(value).replace("/", "_").replace(" ", "_").replace("-", "-")


def subclasslevel1_pseudobulk_group_counts(population, group1, group2):
    query = f"`{subclasslevel1_key}` == {population!r}"
    pb = build_donor_pseudobulk(adata, query=query)
    pb = subset_pseudobulk_to_groups(pb, group1, group2)
    counts = pb["meta"][disease_key].astype(str).value_counts()
    return counts, int(pb["meta"]["n_cells"].sum())


def build_subclasslevel1_targeted_specs(
    adata, min_cells=min_compartment_cells, min_donors=min_donors_per_group
):
    specs = []
    skipped = []
    if subclasslevel1_key not in adata.obs:
        raise KeyError(f"{subclasslevel1_key} is missing from adata.obs")

    for population, n_cells in (
        adata.obs[subclasslevel1_key].astype(str).value_counts().items()
    ):
        if n_cells < min_cells:
            skipped.append(
                {
                    "population": population,
                    "reason": "too_few_cells",
                    "n_cells": int(n_cells),
                }
            )
            continue

        for comp in comparison_specs:
            try:
                donor_counts, retained_cells = subclasslevel1_pseudobulk_group_counts(
                    population, comp["group1"], comp["group2"]
                )
            except Exception as exc:
                skipped.append(
                    {
                        "population": population,
                        "comparison_id": comp["name"],
                        "reason": f"preflight_failed: {exc}",
                        "n_cells": int(n_cells),
                    }
                )
                continue

            n1 = int(donor_counts.get(comp["group1"], 0))
            n2 = int(donor_counts.get(comp["group2"], 0))
            if min(n1, n2) < min_donors:
                skipped.append(
                    {
                        "population": population,
                        "comparison_id": comp["name"],
                        "reason": "too_few_pseudobulk_donors_after_cell_filter",
                        "n_cells": int(n_cells),
                        "retained_cells": retained_cells,
                        "n_donors_group1": n1,
                        "n_donors_group2": n2,
                    }
                )
                continue

            pop_safe = safe_name(population)
            specs.append(
                {
                    "name": f"targeted_SubclassLevel1_{pop_safe}_{comp['name']}",
                    "query": f"`{subclasslevel1_key}` == {population!r}",
                    "group1": comp["group1"],
                    "group2": comp["group2"],
                    "population": population,
                    "n_cells": int(n_cells),
                    "retained_cells_after_pseudobulk_filter": retained_cells,
                    "n_donors_group1": n1,
                    "n_donors_group2": n2,
                }
            )

    return specs, skipped


# %%
# TARGETED SUBCLASSLEVEL1 DE - EXECUTION
# > Runs run_pseudobulk_de for every queued spec, writing
# pseudobulk_de_targeted_SubclassLevel1_{population}_{comparison_id}.csv
# into OUTPUT_DIR (same directory as the global comparisons), which is what
# 04-plotting.py's SubclassLevel1 triptych functions look for.
####
if run_subclasslevel1_targeted_de:
    subclasslevel1_targeted_specs, subclasslevel1_skipped_specs = (
        build_subclasslevel1_targeted_specs(adata)
    )

    subclasslevel1_targeted_specs_df = pd.DataFrame(subclasslevel1_targeted_specs)
    subclasslevel1_skipped_specs_df = pd.DataFrame(subclasslevel1_skipped_specs)
    subclasslevel1_targeted_specs_df.to_csv(
        OUTPUT_DIR / "subclasslevel1_targeted_pseudobulk_specs.csv", index=False
    )
    subclasslevel1_skipped_specs_df.to_csv(
        OUTPUT_DIR / "subclasslevel1_targeted_pseudobulk_skipped_specs.csv", index=False
    )

    print(
        f"SubclassLevel1 targeted specs: {len(subclasslevel1_targeted_specs)} runnable, "
        f"{len(subclasslevel1_skipped_specs)} skipped (see subclasslevel1_targeted_pseudobulk_skipped_specs.csv)"
    )

    subclasslevel1_run_records = []
    subclasslevel1_results = {}
    for spec in subclasslevel1_targeted_specs:
        out_path = OUTPUT_DIR / f"pseudobulk_de_{spec['name']}.csv"
        if out_path.exists():
            print("Exists, skipping", out_path.name)
            subclasslevel1_run_records.append(
                {"name": spec["name"], "status": "exists"}
            )
            continue

        run_spec = {
            "name": spec["name"],
            "query": spec["query"],
            "group1": spec["group1"],
            "group2": spec["group2"],
        }
        try:
            subclasslevel1_results[spec["name"]] = run_pseudobulk_de(run_spec)
            subclasslevel1_run_records.append(
                {"name": spec["name"], "status": "completed"}
            )
        except ValueError as exc:
            print("Skipping underpowered contrast:", spec["name"], exc)
            subclasslevel1_run_records.append(
                {"name": spec["name"], "status": "skipped", "reason": str(exc)}
            )
        except Exception as exc:
            print("Failed contrast:", spec["name"], exc)
            subclasslevel1_run_records.append(
                {"name": spec["name"], "status": "failed", "reason": str(exc)}
            )

    if subclasslevel1_run_records:
        pd.DataFrame(subclasslevel1_run_records).to_csv(
            OUTPUT_DIR / "subclasslevel1_targeted_pseudobulk_run_records.csv",
            index=False,
        )

    print("SubclassLevel1 targeted DE pipeline completed.")
else:
    print(
        "run_subclasslevel1_targeted_de = False; skipping targeted SubclassLevel1 DE."
    )
