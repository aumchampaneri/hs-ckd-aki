# %% scVI Validation
# > Confound analysis (are scores driven by technical covariates?)
# > Linear mixed model follow-up (batch/donor: technical or biological?)
# > Robustness: independent scoring method (AUCell) + leave-one-gene-out
#
# Reads the scored obs saved by 02b_scvi-plot.py (fast) plus adata_scvi.h5ad
# where raw counts are needed (AUCell, LOGO) -- avoids re-running the ~15min
# ULM fits for anything that only needs the already-computed scores.
#
# %% PATH SETUP
from pathlib import Path

PROJECT_DIR = Path.cwd().parent
DATA_DIR = PROJECT_DIR / "data/"
OUTPUT_DIR = PROJECT_DIR / "outputs/scvi/"
VALIDATION_DIR = OUTPUT_DIR / "validation/"
VALIDATION_DIR.mkdir(parents=True, exist_ok=True)

# %% IMPORTS
import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
import decoupler as dc
import statsmodels.formula.api as smf
from scipy.stats import spearmanr, kruskal, chi2

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# ---------------------------------------------------------------------------
# Complement module definitions -- duplicated from 02b so this script is
# self-contained. Must stay in sync with 02b's complement_gene_sets if that
# ever changes.
# ---------------------------------------------------------------------------
complement_gene_sets = {
    "classical": ["C1QA", "C1QB", "C1QC", "C1R", "C1S", "C2", "C4A", "C4B", "C4BPA", "C4BPB"],
    "lectin": ["MBL2", "FCN1", "FCN2", "FCN3", "MASP1", "MASP2", "MASP3"],
    "alternative": ["C3", "CFB", "CFD", "CFP"],
    "terminal": ["C5", "C6", "C7", "C8A", "C8B", "C8G", "C9"],
    "receptor": ["C3AR1", "C5AR1", "C5AR2", "CR1", "CR2", "ITGAM", "ITGAX", "VSIG4"],
    "regulator": ["CFH", "CFI", "CD46", "CD55", "CD59", "SERPING1"],
}
MIN_GENES_PER_MODULE = 2


def build_raw_gene_symbol_map(adata):
    raw_var = adata.raw.var
    if "feature_name" in raw_var.columns:
        return dict(zip(raw_var["feature_name"].astype(str), raw_var.index.astype(str)))
    return dict(zip(raw_var.index.astype(str), raw_var.index.astype(str)))


def build_net(adata, gene_sets, min_genes=MIN_GENES_PER_MODULE):
    gene_map = build_raw_gene_symbol_map(adata)
    net_rows = []
    for module, genes in gene_sets.items():
        genes_resolved = [gene_map[g] for g in genes if g in gene_map]
        if len(genes_resolved) < min_genes:
            raise ValueError(f"{module} has too few resolved genes: {genes_resolved}")
        for gene in genes_resolved:
            net_rows.append({"source": module, "target": gene, "weight": 1.0})
    return pd.DataFrame(net_rows)


SCORE_COLS = [
    "classical_z", "lectin_z", "alternative_z", "terminal_z", "receptor_z",
    "regulator_z", "cfhr_z",
    "production_score", "response_score", "activation_index", "net_complement_load",
]

# =============================================================================
# SECTION A: CONFOUND ANALYSIS (marginal) -- are scores driven by technical covariates?
# =============================================================================
# Reviewer request: demonstrate module scores are not primarily driven by
# library size, gene-detection depth, mitochondrial fraction, modality, or
# batch, rather than assuming ULM/z-scoring resolves this on its own.
#
# NOTE ON SAMPLE SIZE: at ~1.4M cells, essentially every correlation will be
# "statistically significant" (p < 0.05) regardless of practical relevance.
# Report and interpret the effect size (rho, eta) here, not the p-value.
#
# NOTE ON MISSINGNESS: ~783 cells have NaN for any module they had zero
# expression in. spearmanr/kruskal default to nan_policy="propagate", which
# returns NaN for the ENTIRE statistic if even one paired value is missing --
# nan_policy="omit" (spearmanr) and explicit .dropna() (kruskal) restrict
# each test to cells with a valid score instead.
obs = pd.read_parquet(OUTPUT_DIR / "complement_scores_obs.parquet")
print(f"Loaded {obs.shape[0]} cells x {obs.shape[1]} obs columns")

CONTINUOUS_COVARIATES = ["nCount_RNA", "nFeature_RNA", "percent.mt"]
CATEGORICAL_COVARIATES = ["experiment_id", "donor_id", "suspension_type"]  # suspension_type ~ modality; rename if your schema differs

confound_rows = []
for score_col in SCORE_COLS:
    for cov in CONTINUOUS_COVARIATES:
        if cov not in obs.columns:
            print(f"Skipping missing covariate column: {cov}")
            continue
        rho, p = spearmanr(obs[score_col], obs[cov], nan_policy="omit")
        confound_rows.append({"score": score_col, "covariate": cov, "type": "continuous", "spearman_rho": rho, "p_value": p})

confound_continuous = pd.DataFrame(confound_rows)
print("\nCorrelation of module/composite scores with continuous technical covariates (sorted by |rho|):")
print(confound_continuous.sort_values("spearman_rho", key=lambda s: s.abs(), ascending=False).to_string(index=False))
confound_continuous.to_csv(VALIDATION_DIR / "confound_correlations_continuous.csv", index=False)

confound_cat_rows = []
for score_col in SCORE_COLS:
    for cov in CATEGORICAL_COVARIATES:
        if cov not in obs.columns:
            print(f"Skipping missing covariate column: {cov}")
            continue
        groups = [g[score_col].dropna().to_numpy() for _, g in obs.groupby(cov, observed=True)]
        groups = [g for g in groups if len(g) > 1]
        if len(groups) < 2:
            continue
        stat, p = kruskal(*groups)
        n_total = sum(len(g) for g in groups)
        k = len(groups)
        eta_sq = max((stat - k + 1) / (n_total - k), 0)
        confound_cat_rows.append({
            "score": score_col, "covariate": cov, "type": "categorical",
            "kruskal_H": stat, "eta_squared_approx": eta_sq, "p_value": p, "n_groups": k,
        })

confound_categorical = pd.DataFrame(confound_cat_rows)
print("\nAssociation of module/composite scores with categorical technical covariates (sorted by eta^2):")
print(confound_categorical.sort_values("eta_squared_approx", ascending=False).to_string(index=False))
confound_categorical.to_csv(VALIDATION_DIR / "confound_associations_categorical.csv", index=False)

# =============================================================================
# SECTION B: MIXED MODEL FOLLOW-UP -- is the batch/donor association technical or biological?
# =============================================================================
# Fits disease + broad cell class as FIXED effects, donor_id and
# experiment_id as CROSSED random effects, and reports variance components.
# If donor/experiment variance fractions here are much smaller than the
# marginal eta^2 in Section A, that supports "batch effect is largely
# explained by disease/cell-type composition" rather than pure technical
# noise; if they remain large, that supports a genuine residual batch effect.
#
# COMPUTATIONAL NOTE: does not scale to the full ~1.4M cells x 308/223
# random-effect levels in reasonable time -- fits on a stratified subsample.
FIXED_EFFECT_COLS = ["disease", "Class"]
GROUP_COL = "donor_id"
CROSSED_COL = "experiment_id"
N_PER_DONOR = 500


def stratified_subsample(df, group_col, n_per_group, seed=0):
    parts = []
    for _, g in df.groupby(group_col, observed=True):
        n = min(len(g), n_per_group)
        parts.append(g.sample(n=n, random_state=seed))
    return pd.concat(parts, axis=0)


def fit_mixed_model(df, score_col, fixed_cols, group_col, crossed_col):
    cols_needed = [score_col, group_col, crossed_col] + fixed_cols
    d = df[cols_needed].dropna().copy()
    if d[score_col].nunique() <= 1 or d.shape[0] < 100:
        return {"score": score_col, "status": "skipped (insufficient data/variance)"}

    # Genuinely crossed random effects (donor_id, experiment_id) rather than
    # nesting one inside the other: donor_id passed as `groups=` would treat
    # experiment_id's variance component as nested within donor -- unidentifiable
    # for donors that only ever appear in one experiment_id, which is what
    # produced the "index 0 is out of bounds" failure on every score. Instead,
    # use a constant dummy group with re_formula="0" (suppressing its own
    # default random intercept) and put BOTH donor_id and experiment_id into
    # vc_formula as independent variance components -- the standard statsmodels
    # pattern for crossed (non-nested) random effects.
    d["_dummy_group"] = 1
    fixed_formula = " + ".join(f"C({c})" for c in fixed_cols)
    full_formula = f"{score_col} ~ {fixed_formula}"
    null_formula = f"{score_col} ~ 1"
    vc = {
        group_col: f"0 + C({group_col})",
        crossed_col: f"0 + C({crossed_col})",
    }

    try:
        full_reml = smf.mixedlm(full_formula, data=d, groups=d["_dummy_group"], re_formula="0", vc_formula=vc).fit(reml=True)
        full_ml = smf.mixedlm(full_formula, data=d, groups=d["_dummy_group"], re_formula="0", vc_formula=vc).fit(reml=False)
        null_ml = smf.mixedlm(null_formula, data=d, groups=d["_dummy_group"], re_formula="0", vc_formula=vc).fit(reml=False)

        lr_stat = 2 * (full_ml.llf - null_ml.llf)
        df_diff = full_ml.df_modelwc - null_ml.df_modelwc
        p_lr = chi2.sf(lr_stat, df_diff) if df_diff > 0 else np.nan

        if len(full_reml.vcomp) < 2:
            return {"score": score_col, "status": f"unexpected vcomp length {len(full_reml.vcomp)} (expected 2)"}

        donor_var = float(full_reml.vcomp[0])       # first vc term = group_col (donor_id)
        exp_var = float(full_reml.vcomp[1])           # second vc term = crossed_col (experiment_id)
        resid_var = float(full_reml.scale)
        total_var = donor_var + exp_var + resid_var

        return {
            "score": score_col,
            "status": "ok" if full_reml.converged else "fit completed but did not fully converge -- interpret with caution",
            "n_cells": d.shape[0],
            "donor_var": donor_var,
            "experiment_var": exp_var,
            "residual_var": resid_var,
            "donor_var_frac": donor_var / total_var if total_var > 0 else np.nan,
            "experiment_var_frac": exp_var / total_var if total_var > 0 else np.nan,
            "residual_var_frac": resid_var / total_var if total_var > 0 else np.nan,
            "fixed_effects_LR_stat": lr_stat,
            "fixed_effects_LR_df": df_diff,
            "fixed_effects_LR_pvalue": p_lr,
        }
    except Exception as e:
        return {"score": score_col, "status": f"model failed: {e}"}


subsample = stratified_subsample(obs, GROUP_COL, N_PER_DONOR, RANDOM_SEED)
print(f"\nSubsampled to {subsample.shape[0]} cells "
      f"({subsample[GROUP_COL].nunique()} donors, {subsample[CROSSED_COL].nunique()} experiments) for LMM fitting")

lmm_results_list = []
for score_col in SCORE_COLS:
    print(f"Fitting mixed model for {score_col}...")
    res = fit_mixed_model(subsample, score_col, FIXED_EFFECT_COLS, GROUP_COL, CROSSED_COL)
    print(f"  -> {res.get('status')}")
    lmm_results_list.append(res)

lmm_results = pd.DataFrame(lmm_results_list)
if "experiment_var_frac" in lmm_results.columns:
    print("\nVariance partitioning (donor / experiment / residual), sorted by experiment_var_frac:")
    print(lmm_results.sort_values("experiment_var_frac", ascending=False).to_string(index=False))
else:
    print("\nNo model produced variance-fraction output -- printing raw results instead:")
    print(lmm_results.to_string(index=False))
lmm_results.to_csv(VALIDATION_DIR / "confound_mixed_model_variance_partitioning.csv", index=False)

# =============================================================================
# SECTION C: ROBUSTNESS -- independent scoring method (AUCell) + leave-one-gene-out
# =============================================================================
# Reviewer request: robustness assessed with an independent scoring method,
# and leave-one-gene-out to determine whether results are driven mainly by
# C3 or other highly expressed genes. Requires raw counts, so loads the full
# h5ad here rather than reusing the scored-obs parquet.
adata = sc.read_h5ad(DATA_DIR / "adata_scvi.h5ad")
base_net = build_net(adata, complement_gene_sets)

# ---- C1: ULM vs AUCell agreement ----
# ULM: scored on raw counts (consistent with 02b convention).
adata_raw = adata.raw.to_adata()
adata_raw.obs = adata.obs
ulm_result = dc.mt.ulm(data=adata_raw, net=base_net, verbose=True, bsize=50000, tmin=MIN_GENES_PER_MODULE)
if ulm_result is not None:
    adata_raw = ulm_result
ulm_scores = dc.pp.get_obsm(adata=adata_raw, key="score_ulm").to_df()
ulm_scores.columns = [f"{c}_ulm" for c in ulm_scores.columns]
ulm_scores = ulm_scores.reindex(adata.obs_names)

# Free the ULM working copy before building another full-size copy for the
# AUCell subset -- by this point adata (original) + adata_raw (ULM) would
# otherwise both be resident when the AUCell subset copy is made, which is
# what killed the process on the previous run (same OOM pattern as the
# earlier decontX Stage 3 crash).
del adata_raw, ulm_result
import gc
gc.collect()

# AUCell: decoupler recommends normalized/log data, not raw counts, unlike ULM.
#
# NOTE: unlike ULM, AUCell's ranking step appears not to respect `bsize` for
# memory purposes -- a run on the full ~1.37M cells was killed (OOM) right
# after preprocessing, before any batched fitting progress appeared, while
# ULM completed fine at the same bsize. AUCell here is a robustness check,
# not the primary result, so this runs on a large random subsample instead
# of the full dataset to keep memory bounded, with a smaller explicit bsize
# than ULM's since AUCell's per-batch footprint appears heavier at the same
# setting.
AUCELL_N_CELLS = min(300_000, adata.n_obs)
aucell_idx = np.random.choice(adata.n_obs, size=AUCELL_N_CELLS, replace=False)
adata_aucell_subset = adata[aucell_idx].copy()

adata_norm = adata_aucell_subset.raw.to_adata()
adata_norm.obs = adata_aucell_subset.obs
sc.pp.normalize_total(adata_norm)
sc.pp.log1p(adata_norm)
# NOTE: this decoupler version's dc.mt.aucell does not accept a `seed` kwarg
# (unlike some documented versions) -- removed rather than guessed at an
# alternate name. If AUCell has any internal tie-breaking randomness, it's
# governed by the global np.random.seed(RANDOM_SEED) set at the top of this
# script, not a per-call argument here.
#
# tmin matches the ULM calls above -- without it, `alternative` (4 genes)
# gets silently dropped by AUCell's own default threshold too, same issue
# we fixed for ULM earlier.
aucell_result = dc.mt.aucell(data=adata_norm, net=base_net, verbose=True, bsize=10000, tmin=MIN_GENES_PER_MODULE)
if aucell_result is not None:
    adata_norm = aucell_result
aucell_scores = dc.pp.get_obsm(adata=adata_norm, key="score_aucell").to_df()
aucell_scores.columns = [f"{c}_aucell" for c in aucell_scores.columns]
aucell_scores = aucell_scores.reindex(adata_aucell_subset.obs_names)

# ULM scores restricted to the same subsampled cells, for a like-for-like comparison
ulm_scores_subset = ulm_scores.reindex(adata_aucell_subset.obs_names)

method_comparison = pd.DataFrame({
    "module": list(complement_gene_sets.keys()),
    "spearman_rho": [
        spearmanr(ulm_scores_subset[f"{m}_ulm"], aucell_scores[f"{m}_aucell"], nan_policy="omit").correlation
        for m in complement_gene_sets
    ],
})
print("\nULM vs AUCell agreement per module (on the AUCell subsample):")
print(method_comparison.to_string(index=False))
method_comparison.to_csv(VALIDATION_DIR / "ulm_vs_aucell_agreement.csv", index=False)

# ---- C2: leave-one-gene-out (LOGO) sensitivity ----
# Run on a subsample for tractability; each gene removal requires a fresh
# ULM fit. Tests whether any single gene (esp. C3) dominates a module's score.
LOGO_N_CELLS = min(50_000, adata.n_obs)
logo_idx = np.random.choice(adata.n_obs, size=LOGO_N_CELLS, replace=False)
adata_logo = adata[logo_idx].copy()

logo_net = build_net(adata_logo, complement_gene_sets)
gene_map = build_raw_gene_symbol_map(adata_logo)

full_logo_raw = adata_logo.raw.to_adata()
full_logo_raw.obs = adata_logo.obs
full_result = dc.mt.ulm(data=full_logo_raw, net=logo_net, verbose=False, bsize=50000, tmin=MIN_GENES_PER_MODULE)
if full_result is not None:
    full_logo_raw = full_result
full_scores = dc.pp.get_obsm(adata=full_logo_raw, key="score_ulm").to_df().reindex(adata_logo.obs_names)

logo_results = []
for module, genes in complement_gene_sets.items():
    genes_resolved = [gene_map[g] for g in genes if g in gene_map]
    if len(genes_resolved) <= MIN_GENES_PER_MODULE:
        print(f"Skipping LOGO for {module}: too few genes to drop one and stay >= min_genes")
        continue
    for gene in genes_resolved:
        reduced_genes = [g for g in genes_resolved if g != gene]
        reduced_net = pd.DataFrame([{"source": module, "target": g, "weight": 1.0} for g in reduced_genes])

        reduced_raw = adata_logo.raw.to_adata()
        reduced_raw.obs = adata_logo.obs
        reduced_result = dc.mt.ulm(data=reduced_raw, net=reduced_net, verbose=False, bsize=50000, tmin=MIN_GENES_PER_MODULE)
        if reduced_result is not None:
            reduced_raw = reduced_result

        if "score_ulm" not in reduced_raw.obsm:
            logo_results.append({"module": module, "gene_dropped": gene, "spearman_rho_vs_full": np.nan,
                                  "note": "module dropped below decoupler's internal threshold without this gene"})
            continue

        reduced_score = dc.pp.get_obsm(adata=reduced_raw, key="score_ulm").to_df().reindex(adata_logo.obs_names)[module]
        rho = spearmanr(full_scores[module], reduced_score, nan_policy="omit").correlation
        logo_results.append({"module": module, "gene_dropped": gene, "spearman_rho_vs_full": rho, "note": ""})

logo_df = pd.DataFrame(logo_results).sort_values(["module", "spearman_rho_vs_full"])

# gene_map is symbol -> Ensembl ID; invert it so the saved/printed table shows
# readable gene symbols (e.g. "C3") instead of raw Ensembl IDs.
id_to_symbol = {v: k for k, v in gene_map.items()}
logo_df.insert(1, "gene_symbol", logo_df["gene_dropped"].map(id_to_symbol))

print("\nLeave-one-gene-out sensitivity (lower rho = module more dependent on that gene):")
print(logo_df.to_string(index=False))
logo_df.to_csv(VALIDATION_DIR / "leave_one_gene_out_sensitivity.csv", index=False)

LOGO_FLAG_THRESHOLD = 0.90
flagged = logo_df[logo_df["spearman_rho_vs_full"] < LOGO_FLAG_THRESHOLD]
if len(flagged):
    print(f"\nGenes whose removal drops Spearman rho below {LOGO_FLAG_THRESHOLD}:")
    print(flagged.to_string(index=False))
else:
    print(f"\nNo single gene's removal dropped Spearman rho below {LOGO_FLAG_THRESHOLD} in any module.")

print(f"\nAll validation outputs saved to {VALIDATION_DIR}")
