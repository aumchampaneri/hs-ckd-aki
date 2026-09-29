# %% COMPLEMENT TRAJECTORY INTEGRATION
# Bridges frozen 04c06/04c07 trajectory results to the complement-focused paper.
# No trajectory, GAM, tradeSeq test, or pseudobulk DE model is refit here.

from __future__ import annotations

from pathlib import Path
import re
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
from scipy.stats import wilcoxon

PROJECT_ROOT = Path.cwd().parent
TRAJECTORY_DIR = PROJECT_ROOT / "data" / "trajectory"
TRADESEQ_DIR = TRAJECTORY_DIR / "tradeseq"
INPUT_MANIFEST_PATH = TRADESEQ_DIR / "trajectory_tradeseq_input_manifest.csv"
TEST_DIR = TRADESEQ_DIR / "results" / "tests"
FIT_DIR = TRADESEQ_DIR / "results" / "fitGAM"
PSEUDOBULK_DIR = PROJECT_ROOT / "outputs" / "pseudobulk_de"
METACELL_DIR = PROJECT_ROOT / "data" / "metacells"
OUT_DIR = TRAJECTORY_DIR / "complement"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FDR_ALPHA = 0.05
N_PSEUDOTIME_BINS = 10
EARLY_MAX = 1 / 3
LATE_MIN = 2 / 3
MIN_DONOR_METACELLS_PER_REGION = 2
MIN_DONORS_PAIRED_TEST = 5

FULL_COUNT_PATHS = {
    "PT": METACELL_DIR / "PT_metacells.h5ad",
    "FIB": METACELL_DIR / "FIB_metacells.h5ad",
    "Glomerular": METACELL_DIR / "Glomerular_metacells.h5ad",
}

# %% CANONICAL COMPLEMENT ANNOTATION
# Scoring modules are mutually exclusive, following the corrected 02b convention.
# CFHR genes are retained as a separate descriptive family because they do not share
# a uniform inhibitory direction with CFH/CFI/CD46/CD55/CD59/SERPING1.
COMPLEMENT_MODULES = {
    "classical": ["C1QA", "C1QB", "C1QC", "C1R", "C1S", "C2", "C4A", "C4B", "C4BPA", "C4BPB"],
    "lectin": ["MBL2", "FCN1", "FCN2", "FCN3", "MASP1", "MASP2", "MASP3"],
    "alternative": ["C3", "CFB", "CFD", "CFP"],
    "terminal": ["C5", "C6", "C7", "C8A", "C8B", "C8G", "C9"],
    "receptor": ["C3AR1", "C5AR1", "C5AR2", "CR1", "CR2", "ITGAM", "ITGAX", "VSIG4"],
    "regulator": ["CFH", "CFI", "CD46", "CD55", "CD59", "SERPING1"],
    "cfhr": ["CFHR1", "CFHR2", "CFHR3", "CFHR4", "CFHR5"],
}

rows = []
for module, genes in COMPLEMENT_MODULES.items():
    for gene in genes:
        rows.append({
            "gene": gene,
            "module": module,
            "scoring_module": module != "cfhr",
            "annotation_note": (
                "descriptive CFHR family; excluded from directional regulator composite"
                if module == "cfhr" else "canonical non-overlapping complement module"
            ),
        })
complement_annotation = pd.DataFrame(rows)
if complement_annotation["gene"].duplicated().any():
    raise RuntimeError("Canonical complement annotation contains duplicated genes.")
complement_annotation.to_csv(OUT_DIR / "canonical_complement_gene_annotation.csv", index=False)
COMPLEMENT_GENES = set(complement_annotation["gene"])

print("PROJECT ROOT", PROJECT_ROOT)
print("COMPLEMENT OUTPUT", OUT_DIR)
print(f"Canonical complement universe: {len(COMPLEMENT_GENES)} genes")

# %% HELPERS
def sanitize(value: str) -> str:
    return "".join(c if c.isalnum() or c in "._-" else "_" for c in str(value))


def bh_adjust(values: pd.Series) -> pd.Series:
    p = pd.to_numeric(values, errors="coerce").to_numpy(float)
    out = np.full(len(p), np.nan)
    ok = np.isfinite(p) & (p >= 0) & (p <= 1)
    if not ok.any():
        return pd.Series(out, index=values.index)
    pv = p[ok]
    order = np.argsort(pv, kind="mergesort")
    ranked = pv[order]
    q = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    restored = np.empty(len(ranked))
    restored[order] = np.minimum(q, 1.0)
    out[np.where(ok)[0]] = restored
    return pd.Series(out, index=values.index)


def gene_symbol_series(var: pd.DataFrame) -> pd.Series:
    for col in ["gene_symbol", "feature_name", "gene_name", "symbol"]:
        if col in var.columns:
            return var[col].astype(str)
    return pd.Series(var.index.astype(str), index=var.index)


def weighted_mean(x, w):
    x = np.asarray(x, float)
    w = np.asarray(w, float)
    ok = np.isfinite(x) & np.isfinite(w) & (w > 0)
    return np.average(x[ok], weights=w[ok]) if ok.any() else np.nan


def find_bundle_file(bundle_dir: Path, suffix: str) -> Path:
    hits = sorted(bundle_dir.glob(f"*{suffix}"))
    if len(hits) != 1:
        raise RuntimeError(f"Expected exactly one *{suffix} in {bundle_dir}; found {hits}")
    return hits[0]


def infer_lineage_columns(df: pd.DataFrame, prefix: str) -> list[str]:
    cols = [c for c in df.columns if str(c).startswith(prefix)]
    def key(c):
        m = re.search(r"(\d+)$", str(c))
        return int(m.group(1)) if m else str(c)
    return sorted(cols, key=key)


# %% VALIDATE FROZEN INPUTS
for p in [INPUT_MANIFEST_PATH, FIT_DIR / "trajectory_fitGAM_manifest.csv", TEST_DIR / "trajectory_tradeseq_test_summary.csv"]:
    if not p.exists() or p.stat().st_size == 0:
        raise FileNotFoundError(p)

input_manifest = pd.read_csv(INPUT_MANIFEST_PATH)
fit_manifest = pd.read_csv(FIT_DIR / "trajectory_fitGAM_manifest.csv")
test_summary = pd.read_csv(TEST_DIR / "trajectory_tradeseq_test_summary.csv")

pairs = list(zip(input_manifest["compartment"].astype(str), input_manifest["trajectory"].astype(str)))
print("Frozen trajectories:")
for x in pairs:
    print(" ", x)

# %% 1. COMPLETE COMPLEMENT FIT / TEST AUDIT + 04c07 STATISTICS
# Preserve 04c07 q-values exactly. NA global tests remain NA and are never treated as negative.
audit_records = []
test_long_parts = []

for row in input_manifest.itertuples(index=False):
    compartment, trajectory = str(row.compartment), str(row.trajectory)
    bundle = sanitize(f"{compartment}__{trajectory}")
    test_out = TEST_DIR / bundle
    bundle_dir = Path(row.bundle_dir)

    # 04c06/07 retain the raw feature identifiers used by the full-count object
    # (typically Ensembl IDs), whereas the manuscript complement universe is
    # defined by gene symbols. Resolve feature_id -> symbol from the matching
    # 04a full-count object before any complement filtering. Keep both IDs.
    full_var = sc.read_h5ad(FULL_COUNT_PATHS[compartment], backed="r").var.copy()
    full_symbols = gene_symbol_series(full_var)
    feature_to_symbol = dict(zip(full_var.index.astype(str), full_symbols.astype(str)))

    genes_df = pd.read_csv(row.genes_path)
    gene_col = "gene" if "gene" in genes_df.columns else genes_df.columns[0]
    fitted_feature_ids = set(genes_df[gene_col].astype(str))
    fitted_symbols = {feature_to_symbol.get(x, x) for x in fitted_feature_ids}

    # Full expression-filter audit if the historical bundle contains it.
    filter_hits = sorted(bundle_dir.glob("*gene_filter.csv"))
    filter_df = pd.read_csv(filter_hits[0]) if len(filter_hits) == 1 else None
    if filter_df is not None:
        filter_df["feature_id"] = filter_df["gene"].astype(str)
        filter_df["gene"] = filter_df["feature_id"].map(feature_to_symbol).fillna(filter_df["feature_id"])
        filter_lookup = filter_df.drop_duplicates("gene").set_index("gene")
    else:
        filter_lookup = None

    qc = pd.read_csv(test_out / "fit_qc_combined.csv")
    qc["feature_id"] = qc["gene"].astype(str)
    qc["gene"] = qc["feature_id"].map(feature_to_symbol).fillna(qc["feature_id"])
    conv = qc.drop_duplicates("gene").set_index("gene")["converged"].fillna(False).astype(bool).to_dict()

    n_resolved = len(COMPLEMENT_GENES.intersection(fitted_symbols))
    if n_resolved == 0:
        raise RuntimeError(
            f"No canonical complement symbols resolved for {compartment}/{trajectory}; "
            "check feature_id -> gene-symbol mapping in the full-count .var metadata."
        )
    print(f"  {compartment}/{trajectory}: resolved {n_resolved}/{len(COMPLEMENT_GENES)} complement genes in tradeSeq matrix")

    for ann in complement_annotation.itertuples(index=False):
        gene = ann.gene
        rec = {
            "compartment": compartment, "trajectory": trajectory,
            "gene": gene, "module": ann.module,
            "in_tradeseq_filtered_matrix": gene in fitted_symbols,
            "fit_converged": bool(conv.get(gene, False)),
        }
        if filter_lookup is not None and gene in filter_lookup.index:
            fr = filter_lookup.loc[gene]
            rec["total_counts"] = fr.get("total_counts", np.nan)
            rec["n_nonzero_metacells"] = fr.get("n_nonzero_metacells", np.nan)
            rec["keep_tradeseq"] = fr.get("keep_tradeseq", np.nan)
        else:
            rec["total_counts"] = np.nan
            rec["n_nonzero_metacells"] = np.nan
            rec["keep_tradeseq"] = gene in fitted_symbols
        audit_records.append(rec)

    for test_name in ["association", "start_vs_end", "diff_end", "pattern"]:
        path = test_out / f"{test_name}_combined.csv"
        tab = pd.read_csv(path)
        tab["feature_id"] = tab["gene"].astype(str)
        tab["gene"] = tab["feature_id"].map(feature_to_symbol).fillna(tab["feature_id"])
        tab = tab[tab["gene"].isin(COMPLEMENT_GENES)].copy()
        if tab.empty:
            continue
        id_cols = {"compartment", "trajectory", "batch_id", "gene"}
        pcols = [c for c in tab.columns if c.lower().startswith("pvalue")]
        for pcol in pcols:
            qcol = f"padj__{pcol}"
            # Pair statistic column by numeric suffix when present; keep all raw table columns separately below.
            suffix = pcol[len("pvalue"):]
            wald_candidates = [c for c in tab.columns if c.lower().startswith(("wald", "stat")) and c.endswith(suffix)]
            stat_col = wald_candidates[0] if wald_candidates else None
            # startVsEndTest uses pvalue_lineageN / waldStat_lineageN and
            # stores the model-derived endpoint effect separately as logFClineageN.
            effect_col = None
            if test_name == "start_vs_end" and suffix.startswith("_lineage"):
                lineage_token = suffix.removeprefix("_")  # lineageN
                candidate = f"logFC{lineage_token}"
                if candidate in tab.columns:
                    effect_col = candidate

            selected = ["feature_id", "gene", pcol, qcol]
            if stat_col:
                selected.append(stat_col)
            if effect_col:
                selected.append(effect_col)
            tmp = tab[selected].copy()
            tmp.columns = ["feature_id", "gene", "p_value", "q_value"] + (["statistic"] if stat_col else []) + (["effect"] if effect_col else [])
            tmp["compartment"] = compartment
            tmp["trajectory"] = trajectory
            tmp["test"] = test_name
            tmp["contrast"] = pcol
            tmp["test_available"] = pd.to_numeric(tmp["p_value"], errors="coerce").notna()
            tmp["fdr_lt_0_05"] = pd.to_numeric(tmp["q_value"], errors="coerce") < FDR_ALPHA
            test_long_parts.append(tmp)

complement_audit = pd.DataFrame(audit_records)
complement_audit.to_csv(OUT_DIR / "complement_fit_test_audit.csv", index=False)

if not test_long_parts:
    raise RuntimeError(
        "No complement rows were extracted from 04c07 after identifier resolution. "
        "Inspect feature_id/gene-symbol mapping before proceeding."
    )
complement_tests_long = pd.concat(test_long_parts, ignore_index=True)
complement_tests_long = complement_tests_long.merge(
    complement_annotation[["gene", "module"]], on="gene", how="left"
)
complement_tests_long.to_csv(OUT_DIR / "complement_tradeseq_tests_long.csv", index=False)

# Availability/significance summary makes global-test missingness explicit.
test_availability = (
    complement_tests_long.groupby(["compartment", "trajectory", "test", "contrast"], observed=True)
    .agg(n_complement_rows=("gene", "size"), n_test_available=("test_available", "sum"), n_fdr_lt_0_05=("fdr_lt_0_05", "sum"))
    .reset_index()
)
test_availability.to_csv(OUT_DIR / "complement_tradeseq_test_availability.csv", index=False)

# %% 2. OBSERVED COMPLEMENT EXPRESSION ALONG FROZEN PSEUDOTIME
# Descriptive effect layer: log1p(counts per 10k) from full 04a metacell counts,
# summarized using frozen Slingshot pseudotime and lineage weights.
curve_records = []
shape_records = []
donor_delta_records = []

for row in input_manifest.itertuples(index=False):
    compartment, trajectory = str(row.compartment), str(row.trajectory)
    print(f"\nEXPRESSION SHAPE: {compartment} / {trajectory}")
    # The validated 04c06 input manifest does not require a separate
    # *_metacells.csv checkpoint. Metacell identity/order is encoded directly
    # in the row names of the frozen pseudotime and cell-weight matrices; this
    # is also the exact order used by fitGAM. Reuse it rather than inventing a
    # second metadata dependency.
    pt = pd.read_csv(row.pseudotime_path, index_col=0)
    wt = pd.read_csv(row.cellweights_path, index_col=0)
    pt.index = pt.index.astype(str)
    wt.index = wt.index.astype(str)
    if pt.index.tolist() != wt.index.tolist():
        raise RuntimeError(f"Pseudotime/weight metacell order mismatch for {compartment}/{trajectory}")
    metacells = pt.index.tolist()
    if len(metacells) != int(row.n_metacells):
        raise RuntimeError(
            f"Frozen metacell count mismatch for {compartment}/{trajectory}: "
            f"manifest={int(row.n_metacells)}, pseudotime={len(metacells)}"
        )
    if pt.shape != wt.shape:
        raise RuntimeError("Pseudotime/weight shape mismatch")

    lineage_df = pd.read_csv(row.lineages_path)
    if "lineage" in lineage_df.columns:
        lineage_names = lineage_df["lineage"].astype(str).tolist()
    else:
        lineage_names = [str(x) for x in lineage_df.iloc[:, 0].tolist()]
    if len(lineage_names) != pt.shape[1]:
        lineage_names = [str(i + 1) for i in range(pt.shape[1])]

    adata = sc.read_h5ad(FULL_COUNT_PATHS[compartment])
    adata.obs_names = adata.obs_names.astype(str)
    missing = sorted(set(metacells) - set(adata.obs_names))
    if missing:
        raise RuntimeError(f"Missing metacells in full-count object: {missing[:5]}")
    sub = adata[metacells]
    counts = sub.layers["counts"] if "counts" in sub.layers else sub.X
    if not sp.issparse(counts):
        counts = sp.csr_matrix(counts)
    counts = counts.tocsr()
    lib = np.asarray(counts.sum(axis=1)).ravel()
    lib_safe = np.where(lib > 0, lib, 1.0)

    symbols = gene_symbol_series(sub.var)
    symbol_to_idx = {}
    for i, g in enumerate(symbols.astype(str)):
        symbol_to_idx.setdefault(g, i)

    donor_col = "donor_id"
    if donor_col not in sub.obs:
        raise RuntimeError(f"{donor_col} missing from {compartment} metacells")
    donors = sub.obs[donor_col].astype(str).to_numpy()

    for gene in complement_annotation["gene"]:
        if gene not in symbol_to_idx:
            continue
        gi = symbol_to_idx[gene]
        raw = np.asarray(counts[:, gi].toarray()).ravel()
        expr = np.log1p(raw / lib_safe * 1e4)

        for j, lineage in enumerate(lineage_names):
            p = pd.to_numeric(pt.iloc[:, j], errors="coerce").to_numpy(float)
            w = pd.to_numeric(wt.iloc[:, j], errors="coerce").to_numpy(float)
            supported = np.isfinite(p) & np.isfinite(w) & (w > 0)
            if supported.sum() < 3:
                continue
            pmin, pmax = np.nanmin(p[supported]), np.nanmax(p[supported])
            pscaled = np.full(len(p), np.nan)
            if pmax > pmin:
                pscaled[supported] = (p[supported] - pmin) / (pmax - pmin)
            else:
                pscaled[supported] = 0.0

            # Fixed-width bins on scaled pseudotime; no data-driven cutpoint tuning.
            edges = np.linspace(0, 1, N_PSEUDOTIME_BINS + 1)
            bin_means = []
            for b in range(N_PSEUDOTIME_BINS):
                lo, hi = edges[b], edges[b + 1]
                mask = supported & (pscaled >= lo) & ((pscaled < hi) if b < N_PSEUDOTIME_BINS - 1 else (pscaled <= hi))
                val = weighted_mean(expr[mask], w[mask]) if mask.any() else np.nan
                bin_means.append(val)
                curve_records.append({
                    "compartment": compartment, "trajectory": trajectory, "lineage": lineage,
                    "gene": gene, "pseudotime_bin": b + 1,
                    "pseudotime_midpoint": (lo + hi) / 2,
                    "weighted_mean_log1p_cp10k": val,
                    "n_metacells": int(mask.sum()),
                    "sum_lineage_weight": float(np.nansum(w[mask])) if mask.any() else 0.0,
                })

            early = supported & (pscaled <= EARLY_MAX)
            late = supported & (pscaled >= LATE_MIN)
            early_mean = weighted_mean(expr[early], w[early])
            late_mean = weighted_mean(expr[late], w[late])
            delta = late_mean - early_mean if np.isfinite(early_mean) and np.isfinite(late_mean) else np.nan
            arr = np.asarray(bin_means, float)
            peak_bin = int(np.nanargmax(arr) + 1) if np.isfinite(arr).any() else np.nan
            trough_bin = int(np.nanargmin(arr) + 1) if np.isfinite(arr).any() else np.nan
            if np.isfinite(delta):
                direction = "increasing" if delta > 0 else ("decreasing" if delta < 0 else "stable")
            else:
                direction = "unavailable"
            # Descriptive transient flag only; inference remains tradeSeq association/pattern tests.
            finite = arr[np.isfinite(arr)]
            transient = False
            if len(finite) >= 3 and np.isfinite(early_mean) and np.isfinite(late_mean):
                interior = arr[1:-1]
                if np.isfinite(interior).any():
                    imax = np.nanmax(interior); imin = np.nanmin(interior)
                    transient = (imax > max(early_mean, late_mean)) or (imin < min(early_mean, late_mean))

            # Paired donor early-vs-late confirmation.
            donor_deltas = []
            for donor in np.unique(donors):
                dm = donors == donor
                em = dm & early
                lm = dm & late
                if em.sum() < MIN_DONOR_METACELLS_PER_REGION or lm.sum() < MIN_DONOR_METACELLS_PER_REGION:
                    continue
                e = weighted_mean(expr[em], w[em]); l = weighted_mean(expr[lm], w[lm])
                if np.isfinite(e) and np.isfinite(l):
                    donor_deltas.append(l - e)
            if len(donor_deltas) >= MIN_DONORS_PAIRED_TEST and np.any(np.asarray(donor_deltas) != 0):
                donor_p = wilcoxon(donor_deltas, alternative="two-sided").pvalue
            else:
                donor_p = np.nan

            shape_records.append({
                "compartment": compartment, "trajectory": trajectory, "lineage": lineage, "gene": gene,
                "n_supported_metacells": int(supported.sum()),
                "n_supported_donors": int(pd.Series(donors[supported]).nunique()),
                "early_mean_log1p_cp10k": early_mean,
                "late_mean_log1p_cp10k": late_mean,
                "late_minus_early": delta,
                "direction_descriptive": direction,
                "peak_bin": peak_bin, "trough_bin": trough_bin,
                "transient_shape_descriptive": transient,
                "n_donors_with_early_late": len(donor_deltas),
                "donor_median_late_minus_early": np.median(donor_deltas) if donor_deltas else np.nan,
                "donor_wilcoxon_p": donor_p,
            })

    del adata, sub, counts

curve_df = pd.DataFrame(curve_records)
shape_df = pd.DataFrame(shape_records)

# BH donor confirmation within trajectory x lineage across the prespecified complement universe.
shape_df["donor_wilcoxon_q"] = np.nan
for _, idx in shape_df.groupby(["compartment", "trajectory", "lineage"], observed=True).groups.items():
    shape_df.loc[idx, "donor_wilcoxon_q"] = bh_adjust(shape_df.loc[idx, "donor_wilcoxon_p"]).to_numpy()
shape_df["donor_supported_fdr_0_05"] = shape_df["donor_wilcoxon_q"] < FDR_ALPHA

curve_df.to_csv(OUT_DIR / "complement_pseudotime_binned_expression.csv", index=False)
shape_df.to_csv(OUT_DIR / "complement_pseudotime_shape_summary.csv", index=False)

# %% 3. EXISTING 03a DISEASE-DE INTEGRATION (DESCRIPTIVE JOIN ONLY)
# No DE model is rerun. These are the existing global donor-pseudobulk contrasts.
disease_parts = []
for comparison in ["ckd_vs_normal", "aki_vs_normal", "aki_vs_ckd"]:
    path = PSEUDOBULK_DIR / f"pseudobulk_de_{comparison}.csv"
    if not path.exists():
        print(f"NOTE: missing existing disease DE table: {path}")
        continue
    d = pd.read_csv(path)
    symbol_col = next((c for c in ["gene_symbol", "gene", "symbol", "feature_name"] if c in d.columns), None)
    if symbol_col is None:
        print(f"NOTE: cannot identify gene-symbol column in {path.name}; skipping")
        continue
    d["gene"] = d[symbol_col].astype(str)
    d = d[d["gene"].isin(COMPLEMENT_GENES)].copy()
    d["comparison_id_join"] = comparison
    keep = [c for c in [
        "gene", "comparison_id_join", "comparison", "log2FoldChange", "log2FoldChange_shrunk",
        "pvalue", "padj", "deseq2_p_value", "deseq2_q_value", "is_de_primary_q_0_05",
        "n_donors_group1", "n_donors_group2", "n_cells_group1", "n_cells_group2"
    ] if c in d.columns]
    disease_parts.append(d[keep])

if disease_parts:
    disease_df = pd.concat(disease_parts, ignore_index=True)
else:
    disease_df = pd.DataFrame(columns=["gene", "comparison_id_join"])
disease_df.to_csv(OUT_DIR / "complement_existing_disease_de.csv", index=False)

# %% 4. MANUSCRIPT-READY INTEGRATED TABLE
# Lineage-specific association columns are pvalue_1...N in 04c07. Map them by
# lineage order from the frozen input. Global pvalue is retained separately and
# may be NA; NA never means non-significant.
assoc = complement_tests_long[complement_tests_long["test"] == "association"].copy()
start = complement_tests_long[complement_tests_long["test"] == "start_vs_end"].copy()

integrated_parts = []
for row in input_manifest.itertuples(index=False):
    compartment, trajectory = str(row.compartment), str(row.trajectory)
    lineages_df = pd.read_csv(row.lineages_path)
    lineage_names = lineages_df["lineage"].astype(str).tolist() if "lineage" in lineages_df else lineages_df.iloc[:, 0].astype(str).tolist()
    subshape = shape_df[(shape_df.compartment == compartment) & (shape_df.trajectory == trajectory)].copy()
    for i, lineage in enumerate(lineage_names, start=1):
        base = subshape[subshape["lineage"].astype(str) == str(lineage)].copy()
        if base.empty:
            continue
        a = assoc[(assoc.compartment == compartment) & (assoc.trajectory == trajectory) & (assoc.contrast == f"pvalue_{i}")][["gene", "p_value", "q_value"]].rename(columns={"p_value":"association_p", "q_value":"association_q"})
        s = start[(start.compartment == compartment) & (start.trajectory == trajectory) & (start.contrast == f"pvalue_lineage{i}")][["gene", "p_value", "q_value", "effect"]].rename(columns={"p_value":"start_end_p", "q_value":"start_end_q", "effect":"start_end_logFC"})
        base = base.merge(a, on="gene", how="left").merge(s, on="gene", how="left")
        base = base.merge(complement_annotation[["gene", "module"]], on="gene", how="left")
        base["association_fdr_0_05"] = base["association_q"] < FDR_ALPHA
        base["start_end_fdr_0_05"] = base["start_end_q"] < FDR_ALPHA
        integrated_parts.append(base)

integrated = pd.concat(integrated_parts, ignore_index=True)
integrated.to_csv(OUT_DIR / "complement_trajectory_integrated.csv", index=False)

# Compact priority table: statistical trajectory association + effect description + donor confirmation.
priority = integrated.copy()
priority["priority_tier"] = np.select(
    [
        priority["association_fdr_0_05"].fillna(False) & priority["start_end_fdr_0_05"].fillna(False) & priority["donor_supported_fdr_0_05"].fillna(False),
        priority["association_fdr_0_05"].fillna(False) & (priority["start_end_fdr_0_05"].fillna(False) | priority["donor_supported_fdr_0_05"].fillna(False)),
        priority["association_fdr_0_05"].fillna(False),
    ],
    ["trajectory+direction+donor", "trajectory+one_confirmation", "trajectory_only"],
    default="not_trajectory_significant",
)
priority.to_csv(OUT_DIR / "complement_trajectory_priority_table.csv", index=False)

# %% 5. QC / MANIFEST SUMMARY
summary = (
    integrated.groupby(["compartment", "trajectory", "lineage"], observed=True)
    .agg(
        n_complement_genes=("gene", "nunique"),
        n_association_available=("association_p", lambda x: int(pd.to_numeric(x, errors="coerce").notna().sum())),
        n_association_fdr_0_05=("association_fdr_0_05", "sum"),
        n_start_end_available=("start_end_p", lambda x: int(pd.to_numeric(x, errors="coerce").notna().sum())),
        n_start_end_fdr_0_05=("start_end_fdr_0_05", "sum"),
        n_donor_tests_available=("donor_wilcoxon_p", lambda x: int(pd.to_numeric(x, errors="coerce").notna().sum())),
        n_donor_supported_fdr_0_05=("donor_supported_fdr_0_05", "sum"),
    )
    .reset_index()
)
summary.to_csv(OUT_DIR / "complement_trajectory_summary.csv", index=False)

# Explicit denominator audit: canonical genes absent from observed shape summaries.
shape_genes = set(shape_df["gene"].astype(str))
missing_shape_genes = sorted(COMPLEMENT_GENES - shape_genes)
print(f"\nCanonical genes absent from observed-expression shape summaries ({len(missing_shape_genes)}): {missing_shape_genes}")

print("\n" + "=" * 100)
print("04c08 COMPLEMENT TRAJECTORY INTEGRATION COMPLETE")
print("=" * 100)
print(summary.to_string(index=False))
print("\nSaved:")
for name in [
    "canonical_complement_gene_annotation.csv",
    "complement_fit_test_audit.csv",
    "complement_tradeseq_tests_long.csv",
    "complement_tradeseq_test_availability.csv",
    "complement_pseudotime_binned_expression.csv",
    "complement_pseudotime_shape_summary.csv",
    "complement_existing_disease_de.csv",
    "complement_trajectory_integrated.csv",
    "complement_trajectory_priority_table.csv",
    "complement_trajectory_summary.csv",
]:
    print(" ", OUT_DIR / name)
print("\nNo trajectories, GAMs, tradeSeq tests, or pseudobulk DE models were refit.")
