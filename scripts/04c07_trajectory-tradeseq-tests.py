# %% IMPORTS / PROJECT PATHS
from __future__ import annotations

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

from pathlib import Path
import shutil
import subprocess
import numpy as np
import pandas as pd

PROJECT_ROOT = Path.cwd().parent
TRAJECTORY_DIR = PROJECT_ROOT / "data" / "trajectory"
TRADESEQ_DIR = TRAJECTORY_DIR / "tradeseq"
TRADESEQ_RESULT_DIR = TRADESEQ_DIR / "results"
TRADESEQ_FIT_DIR = TRADESEQ_RESULT_DIR / "fitGAM"
TRADESEQ_TEST_DIR = TRADESEQ_RESULT_DIR / "tests"
TRADESEQ_TEST_DIR.mkdir(parents=True, exist_ok=True)

print("PROJECT ROOT")
print(PROJECT_ROOT)
print("\nTRADESEQ TEST OUTPUT")
print(TRADESEQ_TEST_DIR)

# %% FROZEN TEST SPECIFICATION
# fitGAM is frozen. This script never refits a GAM and never changes nknots,
# pseudotime, cell weights, normalization, or the expression-only gene filter.
TRADESEQ_TEST_SEED = 20260923
TRADESEQ_OVERWRITE_TEST_BATCHES = False

# Primary discovery: association with pseudotime.
# Directional interpretation: start vs end.
# Secondary branch-comparison tests: endpoint and full-pattern differences.
TESTS = {
    "association": {"primary": True},
    "start_vs_end": {"primary": False},
    "diff_end": {"primary": False},
    "pattern": {"primary": False},
}

# BH is applied after concatenating ALL batches. Each p-value column within a
# trajectory x test is one predefined family (e.g. global or a lineage/pairwise
# contrast). Batch boundaries are never multiple-testing families.
FDR_ALPHA = 0.05

# %% HELPERS
def sanitize_filename(value: str) -> str:
    return "".join(c if c.isalnum() or c in "._-" else "_" for c in str(value))


def bh_adjust(pvalues: pd.Series) -> pd.Series:
    """Benjamini-Hochberg adjustment preserving NA and original row order."""
    p = pd.to_numeric(pvalues, errors="coerce").to_numpy(dtype=float)
    out = np.full(len(p), np.nan, dtype=float)
    valid = np.isfinite(p) & (p >= 0) & (p <= 1)
    if not valid.any():
        return pd.Series(out, index=pvalues.index)
    pv = p[valid]
    order = np.argsort(pv, kind="mergesort")
    ranked = pv[order]
    m = len(ranked)
    q = ranked * m / np.arange(1, m + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.minimum(q, 1.0)
    restored = np.empty(m, dtype=float)
    restored[order] = q
    out[np.where(valid)[0]] = restored
    return pd.Series(out, index=pvalues.index)


def require_nonempty(path: Path, label: str) -> None:
    if not path.exists() or path.stat().st_size == 0:
        raise FileNotFoundError(f"Missing/empty {label}: {path}")


def batch_test_paths(batch_test_dir: Path) -> dict[str, Path]:
    return {
        "qc": batch_test_dir / "fit_qc.csv",
        "association": batch_test_dir / "association.csv",
        "start_vs_end": batch_test_dir / "start_vs_end.csv",
        "diff_end": batch_test_dir / "diff_end.csv",
        "pattern": batch_test_dir / "pattern.csv",
        "stdout": batch_test_dir / "stdout.txt",
        "stderr": batch_test_dir / "stderr.txt",
        "done": batch_test_dir / "DONE",
    }


def valid_test_checkpoint(batch_test_dir: Path, expected_genes: list[str]) -> bool:
    paths = batch_test_paths(batch_test_dir)
    required = ["qc", "association", "start_vs_end", "diff_end", "pattern", "done"]
    if not all(paths[k].exists() and paths[k].stat().st_size > 0 for k in required):
        return False
    try:
        qc = pd.read_csv(paths["qc"])
    except Exception:
        return False
    if "gene" not in qc.columns or qc["gene"].astype(str).tolist() != expected_genes:
        return False
    return True


# %% VALIDATE FROZEN 04c06 MANIFESTS
fit_manifest_path = TRADESEQ_FIT_DIR / "trajectory_fitGAM_manifest.csv"
batch_manifest_path = TRADESEQ_FIT_DIR / "trajectory_fitGAM_batch_manifest.csv"
require_nonempty(fit_manifest_path, "04c06 fitGAM trajectory manifest")
require_nonempty(batch_manifest_path, "04c06 fitGAM batch manifest")

fit_manifest = pd.read_csv(fit_manifest_path)
batch_manifest = pd.read_csv(batch_manifest_path)

required_fit_cols = {"compartment", "trajectory", "n_genes", "n_metacells", "n_lineages", "nknots"}
required_batch_cols = {
    "compartment", "trajectory", "batch_id", "gene_start", "gene_end", "n_genes",
    "fit_path", "rowdata_path", "genes_path",
}
if missing := required_fit_cols - set(fit_manifest.columns):
    raise RuntimeError(f"fitGAM trajectory manifest missing columns: {sorted(missing)}")
if missing := required_batch_cols - set(batch_manifest.columns):
    raise RuntimeError(f"fitGAM batch manifest missing columns: {sorted(missing)}")

expected_pairs = {
    ("PT", "PT"),
    ("FIB", "contractile"),
    ("FIB", "outer_medullary"),
    ("Glomerular", "podocyte_PEC"),
    ("Glomerular", "endothelial_mesangial"),
}
observed_pairs = set(zip(fit_manifest["compartment"], fit_manifest["trajectory"]))
if observed_pairs != expected_pairs:
    raise RuntimeError(f"Unexpected frozen trajectory set: {sorted(observed_pairs)}")

if not (pd.to_numeric(fit_manifest["nknots"], errors="coerce") == 8).all():
    raise RuntimeError("04c06 manifest does not contain frozen nknots=8 for every trajectory.")

for b in batch_manifest.itertuples(index=False):
    require_nonempty(Path(b.fit_path), f"fit RDS {b.compartment}/{b.trajectory}/{b.batch_id}")
    require_nonempty(Path(b.rowdata_path), f"fit QC {b.compartment}/{b.trajectory}/{b.batch_id}")
    require_nonempty(Path(b.genes_path), f"fit genes {b.compartment}/{b.trajectory}/{b.batch_id}")

print("\nFROZEN 04c06 fitGAM MANIFEST")
print(fit_manifest[["compartment", "trajectory", "n_genes", "n_metacells", "n_lineages", "nknots"]].to_string(index=False))
print(f"\nValidated {len(batch_manifest):,} fitGAM batch checkpoints.")

# %% R ENVIRONMENT
rscript_path = shutil.which("Rscript")
if rscript_path is None:
    raise RuntimeError("Rscript not found on PATH. Run this script inside the project pixi environment.")

r_env_cmd = [
    rscript_path,
    "-e",
    (
        'cat("R_VERSION=", R.version.string, "\\n", sep=""); '
        'cat("TRADESEQ_VERSION=", as.character(packageVersion("tradeSeq")), "\\n", sep=""); '
        'cat("SCE_AVAILABLE=", requireNamespace("SingleCellExperiment", quietly=TRUE), "\\n", sep="")'
    ),
]
r_env = subprocess.run(r_env_cmd, capture_output=True, text=True)
if r_env.returncode != 0:
    raise RuntimeError(r_env.stderr)
print("\nR ENVIRONMENT")
print(r_env.stdout)

# %% WRITE BATCH TEST RUNNER
r_test_script = TRADESEQ_TEST_DIR / "run_tradeseq_tests_batch.R"
r_test_script.write_text(r'''suppressPackageStartupMessages({
  library(tradeSeq)
  library(SingleCellExperiment)
  library(SummarizedExperiment)
})

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 4L) {
  stop("Expected 4 args: fit_rds output_dir batch_id seed")
}
fit_path <- args[[1]]
out_dir <- args[[2]]
batch_id <- args[[3]]
seed <- as.integer(args[[4]])
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
set.seed(seed)

sce <- readRDS(fit_path)
genes <- rownames(sce)
if (is.null(genes) || anyDuplicated(genes)) stop("Missing or duplicated SCE rownames")

# Keep the complete fitted-gene denominator in QC, but test only converged fits.
ts_rd <- SummarizedExperiment::rowData(sce)$tradeSeq
converged <- tryCatch(as.logical(ts_rd$converged), error = function(e) rep(NA, length(genes)))
if (length(converged) != length(genes)) stop("Invalid tradeSeq convergence vector")
converged[is.na(converged)] <- FALSE

qc <- data.frame(
  gene = genes,
  converged = converged,
  batch_id = batch_id,
  stringsAsFactors = FALSE
)
write.csv(qc, file.path(out_dir, "fit_qc.csv"), row.names = FALSE, quote = TRUE)

sce_test <- sce[converged, , drop = FALSE]
if (nrow(sce_test) == 0L) stop("No converged genes in batch")

write_result <- function(x, filename) {
  x <- as.data.frame(x)
  x <- cbind(gene = rownames(x), x, stringsAsFactors = FALSE)
  rownames(x) <- NULL
  write.csv(x, file.path(out_dir, filename), row.names = FALSE, quote = TRUE)
}

# Primary: any association between average expression and pseudotime.
association <- tradeSeq::associationTest(
  sce_test, global = TRUE, lineages = TRUE, l2fc = 0
)
write_result(association, "association.csv")

# Directional interpretation: beginning vs endpoint of each lineage.
start_end <- tradeSeq::startVsEndTest(
  sce_test, global = TRUE, lineages = TRUE, l2fc = 0
)
write_result(start_end, "start_vs_end.csv")

# Secondary branch comparisons. These are not part of the primary dynamic-gene
# definition and will be interpreted together with frozen lineage robustness.
diff_end <- tradeSeq::diffEndTest(
  sce_test, global = TRUE, pairwise = TRUE, l2fc = 0
)
write_result(diff_end, "diff_end.csv")

pattern <- tradeSeq::patternTest(
  sce_test, global = TRUE, pairwise = TRUE, l2fc = 0
)
write_result(pattern, "pattern.csv")

cat("Batch", batch_id, "COMPLETE\n")
cat("Genes:", length(genes), "\n")
cat("Converged/tested:", sum(converged), "\n")
''')
print("Saved R test runner:")
print(r_test_script)

# %% RUN / RESUME BATCH TESTS
batch_records: list[dict] = []

for tr in fit_manifest.itertuples(index=False):
    compartment = str(tr.compartment)
    trajectory = str(tr.trajectory)
    bundle = sanitize_filename(f"{compartment}__{trajectory}")
    trajectory_test_dir = TRADESEQ_TEST_DIR / bundle
    batches_test_dir = trajectory_test_dir / "batches"
    batches_test_dir.mkdir(parents=True, exist_ok=True)

    sub = batch_manifest.loc[
        (batch_manifest["compartment"] == compartment)
        & (batch_manifest["trajectory"] == trajectory)
    ].copy()
    sub = sub.sort_values(["gene_start", "gene_end"]).reset_index(drop=True)

    if int(sub["n_genes"].sum()) != int(tr.n_genes):
        raise RuntimeError(f"Batch gene coverage mismatch for {compartment}/{trajectory}")

    print("\n" + "=" * 100)
    print(f"TRADESEQ TESTS: {compartment} / {trajectory}")
    print("=" * 100)
    print(
        f"genes={int(tr.n_genes):,}; lineages={int(tr.n_lineages)}; "
        f"batches={len(sub):,}; association=PRIMARY"
    )

    for b in sub.itertuples(index=False):
        genes = pd.read_csv(b.genes_path)["gene"].astype(str).tolist()
        if len(genes) != int(b.n_genes):
            raise RuntimeError(f"Gene checkpoint mismatch: {compartment}/{trajectory}/{b.batch_id}")

        out_dir = batches_test_dir / str(b.batch_id)
        out_dir.mkdir(parents=True, exist_ok=True)
        paths = batch_test_paths(out_dir)

        if TRADESEQ_OVERWRITE_TEST_BATCHES:
            for p in paths.values():
                if p.exists():
                    p.unlink()

        if valid_test_checkpoint(out_dir, genes):
            status = "checkpoint"
            print(f"  {b.batch_id}: SKIP valid test checkpoint")
        else:
            # Remove partial inferential outputs from interrupted runs.
            for key in ["qc", "association", "start_vs_end", "diff_end", "pattern", "done"]:
                if paths[key].exists():
                    paths[key].unlink()

            print(f"  {b.batch_id}: TEST {len(genes)} genes")
            cmd = [
                rscript_path,
                str(r_test_script),
                str(b.fit_path),
                str(out_dir),
                str(b.batch_id),
                str(TRADESEQ_TEST_SEED),
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            paths["stdout"].write_text(result.stdout)
            paths["stderr"].write_text(result.stderr)
            if result.returncode != 0:
                print(result.stdout[-3000:])
                print(result.stderr[-5000:])
                raise RuntimeError(
                    f"tradeSeq testing failed for {compartment}/{trajectory}/{b.batch_id}. "
                    f"See {paths['stderr']}"
                )

            for key in ["qc", "association", "start_vs_end", "diff_end", "pattern"]:
                require_nonempty(paths[key], f"{key} output")
            qc = pd.read_csv(paths["qc"])
            if qc["gene"].astype(str).tolist() != genes:
                raise RuntimeError(f"QC gene-order mismatch: {compartment}/{trajectory}/{b.batch_id}")

            done_tmp = out_dir / "DONE.tmp"
            done_tmp.write_text(f"{compartment}\t{trajectory}\t{b.batch_id}\t{len(genes)}\n")
            done_tmp.replace(paths["done"])
            status = "tested"
            print(result.stdout[-800:])

        qc = pd.read_csv(paths["qc"])
        n_converged = int(qc["converged"].fillna(False).astype(bool).sum())
        batch_records.append({
            "compartment": compartment,
            "trajectory": trajectory,
            "batch_id": b.batch_id,
            "n_genes": len(genes),
            "n_converged": n_converged,
            "n_nonconverged": len(genes) - n_converged,
            "status": status,
            "test_dir": str(out_dir),
            **{f"{k}_path": str(paths[k]) for k in ["qc", "association", "start_vs_end", "diff_end", "pattern"]},
        })

# %% COMBINE RESULTS + APPLY PREDEFINED BH FAMILIES
batch_test_manifest = pd.DataFrame(batch_records)
batch_test_manifest_path = TRADESEQ_TEST_DIR / "trajectory_tradeseq_test_batch_manifest.csv"
batch_test_manifest.to_csv(batch_test_manifest_path, index=False)

trajectory_summary_records: list[dict] = []
family_records: list[dict] = []

for tr in fit_manifest.itertuples(index=False):
    compartment = str(tr.compartment)
    trajectory = str(tr.trajectory)
    bundle = sanitize_filename(f"{compartment}__{trajectory}")
    out_dir = TRADESEQ_TEST_DIR / bundle

    sub = batch_test_manifest.loc[
        (batch_test_manifest["compartment"] == compartment)
        & (batch_test_manifest["trajectory"] == trajectory)
    ].copy()

    # Complete fit-QC denominator.
    qc_parts = []
    for b in sub.itertuples(index=False):
        x = pd.read_csv(b.qc_path)
        x.insert(0, "trajectory", trajectory)
        x.insert(0, "compartment", compartment)
        qc_parts.append(x)
    qc_all = pd.concat(qc_parts, ignore_index=True)
    if len(qc_all) != int(tr.n_genes):
        raise RuntimeError(f"QC denominator mismatch for {compartment}/{trajectory}")
    qc_all.to_csv(out_dir / "fit_qc_combined.csv", index=False)

    n_converged = int(qc_all["converged"].fillna(False).astype(bool).sum())

    for test_name in TESTS:
        parts = []
        path_col = f"{test_name}_path"
        for b in sub.itertuples(index=False):
            x = pd.read_csv(getattr(b, path_col))
            x.insert(0, "batch_id", b.batch_id)
            x.insert(0, "trajectory", trajectory)
            x.insert(0, "compartment", compartment)
            parts.append(x)
        combined = pd.concat(parts, ignore_index=True)

        # Only converged genes are returned by the R test runner.
        if len(combined) != n_converged:
            raise RuntimeError(
                f"Tested-gene denominator mismatch for {compartment}/{trajectory}/{test_name}: "
                f"{len(combined)} vs {n_converged} converged"
            )
        if combined["gene"].astype(str).duplicated().any():
            raise RuntimeError(f"Duplicated tested genes: {compartment}/{trajectory}/{test_name}")

        pvalue_cols = [c for c in combined.columns if c.lower().startswith("pvalue")]
        if not pvalue_cols:
            raise RuntimeError(f"No p-value columns found in {compartment}/{trajectory}/{test_name}")

        for pcol in pvalue_cols:
            qcol = f"padj__{pcol}"
            combined[qcol] = bh_adjust(combined[pcol])
            valid = pd.to_numeric(combined[pcol], errors="coerce").between(0, 1, inclusive="both")
            sig = pd.to_numeric(combined[qcol], errors="coerce") < FDR_ALPHA
            family_records.append({
                "compartment": compartment,
                "trajectory": trajectory,
                "test": test_name,
                "pvalue_column": pcol,
                "padj_column": qcol,
                "n_tested": int(valid.sum()),
                "n_fdr_lt_0_05": int(sig.fillna(False).sum()),
                "fdr_method": "Benjamini-Hochberg",
                "fdr_alpha": FDR_ALPHA,
                "family_definition": "trajectory x test x pvalue-column; all batches combined",
            })

        combined_path = out_dir / f"{test_name}_combined.csv"
        combined.to_csv(combined_path, index=False)

    trajectory_summary_records.append({
        "compartment": compartment,
        "trajectory": trajectory,
        "n_genes_fitted": int(tr.n_genes),
        "n_converged": n_converged,
        "n_nonconverged": int(tr.n_genes) - n_converged,
        "convergence_fraction": n_converged / int(tr.n_genes),
        "n_lineages": int(tr.n_lineages),
        "nknots": int(tr.nknots),
        "primary_test": "associationTest(global=TRUE, lineages=TRUE, l2fc=0)",
        "directional_test": "startVsEndTest(global=TRUE, lineages=TRUE, l2fc=0)",
        "secondary_branch_tests": "diffEndTest + patternTest (global + pairwise)",
    })

# %% FINAL MANIFESTS / CHECKPOINT SUMMARY
trajectory_summary = pd.DataFrame(trajectory_summary_records)
family_manifest = pd.DataFrame(family_records)

trajectory_summary_path = TRADESEQ_TEST_DIR / "trajectory_tradeseq_test_summary.csv"
family_manifest_path = TRADESEQ_TEST_DIR / "trajectory_tradeseq_fdr_family_manifest.csv"
trajectory_summary.to_csv(trajectory_summary_path, index=False)
family_manifest.to_csv(family_manifest_path, index=False)

print("\n" + "=" * 100)
print("04c07 TRADESEQ TESTING COMPLETE")
print("=" * 100)
print(trajectory_summary.to_string(index=False))

print("\nPRIMARY associationTest FDR summary")
primary = family_manifest.loc[family_manifest["test"] == "association"].copy()
print(primary[["compartment", "trajectory", "pvalue_column", "n_tested", "n_fdr_lt_0_05"]].to_string(index=False))

print("\nSaved:")
print(" ", batch_test_manifest_path)
print(" ", trajectory_summary_path)
print(" ", family_manifest_path)
print("\nNo fitGAM models were refit or retuned.")
