# %% DECIPHER
#
#
# %% SETUP
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import stats


# %% PATHS / CONFIG
PROJECT_DIR = Path.cwd().parent
DATA_DIR = PROJECT_DIR / "data"
METACELL_DIR = DATA_DIR / "metacells"
OUTPUT_DIR = DATA_DIR / "decipher"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PLOT_DIR = PROJECT_DIR / "outputs" / "decipher"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

# COMPARTMENT = "PT"
# COMPARTMENT = "FIB"
# COMPARTMENT = "Immune"
COMPARTMENT = "Glomerular"
H5AD = METACELL_DIR / f"{COMPARTMENT}_metacells.h5ad"

N_TOP_GENES = 2000
DIM_V = 2
DIM_Z = 10
MAX_EPOCHS = 400
SEED = 0

COLOR_COLS = [
    "dominant_disease",
    "dominant_state",
    "dominant_subclass",
    "dominant_assay",
]

DISEASE_COL = "dominant_disease"
ASSAY_COL = "dominant_assay"
COVID_AKI_COL = "is_covid_aki_metacell"
DONOR_COL = "donor_id"
AKI_LABEL = "Acute Kidney Injury"

SKIP_CONFOUND_CHECKS = False


# %% HELPERS
def run_pairwise_tests(df, group_col, latent_cols, donor_col, label_prefix=""):
    levels = sorted(df[group_col].dropna().unique(), key=str)
    if len(levels) < 2:
        print(f"  {label_prefix}Only {len(levels)} level(s) of '{group_col}' present -- skipping")
        return

    for i, a in enumerate(levels):
        for b in levels[i + 1:]:
            da = df.loc[df[group_col] == a]
            db = df.loc[df[group_col] == b]
            n_a = da[donor_col].nunique()
            n_b = db[donor_col].nunique()

            print(
                f"  {label_prefix}'{a}' (n={len(da)} metacells, {n_a} donors) "
                f"vs '{b}' (n={len(db)} metacells, {n_b} donors):"
            )

            for col in latent_cols:
                p_mc = (
                    stats.mannwhitneyu(da[col], db[col], alternative="two-sided").pvalue
                    if len(da) >= 2 and len(db) >= 2 else np.nan
                )

                donor_a = da.groupby(donor_col)[col].mean()
                donor_b = db.groupby(donor_col)[col].mean()
                p_donor = (
                    stats.mannwhitneyu(donor_a, donor_b, alternative="two-sided").pvalue
                    if len(donor_a) >= 2 and len(donor_b) >= 2 else np.nan
                )

                flag = " <-- " if pd.notna(p_donor) and p_donor < 0.05 else ""
                print(
                    f"    {col}: mean {a}={da[col].mean():.3f}, {b}={db[col].mean():.3f} | "
                    f"metacell-level p={p_mc:.4f}, donor-level p={p_donor:.4f}{flag}"
                )


def run_confound_checks(adata, v_cols, z_cols):
    df = pd.DataFrame(adata.obsm["X_decipher_v"], columns=v_cols)

    for i, col in enumerate(z_cols):
        df[col] = adata.obsm["X_decipher_z"][:, i]

    for col in [DISEASE_COL, DONOR_COL]:
        df[col] = adata.obs[col].to_numpy()

    has_assay = ASSAY_COL in adata.obs
    has_covid = COVID_AKI_COL in adata.obs

    if has_assay:
        df[ASSAY_COL] = adata.obs[ASSAY_COL].to_numpy()
    if has_covid:
        df[COVID_AKI_COL] = adata.obs[COVID_AKI_COL].to_numpy()

    # COVID-AKI vs KPMP-AKI, within AKI only
    print(f"\n{'=' * 70}\nConfound check 1: COVID-AKI vs KPMP-AKI (within {AKI_LABEL})\n{'=' * 70}")

    if not has_covid:
        print(
            f"  '{COVID_AKI_COL}' not found in .obs -- skipping. Was the COVID-AKI "
            "donor set in the 04a_metacell.py run that produced this metacell file?"
        )
    else:
        aki = df.loc[df[DISEASE_COL] == AKI_LABEL].copy()

        if aki.empty:
            print(f"  No metacells with disease label '{AKI_LABEL}' -- skipping")
        else:
            aki["_covid_label"] = aki[COVID_AKI_COL].map(
                {True: "COVID-AKI", False: "KPMP-AKI"}
            )
            n_covid_donors = aki.loc[aki[COVID_AKI_COL], DONOR_COL].nunique()

            if n_covid_donors == 0:
                print(
                    f"  0 COVID-AKI metacells found within {AKI_LABEL} -- either this "
                    "cohort has none in this compartment, or the COVID-AKI donors "
                    "were not set for this 04a run."
                )
            else:
                run_pairwise_tests(aki, "_covid_label", v_cols, DONOR_COL)
                run_pairwise_tests(aki, "_covid_label", z_cols, DONOR_COL)
                print(
                    f"\n  Interpretation: {n_covid_donors} COVID-AKI donor(s) is a small "
                    "sample -- a NON-significant result here is weak evidence "
                    "(underpowered), not confirmation the AKI pattern is cohort-"
                    "independent. A SIGNIFICANT result (rows marked '<--') is more "
                    "informative: it shows AKI's position in v/z is at least partly "
                    "etiology/cohort-dependent, not purely a function of injury severity."
                )

    # Assay platform vs v/z, pooled and within disease groups
    print(f"\n{'=' * 70}\nConfound check 2: {ASSAY_COL} vs v/z\n{'=' * 70}")

    if not has_assay:
        print(
            f"  '{ASSAY_COL}' not found in .obs -- skipping. Re-run 04a_metacell.py "
            "with assay metadata available for this compartment."
        )
        return

    print("\n--- Whole compartment (all disease groups pooled) ---")
    run_pairwise_tests(df, ASSAY_COL, v_cols, DONOR_COL)
    run_pairwise_tests(df, ASSAY_COL, z_cols, DONOR_COL)

    print("\n--- Within each disease group ---")
    print(
        f"(if {ASSAY_COL} still predicts v/z position AFTER conditioning on disease "
        "group, that's evidence of a pervasive platform confound, not just "
        "platform/disease happening to correlate)"
    )

    for disease in sorted(df[DISEASE_COL].dropna().unique()):
        sub = df.loc[df[DISEASE_COL] == disease]
        n_levels = sub[ASSAY_COL].dropna().nunique()

        if n_levels < 2:
            print(
                f"\n  --- {disease}: only {n_levels} level(s) of {ASSAY_COL} "
                "--- skipping ---"
            )
            continue

        print(f"\n  --- {disease} ---")
        run_pairwise_tests(
            sub, ASSAY_COL, v_cols, DONOR_COL, label_prefix=f"  [{disease}] "
        )

    print(
        f"\n{'=' * 70}\nConfound checks done. Rows marked '<--' have donor-level "
        "p < 0.05 -- treat any v/z-based claim touching those dimensions as "
        "confounded until addressed (e.g. by switching the trajectory basis to "
        "a batch-corrected embedding, or explicitly conditioning on platform).\n"
        f"{'=' * 70}"
    )


# %% LOAD
name = H5AD.name.replace("_metacells.h5ad", "")
print(f"Loading {H5AD}...")
adata = sc.read_h5ad(H5AD)
print(f"{name}: {adata.n_obs} metacells x {adata.n_vars} genes")

for col in [DISEASE_COL, "n_cells"]:
    if col not in adata.obs:
        raise ValueError(
            f"Expected column '{col}' not found in .obs -- is this a 04a_metacell.py output?"
        )

if "counts" not in adata.layers:
    raise ValueError("Expected a 'counts' layer -- is this a 04a_metacell.py output?")

print("\nMetacell count by dominant disease group:")
print(adata.obs[DISEASE_COL].value_counts())

if ASSAY_COL in adata.obs:
    print("\nMetacell count by dominant assay platform:")
    print(adata.obs[ASSAY_COL].value_counts())
else:
    print(
        f"\nNOTE: '{ASSAY_COL}' not found in .obs -- assay-platform confound "
        "check will be skipped."
    )


# %% HVGS
print(f"\nSelecting top {N_TOP_GENES} HVGs on metacell-level counts (seurat_v3)...")

sc.pp.highly_variable_genes(
    adata,
    n_top_genes=N_TOP_GENES,
    flavor="seurat_v3",
    layer="counts",
)

n_hvg = int(adata.var["highly_variable"].sum())
print(f"Selected {n_hvg} HVGs (requested {N_TOP_GENES})")
adata = adata[:, adata.var["highly_variable"]].copy()


# %% DECIPHER
print("\nImporting scvi-tools / Decipher...")
import scvi
from scvi.external import Decipher

scvi.settings.seed = SEED

print("Registering AnnData with Decipher.setup_anndata()...")
Decipher.setup_anndata(adata, layer="counts")

print(f"Constructing Decipher model (dim_v={DIM_V}, dim_z={DIM_Z})...")
model = Decipher(adata, dim_v=DIM_V, dim_z=DIM_Z)

print(f"Training for up to {MAX_EPOCHS} epochs...")
model.train(max_epochs=MAX_EPOCHS)

if "elbo_train" in model.history_:
    print(f"Final training ELBO: {model.history_['elbo_train'].iloc[-1].item():.2f}")
if "elbo_validation" in model.history_:
    print(f"Final validation ELBO: {model.history_['elbo_validation'].iloc[-1].item():.2f}")


# %% LATENTS
print("\nExtracting v and z latents...")
v = model.get_latent_representation()
z = model.get_latent_representation(give_z=True)

adata.obsm["X_decipher_v"] = v
adata.obsm["X_decipher_z"] = z

print(f"v shape: {v.shape}, z shape: {z.shape}")


# %% SAVE MODEL / ADATA
model_path = OUTPUT_DIR / f"{name}_decipher_model"
model.save(model_path, overwrite=True)
print(f"\nSaved trained model to {model_path}")

adata_path = OUTPUT_DIR / f"{name}_decipher.h5ad"
adata.write_h5ad(adata_path)
print(f"Saved AnnData with v/z latents to {adata_path}")


# %% PLOTS
print("\nGenerating diagnostic plots...")

fig, ax = plt.subplots(figsize=(7, 5))
if "elbo_train" in model.history_:
    ax.plot(model.history_["elbo_train"].to_numpy(), label="Training ELBO")
if "elbo_validation" in model.history_:
    ax.plot(model.history_["elbo_validation"].to_numpy(), label="Validation ELBO")
ax.set_xlabel("Epoch")
ax.set_ylabel("ELBO")
ax.set_title(f"{name}: Decipher training history")
ax.legend()
fig.tight_layout()

history_path = OUTPUT_DIR / f"{name}_decipher_training_history.png"
fig.savefig(history_path, dpi=150)
plt.close(fig)
print(f"Wrote {history_path}")

color_cols = [c for c in COLOR_COLS if c in adata.obs.columns]

if not color_cols:
    print("  No requested color columns found in .obs -- skipping v-space plot")
    v_plot_path = None
else:
    n_cols = len(color_cols)
    fig, axes = plt.subplots(1, n_cols, figsize=(6.5 * n_cols, 5.5))
    axes = [axes] if n_cols == 1 else axes
    cmap = plt.get_cmap("tab10")

    for ax, col in zip(axes, color_cols):
        categories = adata.obs[col].astype("category")
        present_cats = [c for c in categories.cat.categories if (categories == c).any()]

        for i, cat in enumerate(present_cats):
            mask = (categories == cat).to_numpy()
            ax.scatter(
                v[mask, 0], v[mask, 1],
                color=cmap(i % 10), alpha=0.7, s=8, label=str(cat),
            )

        ax.set_xlabel("v1")
        ax.set_ylabel("v2")
        ax.set_title(f"{name}: v colored by {col}")
        ax.legend(loc="best", fontsize=7, title=col, markerscale=1.5)

    fig.tight_layout()
    v_plot_path = PLOT_DIR / f"{name}_decipher_v_space.png"
    fig.savefig(v_plot_path, dpi=150)
    plt.close(fig)
    print(f"Wrote {v_plot_path}")


# %% V COORDINATES
v_cols = [f"v{i + 1}" for i in range(v.shape[1])]
z_cols = [f"z{i + 1}" for i in range(z.shape[1])]

v_df = pd.DataFrame(v, columns=v_cols)
v_df[DISEASE_COL] = adata.obs[DISEASE_COL].to_numpy()
v_df[DONOR_COL] = adata.obs[DONOR_COL].to_numpy()

print("\n=== Metacell-level v-coordinates by dominant disease group (mean, std) ===")
print(v_df.groupby(DISEASE_COL)[v_cols].agg(["mean", "std"]).round(3))

donor_level = (
    v_df.groupby([DISEASE_COL, DONOR_COL])[v_cols]
    .mean()
    .reset_index()
)

donor_summary = pd.DataFrame({
    "n_donors": donor_level.groupby(DISEASE_COL)[DONOR_COL].nunique()
})

for col in v_cols:
    donor_summary[f"{col}_mean_of_donors"] = donor_level.groupby(DISEASE_COL)[col].mean()
    donor_summary[f"{col}_std_across_donors"] = donor_level.groupby(DISEASE_COL)[col].std()

print("\n=== Donor-level v-coordinates by dominant disease group ===")
print(
    "(mean_of_donors should be close to the metacell-level mean above if metacell "
    "counts are roughly balanced across donors within a group -- a large divergence, "
    "or a large std_across_donors relative to n_donors, means the metacell-level "
    "pattern may be driven by just one or two donors rather than being consistent "
    "across the group)"
)
print(donor_summary.round(3))

donor_level_path = OUTPUT_DIR / f"{name}_decipher_v_by_donor.csv"
donor_level.to_csv(donor_level_path, index=False)
print(f"\nPer-donor mean v-coordinates written to {donor_level_path}")


# %% CONFOUND CHECKS
if SKIP_CONFOUND_CHECKS:
    print("\nSKIP_CONFOUND_CHECKS=True -- skipping COVID-AKI / assay-platform checks")
else:
    run_confound_checks(adata, v_cols, z_cols)


# %% RESULTS
print(
    f"\nDone.\n"
    f"Model: {model_path}\n"
    f"AnnData: {adata_path}\n"
    f"Plots: {history_path}"
    + (f", {v_plot_path}" if v_plot_path else "")
    + f"\nPer-donor v table: {donor_level_path}"
)
