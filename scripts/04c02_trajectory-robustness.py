# %% 04C02 TRAJECTORY-ROBUSTNESS
# Sequential stage extracted from 04c_trajectory.py; frozen upstream outputs are reloaded from disk.

# %% SETUP / SHARED CONFIG
# %% SETUP
import os
import sys
import shutil
import subprocess
import tempfile
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

# Ensure Python finds executables in the Pixi bin directory
pixi_bin = str(Path(sys.prefix) / "bin")
if pixi_bin not in os.environ["PATH"]:
    os.environ["PATH"] = f"{pixi_bin}:{os.environ['PATH']}"

# %% IMPORTS
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
from concurrent.futures import ThreadPoolExecutor, as_completed


# %% PATHS / CONFIG
PROJECT_DIR = Path.cwd().parent
DATA_DIR = PROJECT_DIR / "data"
DECIPHER_DIR = DATA_DIR / "decipher"
TRAJECTORY_DIR = DATA_DIR / "trajectory"
TRAJECTORY_DIR.mkdir(parents=True, exist_ok=True)

PLOT_DIR = PROJECT_DIR / "outputs" / "trajectory"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

COMPARTMENTS = ["PT", "FIB", "Immune", "Glomerular"]

PRIMARY_REP = "X_decipher_v"
SENSITIVITY_REPS = ["X_decipher_z", "X_scVI"]

N_NEIGHBORS = 15
LEIDEN_RESOLUTIONS = [0.4, 0.8]
PRIMARY_RESOLUTION = 0.8

MIN_CLUSTER_SIZE = 10
PAGA_THRESHOLD = 0.03
SEED = 0

ZERO_COUNT_FILTER = True

DISEASE_COL = "dominant_disease"
STATE_COL = "dominant_state"
SUBCLASS_COL = "dominant_subclass"
ASSAY_COL = "dominant_assay"
DONOR_COL = "donor_id"

# %% TRAJECTORY SPECIFICATIONS
SLINGSHOT_APPROX_POINTS = 150

TRAJECTORY_SPECS = {
    "PT": {
        "PT": {
            "subclasses": [
                "PT-S1", "PT-S2", "PT-S3",
                "aPT1", "aPT2",
                "aPT-S1/S2", "aPT-S3",
                "frPT-S1/S2", "frPT-S3",
                "dPT", "dPT-S1", "dPT-S3",
            ],
            "root": {
                "subclass": "PT-S1",
                "cluster": "12",
            },
            "root_sensitivity": [
                {"subclass": "PT-S1", "cluster": "13"},
                {"subclass": "PT-S3", "cluster": "15"},
            ],
        },
    },

    "FIB": {
        "contractile": {
            "subclasses": [
                "C-FIB", "C-FIB-PATH",
                "C/M-FIB", "C-MYOF",
                "C-FIB-OSMRlo",
                "C-FIB-OSMRhi",
            ],
            "root": {
                "subclass": "C-FIB",
                "cluster": "3",
            },
            "root_sensitivity": [
                {"subclass": "C-FIB", "cluster": "1"},
            ],
        },

        "outer_medullary": {
            "subclasses": [
                "OM-FIB",
                "dOM-FIB",
            ],
            "root": {
                "subclass": "OM-FIB",
                "cluster": "8",
            },
            "root_sensitivity": [
                {"subclass": "OM-FIB", "cluster": "6"},
                {"subclass": "OM-FIB", "cluster": "15"},
            ],
        },
    },

    "Glomerular": {
        "podocyte_PEC": {
            "subclasses": [
                "POD",
                "dPOD",
                "PEC",
            ],
            "root": {
                "subclass": "POD",
                "cluster": "12",
            },
            "root_sensitivity": [
                {"subclass": "POD", "cluster": "0"},
                {"subclass": "POD", "cluster": "11"},
            ],
        },

        "endothelial_mesangial": {
            "subclasses": [
                "EC-GC",
                "dEC-GC",
                "EC-GC-FILIP1+",
                "MC",
            ],
            "root": {
                "subclass": "EC-GC",
                "cluster": "14",
            },
            "root_sensitivity": [
                {"subclass": "EC-GC", "cluster": "1"},
                {"subclass": "EC-GC", "cluster": "10"},
            ],
        },
    },
}


# %% SHARED HELPERS REQUIRED BY THIS STAGE
def numeric_rep(adata, key):
    x = np.asarray(adata.obsm[key], dtype=float)
    if x.ndim != 2 or x.shape[0] != adata.n_obs:
        raise ValueError(f"{key} has invalid shape: {x.shape}")
    if not np.isfinite(x).all():
        raise ValueError(f"{key} contains non-finite values")
    return x


def safe_name(s):
    return str(s).replace("-", "_").replace(".", "_").replace(" ", "_")


def check_slingshot():
    rscript = shutil.which("Rscript")
    if rscript is None:
        raise RuntimeError("Rscript executable not found in PATH.")

    cmd = [
        rscript,
        "-e",
        "if (!requireNamespace('slingshot', quietly=TRUE)) quit(status=10)",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)

    if res.returncode != 0:
        raise RuntimeError(
            "R package 'slingshot' is missing or non-functional:\\n"
            f"{res.stderr}"
        )

    return rscript


def run_slingshot(
    rd,
    cluster_labels,
    start_clus,
    approx_points=SLINGSHOT_APPROX_POINTS,
):
    rd = np.asarray(rd, dtype=float)
    cluster_labels = np.asarray(cluster_labels, dtype=str)

    if rd.ndim != 2 or rd.shape[0] != len(cluster_labels):
        raise ValueError(
            f"Invalid Slingshot input: rd={rd.shape}, "
            f"clusters={len(cluster_labels)}"
        )

    if not np.isfinite(rd).all():
        raise ValueError("Slingshot coordinates contain non-finite values.")

    if str(start_clus) not in set(cluster_labels):
        raise ValueError(f"Slingshot start cluster {start_clus!r} is absent.")

    rscript = check_slingshot()

    with tempfile.TemporaryDirectory(prefix="04c_slingshot_") as tmpdir:
        tmp_path = Path(tmpdir)
        input_file = tmp_path / "input.csv"
        output_file = tmp_path / "output.csv"
        lineage_file = tmp_path / "lineages.csv"
        script_file = tmp_path / "run_slingshot.R"

        input_df = pd.DataFrame(
            rd,
            columns=[f"dim{i + 1}" for i in range(rd.shape[1])],
        )
        input_df.insert(0, "row_id", np.arange(rd.shape[0], dtype=int))
        input_df["cluster"] = cluster_labels
        input_df.to_csv(input_file, index=False)

        r_code = """
suppressPackageStartupMessages(
    library(slingshot)
)

args <- commandArgs(trailingOnly=TRUE)

input_file <- args[1]
output_file <- args[2]
lineage_file <- args[3]
root_cluster <- args[4]
seed <- as.integer(args[5])
approx_points <- as.integer(args[6])

set.seed(seed)

d <- read.csv(
    input_file,
    stringsAsFactors=FALSE,
    check.names=FALSE
)

dim_cols <- grep("^dim[0-9]+$", names(d), value=TRUE)
rd <- as.matrix(d[, dim_cols, drop=FALSE])
clusters <- as.character(d$cluster)

if (any(!is.finite(rd))) {
    stop("Reduced dimensions contain non-finite values.")
}

if (!(root_cluster %in% clusters)) {
    stop(paste("Root cluster absent:", root_cluster))
}

cluster_sizes <- table(clusters)
if (any(cluster_sizes < 2)) {
    stop(
        paste(
            "Trajectory-specific clustering contains singleton clusters:",
            paste(names(cluster_sizes)[cluster_sizes < 2], collapse=",")
        )
    )
}

fit <- slingshot(
    rd,
    clusterLabels=clusters,
    start.clus=root_cluster,
    approx_points=approx_points
)

pt <- slingPseudotime(fit, na=TRUE)
wt <- slingCurveWeights(fit, as.probs=FALSE)

if (is.null(dim(pt))) {
    pt <- matrix(pt, ncol=1)
}

if (is.null(dim(wt))) {
    wt <- matrix(wt, ncol=1)
}

if (is.null(colnames(pt))) {
    colnames(pt) <- paste0("Lineage", seq_len(ncol(pt)))
}

if (is.null(colnames(wt))) {
    colnames(wt) <- colnames(pt)
}

pt_names <- colnames(pt)
wt_names <- colnames(wt)

colnames(pt) <- paste0("pt__", pt_names)
colnames(wt) <- paste0("weight__", wt_names)

out <- data.frame(
    row_id=d$row_id,
    pt,
    wt,
    check.names=FALSE
)

write.csv(out, output_file, row.names=FALSE)

lin <- slingLineages(fit)

lin_df <- do.call(
    rbind,
    lapply(
        seq_along(lin),
        function(i) {
            data.frame(
                lineage=paste0("Lineage", i),
                path=paste(as.character(lin[[i]]), collapse="->"),
                stringsAsFactors=FALSE
            )
        }
    )
)

write.csv(lin_df, lineage_file, row.names=FALSE)
"""
        script_file.write_text(r_code)

        cmd = [
            rscript,
            str(script_file),
            str(input_file),
            str(output_file),
            str(lineage_file),
            str(start_clus),
            str(SEED),
            str(approx_points),
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)

        if res.returncode != 0:
            raise RuntimeError(
                "Slingshot execution failed:\\n"
                f"STDOUT:\\n{res.stdout}\\n"
                f"STDERR:\\n{res.stderr}"
            )

        result_df = pd.read_csv(output_file)
        lineage_df = pd.read_csv(lineage_file)

    result_df = result_df.sort_values("row_id").reset_index(drop=True)
    return result_df, lineage_df


def map_root_cluster(
    sub_adata,
    original_cluster_col,
    trajectory_cluster_col,
    original_root_cluster,
):
    original_root_cluster = str(original_root_cluster)

    old_labels = sub_adata.obs[original_cluster_col].astype(str)
    new_labels = sub_adata.obs[trajectory_cluster_col].astype(str)
    root_mask = old_labels.eq(original_root_cluster)

    if root_mask.sum() == 0:
        raise ValueError(
            f"Original root cluster {original_root_cluster} "
            "has no metacells in the trajectory subset."
        )

    overlap = new_labels[root_mask].value_counts()
    mapped_root = str(overlap.index[0])
    n_original_root = int(root_mask.sum())
    n_mapped = int(overlap.iloc[0])
    mapped_fraction = float(n_mapped / n_original_root)

    return mapped_root, n_original_root, n_mapped, mapped_fraction


# %% ROBUSTNESS HELPERS
from scipy.optimize import linear_sum_assignment
from scipy.stats import spearmanr


def finite_spearman(x, y, min_n=10):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    keep = np.isfinite(x) & np.isfinite(y)
    n = int(keep.sum())

    if n < min_n:
        return np.nan, n

    if (
        np.nanstd(x[keep]) == 0
        or np.nanstd(y[keep]) == 0
    ):
        return np.nan, n

    rho = spearmanr(
        x[keep],
        y[keep],
    ).statistic

    return float(rho), n


def lineage_terminal(path):
    path = str(path)

    if not path:
        return None

    return path.split("->")[-1]


def lineage_cluster_set(path):
    return set(
        str(path).split("->")
    )


def path_jaccard(path_a, path_b):
    a = lineage_cluster_set(path_a)
    b = lineage_cluster_set(path_b)

    union = a | b

    if not union:
        return np.nan

    return len(a & b) / len(union)


def primary_pt_columns(
    adata,
    trajectory_name,
):
    prefix = f"pt_{safe_name(trajectory_name)}_"

    cols = [
        col
        for col in adata.obs.columns
        if col.startswith(prefix)
        and not col.endswith("_scaled")
    ]

    return sorted(cols)


def primary_weight_columns(
    adata,
    trajectory_name,
):
    prefix = f"weight_{safe_name(trajectory_name)}_"

    return sorted(
        col
        for col in adata.obs.columns
        if col.startswith(prefix)
    )


def extract_primary_lineage_name(
    col,
    trajectory_name,
):
    prefix = f"pt_{safe_name(trajectory_name)}_"

    return col.replace(
        prefix,
        "",
        1,
    )


def load_primary_lineages(
    compartment,
    trajectory_name,
):
    path = (
        TRAJECTORY_DIR
        / (
            f"{compartment}_"
            f"{trajectory_name}_"
            "slingshot_lineages.csv"
        )
    )

    out = pd.read_csv(path)

    if "compartment" in out.columns:
        out = out[
            out["compartment"].astype(str)
            == str(compartment)
        ].copy()

    if "trajectory" in out.columns:
        out = out[
            out["trajectory"].astype(str)
            == str(trajectory_name)
        ].copy()

    return out.reset_index(drop=True)


def get_trajectory_subset(
    adata,
    trajectory_name,
    spec,
):
    membership_col = (
        f"trajectory_"
        f"{safe_name(trajectory_name)}"
        "_member"
    )

    if membership_col in adata.obs:
        mask = (
            adata.obs[
                membership_col
            ]
            .astype(bool)
        )
    else:
        mask = (
            adata.obs[
                SUBCLASS_COL
            ]
            .astype(str)
            .isin(
                spec["subclasses"]
            )
        )

    return adata[
        mask
    ].copy()


def match_lineages_by_correlation(
    primary_matrix,
    sensitivity_matrix,
    primary_names,
    sensitivity_names,
):
    primary_matrix = np.asarray(
        primary_matrix,
        dtype=float,
    )
    sensitivity_matrix = np.asarray(
        sensitivity_matrix,
        dtype=float,
    )

    score = np.full(
        (
            len(primary_names),
            len(sensitivity_names),
        ),
        np.nan,
        dtype=float,
    )

    n_overlap = np.zeros(
        score.shape,
        dtype=int,
    )

    for i in range(
        len(primary_names)
    ):
        for j in range(
            len(sensitivity_names)
        ):
            rho, n = finite_spearman(
                primary_matrix[:, i],
                sensitivity_matrix[:, j],
            )

            score[i, j] = rho
            n_overlap[i, j] = n

    if (
        score.size == 0
        or np.all(~np.isfinite(score))
    ):
        return pd.DataFrame(
            columns=[
                "primary_lineage",
                "sensitivity_lineage",
                "spearman_rho",
                "n_overlap",
            ]
        )

    # We expect orientation to remain anchored at
    # the same biological root, so maximize positive
    # rank correlation rather than absolute correlation.
    cost = np.where(
        np.isfinite(score),
        -score,
        1e6,
    )

    rows, cols = linear_sum_assignment(
        cost
    )

    records = []

    for i, j in zip(rows, cols):
        records.append(
            {
                "primary_lineage":
                    primary_names[i],
                "sensitivity_lineage":
                    sensitivity_names[j],
                "spearman_rho":
                    score[i, j],
                "n_overlap":
                    int(
                        n_overlap[i, j]
                    ),
            }
        )

    return pd.DataFrame(records)


def trajectory_cluster_qc(
    adata,
    cluster_col,
):
    out = (
        adata.obs
        .assign(
            _cluster=(
                adata.obs[
                    cluster_col
                ].astype(str)
            )
        )
        .groupby(
            "_cluster",
            observed=True,
        )
        .agg(
            n_metacells=(
                "_cluster",
                "size",
            ),
            n_donors=(
                DONOR_COL,
                "nunique",
            ),
        )
        .reset_index()
        .rename(
            columns={
                "_cluster":
                    "cluster"
            }
        )
    )

    return out


# %% ROOT SENSITIVITY
#
# Change only the root.
# Biological subset, X_decipher_v coordinates,
# and primary trajectory-specific clustering remain fixed.

root_sensitivity_summary = []
root_sensitivity_lineages = []

for compartment, specs in TRAJECTORY_SPECS.items():
    print(
        "\n"
        + "=" * 100
    )
    print(f"{compartment} ROOT SENSITIVITY")
    print(
        "=" * 100
    )

    adata = sc.read_h5ad(
        TRAJECTORY_DIR
        / (
            f"{compartment}_"
            "trajectory_primary.h5ad"
        )
    )

    original_cluster_col = (
        f"trajectory_leiden_"
        f"{str(PRIMARY_RESOLUTION).replace('.', 'p')}"
    )

    for trajectory_name, spec in specs.items():
        sensitivities = spec.get(
            "root_sensitivity",
            [],
        )

        if not sensitivities:
            continue

        print(f"\n{trajectory_name}")

        sub = get_trajectory_subset(
            adata,
            trajectory_name,
            spec,
        )

        trajectory_cluster_col = (
            f"trajectory_"
            f"{safe_name(trajectory_name)}"
            "_cluster"
        )

        if (
            trajectory_cluster_col
            not in sub.obs
        ):
            raise KeyError(
                trajectory_cluster_col
            )

        cluster_labels = (
            sub.obs[
                trajectory_cluster_col
            ]
            .astype(str)
            .to_numpy()
        )

        rd = numeric_rep(
            sub,
            PRIMARY_REP,
        )

        primary_lineages = (
            load_primary_lineages(
                compartment,
                trajectory_name,
            )
        )

        primary_pt_cols = (
            primary_pt_columns(
                sub,
                trajectory_name,
            )
        )

        primary_pt_names = [
            extract_primary_lineage_name(
                col,
                trajectory_name,
            )
            for col in primary_pt_cols
        ]

        primary_pt = (
            sub.obs[
                primary_pt_cols
            ]
            .to_numpy(
                dtype=float
            )
        )

        for sensitivity_index, root_spec in enumerate(
            sensitivities,
            start=1,
        ):
            (
                mapped_root,
                n_original_root,
                n_root_mapped,
                mapping_fraction,
            ) = map_root_cluster(
                sub_adata=sub,
                original_cluster_col=(
                    original_cluster_col
                ),
                trajectory_cluster_col=(
                    trajectory_cluster_col
                ),
                original_root_cluster=(
                    root_spec[
                        "cluster"
                    ]
                ),
            )

            print(
                f"  root {sensitivity_index}: "
                f"{root_spec['subclass']} / "
                f"original cluster "
                f"{root_spec['cluster']} "
                f"-> trajectory cluster "
                f"{mapped_root} "
                f"({mapping_fraction:.1%})"
            )

            result_df, lineage_df = (
                run_slingshot(
                    rd=rd,
                    cluster_labels=(
                        cluster_labels
                    ),
                    start_clus=(
                        mapped_root
                    ),
                    approx_points=(
                        SLINGSHOT_APPROX_POINTS
                    ),
                )
            )

            sensitivity_pt_cols = [
                col
                for col
                in result_df.columns
                if col.startswith(
                    "pt__"
                )
            ]

            sensitivity_names = [
                col.replace(
                    "pt__",
                    "",
                    1,
                )
                for col
                in sensitivity_pt_cols
            ]

            sensitivity_pt = (
                result_df[
                    sensitivity_pt_cols
                ]
                .to_numpy(
                    dtype=float
                )
            )

            # Because clustering is unchanged,
            # terminal clusters and paths are
            # directly comparable.
            primary_path_map = {
                str(row.lineage):
                    str(row.path)
                for row
                in primary_lineages.itertuples()
            }

            sensitivity_path_map = {
                str(row.lineage):
                    str(row.path)
                for row
                in lineage_df.itertuples()
            }

            for sensitivity_lineage in (
                sensitivity_names
            ):
                sensitivity_path = (
                    sensitivity_path_map.get(sensitivity_lineage)
                )

                sensitivity_terminal = (
                    lineage_terminal(sensitivity_path)
                )

                same_terminal = [
                    name
                    for name, path
                    in primary_path_map.items()
                    if lineage_terminal(
                        path
                    )
                    == sensitivity_terminal
                ]

                if same_terminal:
                    primary_lineage = same_terminal[0]
                    match_method = (
                        "terminal_cluster"
                    )
                else:
                    candidates = []

                    for (
                        primary_lineage_name,
                        primary_path,
                    ) in (
                        primary_path_map.items()
                    ):
                        candidates.append(
                            (
                                path_jaccard(
                                    primary_path,
                                    sensitivity_path,
                                ),
                                primary_lineage_name,
                            )
                        )

                    candidates.sort(reverse=True)

                    primary_lineage = candidates[0][1]
                    match_method = (
                        "path_jaccard"
                    )

                primary_index = (
                    primary_pt_names.index(primary_lineage)
                )

                sensitivity_index_col = (
                    sensitivity_names.index(sensitivity_lineage)
                )

                rho, n_overlap = (
                    finite_spearman(
                        primary_pt[
                            :,
                            primary_index
                        ],
                        sensitivity_pt[
                            :,
                            sensitivity_index_col
                        ],
                    )
                )

                primary_path = (
                    primary_path_map[
                        primary_lineage
                    ]
                )

                root_sensitivity_lineages.append(
                    {
                        "compartment":
                            compartment,
                        "trajectory":
                            trajectory_name,
                        "sensitivity_root_subclass":
                            root_spec[
                                "subclass"
                            ],
                        "sensitivity_original_root_cluster":
                            str(
                                root_spec[
                                    "cluster"
                                ]
                            ),
                        "sensitivity_mapped_root_cluster":
                            mapped_root,
                        "primary_lineage":
                            primary_lineage,
                        "sensitivity_lineage":
                            sensitivity_lineage,
                        "primary_terminal":
                            lineage_terminal(
                                primary_path
                            ),
                        "sensitivity_terminal":
                            sensitivity_terminal,
                        "match_method":
                            match_method,
                        "path_jaccard":
                            path_jaccard(
                                primary_path,
                                sensitivity_path,
                            ),
                        "spearman_rho":
                            rho,
                        "n_overlap":
                            n_overlap,
                    }
                )

            matched = [
                x
                for x
                in root_sensitivity_lineages
                if (
                    x["compartment"]
                    == compartment
                    and x[
                        "trajectory"
                    ]
                    == trajectory_name
                    and x[
                        "sensitivity_original_root_cluster"
                    ]
                    == str(
                        root_spec[
                            "cluster"
                        ]
                    )
                )
            ]

            rhos = np.asarray(
                [
                    x["spearman_rho"]
                    for x in matched
                ],
                dtype=float,
            )

            jaccards = np.asarray(
                [
                    x["path_jaccard"]
                    for x in matched
                ],
                dtype=float,
            )

            root_sensitivity_summary.append(
                {
                    "compartment":
                        compartment,
                    "trajectory":
                        trajectory_name,
                    "primary_original_root_cluster":
                        str(
                            spec[
                                "root"
                            ][
                                "cluster"
                            ]
                        ),
                    "sensitivity_root_subclass":
                        root_spec[
                            "subclass"
                        ],
                    "sensitivity_original_root_cluster":
                        str(
                            root_spec[
                                "cluster"
                            ]
                        ),
                    "sensitivity_mapped_root_cluster":
                        mapped_root,
                    "root_mapping_fraction":
                        mapping_fraction,
                    "n_original_root_metacells":
                        n_original_root,
                    "n_root_metacells_mapped":
                        n_root_mapped,
                    "n_primary_lineages":
                        len(
                            primary_pt_names
                        ),
                    "n_sensitivity_lineages":
                        len(
                            sensitivity_names
                        ),
                    "median_spearman_rho":
                        (
                            float(
                                np.nanmedian(rhos)
                            )
                            if np.isfinite(
                                rhos
                            ).any()
                            else np.nan
                        ),
                    "min_spearman_rho":
                        (
                            float(
                                np.nanmin(rhos)
                            )
                            if np.isfinite(
                                rhos
                            ).any()
                            else np.nan
                        ),
                    "median_path_jaccard":
                        (
                            float(
                                np.nanmedian(jaccards)
                            )
                            if np.isfinite(
                                jaccards
                            ).any()
                            else np.nan
                        ),
                }
            )


root_sensitivity_summary = pd.DataFrame(
    root_sensitivity_summary
)

root_sensitivity_lineages = pd.DataFrame(
    root_sensitivity_lineages
)

root_sensitivity_summary.to_csv(
    TRAJECTORY_DIR
    / "trajectory_root_sensitivity_summary.csv",
    index=False,
)

root_sensitivity_lineages.to_csv(
    TRAJECTORY_DIR
    / "trajectory_root_sensitivity_lineages.csv",
    index=False,
)

print(
    "\n"
    + "=" * 100
)
print(
    "ROOT SENSITIVITY COMPLETE"
)
print(
    "=" * 100
)

print(
    root_sensitivity_summary.to_string(index=False)
)


# %% REPRESENTATION SENSITIVITY
#
# Change the latent representation while keeping:
#   - biological trajectory definition fixed
#   - audited biological root fixed
#   - Leiden resolution fixed
#
# The KNN graph and Leiden clustering are rebuilt within
# each alternative representation because geometry itself
# is the sensitivity being tested.
#
# IMPORTANT:
# Slingshot uses dist.method="simple" for ALL representation
# sensitivities. This uses Euclidean distances between cluster
# centers and avoids covariance inversion failures in higher-
# dimensional X_decipher_z / X_scVI spaces.
#
# The primary Slingshot result remains unchanged.


def run_slingshot_representation_sensitivity(
    rd,
    cluster_labels,
    start_clus,
    approx_points=SLINGSHOT_APPROX_POINTS,
):
    rd = np.asarray(
        rd,
        dtype=float,
    )

    cluster_labels = np.asarray(
        cluster_labels,
        dtype=str,
    )

    if (
        rd.ndim != 2
        or rd.shape[0]
        != len(cluster_labels)
    ):
        raise ValueError(
            "Invalid Slingshot input: "
            f"rd={rd.shape}, "
            f"clusters={len(cluster_labels)}"
        )

    if not np.isfinite(rd).all():
        raise ValueError(
            "Slingshot coordinates contain "
            "non-finite values."
        )

    if (
        str(start_clus)
        not in set(cluster_labels)
    ):
        raise ValueError(
            f"Slingshot start cluster "
            f"{start_clus!r} is absent."
        )

    rscript = check_slingshot()

    with tempfile.TemporaryDirectory(
        prefix="04c_rep_sensitivity_"
    ) as tmpdir:

        tmp_path = Path(tmpdir)

        input_file = (
            tmp_path
            / "input.csv"
        )

        output_file = (
            tmp_path
            / "output.csv"
        )

        lineage_file = (
            tmp_path
            / "lineages.csv"
        )

        script_file = (
            tmp_path
            / "run_slingshot.R"
        )

        input_df = pd.DataFrame(
            rd,
            columns=[
                f"dim{i + 1}"
                for i
                in range(
                    rd.shape[1]
                )
            ],
        )

        input_df.insert(
            0,
            "row_id",
            np.arange(
                rd.shape[0],
                dtype=int,
            ),
        )

        input_df[
            "cluster"
        ] = cluster_labels

        input_df.to_csv(
            input_file,
            index=False,
        )

        r_code = """
        suppressPackageStartupMessages(
            library(slingshot)
        )

        args <- commandArgs(
            trailingOnly=TRUE
        )

        input_file <- args[1]
        output_file <- args[2]
        lineage_file <- args[3]
        root_cluster <- args[4]
        seed <- as.integer(args[5])
        approx_points <- as.integer(args[6])

        set.seed(seed)

        d <- read.csv(
            input_file,
            stringsAsFactors=FALSE,
            check.names=FALSE
        )

        dim_cols <- grep(
            "^dim[0-9]+$",
            names(d),
            value=TRUE
        )

        if (
            length(dim_cols) == 0
        ) {
            stop(
                "No reduced-dimension columns found."
            )
        }

        rd <- as.matrix(
            d[
                ,
                dim_cols,
                drop=FALSE
            ]
        )

        storage.mode(rd) <- "double"

        clusters <- as.character(
            d$cluster
        )

        if (
            any(
                !is.finite(rd)
            )
        ) {
            stop(
                "Reduced dimensions contain non-finite values."
            )
        }

        if (
            !(root_cluster %in% clusters)
        ) {
            stop(
                paste(
                    "Root cluster absent:",
                    root_cluster
                )
            )
        }

        cluster_sizes <- table(
            clusters
        )

        if (
            any(cluster_sizes < 2)
        ) {
            bad_clusters <- names(
                cluster_sizes
            )[
                cluster_sizes < 2
            ]

            stop(
                paste(
                    "Trajectory-specific clustering contains singleton clusters:",
                    paste(
                        bad_clusters,
                        collapse=","
                    )
                )
            )
        }

        fit <- slingshot(
            rd,
            clusterLabels=clusters,
            start.clus=root_cluster,
            approx_points=approx_points,
            dist.method="simple"
        )

        pt <- slingPseudotime(
            fit,
            na=TRUE
        )

        wt <- slingCurveWeights(
            fit,
            as.probs=FALSE
        )

        if (
            is.null(dim(pt))
        ) {
            pt <- matrix(
                pt,
                ncol=1
            )
        }

        if (
            is.null(dim(wt))
        ) {
            wt <- matrix(
                wt,
                ncol=1
            )
        }

        if (
            is.null(colnames(pt))
        ) {
            colnames(pt) <- paste0(
                "Lineage",
                seq_len(
                    ncol(pt)
                )
            )
        }

        if (
            is.null(colnames(wt))
        ) {
            colnames(wt) <- colnames(
                pt
            )
        }

        if (
            ncol(pt) != ncol(wt)
        ) {
            stop(
                "Pseudotime and curve-weight dimensions disagree."
            )
        }

        pt_names <- colnames(
            pt
        )

        wt_names <- colnames(
            wt
        )

        colnames(pt) <- paste0(
            "pt__",
            pt_names
        )

        colnames(wt) <- paste0(
            "weight__",
            wt_names
        )

        out <- data.frame(
            row_id=d$row_id,
            pt,
            wt,
            check.names=FALSE
        )

        write.csv(
            out,
            output_file,
            row.names=FALSE
        )

        lin <- slingLineages(
            fit
        )

        lin_df <- do.call(
            rbind,
            lapply(
                seq_along(lin),
                function(i) {
                    data.frame(
                        lineage=paste0(
                            "Lineage",
                            i
                        ),
                        path=paste(
                            as.character(
                                lin[[i]]
                            ),
                            collapse="->"
                        ),
                        stringsAsFactors=FALSE
                    )
                }
            )
        )

        write.csv(
            lin_df,
            lineage_file,
            row.names=FALSE
        )
        """

        script_file.write_text(r_code)

        cmd = [
            rscript,
            str(
                script_file
            ),
            str(
                input_file
            ),
            str(
                output_file
            ),
            str(
                lineage_file
            ),
            str(
                start_clus
            ),
            str(
                SEED
            ),
            str(
                approx_points
            ),
        ]

        res = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
        )

        if (
            res.returncode
            != 0
        ):
            raise RuntimeError(
                "Representation-sensitivity "
                "Slingshot failed:\n"
                f"STDOUT:\n"
                f"{res.stdout}\n"
                f"STDERR:\n"
                f"{res.stderr}"
            )

        result_df = (
            pd.read_csv(output_file)
        )

        lineage_df = (
            pd.read_csv(lineage_file)
        )

    result_df = (
        result_df
        .sort_values(
            "row_id"
        )
        .reset_index(
            drop=True
        )
    )

    return (
        result_df,
        lineage_df,
    )


representation_sensitivity_summary = []
representation_sensitivity_lineages = []
representation_cluster_qc = []


for compartment, specs in (
    TRAJECTORY_SPECS.items()
):

    print(
        "\n"
        + "=" * 100
    )

    print(
        f"{compartment} "
        "REPRESENTATION SENSITIVITY"
    )

    print(
        "=" * 100
    )

    adata = sc.read_h5ad(
        TRAJECTORY_DIR
        / (
            f"{compartment}_"
            "trajectory_primary.h5ad"
        )
    )

    original_cluster_col = (
        f"trajectory_leiden_"
        f"{str(PRIMARY_RESOLUTION).replace('.', 'p')}"
    )

    for (
        trajectory_name,
        spec,
    ) in specs.items():

        print(f"\n{trajectory_name}")

        sub = get_trajectory_subset(
            adata,
            trajectory_name,
            spec,
        )

        # ------------------------------------------
        # Primary pseudotime matrix
        # ------------------------------------------

        primary_pt_cols = (
            primary_pt_columns(
                sub,
                trajectory_name,
            )
        )

        if (
            not primary_pt_cols
        ):
            raise ValueError(
                f"{compartment}/"
                f"{trajectory_name}: "
                "no primary pseudotime "
                "columns found."
            )

        primary_names = [
            extract_primary_lineage_name(
                col,
                trajectory_name,
            )
            for col
            in primary_pt_cols
        ]

        primary_pt = (
            sub.obs[
                primary_pt_cols
            ]
            .to_numpy(
                dtype=float
            )
        )

        # ------------------------------------------
        # Alternative representations
        # ------------------------------------------

        for rep in (
            SENSITIVITY_REPS
        ):

            if (
                rep
                not in sub.obsm
            ):
                print(
                    f"  {rep}: "
                    "missing; skipped"
                )
                continue

            print(f"  {rep}")

            rep_safe = (
                safe_name(rep)
            )

            trajectory_safe = (
                safe_name(trajectory_name)
            )

            neighbors_key = (
                "sensitivity_"
                f"{trajectory_safe}_"
                f"{rep_safe}_"
                "neighbors"
            )

            cluster_col = (
                "sensitivity_"
                f"{trajectory_safe}_"
                f"{rep_safe}_"
                "leiden"
            )

            # --------------------------------------
            # Validate full representation
            # --------------------------------------

            rd = numeric_rep(
                sub,
                rep,
            )

            # Use all dimensions:
            # X_decipher_z -> 10D
            # X_scVI       -> 30D

            # --------------------------------------
            # Representation-specific graph
            # --------------------------------------

            sc.pp.neighbors(
                sub,
                n_neighbors=min(
                    N_NEIGHBORS,
                    sub.n_obs - 1,
                ),
                use_rep=rep,
                metric="euclidean",
                random_state=SEED,
                key_added=(
                    neighbors_key
                ),
            )

            # --------------------------------------
            # Representation-specific Leiden
            # --------------------------------------

            sc.tl.leiden(
                sub,
                resolution=(
                    PRIMARY_RESOLUTION
                ),
                key_added=(
                    cluster_col
                ),
                random_state=SEED,
                flavor="igraph",
                n_iterations=2,
                neighbors_key=(
                    neighbors_key
                ),
            )

            # --------------------------------------
            # Cluster QC
            # --------------------------------------

            qc = (
                trajectory_cluster_qc(
                    sub,
                    cluster_col,
                )
            )

            qc.insert(
                0,
                "representation",
                rep,
            )

            qc.insert(
                0,
                "trajectory",
                trajectory_name,
            )

            qc.insert(
                0,
                "compartment",
                compartment,
            )

            representation_cluster_qc.append(qc)

            singleton = (
                qc.loc[
                    qc[
                        "n_metacells"
                    ]
                    < 2,
                    "cluster",
                ]
                .astype(str)
                .tolist()
            )

            if singleton:
                raise ValueError(
                    f"{compartment}/"
                    f"{trajectory_name}/"
                    f"{rep}: "
                    "singleton clusters "
                    f"{singleton}"
                )

            # --------------------------------------
            # Map audited biological root
            # --------------------------------------

            (
                mapped_root,
                n_original_root,
                n_root_mapped,
                mapping_fraction,
            ) = map_root_cluster(
                sub_adata=sub,
                original_cluster_col=(
                    original_cluster_col
                ),
                trajectory_cluster_col=(
                    cluster_col
                ),
                original_root_cluster=(
                    spec[
                        "root"
                    ][
                        "cluster"
                    ]
                ),
            )

            cluster_labels = (
                sub.obs[
                    cluster_col
                ]
                .astype(str)
                .to_numpy()
            )

            # --------------------------------------
            # Slingshot sensitivity fit
            #
            # dist.method="simple" is deliberately
            # used for BOTH z and scVI.
            # --------------------------------------

            (
                result_df,
                lineage_df,
            ) = (
                run_slingshot_representation_sensitivity(
                    rd=rd,
                    cluster_labels=(
                        cluster_labels
                    ),
                    start_clus=(
                        mapped_root
                    ),
                    approx_points=(
                        SLINGSHOT_APPROX_POINTS
                    ),
                )
            )

            # --------------------------------------
            # Extract sensitivity pseudotimes
            # --------------------------------------

            sensitivity_pt_cols = [
                col
                for col
                in result_df.columns
                if col.startswith(
                    "pt__"
                )
            ]

            sensitivity_names = [
                col.replace(
                    "pt__",
                    "",
                    1,
                )
                for col
                in sensitivity_pt_cols
            ]

            if (
                not sensitivity_pt_cols
            ):
                raise ValueError(
                    f"{compartment}/"
                    f"{trajectory_name}/"
                    f"{rep}: "
                    "Slingshot returned "
                    "no pseudotime."
                )

            sensitivity_pt = (
                result_df[
                    sensitivity_pt_cols
                ]
                .to_numpy(
                    dtype=float
                )
            )

            # --------------------------------------
            # Match sensitivity lineages to primary
            #
            # Clusters differ across representations,
            # so cluster-path labels cannot be
            # compared directly. Match by maximal
            # pseudotime rank agreement instead.
            # --------------------------------------

            matches = (
                match_lineages_by_correlation(
                    primary_matrix=(
                        primary_pt
                    ),
                    sensitivity_matrix=(
                        sensitivity_pt
                    ),
                    primary_names=(
                        primary_names
                    ),
                    sensitivity_names=(
                        sensitivity_names
                    ),
                )
            )

            if (
                not matches.empty
            ):
                matches.insert(
                    0,
                    "representation",
                    rep,
                )

                matches.insert(
                    0,
                    "trajectory",
                    trajectory_name,
                )

                matches.insert(
                    0,
                    "compartment",
                    compartment,
                )

                representation_sensitivity_lineages.append(matches)

            # --------------------------------------
            # Summary statistics
            # --------------------------------------

            if (
                not matches.empty
            ):
                rhos = (
                    matches[
                        "spearman_rho"
                    ]
                    .to_numpy(
                        dtype=float
                    )
                )
            else:
                rhos = np.array(
                    [],
                    dtype=float,
                )

            finite_rhos = (
                rhos[
                    np.isfinite(rhos)
                ]
            )

            if (
                finite_rhos.size
                > 0
            ):
                median_rho = float(
                    np.median(finite_rhos)
                )

                min_rho = float(
                    np.min(finite_rhos)
                )

                max_rho = float(
                    np.max(finite_rhos)
                )
            else:
                median_rho = np.nan
                min_rho = np.nan
                max_rho = np.nan

            representation_sensitivity_summary.append(
                {
                    "compartment":
                        compartment,

                    "trajectory":
                        trajectory_name,

                    "representation":
                        rep,

                    "slingshot_dist_method":
                        "simple",

                    "n_dimensions":
                        int(
                            rd.shape[1]
                        ),

                    "n_metacells":
                        int(
                            sub.n_obs
                        ),

                    "n_donors":
                        int(
                            sub.obs[
                                DONOR_COL
                            ].nunique()
                        ),

                    "n_clusters":
                        int(
                            sub.obs[
                                cluster_col
                            ].nunique()
                        ),

                    "mapped_root_cluster":
                        mapped_root,

                    "root_mapping_fraction":
                        mapping_fraction,

                    "n_original_root_metacells":
                        n_original_root,

                    "n_root_metacells_mapped":
                        n_root_mapped,

                    "n_primary_lineages":
                        int(
                            len(primary_names)
                        ),

                    "n_sensitivity_lineages":
                        int(
                            len(sensitivity_names)
                        ),

                    "n_matched_lineages":
                        int(
                            len(matches)
                        ),

                    "median_spearman_rho":
                        median_rho,

                    "min_spearman_rho":
                        min_rho,

                    "max_spearman_rho":
                        max_rho,
                }
            )

            if (
                finite_rhos.size
                > 0
            ):
                print(
                    f"    dims="
                    f"{rd.shape[1]}; "
                    f"clusters="
                    f"{sub.obs[cluster_col].nunique()}; "
                    f"lineages="
                    f"{len(sensitivity_names)}; "
                    f"root map="
                    f"{mapping_fraction:.1%}; "
                    f"median rho="
                    f"{median_rho:.3f}"
                )

            else:
                print(
                    f"    dims="
                    f"{rd.shape[1]}; "
                    f"clusters="
                    f"{sub.obs[cluster_col].nunique()}; "
                    f"lineages="
                    f"{len(sensitivity_names)}; "
                    f"root map="
                    f"{mapping_fraction:.1%}; "
                    "no finite lineage "
                    "correlations"
                )


# ----------------------------------------------
# Combine outputs
# ----------------------------------------------

representation_sensitivity_summary = (
    pd.DataFrame(representation_sensitivity_summary)
)


if (
    representation_sensitivity_lineages
):
    representation_sensitivity_lineages = (
        pd.concat(
            representation_sensitivity_lineages,
            ignore_index=True,
        )
    )
else:
    representation_sensitivity_lineages = (
        pd.DataFrame(
            columns=[
                "compartment",
                "trajectory",
                "representation",
                "primary_lineage",
                "sensitivity_lineage",
                "spearman_rho",
                "n_overlap",
            ]
        )
    )


if (
    representation_cluster_qc
):
    representation_cluster_qc = (
        pd.concat(
            representation_cluster_qc,
            ignore_index=True,
        )
    )
else:
    representation_cluster_qc = (
        pd.DataFrame(
            columns=[
                "compartment",
                "trajectory",
                "representation",
                "cluster",
                "n_metacells",
                "n_donors",
            ]
        )
    )


# ----------------------------------------------
# Save
# ----------------------------------------------

representation_sensitivity_summary.to_csv(
    TRAJECTORY_DIR
    / (
        "trajectory_representation_"
        "sensitivity_summary.csv"
    ),
    index=False,
)


representation_sensitivity_lineages.to_csv(
    TRAJECTORY_DIR
    / (
        "trajectory_representation_"
        "sensitivity_lineages.csv"
    ),
    index=False,
)


representation_cluster_qc.to_csv(
    TRAJECTORY_DIR
    / (
        "trajectory_representation_"
        "cluster_qc.csv"
    ),
    index=False,
)


# ----------------------------------------------
# Final console summary
# ----------------------------------------------

print(
    "\n"
    + "=" * 100
)

print(
    "REPRESENTATION SENSITIVITY COMPLETE"
)

print(
    "=" * 100
)


summary_cols = [
    "compartment",
    "trajectory",
    "representation",
    "slingshot_dist_method",
    "n_dimensions",
    "n_clusters",
    "root_mapping_fraction",
    "n_primary_lineages",
    "n_sensitivity_lineages",
    "n_matched_lineages",
    "median_spearman_rho",
    "min_spearman_rho",
    "max_spearman_rho",
]


print(
    representation_sensitivity_summary[
        summary_cols
    ].to_string(
        index=False
    )
)


# %% DPT SENSITIVITY
#
# Independent trajectory-ordering sensitivity.
#
# DPT uses the primary X_decipher_v geometry.
# The root metacell is chosen as the medoid-like
# observation nearest the centroid of the previously
# audited original root cluster.
#
# Disease labels are not used.

dpt_summary = []
dpt_lineage_comparison = []
dpt_metacell_tables = []

for compartment, specs in TRAJECTORY_SPECS.items():
    print(
        "\n"
        + "=" * 100
    )
    print(f"{compartment} DPT SENSITIVITY")
    print(
        "=" * 100
    )

    adata = sc.read_h5ad(
        TRAJECTORY_DIR
        / (
            f"{compartment}_"
            "trajectory_primary.h5ad"
        )
    )

    original_cluster_col = (
        f"trajectory_leiden_"
        f"{str(PRIMARY_RESOLUTION).replace('.', 'p')}"
    )

    for trajectory_name, spec in specs.items():
        print(f"\n{trajectory_name}")

        sub = get_trajectory_subset(
            adata,
            trajectory_name,
            spec,
        )

        rd = numeric_rep(
            sub,
            PRIMARY_REP,
        )

        original_labels = (
            sub.obs[
                original_cluster_col
            ]
            .astype(str)
        )

        root_mask = (
            original_labels
            == str(
                spec[
                    "root"
                ][
                    "cluster"
                ]
            )
        ).to_numpy()

        if root_mask.sum() == 0:
            raise ValueError(
                f"{compartment}/"
                f"{trajectory_name}: "
                "audited root cluster absent "
                "for DPT."
            )

        root_coords = rd[
            root_mask
        ]

        root_centroid = (
            root_coords.mean(axis=0)
        )

        root_dist = np.linalg.norm(
            root_coords
            - root_centroid,
            axis=1,
        )

        root_subset_indices = (
            np.flatnonzero(root_mask)
        )

        root_index = int(
            root_subset_indices[
                np.argmin(root_dist)
            ]
        )

        root_obs_id = str(
            sub.obs_names[
                root_index
            ]
        )

        dpt_neighbors_key = (
            f"dpt_"
            f"{safe_name(trajectory_name)}"
            "_neighbors"
        )

        sc.pp.neighbors(
            sub,
            n_neighbors=min(
                N_NEIGHBORS,
                sub.n_obs - 1,
            ),
            use_rep=PRIMARY_REP,
            metric="euclidean",
            random_state=SEED,
            key_added=(
                dpt_neighbors_key
            ),
        )

        sub.uns[
            "iroot"
        ] = root_index

        sc.tl.diffmap(
            sub,
            neighbors_key=(
                dpt_neighbors_key
            ),
        )

        sc.tl.dpt(
            sub,
            neighbors_key=(
                dpt_neighbors_key
            ),
        )

        dpt = (
            sub.obs[
                "dpt_pseudotime"
            ]
            .to_numpy(
                dtype=float
            )
        )

        finite_dpt = (
            np.isfinite(dpt)
        )

        primary_cols = (
            primary_pt_columns(
                sub,
                trajectory_name,
            )
        )

        lineage_rhos = []

        for primary_col in primary_cols:
            lineage = (
                extract_primary_lineage_name(
                    primary_col,
                    trajectory_name,
                )
            )

            primary_pt = (
                sub.obs[
                    primary_col
                ]
                .to_numpy(
                    dtype=float
                )
            )

            rho, n_overlap = (
                finite_spearman(
                    primary_pt,
                    dpt,
                )
            )

            lineage_rhos.append(rho)

            dpt_lineage_comparison.append(
                {
                    "compartment":
                        compartment,
                    "trajectory":
                        trajectory_name,
                    "lineage":
                        lineage,
                    "root_obs_id":
                        root_obs_id,
                    "root_subclass":
                        spec[
                            "root"
                        ][
                            "subclass"
                        ],
                    "original_root_cluster":
                        str(
                            spec[
                                "root"
                            ][
                                "cluster"
                            ]
                        ),
                    "spearman_rho":
                        rho,
                    "n_overlap":
                        n_overlap,
                }
            )

        lineage_rhos = np.asarray(
            lineage_rhos,
            dtype=float,
        )

        dpt_summary.append(
            {
                "compartment":
                    compartment,
                "trajectory":
                    trajectory_name,
                "representation":
                    PRIMARY_REP,
                "root_obs_id":
                    root_obs_id,
                "root_subclass":
                    spec[
                        "root"
                    ][
                        "subclass"
                    ],
                "original_root_cluster":
                    str(
                        spec[
                            "root"
                        ][
                            "cluster"
                        ]
                    ),
                "n_root_metacells":
                    int(
                        root_mask.sum()
                    ),
                "n_metacells":
                    int(
                        sub.n_obs
                    ),
                "n_finite_dpt":
                    int(
                        finite_dpt.sum()
                    ),
                "n_primary_lineages":
                    int(
                        len(primary_cols)
                    ),
                "median_spearman_rho":
                    (
                        float(
                            np.nanmedian(lineage_rhos)
                        )
                        if np.isfinite(
                            lineage_rhos
                        ).any()
                        else np.nan
                    ),
                "max_spearman_rho":
                    (
                        float(
                            np.nanmax(lineage_rhos)
                        )
                        if np.isfinite(
                            lineage_rhos
                        ).any()
                        else np.nan
                    ),
            }
        )

        dpt_table = pd.DataFrame(
            {
                "obs_id":
                    sub.obs_names.astype(
                        str
                    ),
                "compartment":
                    compartment,
                "trajectory":
                    trajectory_name,
                "donor_id":
                    sub.obs[
                        DONOR_COL
                    ].astype(
                        str
                    ).to_numpy(),
                "dominant_subclass":
                    sub.obs[
                        SUBCLASS_COL
                    ].astype(
                        str
                    ).to_numpy(),
                "dpt_pseudotime":
                    dpt,
                "is_dpt_root":
                    (
                        np.arange(sub.n_obs)
                        == root_index
                    ),
            }
        )

        dpt_metacell_tables.append(dpt_table)

        print(
            f"root={root_obs_id}; "
            f"finite DPT="
            f"{finite_dpt.sum()}/"
            f"{sub.n_obs}; "
            f"median lineage rho="
            f"{np.nanmedian(lineage_rhos):.3f}"
            if np.isfinite(
                lineage_rhos
            ).any()
            else
            (
                f"root={root_obs_id}; "
                "no finite lineage correlations"
            )
        )


dpt_summary = pd.DataFrame(
    dpt_summary
)

dpt_lineage_comparison = pd.DataFrame(
    dpt_lineage_comparison
)

dpt_metacell_tables = pd.concat(
    dpt_metacell_tables,
    ignore_index=True,
)


dpt_summary.to_csv(
    TRAJECTORY_DIR
    / "trajectory_dpt_sensitivity_summary.csv",
    index=False,
)

dpt_lineage_comparison.to_csv(
    TRAJECTORY_DIR
    / "trajectory_dpt_lineage_comparison.csv",
    index=False,
)

dpt_metacell_tables.to_csv(
    TRAJECTORY_DIR
    / "trajectory_dpt_pseudotime.csv",
    index=False,
)


print(
    "\n"
    + "=" * 100
)
print(
    "DPT SENSITIVITY COMPLETE"
)
print(
    "=" * 100
)

print(
    dpt_summary.to_string(index=False)
)


# %% ROBUSTNESS CHECKPOINT SUMMARY

print(
    "\n"
    + "=" * 100
)
print(
    "04c TRAJECTORY ROBUSTNESS CHECKPOINT COMPLETE"
)
print(
    "=" * 100
)

print(
    "\nROOT SENSITIVITY"
)

print(
    root_sensitivity_summary[
        [
            "compartment",
            "trajectory",
            "sensitivity_original_root_cluster",
            "root_mapping_fraction",
            "n_primary_lineages",
            "n_sensitivity_lineages",
            "median_spearman_rho",
            "min_spearman_rho",
            "median_path_jaccard",
        ]
    ].to_string(
        index=False
    )
)

print(
    "\nREPRESENTATION SENSITIVITY"
)

print(
    representation_sensitivity_summary[
        [
            "compartment",
            "trajectory",
            "representation",
            "n_dimensions",
            "n_clusters",
            "root_mapping_fraction",
            "n_primary_lineages",
            "n_sensitivity_lineages",
            "n_matched_lineages",
            "median_spearman_rho",
            "min_spearman_rho",
        ]
    ].to_string(
        index=False
    )
)

print(
    "\nDPT SENSITIVITY"
)

print(
    dpt_summary[
        [
            "compartment",
            "trajectory",
            "n_metacells",
            "n_finite_dpt",
            "n_primary_lineages",
            "median_spearman_rho",
            "max_spearman_rho",
        ]
    ].to_string(
        index=False
    )
)

