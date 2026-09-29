# %% 04C03 TRAJECTORY-SUBSAMPLING
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


from scipy.optimize import linear_sum_assignment
from scipy.stats import spearmanr

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


# %% DONOR SUBSAMPLING ROBUSTNESS
#
# Purpose
# -------
# Test whether the five frozen primary trajectories remain reproducible
# after perturbing the biological replicate composition.
#
# For each trajectory:
#   1. sample 80% of donors WITHOUT replacement
#   2. retain all metacells from those donors
#   3. rebuild KNN + Leiden in X_decipher_v
#   4. map the frozen audited biological root into the new clustering
#   5. rerun Slingshot
#   6. compare resampled pseudotime against the frozen primary
#      pseudotime on the SAME retained metacells
#
# Important:
#   - disease labels are NOT used
#   - donors, not metacells, are sampled
#   - no replacement is used for geometry robustness
#   - primary trajectories are never modified
#   - lineage labels are NOT assumed to correspond across resamples
#   - lineage matching is one-to-one by pseudotime rank agreement
#
# Slingshot:
#   dist.method="simple" is used for all donor-resampling fits.
#   This avoids covariance-inversion failures in small/resampled clusters
#   without adding jitter or modifying the representation.

# DONOR_SUBSAMPLE_N = 1
# DONOR_SUBSAMPLE_N_JOBS = 1

DONOR_SUBSAMPLE_N = 100
DONOR_SUBSAMPLE_FRACTION = 0.80
DONOR_SUBSAMPLE_BASE_SEED = 20260920

DONOR_SUBSAMPLE_MIN_DONORS = 2
DONOR_SUBSAMPLE_MIN_METACELLS = 20

# Parallelism:
# Slingshot launches R subprocesses, so avoid excessive oversubscription.
DONOR_SUBSAMPLE_N_JOBS = min(
    8,
    max(
        1,
        (os.cpu_count() or 2) // 2,
    ),
)


def run_slingshot_donor_subsample(
    rd,
    cluster_labels,
    start_clus,
    approx_points=SLINGSHOT_APPROX_POINTS,
):
    """
    Slingshot wrapper for donor-subsampling robustness.

    Uses dist.method='simple' consistently for every resample.
    """

    rd = np.asarray(
        rd,
        dtype=float,
    )

    cluster_labels = np.asarray(
        cluster_labels,
        dtype=str,
    )

    if rd.ndim != 2:
        raise ValueError(
            f"rd must be 2D; got {rd.shape}"
        )

    if (
        rd.shape[0]
        != len(cluster_labels)
    ):
        raise ValueError(
            "Coordinate/cluster length mismatch: "
            f"{rd.shape[0]} vs "
            f"{len(cluster_labels)}"
        )

    if not np.isfinite(rd).all():
        raise ValueError(
            "Donor-subsample coordinates "
            "contain non-finite values."
        )

    if (
        str(start_clus)
        not in set(cluster_labels)
    ):
        raise ValueError(
            f"Root cluster {start_clus!r} "
            "is absent."
        )

    rscript = check_slingshot()

    with tempfile.TemporaryDirectory(
        prefix="04c_donor_subsample_"
    ) as tmpdir:

        tmp_path = Path(
            tmpdir
        )

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
                "Donor-subsample "
                "Slingshot failed:\n"
                f"STDOUT:\n"
                f"{res.stdout}\n"
                f"STDERR:\n"
                f"{res.stderr}"
            )

        result_df = pd.read_csv(
            output_file
        )

        lineage_df = pd.read_csv(
            lineage_file
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


def make_donor_subsample_schedule(
    donors,
    n_iterations,
    fraction,
    base_seed,
):
    """
    Precompute deterministic donor subsets.

    This is done outside worker processes so parallel scheduling
    cannot affect random-number generation.
    """

    donors = np.asarray(
        sorted(
            map(
                str,
                donors,
            )
        ),
        dtype=object,
    )

    n_total = len(
        donors
    )

    if (
        n_total
        < DONOR_SUBSAMPLE_MIN_DONORS
    ):
        raise ValueError(
            "Too few donors for "
            "donor subsampling."
        )

    n_keep = int(
        np.floor(
            n_total
            * fraction
        )
    )

    n_keep = max(
        DONOR_SUBSAMPLE_MIN_DONORS,
        n_keep,
    )

    n_keep = min(
        n_total,
        n_keep,
    )

    schedule = []

    for iteration in range(
        1,
        n_iterations + 1,
    ):
        seed = int(
            base_seed
            + iteration
        )

        rng = np.random.default_rng(
            seed
        )

        selected = rng.choice(
            donors,
            size=n_keep,
            replace=False,
        )

        schedule.append(
            {
                "iteration":
                    iteration,
                "seed":
                    seed,
                "donors":
                    tuple(
                        sorted(
                            map(
                                str,
                                selected,
                            )
                        )
                    ),
            }
        )

    return schedule


def match_subsample_lineages(
    primary_pt,
    subsample_pt,
    primary_names,
    subsample_names,
):
    """
    One-to-one lineage matching by positive Spearman agreement.

    Both matrices contain exactly the same retained metacells.

    The Hungarian assignment maximizes rank correlation while
    preventing a single primary lineage from being matched to
    multiple subsample lineages.
    """

    primary_pt = np.asarray(
        primary_pt,
        dtype=float,
    )

    subsample_pt = np.asarray(
        subsample_pt,
        dtype=float,
    )

    n_primary = len(
        primary_names
    )

    n_subsample = len(
        subsample_names
    )

    rho_matrix = np.full(
        (
            n_primary,
            n_subsample,
        ),
        np.nan,
        dtype=float,
    )

    overlap_matrix = np.zeros(
        (
            n_primary,
            n_subsample,
        ),
        dtype=int,
    )

    for i in range(
        n_primary
    ):
        for j in range(
            n_subsample
        ):
            rho, n_overlap = (
                finite_spearman(
                    primary_pt[
                        :,
                        i
                    ],
                    subsample_pt[
                        :,
                        j
                    ],
                )
            )

            rho_matrix[
                i,
                j
            ] = rho

            overlap_matrix[
                i,
                j
            ] = n_overlap

    if (
        rho_matrix.size == 0
        or np.all(
            ~np.isfinite(
                rho_matrix
            )
        )
    ):
        return pd.DataFrame(
            columns=[
                "primary_lineage",
                "subsample_lineage",
                "spearman_rho",
                "n_overlap",
            ]
        )

    # Maximize positive correlation.
    #
    # Non-finite comparisons receive a prohibitive cost.
    cost = np.where(
        np.isfinite(
            rho_matrix
        ),
        -rho_matrix,
        1e6,
    )

    rows, cols = (
        linear_sum_assignment(cost)
    )

    records = []

    for i, j in zip(
        rows,
        cols,
    ):
        rho = (
            rho_matrix[
                i,
                j
            ]
        )

        if not np.isfinite(
            rho
        ):
            continue

        records.append(
            {
                "primary_lineage":
                    str(
                        primary_names[
                            i
                        ]
                    ),
                "subsample_lineage":
                    str(
                        subsample_names[
                            j
                        ]
                    ),
                "spearman_rho":
                    float(
                        rho
                    ),
                "n_overlap":
                    int(
                        overlap_matrix[
                            i,
                            j
                        ]
                    ),
            }
        )

    return pd.DataFrame(
        records
    )


def run_one_donor_subsample(
    compartment,
    trajectory_name,
    spec,
    obs_names,
    donor_values,
    original_cluster_values,
    rd_full,
    primary_pt_full,
    primary_names,
    selected_donors,
    iteration,
    iteration_seed,
):
    """
    Run one donor-subsampling replicate.

    Returns:
        replicate_summary
        lineage_match_records
    """

    selected_donors = set(
        map(
            str,
            selected_donors,
        )
    )

    donor_values = np.asarray(
        donor_values,
        dtype=str,
    )

    keep = np.asarray(
        [
            donor
            in selected_donors
            for donor
            in donor_values
        ],
        dtype=bool,
    )

    n_metacells = int(
        keep.sum()
    )

    n_donors = int(
        len(selected_donors)
    )

    base_record = {
        "compartment":
            compartment,
        "trajectory":
            trajectory_name,
        "iteration":
            int(
                iteration
            ),
        "seed":
            int(
                iteration_seed
            ),
        "fraction_requested":
            float(
                DONOR_SUBSAMPLE_FRACTION
            ),
        "n_donors":
            n_donors,
        "n_metacells":
            n_metacells,
    }

    if (
        n_metacells
        < DONOR_SUBSAMPLE_MIN_METACELLS
    ):
        return (
            {
                **base_record,
                "status":
                    "failed",
                "failure_stage":
                    "minimum_metacells",
                "error":
                    (
                        "Too few retained "
                        "metacells."
                    ),
            },
            [],
        )

    rd = np.asarray(
        rd_full[
            keep
        ],
        dtype=float,
    )

    primary_pt = np.asarray(
        primary_pt_full[
            keep
        ],
        dtype=float,
    )

    original_clusters = np.asarray(
        original_cluster_values[
            keep
        ],
        dtype=str,
    )

    retained_obs_names = np.asarray(
        obs_names[
            keep
        ],
        dtype=str,
    )

    # ------------------------------------------
    # Root availability
    # ------------------------------------------

    audited_root = str(
        spec[
            "root"
        ][
            "cluster"
        ]
    )

    original_root_mask = (
        original_clusters
        == audited_root
    )

    n_root_metacells = int(
        original_root_mask.sum()
    )

    if (
        n_root_metacells
        == 0
    ):
        return (
            {
                **base_record,
                "status":
                    "failed",
                "failure_stage":
                    "root_absent",
                "error":
                    (
                        "No audited root "
                        "metacells retained."
                    ),
                "n_original_root_metacells":
                    0,
            },
            [],
        )

    # ------------------------------------------
    # Build temporary AnnData only for graph
    # and Leiden clustering.
    # ------------------------------------------

    tmp = ad.AnnData(
        X=np.zeros(
            (
                n_metacells,
                1,
            ),
            dtype=np.float32,
        )
    )

    tmp.obs_names = (
        retained_obs_names
    )

    tmp.obsm[
        PRIMARY_REP
    ] = rd.astype(
        np.float32,
        copy=False,
    )

    neighbors_key = (
        "donor_subsample_neighbors"
    )

    cluster_col = (
        "donor_subsample_leiden"
    )

    try:
        sc.pp.neighbors(
            tmp,
            n_neighbors=min(
                N_NEIGHBORS,
                tmp.n_obs - 1,
            ),
            use_rep=PRIMARY_REP,
            metric="euclidean",
            random_state=(
                iteration_seed
            ),
            key_added=(
                neighbors_key
            ),
        )

        sc.tl.leiden(
            tmp,
            resolution=(
                PRIMARY_RESOLUTION
            ),
            key_added=(
                cluster_col
            ),
            random_state=(
                iteration_seed
            ),
            flavor="igraph",
            n_iterations=2,
            neighbors_key=(
                neighbors_key
            ),
        )

    except Exception as exc:
        return (
            {
                **base_record,
                "status":
                    "failed",
                "failure_stage":
                    "graph_or_leiden",
                "error":
                    repr(
                        exc
                    ),
                "n_original_root_metacells":
                    n_root_metacells,
            },
            [],
        )

    cluster_labels = (
        tmp.obs[
            cluster_col
        ]
        .astype(str)
        .to_numpy()
    )

    cluster_counts = (
        pd.Series(cluster_labels)
        .value_counts()
    )

    n_clusters = int(
        len(cluster_counts)
    )

    singleton_clusters = (
        cluster_counts[
            cluster_counts
            < 2
        ]
        .index
        .astype(str)
        .tolist()
    )

    if singleton_clusters:
        return (
            {
                **base_record,
                "status":
                    "failed",
                "failure_stage":
                    "singleton_cluster",
                "error":
                    ",".join(
                        singleton_clusters
                    ),
                "n_original_root_metacells":
                    n_root_metacells,
                "n_clusters":
                    n_clusters,
            },
            [],
        )

    # ------------------------------------------
    # Map frozen audited root into resampled
    # trajectory-specific clustering.
    # ------------------------------------------

    root_cluster_counts = (
        pd.Series(
            cluster_labels[
                original_root_mask
            ]
        )
        .value_counts()
    )

    mapped_root = str(
        root_cluster_counts.index[
            0
        ]
    )

    n_root_mapped = int(
        root_cluster_counts.iloc[
            0
        ]
    )

    root_mapping_fraction = (
        n_root_mapped
        / n_root_metacells
    )

    # ------------------------------------------
    # Slingshot
    # ------------------------------------------

    try:
        (
            result_df,
            lineage_df,
        ) = (
            run_slingshot_donor_subsample(
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

    except Exception as exc:
        return (
            {
                **base_record,
                "status":
                    "failed",
                "failure_stage":
                    "slingshot",
                "error":
                    repr(
                        exc
                    ),
                "n_original_root_metacells":
                    n_root_metacells,
                "mapped_root_cluster":
                    mapped_root,
                "n_root_metacells_mapped":
                    n_root_mapped,
                "root_mapping_fraction":
                    root_mapping_fraction,
                "n_clusters":
                    n_clusters,
            },
            [],
        )

    subsample_pt_cols = [
        col
        for col
        in result_df.columns
        if col.startswith(
            "pt__"
        )
    ]

    subsample_names = [
        col.replace(
            "pt__",
            "",
            1,
        )
        for col
        in subsample_pt_cols
    ]

    if (
        len(subsample_pt_cols)
        == 0
    ):
        return (
            {
                **base_record,
                "status":
                    "failed",
                "failure_stage":
                    "no_pseudotime",
                "error":
                    (
                        "Slingshot returned "
                        "no pseudotime columns."
                    ),
                "n_original_root_metacells":
                    n_root_metacells,
                "mapped_root_cluster":
                    mapped_root,
                "n_root_metacells_mapped":
                    n_root_mapped,
                "root_mapping_fraction":
                    root_mapping_fraction,
                "n_clusters":
                    n_clusters,
            },
            [],
        )

    subsample_pt = (
        result_df[
            subsample_pt_cols
        ]
        .to_numpy(
            dtype=float
        )
    )

    # ------------------------------------------
    # Match resampled lineages to frozen primary
    # lineages on identical retained metacells.
    # ------------------------------------------

    matches = (
        match_subsample_lineages(
            primary_pt=(
                primary_pt
            ),
            subsample_pt=(
                subsample_pt
            ),
            primary_names=(
                primary_names
            ),
            subsample_names=(
                subsample_names
            ),
        )
    )

    lineage_records = []

    if (
        not matches.empty
    ):
        for row in (
            matches.itertuples(index=False)
        ):
            lineage_records.append(
                {
                    "compartment":
                        compartment,
                    "trajectory":
                        trajectory_name,
                    "iteration":
                        int(
                            iteration
                        ),
                    "seed":
                        int(
                            iteration_seed
                        ),
                    "primary_lineage":
                        row.primary_lineage,
                    "subsample_lineage":
                        row.subsample_lineage,
                    "spearman_rho":
                        float(
                            row.spearman_rho
                        ),
                    "n_overlap":
                        int(
                            row.n_overlap
                        ),
                }
            )

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

    replicate_summary = {
        **base_record,

        "status":
            "success",

        "failure_stage":
            "",

        "error":
            "",

        "n_original_root_metacells":
            n_root_metacells,

        "mapped_root_cluster":
            mapped_root,

        "n_root_metacells_mapped":
            n_root_mapped,

        "root_mapping_fraction":
            float(
                root_mapping_fraction
            ),

        "n_clusters":
            n_clusters,

        "n_primary_lineages":
            int(
                len(primary_names)
            ),

        "n_subsample_lineages":
            int(
                len(subsample_names)
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

    return (
        replicate_summary,
        lineage_records,
    )


# %% DONOR SUBSAMPLING: RUN ALL FIVE TRAJECTORIES

donor_subsample_replicates = []
donor_subsample_lineages = []
donor_subsample_schedule_records = []


trajectory_counter = 0

for compartment, specs in (
    TRAJECTORY_SPECS.items()
):

    print(
        "\n"
        + "=" * 100
    )

    print(
        f"{compartment} "
        "DONOR SUBSAMPLING ROBUSTNESS"
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

        trajectory_counter += 1

        print(f"\n{trajectory_name}")

        sub = get_trajectory_subset(
            adata,
            trajectory_name,
            spec,
        )

        # --------------------------------------
        # Frozen primary data
        # --------------------------------------

        rd_full = numeric_rep(
            sub,
            PRIMARY_REP,
        )

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

        primary_pt_full = (
            sub.obs[
                primary_pt_cols
            ]
            .to_numpy(
                dtype=float
            )
        )

        donor_values = (
            sub.obs[
                DONOR_COL
            ]
            .astype(str)
            .to_numpy()
        )

        original_cluster_values = (
            sub.obs[
                original_cluster_col
            ]
            .astype(str)
            .to_numpy()
        )

        obs_names = (
            sub.obs_names
            .astype(str)
            .to_numpy()
        )

        unique_donors = np.unique(
            donor_values
        )

        n_total_donors = int(
            len(unique_donors)
        )

        trajectory_seed = int(
            DONOR_SUBSAMPLE_BASE_SEED
            + (
                trajectory_counter
                * 10000
            )
        )

        schedule = (
            make_donor_subsample_schedule(
                donors=(
                    unique_donors
                ),
                n_iterations=(
                    DONOR_SUBSAMPLE_N
                ),
                fraction=(
                    DONOR_SUBSAMPLE_FRACTION
                ),
                base_seed=(
                    trajectory_seed
                ),
            )
        )

        n_keep = len(
            schedule[
                0
            ][
                "donors"
            ]
        )

        print(
            f"  donors="
            f"{n_total_donors}; "
            f"retain={n_keep} "
            f"({n_keep / n_total_donors:.1%}); "
            f"iterations="
            f"{DONOR_SUBSAMPLE_N}; "
            f"workers="
            f"{DONOR_SUBSAMPLE_N_JOBS}"
        )

        for item in schedule:
            donor_subsample_schedule_records.append(
                {
                    "compartment":
                        compartment,
                    "trajectory":
                        trajectory_name,
                    "iteration":
                        item[
                            "iteration"
                        ],
                    "seed":
                        item[
                            "seed"
                        ],
                    "n_total_donors":
                        n_total_donors,
                    "n_selected_donors":
                        len(
                            item[
                                "donors"
                            ]
                        ),
                    "selected_donors":
                        "|".join(
                            item[
                                "donors"
                            ]
                        ),
                }
            )

        # --------------------------------------
        # Run donor subsets in parallel
        # --------------------------------------

        futures = {}

        with ThreadPoolExecutor(
            max_workers= DONOR_SUBSAMPLE_N_JOBS
        ) as executor:

            for item in schedule:

                future = executor.submit(
                    run_one_donor_subsample,

                    compartment,
                    trajectory_name,
                    spec,

                    obs_names,
                    donor_values,
                    original_cluster_values,

                    rd_full,
                    primary_pt_full,
                    primary_names,

                    item[
                        "donors"
                    ],
                    item[
                        "iteration"
                    ],
                    item[
                        "seed"
                    ],
                )

                futures[
                    future
                ] = item[
                    "iteration"
                ]

            completed = 0

            for future in as_completed(
                futures
            ):
                iteration = (
                    futures[
                        future
                    ]
                )

                try:
                    (
                        replicate_record,
                        lineage_records,
                    ) = future.result()

                except Exception as exc:
                    replicate_record = {
                        "compartment":
                            compartment,
                        "trajectory":
                            trajectory_name,
                        "iteration":
                            int(
                                iteration
                            ),
                        "status":
                            "failed",
                        "failure_stage":
                            "worker_exception",
                        "error":
                            repr(
                                exc
                            ),
                    }

                    lineage_records = []

                donor_subsample_replicates.append(replicate_record)

                donor_subsample_lineages.extend(lineage_records)

                completed += 1

                if (
                    completed % 10
                    == 0
                    or completed
                    == DONOR_SUBSAMPLE_N
                ):
                    current = [
                        x
                        for x
                        in donor_subsample_replicates
                        if (
                            x.get(
                                "compartment"
                            )
                            == compartment
                            and x.get(
                                "trajectory"
                            )
                            == trajectory_name
                        )
                    ]

                    n_success = sum(
                        x.get(
                            "status"
                        )
                        == "success"
                        for x in current
                    )

                    print(
                        f"    completed "
                        f"{completed:3d}/"
                        f"{DONOR_SUBSAMPLE_N}; "
                        f"successful="
                        f"{n_success:3d}"
                    )


# %% DONOR SUBSAMPLING: SAVE RAW RESULTS

donor_subsample_replicates = (
    pd.DataFrame(donor_subsample_replicates)
)

donor_subsample_lineages = (
    pd.DataFrame(donor_subsample_lineages)
)

donor_subsample_schedule = (
    pd.DataFrame(donor_subsample_schedule_records)
)


# Deterministic row ordering after parallel execution.
if (
    not donor_subsample_replicates.empty
):
    donor_subsample_replicates = (
        donor_subsample_replicates
        .sort_values(
            [
                "compartment",
                "trajectory",
                "iteration",
            ]
        )
        .reset_index(
            drop=True
        )
    )


if (
    not donor_subsample_lineages.empty
):
    donor_subsample_lineages = (
        donor_subsample_lineages
        .sort_values(
            [
                "compartment",
                "trajectory",
                "iteration",
                "primary_lineage",
            ]
        )
        .reset_index(
            drop=True
        )
    )


donor_subsample_schedule = (
    donor_subsample_schedule
    .sort_values(
        [
            "compartment",
            "trajectory",
            "iteration",
        ]
    )
    .reset_index(
        drop=True
    )
)


donor_subsample_replicates.to_csv(
    TRAJECTORY_DIR
    / (
        "trajectory_donor_subsampling_"
        "replicates.csv"
    ),
    index=False,
)


donor_subsample_lineages.to_csv(
    TRAJECTORY_DIR
    / (
        "trajectory_donor_subsampling_"
        "lineages.csv"
    ),
    index=False,
)


donor_subsample_schedule.to_csv(
    TRAJECTORY_DIR
    / (
        "trajectory_donor_subsampling_"
        "schedule.csv"
    ),
    index=False,
)


# %% DONOR SUBSAMPLING: TRAJECTORY-LEVEL SUMMARY


def q05(x):
    x = pd.to_numeric(
        x,
        errors="coerce",
    )

    x = x[
        np.isfinite(x)
    ]

    if (
        len(x)
        == 0
    ):
        return np.nan

    return float(
        np.quantile(
            x,
            0.05,
        )
    )


def q25(x):
    x = pd.to_numeric(
        x,
        errors="coerce",
    )

    x = x[
        np.isfinite(x)
    ]

    if (
        len(x)
        == 0
    ):
        return np.nan

    return float(
        np.quantile(
            x,
            0.25,
        )
    )


def q75(x):
    x = pd.to_numeric(
        x,
        errors="coerce",
    )

    x = x[
        np.isfinite(x)
    ]

    if (
        len(x)
        == 0
    ):
        return np.nan

    return float(
        np.quantile(
            x,
            0.75,
        )
    )


def q95(x):
    x = pd.to_numeric(
        x,
        errors="coerce",
    )

    x = x[
        np.isfinite(x)
    ]

    if (
        len(x)
        == 0
    ):
        return np.nan

    return float(
        np.quantile(
            x,
            0.95,
        )
    )


summary_records = []


for (
    compartment,
    trajectory_name,
), group in (
    donor_subsample_replicates
    .groupby(
        [
            "compartment",
            "trajectory",
        ],
        observed=True,
    )
):

    success = group[
        group[
            "status"
        ]
        == "success"
    ].copy()

    n_attempted = int(
        len(group)
    )

    n_success = int(
        len(success)
    )

    n_failed = (
        n_attempted
        - n_success
    )

    record = {
        "compartment":
            compartment,

        "trajectory":
            trajectory_name,

        "n_attempted":
            n_attempted,

        "n_success":
            n_success,

        "n_failed":
            n_failed,

        "success_fraction":
            (
                n_success
                / n_attempted
                if n_attempted
                else np.nan
            ),
    }

    if (
        n_success
        > 0
    ):
        for column in [
            "n_donors",
            "n_metacells",
            "n_clusters",
            "n_subsample_lineages",
            "n_matched_lineages",
            "root_mapping_fraction",
            "median_spearman_rho",
            "min_spearman_rho",
            "max_spearman_rho",
        ]:
            values = pd.to_numeric(
                success[
                    column
                ],
                errors="coerce",
            )

            finite = values[
                np.isfinite(values)
            ]

            if (
                len(finite)
                == 0
            ):
                record[
                    f"{column}_median"
                ] = np.nan

                record[
                    f"{column}_q05"
                ] = np.nan

                record[
                    f"{column}_q95"
                ] = np.nan

                continue

            record[
                f"{column}_median"
            ] = float(
                np.median(finite)
            )

            record[
                f"{column}_q05"
            ] = q05(
                finite
            )

            record[
                f"{column}_q95"
            ] = q95(
                finite
            )

    summary_records.append(record)


donor_subsample_summary = (
    pd.DataFrame(summary_records)
)


donor_subsample_summary.to_csv(
    TRAJECTORY_DIR
    / (
        "trajectory_donor_subsampling_"
        "summary.csv"
    ),
    index=False,
)


# %% DONOR SUBSAMPLING: PRIMARY-LINEAGE STABILITY

lineage_stability_records = []


if (
    not donor_subsample_lineages.empty
):

    for (
        compartment,
        trajectory_name,
        primary_lineage,
    ), group in (
        donor_subsample_lineages
        .groupby(
            [
                "compartment",
                "trajectory",
                "primary_lineage",
            ],
            observed=True,
        )
    ):

        trajectory_success = (
            donor_subsample_replicates[
                (
                    donor_subsample_replicates[
                        "compartment"
                    ]
                    == compartment
                )
                &
                (
                    donor_subsample_replicates[
                        "trajectory"
                    ]
                    == trajectory_name
                )
                &
                (
                    donor_subsample_replicates[
                        "status"
                    ]
                    == "success"
                )
            ]
        )

        n_successful_runs = int(
            len(trajectory_success)
        )

        # Hungarian matching yields at most one match
        # for this primary lineage per successful run.
        n_runs_matched = int(
            group[
                "iteration"
            ]
            .nunique()
        )

        rhos = pd.to_numeric(
            group[
                "spearman_rho"
            ],
            errors="coerce",
        )

        rhos = rhos[
            np.isfinite(rhos)
        ]

        lineage_stability_records.append(
            {
                "compartment":
                    compartment,

                "trajectory":
                    trajectory_name,

                "primary_lineage":
                    primary_lineage,

                "n_successful_runs":
                    n_successful_runs,

                "n_runs_matched":
                    n_runs_matched,

                "match_fraction":
                    (
                        n_runs_matched
                        / n_successful_runs
                        if n_successful_runs
                        else np.nan
                    ),

                "spearman_median":
                    (
                        float(
                            np.median(rhos)
                        )
                        if len(
                            rhos
                        )
                        else np.nan
                    ),

                "spearman_q05":
                    q05(
                        rhos
                    ),

                "spearman_q25":
                    q25(
                        rhos
                    ),

                "spearman_q75":
                    q75(
                        rhos
                    ),

                "spearman_q95":
                    q95(
                        rhos
                    ),

                "spearman_min":
                    (
                        float(
                            np.min(rhos)
                        )
                        if len(
                            rhos
                        )
                        else np.nan
                    ),

                "spearman_max":
                    (
                        float(
                            np.max(rhos)
                        )
                        if len(
                            rhos
                        )
                        else np.nan
                    ),
            }
        )


donor_subsample_lineage_stability = (
    pd.DataFrame(lineage_stability_records)
)


donor_subsample_lineage_stability.to_csv(
    TRAJECTORY_DIR
    / (
        "trajectory_donor_subsampling_"
        "lineage_stability.csv"
    ),
    index=False,
)


# %% DONOR SUBSAMPLING: FAILURE DIAGNOSTICS

failure_diagnostics = (
    donor_subsample_replicates[
        donor_subsample_replicates[
            "status"
        ]
        != "success"
    ]
    .copy()
)


if (
    not failure_diagnostics.empty
):

    failure_summary = (
        failure_diagnostics
        .groupby(
            [
                "compartment",
                "trajectory",
                "failure_stage",
            ],
            observed=True,
        )
        .size()
        .rename(
            "n_failures"
        )
        .reset_index()
    )

else:

    failure_summary = pd.DataFrame(
        columns=[
            "compartment",
            "trajectory",
            "failure_stage",
            "n_failures",
        ]
    )


failure_summary.to_csv(
    TRAJECTORY_DIR
    / (
        "trajectory_donor_subsampling_"
        "failures.csv"
    ),
    index=False,
)


# %% DONOR SUBSAMPLING: FINAL CHECKPOINT

print(
    "\n"
    + "=" * 100
)

print(
    "04c DONOR SUBSAMPLING ROBUSTNESS COMPLETE"
)

print(
    "=" * 100
)


checkpoint_cols = [
    "compartment",
    "trajectory",
    "n_attempted",
    "n_success",
    "n_failed",
    "success_fraction",
    "n_clusters_median",
    "n_clusters_q05",
    "n_clusters_q95",
    "n_subsample_lineages_median",
    "n_subsample_lineages_q05",
    "n_subsample_lineages_q95",
    "root_mapping_fraction_median",
    "root_mapping_fraction_q05",
    "root_mapping_fraction_q95",
    "median_spearman_rho_median",
    "median_spearman_rho_q05",
    "median_spearman_rho_q95",
    "min_spearman_rho_median",
    "min_spearman_rho_q05",
]


available_checkpoint_cols = [
    col
    for col
    in checkpoint_cols
    if col
    in donor_subsample_summary.columns
]


print(
    "\nTRAJECTORY-LEVEL ROBUSTNESS"
)

print(
    donor_subsample_summary[
        available_checkpoint_cols
    ].to_string(
        index=False
    )
)


print(
    "\nPRIMARY-LINEAGE ROBUSTNESS"
)

if (
    donor_subsample_lineage_stability.empty
):
    print(
        "No lineage-stability "
        "records available."
    )

else:
    print(
        donor_subsample_lineage_stability[
            [
                "compartment",
                "trajectory",
                "primary_lineage",
                "n_successful_runs",
                "n_runs_matched",
                "match_fraction",
                "spearman_median",
                "spearman_q05",
                "spearman_q95",
                "spearman_min",
            ]
        ].to_string(
            index=False
        )
    )


print(
    "\nFAILURE DIAGNOSTICS"
)

if (
    failure_summary.empty
):
    print(
        "No failed donor-subsampling runs."
    )

else:
    print(
        failure_summary.to_string(index=False)
    )


print(
    "\nSaved:"
)

for filename in [
    "trajectory_donor_subsampling_replicates.csv",
    "trajectory_donor_subsampling_lineages.csv",
    "trajectory_donor_subsampling_schedule.csv",
    "trajectory_donor_subsampling_summary.csv",
    "trajectory_donor_subsampling_lineage_stability.csv",
    "trajectory_donor_subsampling_failures.csv",
]:
    print(
        " ",
        TRAJECTORY_DIR
        / filename,
    )
# %%
# =============================================================================
# SUPPLEMENTARY FIGURE S7
# Trajectory robustness across sensitivity analyses
# =============================================================================

import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

ROBUSTNESS_PLOT_DIR = PLOT_DIR / "supplementary"
ROBUSTNESS_PLOT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------
# Input tables
# ---------------------------------------------------------------------
root_df = pd.read_csv(
    TRAJECTORY_DIR / "trajectory_root_sensitivity_lineages.csv"
)

rep_df = pd.read_csv(
    TRAJECTORY_DIR / "trajectory_representation_sensitivity_lineages.csv"
)

dpt_df = pd.read_csv(
    TRAJECTORY_DIR / "trajectory_dpt_lineage_comparison.csv"
)

sub_df = pd.read_csv(
    TRAJECTORY_DIR / "trajectory_donor_subsampling_lineage_stability.csv"
)

sub_summary = pd.read_csv(
    TRAJECTORY_DIR / "trajectory_donor_subsampling_summary.csv"
)

fail_df = pd.read_csv(
    TRAJECTORY_DIR / "trajectory_donor_subsampling_failures.csv"
)

# ---------------------------------------------------------------------
# Display order
# ---------------------------------------------------------------------
trajectory_order = [
    ("PT", "PT"),
    ("FIB", "contractile"),
    ("FIB", "outer_medullary"),
    ("Glomerular", "podocyte_PEC"),
    ("Glomerular", "endothelial_mesangial"),
]

trajectory_label = {
    ("PT", "PT"): "PT",
    ("FIB", "contractile"): "Contractile FIB",
    ("FIB", "outer_medullary"): "Outer-medullary FIB",
    ("Glomerular", "podocyte_PEC"): "Podocyte–PEC",
    ("Glomerular", "endothelial_mesangial"): "Endothelial–mesangial",
}

def lineage_number(x):
    x = str(x).replace("Lineage", "")
    try:
        return int(x)
    except Exception:
        return 999

# construct global ordered lineage list
lineage_keys = []

for comp, traj in trajectory_order:
    candidate = set()

    for df, col in [
        (root_df, "primary_lineage"),
        (rep_df, "primary_lineage"),
        (dpt_df, "lineage"),
        (sub_df, "primary_lineage"),
    ]:
        ss = df.loc[
            (df["compartment"] == comp)
            & (df["trajectory"] == traj),
            col,
        ]
        candidate.update(ss.dropna().astype(str))

    for lin in sorted(candidate, key=lineage_number):
        lineage_keys.append((comp, traj, lin))

lineage_to_y = {
    key: i for i, key in enumerate(lineage_keys[::-1])
}

def pretty_lineage(comp, traj, lin):
    return f"{trajectory_label[(comp, traj)]} {str(lin).replace('Lineage', 'L')}"

y_labels = [
    pretty_lineage(*key)
    for key in lineage_keys[::-1]
]

# separator positions between trajectories
separator_y = []
prev = None

for i, key in enumerate(lineage_keys[::-1]):
    group = key[:2]
    if prev is not None and group != prev:
        separator_y.append(i - 0.5)
    prev = group


# ---------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------
fig = plt.figure(figsize=(15, 16))

gs = fig.add_gridspec(
    2,
    2,
    width_ratios=[1.10, 0.90],
    height_ratios=[1.00, 1.05],
    hspace=0.28,
    wspace=0.28,
)

cmap = mpl.cm.get_cmap("coolwarm")
norm = mpl.colors.Normalize(vmin=-1, vmax=1)

# =====================================================================
# A. Alternative-root sensitivity
# =====================================================================
ax = fig.add_subplot(gs[0, 0])

root_specs = (
    root_df[
        [
            "compartment",
            "trajectory",
            "sensitivity_root_subclass",
            "sensitivity_original_root_cluster",
        ]
    ]
    .drop_duplicates()
    .reset_index(drop=True)
)

root_specs["root_label"] = (
    root_specs["sensitivity_root_subclass"].astype(str)
    + " / cluster "
    + root_specs["sensitivity_original_root_cluster"].astype(str)
)

markers = ["o", "s", "^", "D", "P", "X", "v", "<", ">"]

root_marker = {
    label: markers[i % len(markers)]
    for i, label in enumerate(root_specs["root_label"].unique())
}

for _, row in root_df.iterrows():

    key = (
        row["compartment"],
        row["trajectory"],
        row["primary_lineage"],
    )

    if key not in lineage_to_y:
        continue

    root_label = (
        str(row["sensitivity_root_subclass"])
        + " / cluster "
        + str(row["sensitivity_original_root_cluster"])
    )

    # Path Jaccard encoded as point size
    size = 25 + 95 * float(row["path_jaccard"])

    ax.scatter(
        row["spearman_rho"],
        lineage_to_y[key],
        s=size,
        marker=root_marker[root_label],
        c=[cmap(norm(row["spearman_rho"]))],
        edgecolor="black",
        linewidth=0.45,
        alpha=0.9,
    )

ax.axvline(0, color="0.5", lw=0.7)
ax.axvline(0.7, color="0.65", lw=0.8, ls="--")
ax.axvline(0.9, color="0.65", lw=0.8, ls=":")

ax.set_xlim(-1.05, 1.05)
ax.set_yticks(range(len(y_labels)))
ax.set_yticklabels(y_labels, fontsize=7)
ax.set_xlabel("Spearman correlation with primary pseudotime")
ax.set_title(
    "A  Alternative-root sensitivity",
    loc="left",
    fontweight="bold",
)

for y in separator_y:
    ax.axhline(y, color="0.85", lw=0.7)

ax.grid(axis="x", alpha=0.15)

# Do not create huge legend with every root in main plotting area.
# Instead show size key only; root identity remains available in source data.
size_handles = [
    plt.scatter([], [], s=25 + 95*j, facecolor="white",
                edgecolor="black", linewidth=0.5,
                label=f"Jaccard {j:.2f}")
    for j in [0.25, 0.50, 0.75, 1.00]
]

ax.legend(
    handles=size_handles,
    title="Path overlap",
    fontsize=7,
    title_fontsize=8,
    loc="lower left",
    frameon=False,
)


# =====================================================================
# B. Representation sensitivity
# =====================================================================
ax = fig.add_subplot(gs[0, 1])

representations = ["X_decipher_z", "X_scVI"]

rep_display = {
    "X_decipher_z": "DECIPHER z",
    "X_scVI": "scVI",
}

matrix = np.full(
    (len(lineage_keys[::-1]), len(representations)),
    np.nan,
)

for _, row in rep_df.iterrows():

    key = (
        row["compartment"],
        row["trajectory"],
        row["primary_lineage"],
    )

    if key not in lineage_to_y:
        continue

    yi = lineage_to_y[key]

    try:
        xi = representations.index(row["representation"])
    except ValueError:
        continue

    matrix[yi, xi] = row["spearman_rho"]

im = ax.imshow(
    matrix,
    aspect="auto",
    cmap=cmap,
    norm=norm,
    interpolation="none",
)

ax.set_xticks(range(len(representations)))
ax.set_xticklabels(
    [rep_display[x] for x in representations],
    rotation=0,
)
ax.set_yticks(range(len(y_labels)))
ax.set_yticklabels(y_labels, fontsize=7)

for y in separator_y:
    ax.axhline(y, color="white", lw=1.5)

# annotate rho
for yi in range(matrix.shape[0]):
    for xi in range(matrix.shape[1]):
        v = matrix[yi, xi]
        if np.isfinite(v):
            ax.text(
                xi,
                yi,
                f"{v:.2f}",
                ha="center",
                va="center",
                fontsize=6.5,
                color="black" if abs(v) < 0.65 else "white",
            )

ax.set_title(
    "B  Alternative latent representations",
    loc="left",
    fontweight="bold",
)

cbar = fig.colorbar(
    im,
    ax=ax,
    fraction=0.055,
    pad=0.04,
)
cbar.set_label("Spearman correlation")


# =====================================================================
# C. DPT sensitivity
# =====================================================================
ax = fig.add_subplot(gs[1, 0])

for _, row in dpt_df.iterrows():

    key = (
        row["compartment"],
        row["trajectory"],
        row["lineage"],
    )

    if key not in lineage_to_y:
        continue

    rho = float(row["spearman_rho"])

    ax.scatter(
        rho,
        lineage_to_y[key],
        s=55,
        c=[cmap(norm(rho))],
        edgecolor="black",
        linewidth=0.5,
    )

ax.axvline(0, color="0.5", lw=0.7)
ax.axvline(0.7, color="0.65", lw=0.8, ls="--")
ax.axvline(0.9, color="0.65", lw=0.8, ls=":")

ax.set_xlim(-0.05, 1.05)
ax.set_yticks(range(len(y_labels)))
ax.set_yticklabels(y_labels, fontsize=7)
ax.set_xlabel("Spearman correlation with primary Slingshot pseudotime")
ax.set_title(
    "C  Diffusion pseudotime sensitivity",
    loc="left",
    fontweight="bold",
)

for y in separator_y:
    ax.axhline(y, color="0.85", lw=0.7)

ax.grid(axis="x", alpha=0.15)


# =====================================================================
# D. Donor-subsampling stability
# =====================================================================
ax = fig.add_subplot(gs[1, 1])

for _, row in sub_df.iterrows():

    key = (
        row["compartment"],
        row["trajectory"],
        row["primary_lineage"],
    )

    if key not in lineage_to_y:
        continue

    y = lineage_to_y[key]

    med = float(row["spearman_median"])
    q05 = float(row["spearman_q05"])
    q95 = float(row["spearman_q95"])
    match = float(row["match_fraction"])

    # 5th–95th percentile interval
    ax.hlines(
        y,
        q05,
        q95,
        color="0.55",
        lw=1.3,
        zorder=1,
    )

    # median point; size reflects lineage recovery fraction
    size = 25 + 100 * match

    ax.scatter(
        med,
        y,
        s=size,
        c=[cmap(norm(med))],
        edgecolor="black",
        linewidth=0.5,
        zorder=2,
    )

ax.axvline(0, color="0.5", lw=0.7)
ax.axvline(0.7, color="0.65", lw=0.8, ls="--")
ax.axvline(0.9, color="0.65", lw=0.8, ls=":")

ax.set_xlim(-1.05, 1.05)
ax.set_yticks(range(len(y_labels)))
ax.set_yticklabels(y_labels, fontsize=7)
ax.set_xlabel("Spearman correlation")
ax.set_title(
    "D  Donor-subsampling stability",
    loc="left",
    fontweight="bold",
)

for y in separator_y:
    ax.axhline(y, color="0.85", lw=0.7)

ax.grid(axis="x", alpha=0.15)

match_handles = [
    plt.scatter(
        [],
        [],
        s=25 + 100*m,
        facecolor="white",
        edgecolor="black",
        linewidth=0.5,
        label=f"{int(m*100)}%",
    )
    for m in [0.4, 0.7, 1.0]
]

ax.legend(
    handles=match_handles,
    title="Matched runs",
    fontsize=7,
    title_fontsize=8,
    frameon=False,
    loc="lower left",
)


# ---------------------------------------------------------------------
# Overall title / footnote
# ---------------------------------------------------------------------
fig.suptitle(
    "Trajectory robustness across alternative analytical choices",
    fontsize=15,
    fontweight="bold",
    y=0.995,
)

n_failed = int(
    fail_df["n_failures"].sum()
    if not fail_df.empty
    else 0
)

fig.text(
    0.5,
    0.012,
    (
        "Alternative-root and representation analyses compare matched lineages with the frozen "
        "primary Slingshot trajectories. Donor-subsampling points show median rho with 5th–95th "
        f"percentile intervals; point size indicates the fraction of successful runs in which "
        f"the lineage was recovered. {n_failed} donor-subsampling runs failed before lineage comparison."
    ),
    ha="center",
    va="bottom",
    fontsize=8,
)

fig.subplots_adjust(
    left=0.20,
    right=0.96,
    top=0.95,
    bottom=0.06,
)

# ---------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------
out_png = (
    ROBUSTNESS_PLOT_DIR
    / "S7_trajectory_robustness.png"
)

out_pdf = (
    ROBUSTNESS_PLOT_DIR
    / "S7_trajectory_robustness.pdf"
)

fig.savefig(
    out_png,
    dpi=300,
    bbox_inches="tight",
)

fig.savefig(
    out_pdf,
    bbox_inches="tight",
)

plt.close(fig)

print(
    "\nSaved S7 trajectory robustness figure:\n"
    f"  {out_png}\n"
    f"  {out_pdf}"
)
