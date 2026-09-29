# %% 04C06 TRAJECTORY-TRADESEQ
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


DONOR_SUMMARY_MIN_WEIGHT = 1e-8

# %% SHARED HELPERS REQUIRED BY THIS STAGE
def safe_name(s):
    return str(s).replace("-", "_").replace(".", "_").replace(" ", "_")


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


# %% TRAJECTORY GENES: FULL-COUNT ALIGNMENT + TRADESEQ INPUT
#
# Purpose
# -------
# Prepare full-gene metacell counts for tradeSeq WITHOUT
# recomputing or modifying any frozen trajectory.
#
# Frozen quantities reused:
#   - trajectory membership
#   - primary Slingshot pseudotime
#   - primary Slingshot curve weights
#   - metacell identities
#
# Counts come from the original 04a metacell objects because
# the 04b Decipher objects contain only the 2,000 HVGs.
#
# Output unit:
#   one tradeSeq input bundle per compartment × trajectory
#
# Files:
#   <trajectory>_counts.mtx
#   <trajectory>_genes.csv
#   <trajectory>_metacells.csv
#   <trajectory>_pseudotime.csv
#   <trajectory>_cellweights.csv
#   <trajectory>_lineages.csv
#   <trajectory>_gene_filter.csv
#   trajectory_tradeseq_input_manifest.csv
#
# IMPORTANT:
#   No graph, clustering, root, Slingshot, pseudotime,
#   or lineage weight is recomputed here.


from scipy import sparse
from scipy.io import mmwrite

import subprocess
import shutil
import json
import re


# --------------------------------------
# Paths
# --------------------------------------

TRADESEQ_DIR = (
    TRAJECTORY_DIR
    / "tradeseq"
)

TRADESEQ_INPUT_DIR = (
    TRADESEQ_DIR
    / "input"
)

TRADESEQ_RESULT_DIR = (
    TRADESEQ_DIR
    / "results"
)

TRADESEQ_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

TRADESEQ_INPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

TRADESEQ_RESULT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

# %% TRAJECTORY GENES: 04a FULL-COUNT OBJECTS

PROJECT_ROOT = Path.cwd().resolve().parent
METACELL_DIR = PROJECT_ROOT / "data" / "metacells"

FULL_COUNT_PATHS = {
    "PT": METACELL_DIR / "PT_metacells.h5ad",
    "FIB": METACELL_DIR / "FIB_metacells.h5ad",
    "Glomerular": METACELL_DIR / "Glomerular_metacells.h5ad",
}

for compartment, path in FULL_COUNT_PATHS.items():
    if not path.exists():
        raise FileNotFoundError(
            f"Missing 04a full-count object for "
            f"{compartment}: {path}"
        )

print("\nPROJECT ROOT")
print(PROJECT_ROOT)

print("\n04a FULL-COUNT OBJECTS")
for compartment, path in FULL_COUNT_PATHS.items():
    print(f"{compartment:12s} {path}")


# --------------------------------------
# Gene-filter parameters
# --------------------------------------
#
# Filtering is performed BEFORE tradeSeq.
#
# A gene must have:
#   >= 3 raw counts
# in:
#   >= 10 metacells
#
# and:
#   >= 20 total counts
#
# These are deliberately modest filters. They remove genes
# with essentially no information without selecting genes
# based on disease or pseudotime.
#
# The full filter table is exported so this decision remains
# auditable.

TRADESEQ_MIN_COUNT = 3
TRADESEQ_MIN_METACELLS = 10
TRADESEQ_MIN_TOTAL_COUNT = 20


# --------------------------------------
# Helpers
# --------------------------------------

def sanitize_filename(
    value,
):
    value = str(
        value
    )

    value = re.sub(
        r"[^A-Za-z0-9_.-]+",
        "_",
        value,
    )

    return value.strip(
        "_"
    )


def get_raw_count_matrix(
    adata,
):
    """
    Retrieve the original aggregated metacell count matrix.

    04a metacell objects should contain raw counts in X.
    This helper performs strict integer-like QC rather than
    silently guessing or transforming the matrix.
    """

    X = adata.X

    if sparse.issparse(
        X
    ):
        X = X.tocsr(
            copy=True
        )

        values = X.data

    else:
        X = np.asarray(
            X
        )

        values = X.ravel()

    if values.size == 0:
        raise RuntimeError(
            "Count matrix contains no values."
        )

    if not np.all(
        np.isfinite(values)
    ):
        raise RuntimeError(
            "Non-finite values detected in "
            "04a count matrix."
        )

    if np.min(
        values
    ) < 0:
        raise RuntimeError(
            "Negative values detected in "
            "04a count matrix."
        )

    # Raw aggregated UMI counts should be integer-like.
    integer_error = np.max(
        np.abs(
            values
            - np.round(
                values
            )
        )
    )

    if integer_error > 1e-6:
        raise RuntimeError(
            "04a X is not integer-like. "
            "Do not use normalized/log-transformed "
            "expression for tradeSeq."
        )

    if sparse.issparse(
        X
    ):
        X.data = np.round(
            X.data
        ).astype(
            np.int64,
            copy=False,
        )

        X.eliminate_zeros()

    else:
        X = np.round(
            X
        ).astype(
            np.int64,
            copy=False,
        )

    return X


def get_frozen_trajectory_adata(
    compartment,
    trajectory_name,
):
    """
    Load and subset the frozen compartment-level primary
    trajectory H5AD for one directional analysis.

    Uses ONLY original Slingshot quantities:
        pt_<trajectory>_LineageN
        weight_<trajectory>_LineageN

    Explicitly excludes later derived columns such as:
        pt_<trajectory>_LineageN_scaled

    No trajectory quantity is recomputed.
    """

    path = (
        TRAJECTORY_DIR
        / f"{compartment}_trajectory_primary.h5ad"
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Missing frozen primary trajectory object: {path}"
        )

    adata_primary = ad.read_h5ad(
        path
    )

    pt_prefix = f"pt_{trajectory_name}_"

    weight_prefix = f"weight_{trajectory_name}_"

    # ----------------------------------
    # Original Slingshot columns only
    # ----------------------------------

    pt_cols = [
        str(col)
        for col in adata_primary.obs.columns
        if (
            str(col).startswith(pt_prefix)
            and not str(col).endswith("_scaled")
        )
    ]

    weight_cols = [
        str(col)
        for col in adata_primary.obs.columns
        if str(col).startswith(weight_prefix)
    ]

    if len(pt_cols) == 0:
        raise RuntimeError(
            f"No original Slingshot pseudotime columns "
            f"found for {compartment}/{trajectory_name}."
        )

    if len(weight_cols) == 0:
        raise RuntimeError(
            f"No Slingshot weight columns found for "
            f"{compartment}/{trajectory_name}."
        )

    # ----------------------------------
    # Extract lineage IDs
    # ----------------------------------

    pt_lineages = {
        col[len(pt_prefix):]
        for col in pt_cols
    }

    weight_lineages = {
        col[len(weight_prefix):]
        for col in weight_cols
    }

    if pt_lineages != weight_lineages:
        raise RuntimeError(
            f"Original Slingshot pseudotime/weight "
            f"lineages differ for "
            f"{compartment}/{trajectory_name}.\n"
            f"Pseudotime only: "
            f"{sorted(pt_lineages - weight_lineages)}\n"
            f"Weight only: "
            f"{sorted(weight_lineages - pt_lineages)}"
        )

    # ----------------------------------
    # Validate lineage names
    # ----------------------------------

    unexpected = [
        lineage
        for lineage in pt_lineages
        if re.fullmatch(
            r"Lineage\d+",
            lineage,
        ) is None
    ]

    if unexpected:
        raise RuntimeError(
            "Unexpected frozen lineage identifiers for "
            f"{compartment}/{trajectory_name}: "
            f"{sorted(unexpected)}"
        )

    # ----------------------------------
    # Recover exact directional subset
    # ----------------------------------
    #
    # Primary H5AD is compartment-level. Cells outside this
    # directional analysis have no finite original Slingshot
    # pseudotime and no positive curve weight for its
    # lineages.

    pt_matrix = (
        adata_primary.obs[
            pt_cols
        ]
        .apply(
            pd.to_numeric,
            errors="coerce",
        )
        .to_numpy(
            dtype=float
        )
    )

    weight_matrix = (
        adata_primary.obs[
            weight_cols
        ]
        .apply(
            pd.to_numeric,
            errors="coerce",
        )
        .to_numpy(
            dtype=float
        )
    )

    finite_pt = np.any(
        np.isfinite(
            pt_matrix
        ),
        axis=1,
    )

    positive_weight = np.any(
        (
            np.isfinite(weight_matrix)
            & (
                weight_matrix
                > DONOR_SUMMARY_MIN_WEIGHT
            )
        ),
        axis=1,
    )

    trajectory_mask = (
        finite_pt
        | positive_weight
    )

    n_selected = int(
        trajectory_mask.sum()
    )

    if n_selected == 0:
        raise RuntimeError(
            f"No frozen trajectory metacells found for "
            f"{compartment}/{trajectory_name}."
        )

    trajectory_adata = (
        adata_primary[
            trajectory_mask
        ]
        .copy()
    )

    # ----------------------------------
    # Frozen-size assertions
    # ----------------------------------
    #
    # These are established primary trajectory sizes.
    # Any mismatch means we are not recovering the same
    # directional analysis and should stop before tradeSeq.

    expected_sizes = {
        ("PT", "PT"): 6785,
        ("FIB", "contractile"): 722,
        ("FIB", "outer_medullary"): 278,
        ("Glomerular", "podocyte_PEC"): 610,
        ("Glomerular", "endothelial_mesangial"): 312,
    }

    key = (
        compartment,
        trajectory_name,
    )

    if key not in expected_sizes:
        raise RuntimeError(
            f"Unexpected directional trajectory: {key}"
        )

    expected_n = expected_sizes[
        key
    ]

    if trajectory_adata.n_obs != expected_n:
        raise RuntimeError(
            f"Frozen trajectory subset size mismatch for "
            f"{compartment}/{trajectory_name}: "
            f"recovered {trajectory_adata.n_obs:,}, "
            f"expected {expected_n:,}."
        )

    # ----------------------------------
    # Final column-level validation
    # ----------------------------------

    expected_pt_cols = {
        f"pt_{trajectory_name}_{lineage}"
        for lineage in pt_lineages
    }

    expected_weight_cols = {
        f"weight_{trajectory_name}_{lineage}"
        for lineage in pt_lineages
    }

    if set(pt_cols) != expected_pt_cols:
        raise RuntimeError(
            "Unexpected original pseudotime column set."
        )

    if set(weight_cols) != expected_weight_cols:
        raise RuntimeError(
            "Unexpected original weight column set."
        )

    print(
        f"frozen primary: {path.name}; "
        f"trajectory={trajectory_name}; "
        f"metacells={trajectory_adata.n_obs:,}; "
        f"lineages={len(pt_lineages)}"
    )

    return trajectory_adata


def align_full_counts_to_trajectory(
    full_adata,
    trajectory_adata,
):
    """
    Align 04a full-gene counts to the exact ordered
    metacell identities in the frozen trajectory.
    """

    full_names = pd.Index(
        full_adata.obs_names.astype(str)
    )

    trajectory_names = pd.Index(
        trajectory_adata.obs_names.astype(str)
    )

    if full_names.has_duplicates:
        raise RuntimeError(
            "Duplicate metacell IDs in 04a object."
        )

    if trajectory_names.has_duplicates:
        raise RuntimeError(
            "Duplicate metacell IDs in frozen "
            "trajectory object."
        )

    missing = (
        trajectory_names
        .difference(
            full_names
        )
    )

    if len(
        missing
    ):
        raise RuntimeError(
            f"{len(missing)} frozen trajectory metacells "
            "are absent from the 04a full-count object. "
            f"Examples: {missing[:10].tolist()}"
        )

    indexer = full_names.get_indexer(
        trajectory_names
    )

    if np.any(
        indexer
        < 0
    ):
        raise RuntimeError(
            "Failed exact metacell alignment."
        )

    counts = get_raw_count_matrix(
        full_adata
    )

    counts = counts[
        indexer,
        :
    ]

    # Verify alignment explicitly.
    aligned_names = full_names[
        indexer
    ]

    if not np.array_equal(
        aligned_names.to_numpy(),
        trajectory_names.to_numpy(),
    ):
        raise RuntimeError(
            "04a count ordering does not exactly match "
            "the frozen trajectory after alignment."
        )

    return (
        counts,
        trajectory_names,
    )


def build_tradeseq_matrices(
    trajectory_adata,
    trajectory_name,
):
    """
    Extract frozen Slingshot pseudotime and curve weights.

    Pseudotime columns:
        pt_<trajectory>_<lineage>

    Weight columns:
        weight_<trajectory>_<lineage>

    Returns:
      pseudotime: n_metacells × n_lineages
      weights:    n_metacells × n_lineages
      lineage_names
      pt_columns
      weight_columns
    """

    pt_cols = list(
        primary_pt_columns(
            trajectory_adata,
            trajectory_name,
        )
    )

    weight_cols = list(
        primary_weight_columns(
            trajectory_adata,
            trajectory_name,
        )
    )

    if len(pt_cols) == 0:
        raise RuntimeError(
            f"No primary pseudotime columns found "
            f"for {trajectory_name}."
        )

    if len(weight_cols) == 0:
        raise RuntimeError(
            f"No primary weight columns found "
            f"for {trajectory_name}."
        )

    # ----------------------------------
    # Extract lineage identifiers
    # ----------------------------------

    pt_prefix = f"pt_{trajectory_name}_"

    weight_prefix = f"weight_{trajectory_name}_"

    pt_lineages = []

    for col in pt_cols:

        if not col.startswith(
            pt_prefix
        ):
            raise RuntimeError(
                "Unexpected primary pseudotime "
                f"column name: {col!r}; "
                f"expected prefix {pt_prefix!r}."
            )

        lineage = col[
            len(pt_prefix):
        ]

        if not lineage:
            raise RuntimeError(
                f"Could not extract lineage from "
                f"pseudotime column {col!r}."
            )

        pt_lineages.append(lineage)

    weight_lineages = []

    for col in weight_cols:

        if not col.startswith(
            weight_prefix
        ):
            raise RuntimeError(
                "Unexpected primary weight "
                f"column name: {col!r}; "
                f"expected prefix {weight_prefix!r}."
            )

        lineage = col[
            len(weight_prefix):
        ]

        if not lineage:
            raise RuntimeError(
                f"Could not extract lineage from "
                f"weight column {col!r}."
            )

        weight_lineages.append(lineage)

    # ----------------------------------
    # Validate one-to-one lineage mapping
    # ----------------------------------

    if len(
        set(pt_lineages)
    ) != len(
        pt_lineages
    ):
        raise RuntimeError(
            "Duplicate lineage identifiers in "
            "primary pseudotime columns."
        )

    if len(
        set(weight_lineages)
    ) != len(
        weight_lineages
    ):
        raise RuntimeError(
            "Duplicate lineage identifiers in "
            "primary weight columns."
        )

    missing_weights = (
        set(pt_lineages)
        - set(weight_lineages)
    )

    extra_weights = (
        set(weight_lineages)
        - set(pt_lineages)
    )

    if (
        missing_weights
        or extra_weights
    ):
        raise RuntimeError(
            "Primary pseudotime and weight lineage "
            "sets genuinely differ.\n"
            f"Missing weights: "
            f"{sorted(missing_weights)}\n"
            f"Extra weights: "
            f"{sorted(extra_weights)}"
        )

    # ----------------------------------
    # Canonical lineage ordering
    # ----------------------------------
    #
    # Use natural numeric ordering:
    # Lineage1, Lineage2, ..., Lineage10
    #
    # Do not rely on lexicographic column ordering.

    def lineage_sort_key(
        lineage,
    ):
        match = re.fullmatch(
            r"Lineage(\d+)",
            lineage,
        )

        if match is not None:
            return (
                0,
                int(
                    match.group(1)
                ),
            )

        return (
            1,
            lineage,
        )

    lineage_names = sorted(
        pt_lineages,
        key=lineage_sort_key,
    )

    pt_lookup = {
        lineage: col
        for lineage, col
        in zip(
            pt_lineages,
            pt_cols,
        )
    }

    weight_lookup = {
        lineage: col
        for lineage, col
        in zip(
            weight_lineages,
            weight_cols,
        )
    }

    ordered_pt_cols = [
        pt_lookup[
            lineage
        ]
        for lineage
        in lineage_names
    ]

    ordered_weight_cols = [
        weight_lookup[
            lineage
        ]
        for lineage
        in lineage_names
    ]

    # Final explicit pairing check.
    for (
        lineage,
        pt_col,
        weight_col,
    ) in zip(
        lineage_names,
        ordered_pt_cols,
        ordered_weight_cols,
    ):

        expected_pt = f"pt_{trajectory_name}_{lineage}"

        expected_weight = f"weight_{trajectory_name}_{lineage}"

        if pt_col != expected_pt:
            raise RuntimeError(
                f"Pseudotime pairing failure for "
                f"{lineage}: {pt_col!r} != "
                f"{expected_pt!r}"
            )

        if weight_col != expected_weight:
            raise RuntimeError(
                f"Weight pairing failure for "
                f"{lineage}: {weight_col!r} != "
                f"{expected_weight!r}"
            )

    # ----------------------------------
    # Extract matrices
    # ----------------------------------

    pseudotime = (
        trajectory_adata.obs[
            ordered_pt_cols
        ]
        .apply(
            pd.to_numeric,
            errors="coerce",
        )
        .to_numpy(
            dtype=float
        )
    )

    weights = (
        trajectory_adata.obs[
            ordered_weight_cols
        ]
        .apply(
            pd.to_numeric,
            errors="coerce",
        )
        .to_numpy(
            dtype=float
        )
    )

    if pseudotime.shape != weights.shape:
        raise RuntimeError(
            "Pseudotime and cell-weight matrices "
            "have different shapes."
        )

    if (
        pseudotime.shape[1]
        != len(lineage_names)
    ):
        raise RuntimeError(
            "Unexpected lineage dimension."
        )

    # ----------------------------------
    # Weight QC
    # ----------------------------------

    if not np.all(
        np.isfinite(weights)
    ):
        raise RuntimeError(
            "Non-finite primary Slingshot weights."
        )

    if np.any(
        weights < -1e-8
    ):
        raise RuntimeError(
            "Negative primary Slingshot weights."
        )

    if np.any(
        weights > 1.0 + 1e-8
    ):
        raise RuntimeError(
            "Primary Slingshot weights exceed 1."
        )

    weights = np.clip(
        weights,
        0.0,
        1.0,
    )

    # ----------------------------------
    # Pseudotime QC
    # ----------------------------------

    invalid_supported_pt = (
        (
            weights
            > DONOR_SUMMARY_MIN_WEIGHT
        )
        & ~np.isfinite(
            pseudotime
        )
    )

    if np.any(
        invalid_supported_pt
    ):
        bad = np.argwhere(
            invalid_supported_pt
        )

        raise RuntimeError(
            "Supported lineage memberships contain "
            "non-finite primary pseudotime values. "
            f"First indices: {bad[:10].tolist()}"
        )

    # Missing pseudotime is permissible only when that
    # metacell contributes no meaningful weight to the
    # corresponding lineage. Set those unused entries to
    # zero for the exported numeric matrix.
    zero_weight_nonfinite = (
        ~np.isfinite(
            pseudotime
        )
        & (
            weights
            <= DONOR_SUMMARY_MIN_WEIGHT
        )
    )

    pseudotime[
        zero_weight_nonfinite
    ] = 0.0

    if not np.all(
        np.isfinite(pseudotime)
    ):
        raise RuntimeError(
            "Non-finite pseudotime remains after "
            "zero-weight handling."
        )

    return (
        pseudotime,
        weights,
        lineage_names,
        ordered_pt_cols,
        ordered_weight_cols,
    )


def compute_gene_filter(
    counts,
    gene_names,
):
    """
    Expression-only gene filter.

    No disease or pseudotime information enters filtering.
    """

    if sparse.issparse(
        counts
    ):
        counts_csc = counts.tocsc()

        n_metacells_ge_min = np.asarray(
            (
                counts_csc
                >= TRADESEQ_MIN_COUNT
            )
            .sum(
                axis=0
            )
        ).ravel()

        total_counts = np.asarray(
            counts_csc.sum(axis=0)
        ).ravel()

        n_nonzero = np.asarray(
            (
                counts_csc
                > 0
            )
            .sum(
                axis=0
            )
        ).ravel()

    else:
        n_metacells_ge_min = (
            counts
            >= TRADESEQ_MIN_COUNT
        ).sum(
            axis=0
        )

        total_counts = counts.sum(
            axis=0
        )

        n_nonzero = (
            counts
            > 0
        ).sum(
            axis=0
        )

    keep = (
        (
            n_metacells_ge_min
            >= TRADESEQ_MIN_METACELLS
        )
        & (
            total_counts
            >= TRADESEQ_MIN_TOTAL_COUNT
        )
    )

    filter_df = pd.DataFrame(
        {
            "gene":
                np.asarray(
                    gene_names,
                    dtype=str,
                ),

            "total_counts":
                total_counts.astype(
                    np.int64
                ),

            "n_nonzero_metacells":
                n_nonzero.astype(
                    np.int64
                ),

            f"n_metacells_count_ge_{TRADESEQ_MIN_COUNT}":
                n_metacells_ge_min.astype(
                    np.int64
                ),

            "keep_tradeseq":
                keep.astype(
                    bool
                ),
        }
    )

    return (
        keep,
        filter_df,
    )



# %% TRADESEQ PRODUCTION CHECKPOINT CONFIG
# Input bundles and evaluateK are already validated/frozen. This production
# stage reuses those durable files and never retains all count matrices in Python.
TRADESEQ_EVALK_DIR = TRADESEQ_DIR / "evaluateK"
TRADESEQ_FIT_DIR = TRADESEQ_RESULT_DIR / "fitGAM"
TRADESEQ_FIT_DIR.mkdir(parents=True, exist_ok=True)

TRADESEQ_NKNOTS = 8
TRADESEQ_BATCH_SIZE = 250
TRADESEQ_FIT_N_WORKERS = 1
TRADESEQ_FIT_SEED = 20260920

# Set True only if you intentionally want to discard valid completed fit batches.
TRADESEQ_OVERWRITE_FIT_BATCHES = False

# %% LOAD + VALIDATE EXISTING TRADESEQ INPUT MANIFEST
manifest_path = TRADESEQ_DIR / "trajectory_tradeseq_input_manifest.csv"
if not manifest_path.exists():
    raise FileNotFoundError(
        f"Missing validated tradeSeq input manifest: {manifest_path}\n"
        "Run the already-validated input-preparation stage once before this production fit stage."
    )

tradeseq_input_manifest = pd.read_csv(manifest_path)
required_manifest_cols = {
    "compartment", "trajectory", "bundle", "bundle_dir",
    "n_metacells", "n_genes_filtered", "n_lineages",
    "counts_path", "genes_path", "pseudotime_path",
    "cellweights_path", "lineages_path",
}
missing_manifest_cols = required_manifest_cols - set(tradeseq_input_manifest.columns)
if missing_manifest_cols:
    raise RuntimeError(
        "tradeSeq input manifest is missing required columns: "
        f"{sorted(missing_manifest_cols)}"
    )

expected_trajectories = {
    (compartment, trajectory_name)
    for compartment, trajectory_dict in TRAJECTORY_SPECS.items()
    for trajectory_name in trajectory_dict.keys()
}
observed_trajectories = set(zip(
    tradeseq_input_manifest["compartment"],
    tradeseq_input_manifest["trajectory"],
))
if observed_trajectories != expected_trajectories:
    raise RuntimeError(
        "Existing tradeSeq manifest does not contain exactly the five frozen directional analyses."
    )

for row in tradeseq_input_manifest.itertuples(index=False):
    for col in [
        "counts_path", "genes_path", "pseudotime_path",
        "cellweights_path", "lineages_path",
    ]:
        path = Path(getattr(row, col))
        if not path.exists():
            raise FileNotFoundError(
                f"Missing tradeSeq bundle file for {row.compartment}/{row.trajectory}: {path}"
            )

print("\nVALIDATED TRADESEQ INPUT MANIFEST")
print(
    tradeseq_input_manifest[
        ["compartment", "trajectory", "n_metacells", "n_genes_filtered", "n_lineages"]
    ].to_string(index=False)
)

# %% R + TRADESEQ ENVIRONMENT AUDIT
rscript_path = shutil.which("Rscript")
if rscript_path is None:
    raise RuntimeError("Rscript was not found on PATH. tradeSeq fitting cannot begin.")

r_audit_script = r'''
suppressPackageStartupMessages({
    has_tradeSeq <- requireNamespace("tradeSeq", quietly = TRUE)
    has_matrix <- requireNamespace("Matrix", quietly = TRUE)
    has_biocparallel <- requireNamespace("BiocParallel", quietly = TRUE)
})
cat("R_VERSION=", R.version.string, "\n", sep = "")
cat("TRADESEQ_AVAILABLE=", has_tradeSeq, "\n", sep = "")
cat("MATRIX_AVAILABLE=", has_matrix, "\n", sep = "")
cat("BIOCPARALLEL_AVAILABLE=", has_biocparallel, "\n", sep = "")
if (has_tradeSeq) cat("TRADESEQ_VERSION=", as.character(packageVersion("tradeSeq")), "\n", sep = "")
if (has_matrix) cat("MATRIX_VERSION=", as.character(packageVersion("Matrix")), "\n", sep = "")
if (has_biocparallel) cat("BIOCPARALLEL_VERSION=", as.character(packageVersion("BiocParallel")), "\n", sep = "")
'''
r_audit = subprocess.run(
    [rscript_path, "-e", r_audit_script],
    capture_output=True,
    text=True,
)
if r_audit.returncode != 0:
    raise RuntimeError("R environment audit failed:\n" + r_audit.stderr)
print("\nR ENVIRONMENT")
print(r_audit.stdout)

# %% VALIDATE FROZEN evaluateK CHECKPOINT
# We do not rerun evaluateK. nknots=8 is already frozen from the completed analysis.
evalk_summary_path = TRADESEQ_EVALK_DIR / "trajectory_evaluateK_summary.csv"
if not evalk_summary_path.exists():
    raise FileNotFoundError(
        f"Missing completed evaluateK summary: {evalk_summary_path}"
    )
tradeseq_evalk_summary = pd.read_csv(evalk_summary_path)
for key in expected_trajectories:
    compartment, trajectory_name = key
    subset = tradeseq_evalk_summary[
        (tradeseq_evalk_summary["compartment"] == compartment)
        & (tradeseq_evalk_summary["trajectory"] == trajectory_name)
    ]
    if set(pd.to_numeric(subset["nknots"], errors="coerce").dropna().astype(int)) != {3, 4, 5, 6, 7, 8}:
        raise RuntimeError(f"Incomplete evaluateK checkpoint for {compartment}/{trajectory_name}.")
    if (pd.to_numeric(subset["n_genes_finite"], errors="coerce") <= 0).any():
        raise RuntimeError(f"Non-finite evaluateK checkpoint for {compartment}/{trajectory_name}.")

print("\nevaluateK checkpoint validated; not rerunning evaluateK.")
print(f"Frozen nknots: {TRADESEQ_NKNOTS}")

# %% FREEZE / EXPORT TRADESEQ MODEL SPECIFICATION
tradeseq_model_spec = pd.DataFrame([
    {
        "parameter": "nknots",
        "value": TRADESEQ_NKNOTS,
        "rationale": "Common spline complexity selected from evaluateK across all five frozen trajectories",
    },
    {
        "parameter": "gene_filter",
        "value": "count>=3 in >=10 metacells; total_count>=20",
        "rationale": "Prespecified expression-only filter; no disease or pseudotime selection",
    },
    {
        "parameter": "trajectory_representation",
        "value": "frozen primary Slingshot",
        "rationale": "Original unscaled Slingshot pseudotime and curve weights",
    },
    {
        "parameter": "disease_covariate",
        "value": "none",
        "rationale": "Primary trajectory-gene discovery is independent of disease",
    },
    {
        "parameter": "normalization",
        "value": "full-filtered-matrix TMM offset",
        "rationale": "Computed once from all filtered genes and reused unchanged in every memory-bounded batch",
    },
    {
        "parameter": "fit_batch_size",
        "value": TRADESEQ_BATCH_SIZE,
        "rationale": "Memory-bounded computational batching; gene-wise GAM specification is unchanged",
    },
    {
        "parameter": "fit_workers",
        "value": TRADESEQ_FIT_N_WORKERS,
        "rationale": "Serial fitting within each batch to bound peak memory and avoid fork/SOCK worker duplication",
    },
])
tradeseq_model_spec.to_csv(
    TRADESEQ_RESULT_DIR / "trajectory_tradeseq_model_specification.csv",
    index=False,
)
print("\nTRADESEQ MODEL SPECIFICATION")
print(tradeseq_model_spec.to_string(index=False))

# %% R FULL-MATRIX NORMALIZATION + fitGAM BATCH RUNNERS
# tradeSeq's sparse-matrix method internally converts the ENTIRE matrix to a
# dense R matrix before fitting. To keep memory bounded while preserving the
# full-matrix normalization used by the original model, we therefore:
#   1) compute TMM normalization offsets once per trajectory directly from the
#      sparse Matrix Market input without densifying the whole matrix;
#   2) subset to a small gene batch;
#   3) pass the fixed full-matrix offset to fitGAM for that batch.
#
# The TMM calculation below uses edgeR's installed .calcFactorTMM routine for
# each library/reference pair and reproduces edgeR's reference-selection logic
# without calling calcNormFactors() on the full matrix (which calls as.matrix()).
normalization_r_script = TRADESEQ_FIT_DIR / "compute_full_TMM_offset.R"
normalization_r_code = r'''
suppressPackageStartupMessages({
    library(Matrix)
})

if (!requireNamespace("edgeR", quietly = TRUE)) {
    stop("edgeR is required to reproduce tradeSeq TMM normalization.")
}

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 3) {
    stop(sprintf(
        "Expected 3 arguments: counts.mtx genes.csv offset.csv; received %d.",
        length(args)
    ))
}

counts_path <- args[[1]]
genes_path <- args[[2]]
offset_path <- args[[3]]

cat("Reading full sparse count matrix for normalization...\n")
counts <- as(readMM(counts_path), "CsparseMatrix")
genes <- read.csv(genes_path, stringsAsFactors = FALSE, check.names = FALSE)
if (nrow(genes) != nrow(counts)) {
    stop("Gene metadata and count matrix dimensions differ during normalization.")
}
if (any(counts@x < 0) || any(!is.finite(counts@x))) {
    stop("Invalid values in count matrix during normalization.")
}

n_genes <- nrow(counts)
n_cells <- ncol(counts)
lib_size <- as.numeric(Matrix::colSums(counts))
if (any(!is.finite(lib_size))) stop("Non-finite library sizes.")
if (any(lib_size <= 0)) {
    stop("Full filtered count matrix contains zero-count metacells; cannot compute TMM offset.")
}

# Exact type=7 75th percentile for a sparse non-negative column without
# materializing all zeros. This matches stats::quantile(..., probs=0.75)
# for the edgeR TMM reference-library selection step.
sparse_q75 <- function(x, n_total) {
    vals <- as.numeric(x@x)
    n_zero <- n_total - length(vals)
    h <- (n_total - 1) * 0.75 + 1
    j <- floor(h)
    g <- h - j

    kth <- function(k) {
        if (k <= n_zero) return(0)
        pos <- k - n_zero
        sort(vals, partial = pos)[pos]
    }

    xj <- kth(j)
    if (g == 0 || j >= n_total) return(xj)
    xj1 <- kth(j + 1)
    (1 - g) * xj + g * xj1
}

cat("Selecting TMM reference library from sparse counts...\n")
f75 <- numeric(n_cells)
for (j in seq_len(n_cells)) {
    f75[j] <- sparse_q75(counts[, j, drop = FALSE], n_genes) / lib_size[j]
}

if (median(f75) < 1e-20) {
    sqrt_counts <- counts
    sqrt_counts@x <- sqrt(sqrt_counts@x)
    ref_column <- which.max(Matrix::colSums(sqrt_counts))
    rm(sqrt_counts)
} else {
    ref_column <- which.min(abs(f75 - mean(f75)))
}

cat(sprintf("TMM reference column: %d / %d\n", ref_column, n_cells))
ref <- as.numeric(counts[, ref_column])
calc_tmm <- getFromNamespace(".calcFactorTMM", "edgeR")

norm_factors <- numeric(n_cells)
for (j in seq_len(n_cells)) {
    obs <- as.numeric(counts[, j])
    norm_factors[j] <- calc_tmm(
        obs = obs,
        ref = ref,
        libsize.obs = lib_size[j],
        libsize.ref = lib_size[ref_column],
        logratioTrim = 0.3,
        sumTrim = 0.05,
        doWeighting = TRUE,
        Acutoff = -1e10
    )
    if (j %% 500 == 0 || j == n_cells) {
        cat(sprintf("  normalized %d / %d metacells\n", j, n_cells))
    }
}

if (any(!is.finite(norm_factors)) || any(norm_factors <= 0)) {
    stop("Invalid TMM normalization factors.")
}

# edgeR scales factors to geometric mean 1.
norm_factors <- norm_factors / exp(mean(log(norm_factors)))
effective_lib_size <- lib_size * norm_factors
if (any(!is.finite(effective_lib_size)) || any(effective_lib_size <= 0)) {
    stop("Invalid effective library sizes.")
}
offset <- log(effective_lib_size)

out <- data.frame(
    metacell_index = seq_len(n_cells),
    library_size = lib_size,
    norm_factor = norm_factors,
    effective_library_size = effective_lib_size,
    offset = offset,
    tmm_reference_column = ref_column,
    stringsAsFactors = FALSE
)
write.csv(out, offset_path, row.names = FALSE, quote = FALSE)
cat("Full-matrix TMM offset COMPLETE\n")
'''
normalization_r_script.write_text(normalization_r_code)

fitgam_r_script = TRADESEQ_FIT_DIR / "run_fitGAM_batch.R"
fitgam_r_code = r'''
suppressPackageStartupMessages({
    library(tradeSeq)
    library(Matrix)
    library(BiocParallel)
})

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 11) {
    stop(sprintf(
        paste(
            "Expected 11 arguments:",
            "counts.mtx genes.csv pseudotime.csv cellweights.csv offset.csv",
            "output_dir batch_start batch_end nknots seed batch_id; received %d."
        ),
        length(args)
    ))
}

counts_path <- args[[1]]
genes_path <- args[[2]]
pseudotime_path <- args[[3]]
weights_path <- args[[4]]
offset_path <- args[[5]]
output_dir <- args[[6]]
batch_start <- as.integer(args[[7]])
batch_end <- as.integer(args[[8]])
nknots <- as.integer(args[[9]])
seed <- as.integer(args[[10]])
batch_id <- args[[11]]

dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

cat("Reading sparse counts...\n")
counts <- as(readMM(counts_path), "CsparseMatrix")
genes <- read.csv(
    genes_path,
    stringsAsFactors = FALSE,
    check.names = FALSE
)
if (nrow(genes) != nrow(counts)) {
    stop("Gene metadata and count matrix dimensions differ.")
}
if (batch_start < 1L || batch_end > nrow(counts) || batch_start > batch_end) {
    stop("Invalid gene-batch indices.")
}

# Critical memory boundary: subset BEFORE fitGAM. The full sparse matrix is
# discarded immediately after subsetting so tradeSeq can only densify this
# small batch.
idx <- seq.int(batch_start, batch_end)
batch_genes <- genes$gene[idx]
counts <- counts[idx, , drop = FALSE]
rownames(counts) <- batch_genes
rm(genes)
invisible(gc())

cat("Reading pseudotime, weights, and frozen full-matrix offset...\n")
pseudotime <- as.matrix(read.csv(
    pseudotime_path,
    row.names = 1,
    check.names = FALSE
))
cellWeights <- as.matrix(read.csv(
    weights_path,
    row.names = 1,
    check.names = FALSE
))
offset_df <- read.csv(offset_path, stringsAsFactors = FALSE, check.names = FALSE)
if (!("offset" %in% colnames(offset_df))) stop("Normalization file lacks offset column.")
offset <- as.numeric(offset_df$offset)

storage.mode(pseudotime) <- "double"
storage.mode(cellWeights) <- "double"

if (!identical(rownames(pseudotime), rownames(cellWeights))) {
    stop("Pseudotime and weight metacell order differs.")
}
if (!identical(colnames(pseudotime), colnames(cellWeights))) {
    stop("Pseudotime and weight lineage order differs.")
}
if (ncol(counts) != nrow(pseudotime)) {
    stop("Count matrix metacells do not match pseudotime rows.")
}
if (length(offset) != ncol(counts)) {
    stop("Full-matrix normalization offset length does not match metacells.")
}
if (any(!is.finite(offset))) stop("Non-finite normalization offset reached fitGAM.")
if (any(!is.finite(pseudotime))) stop("Non-finite pseudotime reached fitGAM.")
if (any(!is.finite(cellWeights))) stop("Non-finite cell weights reached fitGAM.")
if (any(cellWeights < 0) || any(cellWeights > 1)) stop("Cell weights outside [0,1].")

cat(sprintf(
    "Batch %s: genes %d-%d (%d genes); metacells=%d; lineages=%d; nknots=%d\n",
    batch_id, batch_start, batch_end, nrow(counts), ncol(counts),
    ncol(pseudotime), nknots
))
cat("Normalization: fixed TMM offset computed from ALL filtered genes.\n")

BPPARAM <- SerialParam(progressbar = TRUE, RNGseed = seed)
set.seed(seed)

sce <- fitGAM(
    counts = counts,
    pseudotime = pseudotime,
    cellWeights = cellWeights,
    offset = offset,
    nknots = nknots,
    verbose = TRUE,
    parallel = FALSE,
    BPPARAM = BPPARAM
)

if (!inherits(sce, "SingleCellExperiment")) {
    stop("fitGAM did not return a SingleCellExperiment.")
}
if (nrow(sce) != length(batch_genes)) {
    stop("fitGAM result has unexpected gene count.")
}
if (!identical(rownames(sce), batch_genes)) {
    stop("fitGAM result gene order differs from requested batch.")
}

# tradeSeq stores nested objects (Sigma matrices and beta objects) in rowData;
# they are intentionally retained in the RDS and NOT flattened to CSV.
ts_rd <- SummarizedExperiment::rowData(sce)$tradeSeq
if (is.null(ts_rd)) stop("tradeSeq rowData is missing from fitted SCE.")

converged <- tryCatch(
    as.logical(ts_rd$converged),
    error = function(e) rep(NA, length(batch_genes))
)
if (length(converged) != length(batch_genes)) {
    converged <- rep(NA, length(batch_genes))
}

qc <- data.frame(
    gene = batch_genes,
    converged = converged,
    batch_id = batch_id,
    batch_start = batch_start,
    batch_end = batch_end,
    stringsAsFactors = FALSE
)
write.csv(
    qc,
    file.path(output_dir, "fitGAM_rowData.csv"),
    row.names = FALSE,
    quote = FALSE
)

saveRDS(
    sce,
    file.path(output_dir, "fitGAM_sce.rds"),
    compress = FALSE
)

write.csv(
    data.frame(
        gene = batch_genes,
        batch_id = batch_id,
        stringsAsFactors = FALSE
    ),
    file.path(output_dir, "genes.csv"),
    row.names = FALSE,
    quote = FALSE
)

cat(sprintf(
    "Converged genes: %d / %d\n",
    sum(converged %in% TRUE, na.rm = TRUE),
    length(converged)
))
cat("fitGAM batch COMPLETE\n")
'''
fitgam_r_script.write_text(fitgam_r_code)
print("\nSaved sparse full-matrix TMM runner:")
print(normalization_r_script)
print("Saved memory-bounded fitGAM runner:")
print(fitgam_r_script)

# %% FITGAM CHECKPOINT HELPERS
def _batch_paths(batch_dir):
    return {
        "rds": batch_dir / "fitGAM_sce.rds",
        "rowdata": batch_dir / "fitGAM_rowData.csv",
        "genes": batch_dir / "genes.csv",
        "stdout": batch_dir / "stdout.txt",
        "stderr": batch_dir / "stderr.txt",
        "done": batch_dir / "DONE",
    }


def validate_completed_batch(batch_dir, expected_genes):
    paths = _batch_paths(batch_dir)
    if not paths["done"].exists():
        return False
    for key in ["rds", "rowdata", "genes"]:
        if not paths[key].exists() or paths[key].stat().st_size == 0:
            return False
    try:
        observed = pd.read_csv(paths["genes"])["gene"].astype(str).tolist()
    except Exception:
        return False
    return observed == list(map(str, expected_genes))


# %% RUN MEMORY-BOUNDED / RESUMABLE fitGAM
fitgam_batch_records = []
fitgam_failures = []

for row in tradeseq_input_manifest.itertuples(index=False):
    compartment = row.compartment
    trajectory_name = row.trajectory
    bundle_name = sanitize_filename(f"{compartment}__{trajectory_name}")
    trajectory_dir = TRADESEQ_FIT_DIR / bundle_name
    batches_dir = trajectory_dir / "batches"
    batches_dir.mkdir(parents=True, exist_ok=True)

    genes_df = pd.read_csv(row.genes_path)
    if "gene" not in genes_df.columns:
        raise RuntimeError(f"Missing gene column: {row.genes_path}")
    genes = genes_df["gene"].astype(str).tolist()
    if len(genes) != int(row.n_genes_filtered):
        raise RuntimeError(
            f"Gene-count mismatch for {compartment}/{trajectory_name}: "
            f"genes.csv={len(genes):,}, manifest={int(row.n_genes_filtered):,}."
        )

    n_batches = (len(genes) + TRADESEQ_BATCH_SIZE - 1) // TRADESEQ_BATCH_SIZE
    print("\n" + "=" * 100)
    print(f"TRADESEQ BATCHED fitGAM: {compartment} / {trajectory_name}")
    print("=" * 100)
    print(
        f"genes={len(genes):,}; metacells={int(row.n_metacells):,}; "
        f"lineages={int(row.n_lineages)}; nknots={TRADESEQ_NKNOTS}; "
        f"batch_size={TRADESEQ_BATCH_SIZE}; batches={n_batches}; workers=1"
    )

    # Compute the tradeSeq-equivalent TMM offset once from ALL filtered genes.
    # This file is reused on restart and prevents per-batch normalization changes.
    offset_path = trajectory_dir / "full_matrix_TMM_offset.csv"
    offset_stdout = trajectory_dir / "normalization_stdout.txt"
    offset_stderr = trajectory_dir / "normalization_stderr.txt"

    offset_valid = False
    if offset_path.exists() and offset_path.stat().st_size > 0:
        try:
            offset_df = pd.read_csv(offset_path)
            offset_valid = (
                len(offset_df) == int(row.n_metacells)
                and "offset" in offset_df.columns
                and np.isfinite(pd.to_numeric(offset_df["offset"], errors="coerce")).all()
            )
        except Exception:
            offset_valid = False

    if offset_valid:
        print("  normalization: SKIP valid full-matrix TMM checkpoint")
    else:
        if offset_path.exists():
            offset_path.unlink()
        print("  normalization: COMPUTE full-matrix TMM offset (sparse, one-time)")
        norm_cmd = [
            rscript_path,
            str(normalization_r_script),
            str(row.counts_path),
            str(row.genes_path),
            str(offset_path),
        ]
        norm_result = subprocess.run(norm_cmd, capture_output=True, text=True)
        offset_stdout.write_text(norm_result.stdout)
        offset_stderr.write_text(norm_result.stderr)
        if norm_result.returncode != 0:
            print(norm_result.stdout[-3000:])
            print(norm_result.stderr[-5000:])
            raise RuntimeError(
                f"Full-matrix TMM normalization failed for {compartment}/{trajectory_name}. "
                f"See {offset_stderr}"
            )
        if not offset_path.exists() or offset_path.stat().st_size == 0:
            raise RuntimeError(
                f"Normalization completed without offset file for {compartment}/{trajectory_name}."
            )
        offset_df = pd.read_csv(offset_path)
        if (
            len(offset_df) != int(row.n_metacells)
            or "offset" not in offset_df.columns
            or not np.isfinite(pd.to_numeric(offset_df["offset"], errors="coerce")).all()
        ):
            raise RuntimeError(
                f"Invalid normalization checkpoint for {compartment}/{trajectory_name}."
            )
        print(norm_result.stdout[-1200:])

    for batch_number, start0 in enumerate(range(0, len(genes), TRADESEQ_BATCH_SIZE), start=1):
        end0 = min(start0 + TRADESEQ_BATCH_SIZE, len(genes))
        batch_genes = genes[start0:end0]
        batch_id = f"batch_{batch_number:04d}"
        batch_dir = batches_dir / batch_id
        batch_dir.mkdir(parents=True, exist_ok=True)
        paths = _batch_paths(batch_dir)

        if TRADESEQ_OVERWRITE_FIT_BATCHES and paths["done"].exists():
            paths["done"].unlink()

        if validate_completed_batch(batch_dir, batch_genes):
            print(
                f"  {batch_id}: SKIP valid checkpoint "
                f"({start0 + 1}-{end0}; {len(batch_genes)} genes)"
            )
            status = "checkpoint"
        else:
            # Never trust partial artifacts from an interrupted/crashed batch.
            for key in ["rds", "rowdata", "genes", "done"]:
                if paths[key].exists():
                    paths[key].unlink()

            print(
                f"  {batch_id}: FIT genes {start0 + 1}-{end0} "
                f"({len(batch_genes)} genes)"
            )
            cmd = [
                rscript_path,
                str(fitgam_r_script),
                str(row.counts_path),
                str(row.genes_path),
                str(row.pseudotime_path),
                str(row.cellweights_path),
                str(offset_path),
                str(batch_dir),
                str(start0 + 1),       # R 1-based inclusive
                str(end0),             # R 1-based inclusive
                str(TRADESEQ_NKNOTS),
                str(TRADESEQ_FIT_SEED),
                batch_id,
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            paths["stdout"].write_text(result.stdout)
            paths["stderr"].write_text(result.stderr)

            if result.returncode != 0:
                fitgam_failures.append({
                    "compartment": compartment,
                    "trajectory": trajectory_name,
                    "batch_id": batch_id,
                    "gene_start": start0 + 1,
                    "gene_end": end0,
                    "returncode": result.returncode,
                    "stderr": result.stderr[-5000:],
                })
                pd.DataFrame(fitgam_failures).to_csv(
                    TRADESEQ_FIT_DIR / "trajectory_fitGAM_failures.csv",
                    index=False,
                )
                print(result.stdout[-3000:])
                print(result.stderr[-5000:])
                raise RuntimeError(
                    f"tradeSeq fitGAM failed for {compartment}/{trajectory_name} "
                    f"{batch_id}. See {paths['stderr']}"
                )

            if not all(paths[key].exists() and paths[key].stat().st_size > 0
                       for key in ["rds", "rowdata", "genes"]):
                raise RuntimeError(
                    f"fitGAM batch returned successfully but required outputs are missing: "
                    f"{compartment}/{trajectory_name}/{batch_id}"
                )
            observed = pd.read_csv(paths["genes"])["gene"].astype(str).tolist()
            if observed != batch_genes:
                raise RuntimeError(
                    f"Gene checkpoint mismatch for {compartment}/{trajectory_name}/{batch_id}."
                )

            # DONE is deliberately written last and atomically renamed.
            done_tmp = batch_dir / "DONE.tmp"
            done_tmp.write_text(
                f"{compartment}\t{trajectory_name}\t{batch_id}\t"
                f"{start0 + 1}\t{end0}\t{len(batch_genes)}\n"
            )
            done_tmp.replace(paths["done"])
            status = "fit"
            print(result.stdout[-1200:])

        fitgam_batch_records.append({
            "compartment": compartment,
            "trajectory": trajectory_name,
            "batch_id": batch_id,
            "gene_start": start0 + 1,
            "gene_end": end0,
            "n_genes": len(batch_genes),
            "status": status,
            "fit_path": str(paths["rds"]),
            "rowdata_path": str(paths["rowdata"]),
            "genes_path": str(paths["genes"]),
            "stdout_path": str(paths["stdout"]),
            "stderr_path": str(paths["stderr"]),
        })

    # Trajectory checkpoint: every batch must now be valid.
    trajectory_records = [
        x for x in fitgam_batch_records
        if x["compartment"] == compartment and x["trajectory"] == trajectory_name
    ]
    if sum(x["n_genes"] for x in trajectory_records) != len(genes):
        raise RuntimeError(f"Incomplete fitted gene coverage for {compartment}/{trajectory_name}.")

    pd.DataFrame(trajectory_records).to_csv(
        trajectory_dir / "fitGAM_batch_manifest.csv",
        index=False,
    )
    print(f"Completed/checkpointed all {n_batches} batches for {compartment}/{trajectory_name}.")

# %% COMBINE fitGAM QC / FINAL MANIFEST
tradeseq_fitgam_batch_manifest = pd.DataFrame(fitgam_batch_records)
batch_manifest_path = TRADESEQ_FIT_DIR / "trajectory_fitGAM_batch_manifest.csv"
tradeseq_fitgam_batch_manifest.to_csv(batch_manifest_path, index=False)

# Combine rowData only; do NOT combine SCEs, which would recreate the memory problem.
combined_rowdata_records = []
trajectory_manifest_records = []
for row in tradeseq_input_manifest.itertuples(index=False):
    mask = (
        (tradeseq_fitgam_batch_manifest["compartment"] == row.compartment)
        & (tradeseq_fitgam_batch_manifest["trajectory"] == row.trajectory)
    )
    sub = tradeseq_fitgam_batch_manifest.loc[mask].copy()
    rowdata_parts = []
    for batch in sub.itertuples(index=False):
        rd = pd.read_csv(batch.rowdata_path)
        rd.insert(0, "trajectory", row.trajectory)
        rd.insert(0, "compartment", row.compartment)
        rowdata_parts.append(rd)
    combined = pd.concat(rowdata_parts, ignore_index=True)
    if len(combined) != int(row.n_genes_filtered):
        raise RuntimeError(
            f"Combined rowData gene count mismatch for {row.compartment}/{row.trajectory}."
        )
    combined_path = (
        TRADESEQ_FIT_DIR / sanitize_filename(f"{row.compartment}__{row.trajectory}")
        / "fitGAM_rowData_combined.csv"
    )
    combined.to_csv(combined_path, index=False)
    combined_rowdata_records.append(combined)
    trajectory_manifest_records.append({
        "compartment": row.compartment,
        "trajectory": row.trajectory,
        "n_genes": int(row.n_genes_filtered),
        "n_metacells": int(row.n_metacells),
        "n_lineages": int(row.n_lineages),
        "nknots": TRADESEQ_NKNOTS,
        "batch_size": TRADESEQ_BATCH_SIZE,
        "n_batches": len(sub),
        "fit_storage": "per_batch_rds",
        "normalization": "full-filtered-matrix TMM offset",
        "normalization_offset_path": str(
            TRADESEQ_FIT_DIR / sanitize_filename(f"{row.compartment}__{row.trajectory}")
            / "full_matrix_TMM_offset.csv"
        ),
        "batch_manifest_path": str(
            TRADESEQ_FIT_DIR / sanitize_filename(f"{row.compartment}__{row.trajectory}")
            / "fitGAM_batch_manifest.csv"
        ),
        "rowdata_path": str(combined_path),
    })

tradeseq_fitgam_manifest = pd.DataFrame(trajectory_manifest_records)
fitgam_manifest_path = TRADESEQ_FIT_DIR / "trajectory_fitGAM_manifest.csv"
tradeseq_fitgam_manifest.to_csv(fitgam_manifest_path, index=False)

print("\n" + "=" * 100)
print("TRADESEQ MEMORY-BOUNDED fitGAM COMPLETE")
print("=" * 100)
print(
    tradeseq_fitgam_manifest[
        ["compartment", "trajectory", "n_genes", "n_metacells", "n_lineages", "nknots", "n_batches"]
    ].to_string(index=False)
)
print("\nSaved:")
print(" ", fitgam_manifest_path)
print(" ", batch_manifest_path)
print("\nNo monolithic all-gene SCE was created; fitted objects are checkpointed per batch.")
# %%
