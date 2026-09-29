# %% 04C05 TRAJECTORY-DISEASE
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

# %% DISEASE ASSOCIATION: DESIGN + DONOR CONTINGENCY
#
# Purpose
# -------
# Audit the donor-level structure BEFORE fitting disease-association
# models.
#
# This section:
#   1. Uses the frozen donor-level trajectory summary.
#   2. Confirms disease is donor-consistent.
#   3. Constructs one-row-per-donor metadata tables.
#   4. Audits disease × assay overlap.
#   5. Audits lineage-specific disease sample sizes among donors with
#      actual lineage support / finite weighted pseudotime.
#   6. Quantifies disease–assay confounding/identifiability.
#   7. Saves tidy design tables for the next inferential section.
#
# This section DOES NOT:
#   - modify trajectories
#   - filter trajectories based on disease
#   - fit disease models
#   - choose significant lineages
#
# Important:
#   dominant_state is deliberately NOT required to be donor-consistent.
#   It is a trajectory/metacell phenotype and is not a donor-level
#   adjustment variable here.


# --------------------------------------
# Configuration
# --------------------------------------

DISEASE_MIN_DONORS_PER_GROUP = 5

# Minimum number of donors from a disease group that must have finite
# weighted pseudotime on a lineage for that disease group to be
# considered adequately represented for conditional-position testing.
POSITION_MIN_DONORS_PER_GROUP = 5

# Minimum donor count in a disease × assay cell for that cell to be
# considered substantively represented in the overlap audit.
ASSAY_CELL_MIN_DONORS = 3


# --------------------------------------
# Load frozen donor-level summary
# --------------------------------------

donor_summary_file = (
    TRAJECTORY_DIR
    / "trajectory_donor_level_summary.csv"
)

if not donor_summary_file.exists():
    raise FileNotFoundError(
        donor_summary_file
    )


disease_df = pd.read_csv(
    donor_summary_file,
    dtype={
        DONOR_COL: str,
    },
)


required_cols = [
    "compartment",
    "trajectory",
    "lineage",
    DONOR_COL,
    DISEASE_COL,
    ASSAY_COL,
    "n_metacells",
    "n_finite_pseudotime",
    "n_lineage_supported_metacells",
    "fraction_lineage_supported",
    "lineage_weight_sum",
    "lineage_weight_mean",
    "lineage_effective_metacells",
    "weighted_mean_pseudotime",
    "median_pseudotime",
    "metadata_consistent",
    "robustness_match_fraction",
    "robustness_spearman_median",
    "robustness_spearman_q05",
]


missing_cols = [
    col
    for col
    in required_cols
    if col not in disease_df.columns
]


if missing_cols:
    raise ValueError(
        "Missing required donor-level "
        "columns:\n"
        f"{missing_cols}"
    )


# --------------------------------------
# Normalize grouping variables
# --------------------------------------

for col in [
    DONOR_COL,
    DISEASE_COL,
    ASSAY_COL,
]:
    disease_df[col] = (
        disease_df[col]
        .astype("string")
        .str.strip()
    )


def valid_label_mask(
    series,
):
    series = (
        series
        .astype("string")
        .str.strip()
    )

    return (
        series.notna()
        & ~series.isin(
            [
                "",
                "nan",
                "None",
                "<NA>",
            ]
        )
    )


# Disease must exist for disease inference.
valid_disease = valid_label_mask(
    disease_df[
        DISEASE_COL
    ]
)


if not valid_disease.all():
    print(
        "WARNING: excluding "
        f"{int((~valid_disease).sum())} "
        "donor-lineage rows with missing "
        "disease labels."
    )

    disease_df = (
        disease_df.loc[
            valid_disease
        ]
        .copy()
    )


# --------------------------------------
# Check donor-level disease consistency
# independently of the earlier combined
# metadata_consistent flag.
# --------------------------------------

donor_disease_check = (
    disease_df[
        [
            DONOR_COL,
            DISEASE_COL,
        ]
    ]
    .drop_duplicates()
    .groupby(
        DONOR_COL,
        observed=True,
    )[
        DISEASE_COL
    ]
    .nunique(
        dropna=True
    )
)


bad_disease_donors = (
    donor_disease_check[
        donor_disease_check
        != 1
    ]
)


if not bad_disease_donors.empty:
    raise ValueError(
        "Disease is not uniquely defined "
        "for the following donors:\n"
        f"{bad_disease_donors.to_string()}"
    )


# --------------------------------------
# Construct global one-row-per-donor
# metadata table.
#
# Assay is retained only if a donor has
# one unique non-missing assay across the
# available trajectory records.
# --------------------------------------

global_donor_records = []


for donor_id, group in disease_df.groupby(
    DONOR_COL,
    observed=True,
    sort=True,
):

    disease_values = (
        group[
            DISEASE_COL
        ]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    if len(disease_values) != 1:
        raise ValueError(
            f"Donor {donor_id!r} has "
            "non-unique disease labels."
        )

    assay_values = (
        group.loc[
            valid_label_mask(
                group[
                    ASSAY_COL
                ]
            ),
            ASSAY_COL,
        ]
        .astype(str)
        .unique()
        .tolist()
    )

    assay_consistent = (
        len(assay_values)
        == 1
    )

    assay_value = (
        assay_values[0]
        if assay_consistent
        else pd.NA
    )

    global_donor_records.append(
        {
            DONOR_COL:
                str(
                    donor_id
                ),

            DISEASE_COL:
                disease_values[
                    0
                ],

            ASSAY_COL:
                assay_value,

            "assay_consistent":
                assay_consistent,

            "n_unique_assays":
                len(
                    assay_values
                ),
        }
    )


global_donor_metadata = (
    pd.DataFrame(global_donor_records)
    .sort_values(
        DONOR_COL
    )
    .reset_index(
        drop=True
    )
)


# --------------------------------------
# Global disease counts
# --------------------------------------

global_disease_counts = (
    global_donor_metadata
    .groupby(
        DISEASE_COL,
        observed=True,
    )
    .agg(
        n_donors=(
            DONOR_COL,
            "nunique",
        ),
    )
    .reset_index()
    .sort_values(
        [
            "n_donors",
            DISEASE_COL,
        ],
        ascending=[
            False,
            True,
        ],
    )
    .reset_index(
        drop=True
    )
)


global_disease_counts[
    "adequate_global_n"
] = (
    global_disease_counts[
        "n_donors"
    ]
    >= DISEASE_MIN_DONORS_PER_GROUP
)


# --------------------------------------
# Global disease × assay contingency
#
# Only uniquely assigned assays are used
# for assay-aware inference.
# --------------------------------------

global_assay_eligible = (
    global_donor_metadata[
        global_donor_metadata[
            "assay_consistent"
        ]
        & global_donor_metadata[
            ASSAY_COL
        ].notna()
    ]
    .copy()
)


global_disease_assay = (
    global_assay_eligible
    .groupby(
        [
            DISEASE_COL,
            ASSAY_COL,
        ],
        observed=True,
    )
    .agg(
        n_donors=(
            DONOR_COL,
            "nunique",
        ),
    )
    .reset_index()
)


global_disease_assay[
    "adequate_cell_n"
] = (
    global_disease_assay[
        "n_donors"
    ]
    >= ASSAY_CELL_MIN_DONORS
)


# Wide version for easy inspection.
global_disease_assay_wide = (
    global_disease_assay
    .pivot(
        index=DISEASE_COL,
        columns=ASSAY_COL,
        values="n_donors",
    )
    .fillna(
        0
    )
    .astype(
        int
    )
    .reset_index()
)


# --------------------------------------
# Trajectory-level donor metadata
#
# A donor may be assay-consistent globally
# but the trajectory-specific table is
# what matters for each model.
# --------------------------------------

trajectory_donor_records = []


for (
    compartment,
    trajectory,
    donor_id,
), group in disease_df.groupby(
    [
        "compartment",
        "trajectory",
        DONOR_COL,
    ],
    observed=True,
    sort=True,
):

    disease_values = (
        group[
            DISEASE_COL
        ]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    if len(disease_values) != 1:
        raise ValueError(
            f"{compartment}/"
            f"{trajectory}/"
            f"{donor_id}: "
            "disease is not unique."
        )

    assay_values = (
        group.loc[
            valid_label_mask(
                group[
                    ASSAY_COL
                ]
            ),
            ASSAY_COL,
        ]
        .astype(str)
        .unique()
        .tolist()
    )

    assay_consistent = (
        len(assay_values)
        == 1
    )

    trajectory_donor_records.append(
        {
            "compartment":
                compartment,

            "trajectory":
                trajectory,

            DONOR_COL:
                str(
                    donor_id
                ),

            DISEASE_COL:
                disease_values[
                    0
                ],

            ASSAY_COL:
                (
                    assay_values[
                        0
                    ]
                    if assay_consistent
                    else pd.NA
                ),

            "assay_consistent":
                assay_consistent,

            "n_unique_assays":
                len(
                    assay_values
                ),
        }
    )


trajectory_donor_metadata = (
    pd.DataFrame(trajectory_donor_records)
)


# --------------------------------------
# Trajectory-level disease counts
# --------------------------------------

trajectory_disease_counts = (
    trajectory_donor_metadata
    .groupby(
        [
            "compartment",
            "trajectory",
            DISEASE_COL,
        ],
        observed=True,
    )
    .agg(
        n_donors=(
            DONOR_COL,
            "nunique",
        ),
    )
    .reset_index()
)


trajectory_disease_counts[
    "adequate_disease_n"
] = (
    trajectory_disease_counts[
        "n_donors"
    ]
    >= DISEASE_MIN_DONORS_PER_GROUP
)


# --------------------------------------
# Trajectory disease × assay counts
# --------------------------------------

trajectory_assay_eligible = (
    trajectory_donor_metadata[
        trajectory_donor_metadata[
            "assay_consistent"
        ]
        & trajectory_donor_metadata[
            ASSAY_COL
        ].notna()
    ]
    .copy()
)


trajectory_disease_assay = (
    trajectory_assay_eligible
    .groupby(
        [
            "compartment",
            "trajectory",
            DISEASE_COL,
            ASSAY_COL,
        ],
        observed=True,
    )
    .agg(
        n_donors=(
            DONOR_COL,
            "nunique",
        ),
    )
    .reset_index()
)


trajectory_disease_assay[
    "adequate_cell_n"
] = (
    trajectory_disease_assay[
        "n_donors"
    ]
    >= ASSAY_CELL_MIN_DONORS
)


# --------------------------------------
# Quantify disease × assay overlap
# per trajectory.
#
# These are design diagnostics, not
# hypothesis tests.
# --------------------------------------

assay_overlap_records = []


for (
    compartment,
    trajectory,
), group in trajectory_assay_eligible.groupby(
    [
        "compartment",
        "trajectory",
    ],
    observed=True,
):

    diseases = sorted(
        group[
            DISEASE_COL
        ]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    assays = sorted(
        group[
            ASSAY_COL
        ]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    contingency = pd.crosstab(
        group[
            DISEASE_COL
        ],
        group[
            ASSAY_COL
        ],
    )

    n_donors = int(
        group[
            DONOR_COL
        ]
        .nunique()
    )

    n_diseases = len(
        diseases
    )

    n_assays = len(
        assays
    )

    # Number of disease groups represented
    # in each assay.
    diseases_per_assay = (
        contingency
        .gt(
            0
        )
        .sum(
            axis=0
        )
    )

    # Number of assays represented in each
    # disease group.
    assays_per_disease = (
        contingency
        .gt(
            0
        )
        .sum(
            axis=1
        )
    )

    n_shared_assays = int(
        (
            diseases_per_assay
            >= 2
        )
        .sum()
    )

    n_diseases_mult_assay = int(
        (
            assays_per_disease
            >= 2
        )
        .sum()
    )

    # Stronger criterion: assay contains at
    # least two disease groups with >= the
    # minimum useful cell count.
    adequately_populated = (
        contingency
        >= ASSAY_CELL_MIN_DONORS
    )

    adequate_diseases_per_assay = (
        adequately_populated
        .sum(
            axis=0
        )
    )

    n_adequately_shared_assays = int(
        (
            adequate_diseases_per_assay
            >= 2
        )
        .sum()
    )

    # A disease is assay-exclusive when all
    # of its assay-eligible donors occur in
    # exactly one assay.
    n_assay_exclusive_diseases = int(
        (
            assays_per_disease
            == 1
        )
        .sum()
    )

    # Complete confounding in the strongest
    # descriptive sense: no assay contains
    # more than one disease group.
    complete_disease_assay_separation = (
        n_shared_assays
        == 0
    )

    assay_overlap_records.append(
        {
            "compartment":
                compartment,

            "trajectory":
                trajectory,

            "n_assay_eligible_donors":
                n_donors,

            "n_diseases":
                n_diseases,

            "n_assays":
                n_assays,

            "n_shared_assays":
                n_shared_assays,

            "n_adequately_shared_assays":
                n_adequately_shared_assays,

            "n_diseases_represented_in_multiple_assays":
                n_diseases_mult_assay,

            "n_assay_exclusive_diseases":
                n_assay_exclusive_diseases,

            "complete_disease_assay_separation":
                bool(
                    complete_disease_assay_separation
                ),

            "assay_adjustment_potentially_identifiable":
                bool(
                    (
                        n_assays
                        >= 2
                    )
                    and (
                        n_shared_assays
                        >= 1
                    )
                ),

            "well_populated_assay_overlap":
                bool(
                    n_adequately_shared_assays
                    >= 1
                ),
        }
    )


trajectory_assay_overlap = (
    pd.DataFrame(assay_overlap_records)
)


# --------------------------------------
# Lineage-specific disease coverage
#
# Two questions are kept separate:
#
# A) Occupancy/support:
#    all donors in the trajectory can
#    contribute to this analysis.
#
# B) Conditional position:
#    only donors with actual lineage
#    support and finite weighted PT can
#    contribute.
# --------------------------------------

lineage_disease_records = []


for (
    compartment,
    trajectory,
    lineage,
    disease,
), group in disease_df.groupby(
    [
        "compartment",
        "trajectory",
        "lineage",
        DISEASE_COL,
    ],
    observed=True,
    sort=True,
):

    n_donors = int(
        group[
            DONOR_COL
        ]
        .nunique()
    )

    lineage_weight = pd.to_numeric(
        group[
            "lineage_weight_sum"
        ],
        errors="coerce",
    )

    weighted_pt = pd.to_numeric(
        group[
            "weighted_mean_pseudotime"
        ],
        errors="coerce",
    )

    median_pt = pd.to_numeric(
        group[
            "median_pseudotime"
        ],
        errors="coerce",
    )

    positive_support = (
        np.isfinite(lineage_weight)
        & (
            lineage_weight
            > DONOR_SUMMARY_MIN_WEIGHT
        )
    )

    finite_weighted_pt = (
        np.isfinite(weighted_pt)
        & positive_support
    )

    finite_median_pt = (
        np.isfinite(median_pt)
        & positive_support
    )

    n_supported = int(
        positive_support.sum()
    )

    n_position = int(
        finite_weighted_pt.sum()
    )

    n_median_position = int(
        finite_median_pt.sum()
    )

    assay_consistent = (
        group[
            ASSAY_COL
        ]
        .notna()
    )

    n_assay_consistent = int(
        assay_consistent.sum()
    )

    n_position_assay_consistent = int(
        (
            finite_weighted_pt
            & assay_consistent
        )
        .sum()
    )

    lineage_disease_records.append(
        {
            "compartment":
                compartment,

            "trajectory":
                trajectory,

            "lineage":
                lineage,

            DISEASE_COL:
                disease,

            "n_donors_total":
                n_donors,

            "n_donors_lineage_supported":
                n_supported,

            "fraction_donors_lineage_supported":
                (
                    n_supported
                    / n_donors
                    if n_donors
                    else np.nan
                ),

            "n_donors_finite_weighted_pseudotime":
                n_position,

            "fraction_donors_finite_weighted_pseudotime":
                (
                    n_position
                    / n_donors
                    if n_donors
                    else np.nan
                ),

            "n_donors_finite_median_pseudotime":
                n_median_position,

            "n_donors_assay_consistent":
                n_assay_consistent,

            "n_donors_finite_weighted_pseudotime_assay_consistent":
                n_position_assay_consistent,

            "adequate_occupancy_group_n":
                bool(
                    n_donors
                    >= DISEASE_MIN_DONORS_PER_GROUP
                ),

            "adequate_position_group_n":
                bool(
                    n_position
                    >= POSITION_MIN_DONORS_PER_GROUP
                ),

            "robustness_match_fraction":
                float(
                    group[
                        "robustness_match_fraction"
                    ]
                    .iloc[
                        0
                    ]
                ),

            "robustness_spearman_median":
                float(
                    group[
                        "robustness_spearman_median"
                    ]
                    .iloc[
                        0
                    ]
                ),

            "robustness_spearman_q05":
                float(
                    group[
                        "robustness_spearman_q05"
                    ]
                    .iloc[
                        0
                    ]
                ),
        }
    )


lineage_disease_coverage = (
    pd.DataFrame(lineage_disease_records)
)


# --------------------------------------
# Lineage-level eligibility summary
#
# This is descriptive eligibility only.
# We do NOT filter on robustness here.
# --------------------------------------

lineage_design_records = []


for (
    compartment,
    trajectory,
    lineage,
), group in lineage_disease_coverage.groupby(
    [
        "compartment",
        "trajectory",
        "lineage",
    ],
    observed=True,
    sort=True,
):

    n_disease_groups = int(
        group[
            DISEASE_COL
        ]
        .nunique()
    )

    n_adequate_occupancy_groups = int(
        group[
            "adequate_occupancy_group_n"
        ]
        .sum()
    )

    n_adequate_position_groups = int(
        group[
            "adequate_position_group_n"
        ]
        .sum()
    )

    total_donors = int(
        group[
            "n_donors_total"
        ]
        .sum()
    )

    total_position_donors = int(
        group[
            "n_donors_finite_weighted_pseudotime"
        ]
        .sum()
    )

    lineage_design_records.append(
        {
            "compartment":
                compartment,

            "trajectory":
                trajectory,

            "lineage":
                lineage,

            "n_disease_groups":
                n_disease_groups,

            "n_adequate_occupancy_groups":
                n_adequate_occupancy_groups,

            "n_adequate_position_groups":
                n_adequate_position_groups,

            "n_donors_total":
                total_donors,

            "n_position_donors_total":
                total_position_donors,

            "occupancy_test_eligible":
                bool(
                    n_adequate_occupancy_groups
                    >= 2
                ),

            "position_test_eligible":
                bool(
                    n_adequate_position_groups
                    >= 2
                ),

            "robustness_match_fraction":
                float(
                    group[
                        "robustness_match_fraction"
                    ]
                    .iloc[
                        0
                    ]
                ),

            "robustness_spearman_median":
                float(
                    group[
                        "robustness_spearman_median"
                    ]
                    .iloc[
                        0
                    ]
                ),

            "robustness_spearman_q05":
                float(
                    group[
                        "robustness_spearman_q05"
                    ]
                    .iloc[
                        0
                    ]
                ),
        }
    )


lineage_disease_design = (
    pd.DataFrame(lineage_design_records)
)


# --------------------------------------
# Disease × assay overlap specifically
# among donors contributing position data
# to each lineage.
# --------------------------------------

lineage_position_assay_records = []


for (
    compartment,
    trajectory,
    lineage,
), group in disease_df.groupby(
    [
        "compartment",
        "trajectory",
        "lineage",
    ],
    observed=True,
    sort=True,
):

    lineage_weight = pd.to_numeric(
        group[
            "lineage_weight_sum"
        ],
        errors="coerce",
    )

    weighted_pt = pd.to_numeric(
        group[
            "weighted_mean_pseudotime"
        ],
        errors="coerce",
    )

    eligible = (
        np.isfinite(lineage_weight)
        & (
            lineage_weight
            > DONOR_SUMMARY_MIN_WEIGHT
        )
        & np.isfinite(
            weighted_pt
        )
        & valid_label_mask(
            group[
                ASSAY_COL
            ]
        )
    )

    position_group = (
        group.loc[
            eligible
        ]
        .copy()
    )

    if position_group.empty:
        lineage_position_assay_records.append(
            {
                "compartment":
                    compartment,

                "trajectory":
                    trajectory,

                "lineage":
                    lineage,

                "n_position_assay_donors":
                    0,

                "n_diseases":
                    0,

                "n_assays":
                    0,

                "n_shared_assays":
                    0,

                "n_adequately_shared_assays":
                    0,

                "assay_adjustment_potentially_identifiable":
                    False,

                "well_populated_assay_overlap":
                    False,
            }
        )

        continue

    contingency = pd.crosstab(
        position_group[
            DISEASE_COL
        ],
        position_group[
            ASSAY_COL
        ],
    )

    diseases_per_assay = (
        contingency
        .gt(
            0
        )
        .sum(
            axis=0
        )
    )

    adequate_diseases_per_assay = (
        (
            contingency
            >= ASSAY_CELL_MIN_DONORS
        )
        .sum(
            axis=0
        )
    )

    n_shared_assays = int(
        (
            diseases_per_assay
            >= 2
        )
        .sum()
    )

    n_adequately_shared_assays = int(
        (
            adequate_diseases_per_assay
            >= 2
        )
        .sum()
    )

    n_assays = int(
        position_group[
            ASSAY_COL
        ]
        .nunique()
    )

    n_diseases = int(
        position_group[
            DISEASE_COL
        ]
        .nunique()
    )

    lineage_position_assay_records.append(
        {
            "compartment":
                compartment,

            "trajectory":
                trajectory,

            "lineage":
                lineage,

            "n_position_assay_donors":
                int(
                    len(position_group)
                ),

            "n_diseases":
                n_diseases,

            "n_assays":
                n_assays,

            "n_shared_assays":
                n_shared_assays,

            "n_adequately_shared_assays":
                n_adequately_shared_assays,

            "assay_adjustment_potentially_identifiable":
                bool(
                    (
                        n_assays
                        >= 2
                    )
                    and (
                        n_shared_assays
                        >= 1
                    )
                ),

            "well_populated_assay_overlap":
                bool(
                    n_adequately_shared_assays
                    >= 1
                ),
        }
    )


lineage_position_assay_design = (
    pd.DataFrame(lineage_position_assay_records)
)


# --------------------------------------
# Attach position-assay design to the
# lineage-level design table.
# --------------------------------------

lineage_disease_design = (
    lineage_disease_design
    .merge(
        lineage_position_assay_design,
        on=[
            "compartment",
            "trajectory",
            "lineage",
        ],
        how="left",
        validate="one_to_one",
        suffixes=(
            "",
            "_position_assay",
        ),
    )
)


# --------------------------------------
# QC
# --------------------------------------

expected_lineages = (
    disease_df[
        [
            "compartment",
            "trajectory",
            "lineage",
        ]
    ]
    .drop_duplicates()
    .shape[
        0
    ]
)


if (
    len(lineage_disease_design)
    != expected_lineages
):
    raise RuntimeError(
        "Lineage design table does not "
        "contain exactly one row per "
        "frozen primary lineage."
    )


if (
    disease_df[
        [
            "compartment",
            "trajectory",
            "lineage",
            DONOR_COL,
        ]
    ]
    .duplicated()
    .any()
):
    raise RuntimeError(
        "Duplicate donor × trajectory × "
        "lineage records detected."
    )


# --------------------------------------
# Sort outputs
# --------------------------------------

trajectory_disease_counts = (
    trajectory_disease_counts
    .sort_values(
        [
            "compartment",
            "trajectory",
            "n_donors",
            DISEASE_COL,
        ],
        ascending=[
            True,
            True,
            False,
            True,
        ],
    )
    .reset_index(
        drop=True
    )
)


trajectory_disease_assay = (
    trajectory_disease_assay
    .sort_values(
        [
            "compartment",
            "trajectory",
            DISEASE_COL,
            ASSAY_COL,
        ]
    )
    .reset_index(
        drop=True
    )
)


trajectory_assay_overlap = (
    trajectory_assay_overlap
    .sort_values(
        [
            "compartment",
            "trajectory",
        ]
    )
    .reset_index(
        drop=True
    )
)


lineage_disease_coverage = (
    lineage_disease_coverage
    .sort_values(
        [
            "compartment",
            "trajectory",
            "lineage",
            DISEASE_COL,
        ]
    )
    .reset_index(
        drop=True
    )
)


lineage_disease_design = (
    lineage_disease_design
    .sort_values(
        [
            "compartment",
            "trajectory",
            "lineage",
        ]
    )
    .reset_index(
        drop=True
    )
)


# --------------------------------------
# Save
# --------------------------------------

global_donor_metadata.to_csv(
    TRAJECTORY_DIR
    / "trajectory_disease_global_donor_metadata.csv",
    index=False,
)


global_disease_counts.to_csv(
    TRAJECTORY_DIR
    / "trajectory_disease_global_counts.csv",
    index=False,
)


global_disease_assay.to_csv(
    TRAJECTORY_DIR
    / "trajectory_disease_global_assay_contingency.csv",
    index=False,
)


trajectory_donor_metadata.to_csv(
    TRAJECTORY_DIR
    / "trajectory_disease_trajectory_donor_metadata.csv",
    index=False,
)


trajectory_disease_counts.to_csv(
    TRAJECTORY_DIR
    / "trajectory_disease_counts.csv",
    index=False,
)


trajectory_disease_assay.to_csv(
    TRAJECTORY_DIR
    / "trajectory_disease_assay_contingency.csv",
    index=False,
)


trajectory_assay_overlap.to_csv(
    TRAJECTORY_DIR
    / "trajectory_disease_assay_overlap.csv",
    index=False,
)


lineage_disease_coverage.to_csv(
    TRAJECTORY_DIR
    / "trajectory_disease_lineage_coverage.csv",
    index=False,
)


lineage_disease_design.to_csv(
    TRAJECTORY_DIR
    / "trajectory_disease_lineage_design.csv",
    index=False,
)


# --------------------------------------
# Checkpoint
# --------------------------------------

print(
    "\n"
    + "=" * 100
)

print(
    "04c DISEASE ASSOCIATION DESIGN "
    "AUDIT COMPLETE"
)

print(
    "=" * 100
)


print(
    "\nGLOBAL DISEASE COUNTS"
)

print(
    global_disease_counts
    .to_string(
        index=False
    )
)


print(
    "\nGLOBAL DISEASE × ASSAY"
)

print(
    global_disease_assay_wide
    .to_string(
        index=False
    )
)


print(
    "\nTRAJECTORY DISEASE COUNTS"
)

print(
    trajectory_disease_counts[
        [
            "compartment",
            "trajectory",
            DISEASE_COL,
            "n_donors",
            "adequate_disease_n",
        ]
    ]
    .to_string(
        index=False
    )
)


print(
    "\nTRAJECTORY DISEASE × ASSAY "
    "IDENTIFIABILITY"
)

print(
    trajectory_assay_overlap[
        [
            "compartment",
            "trajectory",
            "n_assay_eligible_donors",
            "n_diseases",
            "n_assays",
            "n_shared_assays",
            "n_adequately_shared_assays",
            "n_assay_exclusive_diseases",
            "complete_disease_assay_separation",
            "assay_adjustment_potentially_identifiable",
            "well_populated_assay_overlap",
        ]
    ]
    .to_string(
        index=False
    )
)


print(
    "\nLINEAGE DISEASE-TEST DESIGN"
)

print(
    lineage_disease_design[
        [
            "compartment",
            "trajectory",
            "lineage",
            "n_disease_groups",
            "n_adequate_occupancy_groups",
            "n_adequate_position_groups",
            "n_donors_total",
            "n_position_donors_total",
            "occupancy_test_eligible",
            "position_test_eligible",
            "robustness_match_fraction",
            "robustness_spearman_median",
            "n_position_assay_donors",
            "n_shared_assays",
            "n_adequately_shared_assays",
            "assay_adjustment_potentially_identifiable",
            "well_populated_assay_overlap",
        ]
    ]
    .to_string(
        index=False
    )
)


print(
    "\nLINEAGE × DISEASE POSITION COVERAGE"
)

print(
    lineage_disease_coverage[
        [
            "compartment",
            "trajectory",
            "lineage",
            DISEASE_COL,
            "n_donors_total",
            "n_donors_lineage_supported",
            "n_donors_finite_weighted_pseudotime",
            "n_donors_finite_weighted_pseudotime_assay_consistent",
            "adequate_occupancy_group_n",
            "adequate_position_group_n",
        ]
    ]
    .to_string(
        index=False
    )
)


print(
    "\nSaved:"
)

for filename in [
    "trajectory_disease_global_donor_metadata.csv",
    "trajectory_disease_global_counts.csv",
    "trajectory_disease_global_assay_contingency.csv",
    "trajectory_disease_trajectory_donor_metadata.csv",
    "trajectory_disease_counts.csv",
    "trajectory_disease_assay_contingency.csv",
    "trajectory_disease_assay_overlap.csv",
    "trajectory_disease_lineage_coverage.csv",
    "trajectory_disease_lineage_design.csv",
]:
    print(
        " ",
        TRAJECTORY_DIR
        / filename,
    )

# %% DISEASE ASSOCIATION: DONOR-LEVEL INFERENCE
#
# Primary inference:
#   donor is the biological replicate
#
# Two distinct endpoints:
#
#   1. LINEAGE OCCUPANCY / SUPPORT
#      fraction_lineage_supported
#      All adequately represented donors in the trajectory.
#
#   2. CONDITIONAL TRAJECTORY POSITION
#      weighted_mean_pseudotime
#      Only donors with actual lineage support and finite pseudotime.
#
# Primary tests:
#   - Kruskal-Wallis omnibus
#   - pairwise Mann-Whitney U
#   - rank-biserial effect size
#
# Assay sensitivity:
#   - OLS + HC3 robust SE
#   - outcome ~ C(disease) + C(assay)
#
# Multiple testing:
#   - BH across lineage-level omnibus tests separately by endpoint
#   - BH across pairwise tests separately by endpoint
#   - BH across assay-adjusted disease coefficients separately by endpoint
#
# No trajectory is selected or modified based on these results.


from itertools import combinations

from scipy.stats import (
    kruskal,
    mannwhitneyu,
)

import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests


DISEASE_PRIMARY_ENDPOINTS = {
    "occupancy": {
        "column":
            "fraction_lineage_supported",

        "min_group_n":
            DISEASE_MIN_DONORS_PER_GROUP,

        "require_lineage_support":
            False,
    },

    "position": {
        "column":
            "weighted_mean_pseudotime",

        "min_group_n":
            POSITION_MIN_DONORS_PER_GROUP,

        "require_lineage_support":
            True,
    },
}


# --------------------------------------
# Helpers
# --------------------------------------

def bh_adjust(
    pvalues,
):
    """
    BH-adjust finite p-values while preserving
    NaN positions.
    """

    pvalues = np.asarray(
        pvalues,
        dtype=float,
    )

    adjusted = np.full(
        pvalues.shape,
        np.nan,
        dtype=float,
    )

    finite = np.isfinite(
        pvalues
    )

    if np.any(
        finite
    ):
        adjusted[
            finite
        ] = multipletests(
            pvalues[
                finite
            ],
            method="fdr_bh",
        )[1]

    return adjusted


def rank_biserial_from_u(
    u_statistic,
    n1,
    n2,
):
    """
    Rank-biserial correlation oriented as:

        disease_1 > disease_2  -> positive

    scipy mannwhitneyu(x, y) returns U for x.
    """

    if (
        n1 <= 0
        or n2 <= 0
    ):
        return np.nan

    return float(
        (
            2.0
            * float(
                u_statistic
            )
            / (
                n1
                * n2
            )
        )
        - 1.0
    )


def disease_group_values(
    frame,
    outcome_col,
    min_group_n,
):
    """
    Return adequately represented disease groups
    with finite outcome values.
    """

    working = frame[
        [
            DISEASE_COL,
            outcome_col,
        ]
    ].copy()

    working[
        outcome_col
    ] = pd.to_numeric(
        working[
            outcome_col
        ],
        errors="coerce",
    )

    working = working[
        valid_label_mask(
            working[
                DISEASE_COL
            ]
        )
        & np.isfinite(
            working[
                outcome_col
            ]
        )
    ].copy()

    counts = (
        working
        .groupby(
            DISEASE_COL,
            observed=True,
        )
        .size()
    )

    eligible_diseases = (
        counts[
            counts
            >= min_group_n
        ]
        .index
        .astype(str)
        .tolist()
    )

    return (
        working[
            working[
                DISEASE_COL
            ]
            .astype(str)
            .isin(
                eligible_diseases
            )
        ]
        .copy(),
        eligible_diseases,
    )


def fit_assay_adjusted_model(
    frame,
    outcome_col,
):
    """
    Assay-adjusted sensitivity analysis.

    Uses only donors with:
      - finite outcome
      - valid disease
      - uniquely defined assay

    Returns:
      model
      analysis frame

    Raises ValueError when disease and assay are not
    sufficiently represented for adjustment.
    """

    model_df = frame[
        [
            DONOR_COL,
            DISEASE_COL,
            ASSAY_COL,
            outcome_col,
        ]
    ].copy()

    model_df[
        outcome_col
    ] = pd.to_numeric(
        model_df[
            outcome_col
        ],
        errors="coerce",
    )

    keep = (
        np.isfinite(
            model_df[
                outcome_col
            ]
        )
        & valid_label_mask(
            model_df[
                DISEASE_COL
            ]
        )
        & valid_label_mask(
            model_df[
                ASSAY_COL
            ]
        )
    )

    model_df = (
        model_df.loc[
            keep
        ]
        .copy()
    )

    disease_counts = (
        model_df
        .groupby(
            DISEASE_COL,
            observed=True,
        )
        .size()
    )

    eligible_diseases = (
        disease_counts[
            disease_counts
            >= DISEASE_MIN_DONORS_PER_GROUP
        ]
        .index
        .astype(str)
        .tolist()
    )

    model_df = (
        model_df[
            model_df[
                DISEASE_COL
            ]
            .astype(str)
            .isin(
                eligible_diseases
            )
        ]
        .copy()
    )

    if (
        model_df[
            DISEASE_COL
        ]
        .nunique()
        < 2
    ):
        raise ValueError(
            "Fewer than two adequately "
            "represented disease groups."
        )

    if (
        model_df[
            ASSAY_COL
        ]
        .nunique()
        < 2
    ):
        raise ValueError(
            "Fewer than two assays."
        )

    contingency = pd.crosstab(
        model_df[
            DISEASE_COL
        ],
        model_df[
            ASSAY_COL
        ],
    )

    shared_assays = (
        contingency
        .gt(
            0
        )
        .sum(
            axis=0
        )
        >= 2
    )

    if not np.any(
        shared_assays
    ):
        raise ValueError(
            "Disease and assay are "
            "completely separated."
        )

    # Explicit categorical ordering makes the
    # reference level deterministic.
    #
    # Prefer Healthy Reference as the baseline
    # when present. Otherwise use Reference,
    # then alphabetical fallback.
    disease_levels = sorted(
        model_df[
            DISEASE_COL
        ]
        .astype(str)
        .unique()
        .tolist()
    )

    if (
        "Healthy Reference"
        in disease_levels
    ):
        disease_reference = (
            "Healthy Reference"
        )

    elif (
        "Reference"
        in disease_levels
    ):
        disease_reference = (
            "Reference"
        )

    else:
        disease_reference = (
            disease_levels[
                0
            ]
        )

    disease_levels = [
        disease_reference
    ] + [
        x
        for x
        in disease_levels
        if x
        != disease_reference
    ]

    assay_levels = sorted(
        model_df[
            ASSAY_COL
        ]
        .astype(str)
        .unique()
        .tolist()
    )

    model_df[
        DISEASE_COL
    ] = pd.Categorical(
        model_df[
            DISEASE_COL
        ].astype(str),
        categories=disease_levels,
    )

    model_df[
        ASSAY_COL
    ] = pd.Categorical(
        model_df[
            ASSAY_COL
        ].astype(str),
        categories=assay_levels,
    )

    formula = (
        f'Q("{outcome_col}") ~ '
        f'C(Q("{DISEASE_COL}")) + '
        f'C(Q("{ASSAY_COL}"))'
    )

    model = (
        smf.ols(
            formula=formula,
            data=model_df,
        )
        .fit(
            cov_type="HC3"
        )
    )

    return (
        model,
        model_df,
        disease_reference,
        assay_levels[
            0
        ],
    )


# %% DISEASE ASSOCIATION: PRIMARY OMNIBUS + PAIRWISE


disease_omnibus_records = []
disease_pairwise_records = []


for (
    compartment,
    trajectory,
    lineage,
), lineage_df in disease_df.groupby(
    [
        "compartment",
        "trajectory",
        "lineage",
    ],
    observed=True,
    sort=True,
):

    robustness = {
        "robustness_match_fraction":
            float(
                lineage_df[
                    "robustness_match_fraction"
                ]
                .iloc[
                    0
                ]
            ),

        "robustness_spearman_median":
            float(
                lineage_df[
                    "robustness_spearman_median"
                ]
                .iloc[
                    0
                ]
            ),

        "robustness_spearman_q05":
            float(
                lineage_df[
                    "robustness_spearman_q05"
                ]
                .iloc[
                    0
                ]
            ),
    }

    for (
        endpoint_name,
        endpoint_spec,
    ) in DISEASE_PRIMARY_ENDPOINTS.items():

        outcome_col = (
            endpoint_spec[
                "column"
            ]
        )

        min_group_n = int(
            endpoint_spec[
                "min_group_n"
            ]
        )

        analysis_df = (
            lineage_df
            .copy()
        )

        if endpoint_spec[
            "require_lineage_support"
        ]:
            lineage_weight = pd.to_numeric(
                analysis_df[
                    "lineage_weight_sum"
                ],
                errors="coerce",
            )

            analysis_df = (
                analysis_df[
                    np.isfinite(lineage_weight)
                    & (
                        lineage_weight
                        > DONOR_SUMMARY_MIN_WEIGHT
                    )
                ]
                .copy()
            )

        (
            analysis_df,
            eligible_diseases,
        ) = disease_group_values(
            frame=analysis_df,
            outcome_col=outcome_col,
            min_group_n=min_group_n,
        )

        group_counts = (
            analysis_df
            .groupby(
                DISEASE_COL,
                observed=True,
            )
            .size()
            .to_dict()
        )

        group_medians = (
            analysis_df
            .groupby(
                DISEASE_COL,
                observed=True,
            )[
                outcome_col
            ]
            .median()
            .to_dict()
        )

        group_means = (
            analysis_df
            .groupby(
                DISEASE_COL,
                observed=True,
            )[
                outcome_col
            ]
            .mean()
            .to_dict()
        )

        if (
            len(eligible_diseases)
            >= 2
        ):
            arrays = [
                pd.to_numeric(
                    analysis_df.loc[
                        analysis_df[
                            DISEASE_COL
                        ].astype(str)
                        == disease,
                        outcome_col,
                    ],
                    errors="coerce",
                ).to_numpy(
                    dtype=float
                )
                for disease
                in eligible_diseases
            ]

            try:
                kw = kruskal(
                    *arrays,
                    nan_policy="omit",
                )

                kw_statistic = float(
                    kw.statistic
                )

                kw_pvalue = float(
                    kw.pvalue
                )

                omnibus_status = (
                    "tested"
                )

            except ValueError as exc:
                kw_statistic = np.nan
                kw_pvalue = np.nan
                omnibus_status = (
                    "failed:"
                    f"{type(exc).__name__}"
                )

        else:
            kw_statistic = np.nan
            kw_pvalue = np.nan
            omnibus_status = (
                "insufficient_groups"
            )

        disease_omnibus_records.append(
            {
                "compartment":
                    compartment,

                "trajectory":
                    trajectory,

                "lineage":
                    lineage,

                "endpoint":
                    endpoint_name,

                "outcome":
                    outcome_col,

                "n_donors_analyzed":
                    int(
                        len(analysis_df)
                    ),

                "n_disease_groups_tested":
                    int(
                        len(eligible_diseases)
                    ),

                "disease_groups_tested":
                    "|".join(
                        eligible_diseases
                    ),

                "group_counts":
                    "|".join(
                        [
                            f"{x}:"
                            f"{int(group_counts[x])}"
                            for x
                            in eligible_diseases
                        ]
                    ),

                "group_medians":
                    "|".join(
                        [
                            f"{x}:"
                            f"{float(group_medians[x]):.6g}"
                            for x
                            in eligible_diseases
                        ]
                    ),

                "group_means":
                    "|".join(
                        [
                            f"{x}:"
                            f"{float(group_means[x]):.6g}"
                            for x
                            in eligible_diseases
                        ]
                    ),

                "kruskal_statistic":
                    kw_statistic,

                "pvalue":
                    kw_pvalue,

                "test_status":
                    omnibus_status,

                **robustness,
            }
        )

        # ----------------------------------
        # Pairwise disease comparisons
        # ----------------------------------

        if (
            len(eligible_diseases)
            >= 2
        ):
            for (
                disease_1,
                disease_2,
            ) in combinations(
                eligible_diseases,
                2,
            ):

                x = pd.to_numeric(
                    analysis_df.loc[
                        analysis_df[
                            DISEASE_COL
                        ].astype(str)
                        == disease_1,
                        outcome_col,
                    ],
                    errors="coerce",
                ).to_numpy(
                    dtype=float
                )

                y = pd.to_numeric(
                    analysis_df.loc[
                        analysis_df[
                            DISEASE_COL
                        ].astype(str)
                        == disease_2,
                        outcome_col,
                    ],
                    errors="coerce",
                ).to_numpy(
                    dtype=float
                )

                x = x[
                    np.isfinite(x)
                ]

                y = y[
                    np.isfinite(y)
                ]

                if (
                    len(x)
                    < min_group_n
                    or len(
                        y
                    )
                    < min_group_n
                ):
                    continue

                mw = mannwhitneyu(
                    x,
                    y,
                    alternative="two-sided",
                    method="auto",
                )

                rank_biserial = (
                    rank_biserial_from_u(
                        mw.statistic,
                        len(
                            x
                        ),
                        len(
                            y
                        ),
                    )
                )

                disease_pairwise_records.append(
                    {
                        "compartment":
                            compartment,

                        "trajectory":
                            trajectory,

                        "lineage":
                            lineage,

                        "endpoint":
                            endpoint_name,

                        "outcome":
                            outcome_col,

                        "disease_1":
                            disease_1,

                        "disease_2":
                            disease_2,

                        "n_disease_1":
                            int(
                                len(x)
                            ),

                        "n_disease_2":
                            int(
                                len(y)
                            ),

                        "median_disease_1":
                            float(
                                np.median(x)
                            ),

                        "median_disease_2":
                            float(
                                np.median(y)
                            ),

                        "median_difference":
                            float(
                                np.median(x)
                                - np.median(
                                    y
                                )
                            ),

                        "mean_disease_1":
                            float(
                                np.mean(x)
                            ),

                        "mean_disease_2":
                            float(
                                np.mean(y)
                            ),

                        "mannwhitney_u":
                            float(
                                mw.statistic
                            ),

                        "rank_biserial":
                            rank_biserial,

                        "pvalue":
                            float(
                                mw.pvalue
                            ),

                        **robustness,
                    }
                )


disease_omnibus = pd.DataFrame(
    disease_omnibus_records
)

disease_pairwise = pd.DataFrame(
    disease_pairwise_records
)


# %% DISEASE ASSOCIATION: MULTIPLE TESTING


disease_omnibus[
    "pvalue_bh"
] = np.nan


for endpoint in (
    disease_omnibus[
        "endpoint"
    ]
    .dropna()
    .unique()
):

    mask = (
        disease_omnibus[
            "endpoint"
        ]
        == endpoint
    )

    disease_omnibus.loc[
        mask,
        "pvalue_bh",
    ] = bh_adjust(
        disease_omnibus.loc[
            mask,
            "pvalue",
        ]
    )


if not disease_pairwise.empty:

    disease_pairwise[
        "pvalue_bh"
    ] = np.nan

    for endpoint in (
        disease_pairwise[
            "endpoint"
        ]
        .dropna()
        .unique()
    ):

        mask = (
            disease_pairwise[
                "endpoint"
            ]
            == endpoint
        )

        disease_pairwise.loc[
            mask,
            "pvalue_bh",
        ] = bh_adjust(
            disease_pairwise.loc[
                mask,
                "pvalue",
            ]
        )


# %% DISEASE ASSOCIATION: ASSAY-ADJUSTED SENSITIVITY


assay_adjusted_records = []
assay_adjusted_failures = []


for (
    compartment,
    trajectory,
    lineage,
), lineage_df in disease_df.groupby(
    [
        "compartment",
        "trajectory",
        "lineage",
    ],
    observed=True,
    sort=True,
):

    for (
        endpoint_name,
        endpoint_spec,
    ) in DISEASE_PRIMARY_ENDPOINTS.items():

        outcome_col = (
            endpoint_spec[
                "column"
            ]
        )

        model_input = (
            lineage_df
            .copy()
        )

        if endpoint_spec[
            "require_lineage_support"
        ]:

            lineage_weight = pd.to_numeric(
                model_input[
                    "lineage_weight_sum"
                ],
                errors="coerce",
            )

            model_input = (
                model_input[
                    np.isfinite(lineage_weight)
                    & (
                        lineage_weight
                        > DONOR_SUMMARY_MIN_WEIGHT
                    )
                ]
                .copy()
            )

        try:
            (
                model,
                model_df,
                disease_reference,
                assay_reference,
            ) = fit_assay_adjusted_model(
                frame=model_input,
                outcome_col=outcome_col,
            )

        except Exception as exc:

            assay_adjusted_failures.append(
                {
                    "compartment":
                        compartment,

                    "trajectory":
                        trajectory,

                    "lineage":
                        lineage,

                    "endpoint":
                        endpoint_name,

                    "failure":
                        repr(
                            exc
                        ),
                }
            )

            continue

        params = model.params
        conf_int = model.conf_int()
        pvalues = model.pvalues
        bse = model.bse

        # Disease coefficients only.
        for term in params.index:

            term_string = str(
                term
            )

            disease_prefix = (
                f'C(Q("{DISEASE_COL}"))'
                "[T."
            )

            if not term_string.startswith(
                disease_prefix
            ):
                continue

            disease_level = (
                term_string[
                    len(
                        disease_prefix
                    ):
                ]
                .rstrip(
                    "]"
                )
            )

            disease_n = int(
                (
                    model_df[
                        DISEASE_COL
                    ]
                    .astype(str)
                    == disease_level
                )
                .sum()
            )

            reference_n = int(
                (
                    model_df[
                        DISEASE_COL
                    ]
                    .astype(str)
                    == disease_reference
                )
                .sum()
            )

            assay_adjusted_records.append(
                {
                    "compartment":
                        compartment,

                    "trajectory":
                        trajectory,

                    "lineage":
                        lineage,

                    "endpoint":
                        endpoint_name,

                    "outcome":
                        outcome_col,

                    "disease":
                        disease_level,

                    "disease_reference":
                        disease_reference,

                    "assay_reference":
                        assay_reference,

                    "n_model_donors":
                        int(
                            len(model_df)
                        ),

                    "n_disease":
                        disease_n,

                    "n_reference":
                        reference_n,

                    "coefficient":
                        float(
                            params[
                                term
                            ]
                        ),

                    "standard_error_hc3":
                        float(
                            bse[
                                term
                            ]
                        ),

                    "ci95_low":
                        float(
                            conf_int.loc[
                                term,
                                0,
                            ]
                        ),

                    "ci95_high":
                        float(
                            conf_int.loc[
                                term,
                                1,
                            ]
                        ),

                    "pvalue":
                        float(
                            pvalues[
                                term
                            ]
                        ),

                    "robustness_match_fraction":
                        float(
                            lineage_df[
                                "robustness_match_fraction"
                            ]
                            .iloc[
                                0
                            ]
                        ),

                    "robustness_spearman_median":
                        float(
                            lineage_df[
                                "robustness_spearman_median"
                            ]
                            .iloc[
                                0
                            ]
                        ),

                    "robustness_spearman_q05":
                        float(
                            lineage_df[
                                "robustness_spearman_q05"
                            ]
                            .iloc[
                                0
                            ]
                        ),
                }
            )


assay_adjusted_results = (
    pd.DataFrame(assay_adjusted_records)
)

assay_adjusted_failures = (
    pd.DataFrame(assay_adjusted_failures)
)


if not assay_adjusted_results.empty:

    assay_adjusted_results[
        "pvalue_bh"
    ] = np.nan

    for endpoint in (
        assay_adjusted_results[
            "endpoint"
        ]
        .dropna()
        .unique()
    ):

        mask = (
            assay_adjusted_results[
                "endpoint"
            ]
            == endpoint
        )

        assay_adjusted_results.loc[
            mask,
            "pvalue_bh",
        ] = bh_adjust(
            assay_adjusted_results.loc[
                mask,
                "pvalue",
            ]
        )


# %% DISEASE ASSOCIATION: PRIMARY ↔ ASSAY SENSITIVITY CONCORDANCE
#
# Match each assay-adjusted disease-vs-reference coefficient
# to the equivalent unadjusted Mann-Whitney comparison.
#
# Rank-biserial sign is oriented disease - reference.
# OLS coefficient is also disease - reference.


concordance_records = []


if (
    not assay_adjusted_results.empty
    and not disease_pairwise.empty
):

    for _, row in (
        assay_adjusted_results
        .iterrows()
    ):

        pairwise_candidates = (
            disease_pairwise[
                (
                    disease_pairwise[
                        "compartment"
                    ]
                    == row[
                        "compartment"
                    ]
                )
                & (
                    disease_pairwise[
                        "trajectory"
                    ]
                    == row[
                        "trajectory"
                    ]
                )
                & (
                    disease_pairwise[
                        "lineage"
                    ]
                    == row[
                        "lineage"
                    ]
                )
                & (
                    disease_pairwise[
                        "endpoint"
                    ]
                    == row[
                        "endpoint"
                    ]
                )
            ]
        )

        disease = str(
            row[
                "disease"
            ]
        )

        reference = str(
            row[
                "disease_reference"
            ]
        )

        match_forward = (
            pairwise_candidates[
                (
                    pairwise_candidates[
                        "disease_1"
                    ]
                    == disease
                )
                & (
                    pairwise_candidates[
                        "disease_2"
                    ]
                    == reference
                )
            ]
        )

        match_reverse = (
            pairwise_candidates[
                (
                    pairwise_candidates[
                        "disease_1"
                    ]
                    == reference
                )
                & (
                    pairwise_candidates[
                        "disease_2"
                    ]
                    == disease
                )
            ]
        )

        if (
            len(match_forward)
            == 1
        ):
            pw = (
                match_forward
                .iloc[
                    0
                ]
            )

            unadjusted_effect = float(
                pw[
                    "rank_biserial"
                ]
            )

            unadjusted_median_difference = float(
                pw[
                    "median_difference"
                ]
            )

            unadjusted_pvalue = float(
                pw[
                    "pvalue"
                ]
            )

            unadjusted_pvalue_bh = float(
                pw[
                    "pvalue_bh"
                ]
            )

        elif (
            len(match_reverse)
            == 1
        ):
            pw = (
                match_reverse
                .iloc[
                    0
                ]
            )

            unadjusted_effect = -float(
                pw[
                    "rank_biserial"
                ]
            )

            unadjusted_median_difference = -float(
                pw[
                    "median_difference"
                ]
            )

            unadjusted_pvalue = float(
                pw[
                    "pvalue"
                ]
            )

            unadjusted_pvalue_bh = float(
                pw[
                    "pvalue_bh"
                ]
            )

        else:
            continue

        adjusted_effect = float(
            row[
                "coefficient"
            ]
        )

        sign_concordant = bool(
            (
                unadjusted_effect
                == 0
            )
            or (
                adjusted_effect
                == 0
            )
            or (
                np.sign(unadjusted_effect)
                == np.sign(
                    adjusted_effect
                )
            )
        )

        concordance_records.append(
            {
                "compartment":
                    row[
                        "compartment"
                    ],

                "trajectory":
                    row[
                        "trajectory"
                    ],

                "lineage":
                    row[
                        "lineage"
                    ],

                "endpoint":
                    row[
                        "endpoint"
                    ],

                "disease":
                    disease,

                "reference":
                    reference,

                "unadjusted_rank_biserial":
                    unadjusted_effect,

                "unadjusted_median_difference":
                    unadjusted_median_difference,

                "unadjusted_pvalue":
                    unadjusted_pvalue,

                "unadjusted_pvalue_bh":
                    unadjusted_pvalue_bh,

                "assay_adjusted_coefficient":
                    adjusted_effect,

                "assay_adjusted_pvalue":
                    float(
                        row[
                            "pvalue"
                        ]
                    ),

                "assay_adjusted_pvalue_bh":
                    float(
                        row[
                            "pvalue_bh"
                        ]
                    ),

                "effect_direction_concordant":
                    sign_concordant,

                "robustness_match_fraction":
                    float(
                        row[
                            "robustness_match_fraction"
                        ]
                    ),

                "robustness_spearman_median":
                    float(
                        row[
                            "robustness_spearman_median"
                        ]
                    ),

                "robustness_spearman_q05":
                    float(
                        row[
                            "robustness_spearman_q05"
                        ]
                    ),
            }
        )


disease_assay_concordance = (
    pd.DataFrame(concordance_records)
)


# %% DISEASE ASSOCIATION: DESCRIPTIVE GROUP SUMMARIES


disease_group_summary_records = []


for (
    compartment,
    trajectory,
    lineage,
    disease,
), group in disease_df.groupby(
    [
        "compartment",
        "trajectory",
        "lineage",
        DISEASE_COL,
    ],
    observed=True,
    sort=True,
):

    occupancy = pd.to_numeric(
        group[
            "fraction_lineage_supported"
        ],
        errors="coerce",
    )

    position = pd.to_numeric(
        group[
            "weighted_mean_pseudotime"
        ],
        errors="coerce",
    )

    weight = pd.to_numeric(
        group[
            "lineage_weight_sum"
        ],
        errors="coerce",
    )

    position_keep = (
        np.isfinite(position)
        & np.isfinite(
            weight
        )
        & (
            weight
            > DONOR_SUMMARY_MIN_WEIGHT
        )
    )

    occupancy_finite = occupancy[
        np.isfinite(occupancy)
    ].to_numpy(
        dtype=float
    )

    position_finite = position[
        position_keep
    ].to_numpy(
        dtype=float
    )

    disease_group_summary_records.append(
        {
            "compartment":
                compartment,

            "trajectory":
                trajectory,

            "lineage":
                lineage,

            DISEASE_COL:
                disease,

            "n_donors_total":
                int(
                    len(group)
                ),

            "occupancy_n":
                int(
                    len(occupancy_finite)
                ),

            "occupancy_mean":
                (
                    float(
                        np.mean(occupancy_finite)
                    )
                    if len(
                        occupancy_finite
                    )
                    else np.nan
                ),

            "occupancy_median":
                (
                    float(
                        np.median(occupancy_finite)
                    )
                    if len(
                        occupancy_finite
                    )
                    else np.nan
                ),

            "occupancy_q25":
                (
                    float(
                        np.quantile(
                            occupancy_finite,
                            0.25,
                        )
                    )
                    if len(
                        occupancy_finite
                    )
                    else np.nan
                ),

            "occupancy_q75":
                (
                    float(
                        np.quantile(
                            occupancy_finite,
                            0.75,
                        )
                    )
                    if len(
                        occupancy_finite
                    )
                    else np.nan
                ),

            "position_n":
                int(
                    len(position_finite)
                ),

            "position_mean":
                (
                    float(
                        np.mean(position_finite)
                    )
                    if len(
                        position_finite
                    )
                    else np.nan
                ),

            "position_median":
                (
                    float(
                        np.median(position_finite)
                    )
                    if len(
                        position_finite
                    )
                    else np.nan
                ),

            "position_q25":
                (
                    float(
                        np.quantile(
                            position_finite,
                            0.25,
                        )
                    )
                    if len(
                        position_finite
                    )
                    else np.nan
                ),

            "position_q75":
                (
                    float(
                        np.quantile(
                            position_finite,
                            0.75,
                        )
                    )
                    if len(
                        position_finite
                    )
                    else np.nan
                ),

            "robustness_match_fraction":
                float(
                    group[
                        "robustness_match_fraction"
                    ]
                    .iloc[
                        0
                    ]
                ),

            "robustness_spearman_median":
                float(
                    group[
                        "robustness_spearman_median"
                    ]
                    .iloc[
                        0
                    ]
                ),
        }
    )


disease_group_summary = (
    pd.DataFrame(disease_group_summary_records)
)


# %% DISEASE ASSOCIATION: QC


# Exactly one omnibus record per lineage × endpoint.
expected_omnibus = (
    disease_df[
        [
            "compartment",
            "trajectory",
            "lineage",
        ]
    ]
    .drop_duplicates()
    .shape[
        0
    ]
    * len(
        DISEASE_PRIMARY_ENDPOINTS
    )
)


if (
    len(disease_omnibus)
    != expected_omnibus
):
    raise RuntimeError(
        "Unexpected number of disease "
        "omnibus tests: "
        f"{len(disease_omnibus)} "
        f"!= {expected_omnibus}"
    )


# No pseudoreplication: donor must remain unique
# within each lineage.
if (
    disease_df[
        [
            "compartment",
            "trajectory",
            "lineage",
            DONOR_COL,
        ]
    ]
    .duplicated()
    .any()
):
    raise RuntimeError(
        "Duplicate donor records detected "
        "during disease inference."
    )


# Position analyses must never include donors
# without lineage support.
position_rows = (
    disease_omnibus[
        disease_omnibus[
            "endpoint"
        ]
        == "position"
    ]
)

if (
    position_rows[
        "n_donors_analyzed"
    ]
    .isna()
    .any()
):
    raise RuntimeError(
        "Invalid position analysis counts."
    )


# --------------------------------------
# Sort
# --------------------------------------

disease_omnibus = (
    disease_omnibus
    .sort_values(
        [
            "endpoint",
            "pvalue_bh",
            "compartment",
            "trajectory",
            "lineage",
        ],
        na_position="last",
    )
    .reset_index(
        drop=True
    )
)


if not disease_pairwise.empty:

    disease_pairwise = (
        disease_pairwise
        .sort_values(
            [
                "endpoint",
                "pvalue_bh",
                "compartment",
                "trajectory",
                "lineage",
                "disease_1",
                "disease_2",
            ],
            na_position="last",
        )
        .reset_index(
            drop=True
        )
    )


if not assay_adjusted_results.empty:

    assay_adjusted_results = (
        assay_adjusted_results
        .sort_values(
            [
                "endpoint",
                "pvalue_bh",
                "compartment",
                "trajectory",
                "lineage",
                "disease",
            ],
            na_position="last",
        )
        .reset_index(
            drop=True
        )
    )


if not disease_assay_concordance.empty:

    disease_assay_concordance = (
        disease_assay_concordance
        .sort_values(
            [
                "endpoint",
                "assay_adjusted_pvalue_bh",
                "compartment",
                "trajectory",
                "lineage",
                "disease",
            ],
            na_position="last",
        )
        .reset_index(
            drop=True
        )
    )


# %% DISEASE ASSOCIATION: SAVE


disease_omnibus.to_csv(
    TRAJECTORY_DIR
    / "trajectory_disease_omnibus.csv",
    index=False,
)


disease_pairwise.to_csv(
    TRAJECTORY_DIR
    / "trajectory_disease_pairwise.csv",
    index=False,
)


disease_group_summary.to_csv(
    TRAJECTORY_DIR
    / "trajectory_disease_group_summary.csv",
    index=False,
)


assay_adjusted_results.to_csv(
    TRAJECTORY_DIR
    / "trajectory_disease_assay_adjusted.csv",
    index=False,
)


assay_adjusted_failures.to_csv(
    TRAJECTORY_DIR
    / "trajectory_disease_assay_adjusted_failures.csv",
    index=False,
)


disease_assay_concordance.to_csv(
    TRAJECTORY_DIR
    / "trajectory_disease_assay_concordance.csv",
    index=False,
)


# %% DISEASE ASSOCIATION: CHECKPOINT


print(
    "\n"
    + "=" * 100
)

print(
    "04c DONOR-LEVEL DISEASE "
    "ASSOCIATION COMPLETE"
)

print(
    "=" * 100
)


print(
    "\nOMNIBUS DISEASE ASSOCIATION"
)

print(
    disease_omnibus[
        [
            "compartment",
            "trajectory",
            "lineage",
            "endpoint",
            "n_donors_analyzed",
            "n_disease_groups_tested",
            "kruskal_statistic",
            "pvalue",
            "pvalue_bh",
            "robustness_match_fraction",
            "robustness_spearman_median",
        ]
    ]
    .to_string(
        index=False
    )
)


print(
    "\nPAIRWISE DISEASE ASSOCIATIONS "
    "(BH < 0.05)"
)

significant_pairwise = (
    disease_pairwise[
        disease_pairwise[
            "pvalue_bh"
        ]
        < 0.05
    ]
    if not disease_pairwise.empty
    else disease_pairwise
)


if significant_pairwise.empty:
    print(
        "None."
    )

else:
    print(
        significant_pairwise[
            [
                "compartment",
                "trajectory",
                "lineage",
                "endpoint",
                "disease_1",
                "disease_2",
                "n_disease_1",
                "n_disease_2",
                "median_difference",
                "rank_biserial",
                "pvalue",
                "pvalue_bh",
                "robustness_match_fraction",
                "robustness_spearman_median",
            ]
        ]
        .to_string(
            index=False
        )
    )


print(
    "\nASSAY-ADJUSTED DISEASE EFFECTS "
    "(BH < 0.05)"
)

if assay_adjusted_results.empty:
    print(
        "No assay-adjusted models succeeded."
    )

else:
    significant_adjusted = (
        assay_adjusted_results[
            assay_adjusted_results[
                "pvalue_bh"
            ]
            < 0.05
        ]
    )

    if significant_adjusted.empty:
        print(
            "None."
        )

    else:
        print(
            significant_adjusted[
                [
                    "compartment",
                    "trajectory",
                    "lineage",
                    "endpoint",
                    "disease",
                    "disease_reference",
                    "n_model_donors",
                    "coefficient",
                    "ci95_low",
                    "ci95_high",
                    "pvalue",
                    "pvalue_bh",
                    "robustness_match_fraction",
                    "robustness_spearman_median",
                ]
            ]
            .to_string(
                index=False
            )
        )


print(
    "\nPRIMARY ↔ ASSAY-ADJUSTED "
    "DIRECTION CONCORDANCE"
)

if disease_assay_concordance.empty:
    print(
        "No concordance records."
    )

else:
    concordance_summary = (
        disease_assay_concordance
        .groupby(
            "endpoint",
            observed=True,
        )
        .agg(
            n_comparisons=(
                "effect_direction_concordant",
                "size",
            ),

            n_direction_concordant=(
                "effect_direction_concordant",
                "sum",
            ),

            fraction_direction_concordant=(
                "effect_direction_concordant",
                "mean",
            ),
        )
        .reset_index()
    )

    print(
        concordance_summary
        .to_string(
            index=False
        )
    )


print(
    "\nASSAY-ADJUSTED MODEL FAILURES"
)

if assay_adjusted_failures.empty:
    print(
        "None."
    )

else:
    print(
        assay_adjusted_failures
        .to_string(
            index=False
        )
    )


print(
    "\nSaved:"
)

for filename in [
    "trajectory_disease_omnibus.csv",
    "trajectory_disease_pairwise.csv",
    "trajectory_disease_group_summary.csv",
    "trajectory_disease_assay_adjusted.csv",
    "trajectory_disease_assay_adjusted_failures.csv",
    "trajectory_disease_assay_concordance.csv",
]:
    print(
        " ",
        TRAJECTORY_DIR
        / filename,
    )

