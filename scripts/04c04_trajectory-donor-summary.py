# %% 04C04 TRAJECTORY-DONOR-SUMMARY
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


# %% DONOR-LEVEL TRAJECTORY SUMMARIES
#
# Frozen-input stage:
#   - no graph rebuilding
#   - no reclustering
#   - no Slingshot refitting
#   - no root changes
#
# Inferential unit:
#   donor × compartment × trajectory × lineage
#
# Pseudotime:
#   uses frozen primary Slingshot pseudotime scaled to [0, 1]
#
# Lineage support:
#   uses frozen Slingshot curve weights
#
# Metadata:
#   donor-level disease / assay / state are retained only when
#   internally consistent across that donor's metacells in the
#   trajectory. Mixed labels are explicitly flagged rather than
#   silently collapsed.


DONOR_SUMMARY_METADATA_COLS = [
    DISEASE_COL,
    ASSAY_COL,
    STATE_COL,
]

DONOR_SUMMARY_MIN_WEIGHT = 1e-8


def weighted_mean_finite(
    values,
    weights,
):
    values = np.asarray(
        values,
        dtype=float,
    )

    weights = np.asarray(
        weights,
        dtype=float,
    )

    keep = (
        np.isfinite(values)
        & np.isfinite(weights)
        & (weights > 0)
    )

    if not np.any(
        keep
    ):
        return np.nan

    values = values[
        keep
    ]

    weights = weights[
        keep
    ]

    weight_sum = float(
        weights.sum()
    )

    if (
        weight_sum
        <= 0
    ):
        return np.nan

    return float(
        np.sum(
            values
            * weights
        )
        / weight_sum
    )


def weighted_sd_finite(
    values,
    weights,
):
    values = np.asarray(
        values,
        dtype=float,
    )

    weights = np.asarray(
        weights,
        dtype=float,
    )

    keep = (
        np.isfinite(values)
        & np.isfinite(weights)
        & (weights > 0)
    )

    if (
        keep.sum()
        < 2
    ):
        return np.nan

    values = values[
        keep
    ]

    weights = weights[
        keep
    ]

    weight_sum = float(
        weights.sum()
    )

    if (
        weight_sum
        <= 0
    ):
        return np.nan

    mean = (
        np.sum(
            values
            * weights
        )
        / weight_sum
    )

    variance = (
        np.sum(
            weights
            * (
                values
                - mean
            ) ** 2
        )
        / weight_sum
    )

    return float(
        np.sqrt(
            max(
                variance,
                0.0,
            )
        )
    )


def effective_sample_size(
    weights,
):
    """
    Kish-style effective sample size for lineage weights.

    This is descriptive only; it is not treated as the number of
    independent biological replicates.
    """

    weights = np.asarray(
        weights,
        dtype=float,
    )

    weights = weights[
        np.isfinite(weights)
        & (
            weights
            > 0
        )
    ]

    if (
        weights.size
        == 0
    ):
        return 0.0

    numerator = (
        weights.sum()
        ** 2
    )

    denominator = np.sum(
        weights ** 2
    )

    if (
        denominator
        <= 0
    ):
        return 0.0

    return float(
        numerator
        / denominator
    )


def consistent_donor_metadata(
    values,
):
    """
    Return a donor-level label only if all non-missing metacell
    values agree.

    Returns:
        value
        n_unique
        is_consistent
    """

    series = pd.Series(
        values
    )

    series = (
        series
        .dropna()
        .astype(str)
    )

    series = series[
        ~series.isin(
            [
                "",
                "nan",
                "None",
                "<NA>",
            ]
        )
    ]

    unique_values = (
        pd.unique(series)
        .tolist()
    )

    n_unique = len(
        unique_values
    )

    if (
        n_unique
        == 0
    ):
        return (
            pd.NA,
            0,
            True,
        )

    if (
        n_unique
        == 1
    ):
        return (
            unique_values[
                0
            ],
            1,
            True,
        )

    return (
        pd.NA,
        n_unique,
        False,
    )


def summarize_donor_lineage(
    donor_df,
    pt_col,
    weight_col,
):
    """
    Summarize one donor on one frozen primary lineage.
    """

    pt = pd.to_numeric(
        donor_df[
            pt_col
        ],
        errors="coerce",
    ).to_numpy(
        dtype=float
    )

    weights = pd.to_numeric(
        donor_df[
            weight_col
        ],
        errors="coerce",
    ).to_numpy(
        dtype=float
    )

    finite_pt = np.isfinite(
        pt
    )

    finite_weight = np.isfinite(
        weights
    )

    positive_weight = (
        finite_weight
        & (
            weights
            > DONOR_SUMMARY_MIN_WEIGHT
        )
    )

    supported = (
        finite_pt
        & positive_weight
    )

    n_metacells = int(
        len(donor_df)
    )

    n_finite_pt = int(
        finite_pt.sum()
    )

    n_supported = int(
        supported.sum()
    )

    if (
        n_finite_pt
        > 0
    ):
        finite_values = pt[
            finite_pt
        ]

        median_pt = float(
            np.median(finite_values)
        )

        q25_pt = float(
            np.quantile(
                finite_values,
                0.25,
            )
        )

        q75_pt = float(
            np.quantile(
                finite_values,
                0.75,
            )
        )

        min_pt = float(
            np.min(finite_values)
        )

        max_pt = float(
            np.max(finite_values)
        )

    else:
        median_pt = np.nan
        q25_pt = np.nan
        q75_pt = np.nan
        min_pt = np.nan
        max_pt = np.nan

    finite_weights = weights[
        finite_weight
    ]

    if (
        finite_weights.size
        > 0
    ):
        weight_sum = float(
            np.sum(finite_weights)
        )

        weight_mean = float(
            np.mean(finite_weights)
        )

        weight_median = float(
            np.median(finite_weights)
        )

        weight_max = float(
            np.max(finite_weights)
        )

    else:
        weight_sum = np.nan
        weight_mean = np.nan
        weight_median = np.nan
        weight_max = np.nan

    weighted_mean_pt = (
        weighted_mean_finite(
            pt,
            weights,
        )
    )

    weighted_sd_pt = (
        weighted_sd_finite(
            pt,
            weights,
        )
    )

    lineage_ess = (
        effective_sample_size(
            weights[
                finite_pt
            ]
        )
    )

    return {
        "n_metacells":
            n_metacells,

        "n_finite_pseudotime":
            n_finite_pt,

        "fraction_finite_pseudotime":
            (
                n_finite_pt
                / n_metacells
                if n_metacells
                else np.nan
            ),

        "n_lineage_supported_metacells":
            n_supported,

        "fraction_lineage_supported":
            (
                n_supported
                / n_metacells
                if n_metacells
                else np.nan
            ),

        "lineage_weight_sum":
            weight_sum,

        "lineage_weight_mean":
            weight_mean,

        "lineage_weight_median":
            weight_median,

        "lineage_weight_max":
            weight_max,

        "lineage_effective_metacells":
            lineage_ess,

        "weighted_mean_pseudotime":
            weighted_mean_pt,

        "weighted_sd_pseudotime":
            weighted_sd_pt,

        "median_pseudotime":
            median_pt,

        "pseudotime_q25":
            q25_pt,

        "pseudotime_q75":
            q75_pt,

        "pseudotime_iqr":
            (
                q75_pt
                - q25_pt
                if (
                    np.isfinite(q25_pt)
                    and np.isfinite(
                        q75_pt
                    )
                )
                else np.nan
            ),

        "min_pseudotime":
            min_pt,

        "max_pseudotime":
            max_pt,
    }


# %% DONOR-LEVEL SUMMARIES: BUILD

donor_trajectory_records = []
donor_metadata_qc_records = []


for (
    compartment,
    specs,
) in TRAJECTORY_SPECS.items():

    print(
        "\n"
        + "=" * 100
    )

    print(
        f"{compartment} "
        "DONOR-LEVEL TRAJECTORY SUMMARIES"
    )

    print(
        "=" * 100
    )

    primary_file = (
        TRAJECTORY_DIR
        / (
            f"{compartment}_"
            "trajectory_primary.h5ad"
        )
    )

    if (
        not primary_file.exists()
    ):
        raise FileNotFoundError(
            primary_file
        )

    adata = sc.read_h5ad(
        primary_file
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

        trajectory_safe = (
            trajectory_name
            .replace(
                " ",
                "_",
            )
            .replace(
                "/",
                "_",
            )
        )

        pt_cols = (
            primary_pt_columns(
                sub,
                trajectory_name,
            )
        )

        if (
            len(pt_cols)
            == 0
        ):
            raise ValueError(
                f"{compartment}/"
                f"{trajectory_name}: "
                "no frozen primary "
                "pseudotime columns found."
            )

        # --------------------------------------
        # Resolve matching weight columns
        # --------------------------------------

        lineage_specs = []

        for pt_col in pt_cols:

            lineage = (
                extract_primary_lineage_name(
                    pt_col,
                    trajectory_name,
                )
            )

            weight_col = (
                f"weight_"
                f"{trajectory_safe}_"
                f"{lineage}"
            )

            if (
                weight_col
                not in sub.obs.columns
            ):
                raise ValueError(
                    f"{compartment}/"
                    f"{trajectory_name}/"
                    f"{lineage}: "
                    f"missing frozen weight "
                    f"column {weight_col!r}."
                )

            scaled_pt_col = (
                f"pt_"
                f"{trajectory_safe}_"
                f"{lineage}_scaled"
            )

            # primary_pt_columns() may already
            # return scaled columns depending on
            # the frozen helper implementation.
            #
            # For donor inference we explicitly
            # prefer the frozen [0,1] version.
            if (
                scaled_pt_col
                in sub.obs.columns
            ):
                donor_pt_col = scaled_pt_col

            elif (
                pt_col.endswith(
                    "_scaled"
                )
            ):
                donor_pt_col = pt_col

            else:
                # Fallback should rarely be needed,
                # but preserves compatibility with
                # the frozen primary helper.
                raw = pd.to_numeric(
                    sub.obs[
                        pt_col
                    ],
                    errors="coerce",
                ).to_numpy(
                    dtype=float
                )

                finite = np.isfinite(
                    raw
                )

                scaled = np.full(
                    len(
                        raw
                    ),
                    np.nan,
                    dtype=float,
                )

                if np.any(
                    finite
                ):
                    lo = float(
                        np.min(
                            raw[
                                finite
                            ]
                        )
                    )

                    hi = float(
                        np.max(
                            raw[
                                finite
                            ]
                        )
                    )

                    if (
                        hi
                        > lo
                    ):
                        scaled[
                            finite
                        ] = (
                            raw[
                                finite
                            ]
                            - lo
                        ) / (
                            hi
                            - lo
                        )

                    else:
                        scaled[
                            finite
                        ] = 0.0

                donor_pt_col = (
                    f"__donor_scaled_"
                    f"{trajectory_safe}_"
                    f"{lineage}"
                )

                sub.obs[
                    donor_pt_col
                ] = scaled

            lineage_specs.append(
                {
                    "lineage":
                        lineage,
                    "pt_col":
                        donor_pt_col,
                    "weight_col":
                        weight_col,
                }
            )

        # --------------------------------------
        # Metadata availability
        # --------------------------------------

        available_metadata = [
            col
            for col
            in DONOR_SUMMARY_METADATA_COLS
            if col
            in sub.obs.columns
        ]

        # --------------------------------------
        # Donor loop
        # --------------------------------------

        grouped = sub.obs.groupby(
            DONOR_COL,
            observed=True,
            sort=True,
        )

        n_donors = int(
            sub.obs[
                DONOR_COL
            ]
            .nunique()
        )

        for (
            donor_id,
            donor_df,
        ) in grouped:

            donor_id = str(
                donor_id
            )

            metadata_values = {}

            metadata_consistent_all = True

            metadata_qc = {
                "compartment":
                    compartment,
                "trajectory":
                    trajectory_name,
                DONOR_COL:
                    donor_id,
                "n_metacells":
                    int(
                        len(donor_df)
                    ),
            }

            for metadata_col in (
                available_metadata
            ):

                (
                    value,
                    n_unique,
                    is_consistent,
                ) = (
                    consistent_donor_metadata(
                        donor_df[
                            metadata_col
                        ]
                    )
                )

                metadata_values[
                    metadata_col
                ] = value

                metadata_qc[
                    f"{metadata_col}_n_unique"
                ] = int(
                    n_unique
                )

                metadata_qc[
                    f"{metadata_col}_consistent"
                ] = bool(
                    is_consistent
                )

                if (
                    not is_consistent
                ):
                    metadata_consistent_all = False

            metadata_qc[
                "all_metadata_consistent"
            ] = (
                metadata_consistent_all
            )

            donor_metadata_qc_records.append(metadata_qc)

            for lineage_spec in (
                lineage_specs
            ):

                lineage = (
                    lineage_spec[
                        "lineage"
                    ]
                )

                stats = (
                    summarize_donor_lineage(
                        donor_df=(
                            donor_df
                        ),
                        pt_col=(
                            lineage_spec[
                                "pt_col"
                            ]
                        ),
                        weight_col=(
                            lineage_spec[
                                "weight_col"
                            ]
                        ),
                    )
                )

                record = {
                    "compartment":
                        compartment,

                    "trajectory":
                        trajectory_name,

                    "lineage":
                        lineage,

                    DONOR_COL:
                        donor_id,

                    **metadata_values,

                    "metadata_consistent":
                        metadata_consistent_all,

                    **stats,
                }

                donor_trajectory_records.append(record)

        print(
            f"  donors={n_donors}; "
            f"lineages="
            f"{len(lineage_specs)}; "
            f"rows="
            f"{n_donors * len(lineage_specs)}"
        )


donor_trajectory_summary = (
    pd.DataFrame(donor_trajectory_records)
)

donor_metadata_qc = (
    pd.DataFrame(donor_metadata_qc_records)
)


# %% DONOR-LEVEL SUMMARIES: ATTACH LINEAGE ROBUSTNESS
#
# Donor-subsampling robustness is descriptive annotation.
# It is NOT used here as a hard filter.


robustness_file = (
    TRAJECTORY_DIR
    / (
        "trajectory_donor_subsampling_"
        "lineage_stability.csv"
    )
)


if (
    not robustness_file.exists()
):
    raise FileNotFoundError(
        robustness_file
    )


lineage_robustness = pd.read_csv(
    robustness_file
)


required_robustness_cols = [
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


missing_robustness_cols = [
    col
    for col
    in required_robustness_cols
    if col
    not in lineage_robustness.columns
]


if (
    missing_robustness_cols
):
    raise ValueError(
        "Missing donor-subsampling "
        "robustness columns: "
        f"{missing_robustness_cols}"
    )


lineage_robustness = (
    lineage_robustness[
        required_robustness_cols
    ]
    .rename(
        columns={
            "primary_lineage":
                "lineage",

            "n_successful_runs":
                "robustness_n_successful_runs",

            "n_runs_matched":
                "robustness_n_runs_matched",

            "match_fraction":
                "robustness_match_fraction",

            "spearman_median":
                "robustness_spearman_median",

            "spearman_q05":
                "robustness_spearman_q05",

            "spearman_q95":
                "robustness_spearman_q95",

            "spearman_min":
                "robustness_spearman_min",
        }
    )
)


donor_trajectory_summary = (
    donor_trajectory_summary
    .merge(
        lineage_robustness,
        on=[
            "compartment",
            "trajectory",
            "lineage",
        ],
        how="left",
        validate="many_to_one",
    )
)


# %% DONOR-LEVEL SUMMARIES: QC


if (
    donor_trajectory_summary.empty
):
    raise RuntimeError(
        "No donor-level trajectory "
        "summary rows were generated."
    )


duplicate_keys = (
    donor_trajectory_summary
    .duplicated(
        subset=[
            "compartment",
            "trajectory",
            "lineage",
            DONOR_COL,
        ],
        keep=False,
    )
)


if np.any(
    duplicate_keys
):
    duplicated = (
        donor_trajectory_summary
        .loc[
            duplicate_keys,
            [
                "compartment",
                "trajectory",
                "lineage",
                DONOR_COL,
            ],
        ]
        .sort_values(
            [
                "compartment",
                "trajectory",
                "lineage",
                DONOR_COL,
            ]
        )
    )

    raise ValueError(
        "Duplicate donor × trajectory × "
        "lineage rows detected:\n"
        f"{duplicated.head(20)}"
    )


# Check bounded scaled pseudotime.
for column in [
    "weighted_mean_pseudotime",
    "median_pseudotime",
    "pseudotime_q25",
    "pseudotime_q75",
    "min_pseudotime",
    "max_pseudotime",
]:
    values = pd.to_numeric(
        donor_trajectory_summary[
            column
        ],
        errors="coerce",
    )

    finite = values[
        np.isfinite(values)
    ]

    if (
        len(finite)
        > 0
        and (
            finite.min()
            < -1e-8
            or finite.max()
            > 1 + 1e-8
        )
    ):
        raise ValueError(
            f"{column} falls outside "
            "[0,1]."
        )


# Check weights.
for column in [
    "lineage_weight_sum",
    "lineage_weight_mean",
    "lineage_weight_median",
    "lineage_weight_max",
    "lineage_effective_metacells",
]:
    values = pd.to_numeric(
        donor_trajectory_summary[
            column
        ],
        errors="coerce",
    )

    finite = values[
        np.isfinite(values)
    ]

    if (
        len(finite)
        > 0
        and finite.min()
        < -1e-8
    ):
        raise ValueError(
            f"Negative values detected "
            f"in {column}."
        )


# Ensure robustness annotation exists for every
# frozen primary lineage.
robustness_missing = (
    donor_trajectory_summary[
        "robustness_match_fraction"
    ]
    .isna()
)


if np.any(
    robustness_missing
):
    missing = (
        donor_trajectory_summary
        .loc[
            robustness_missing,
            [
                "compartment",
                "trajectory",
                "lineage",
            ],
        ]
        .drop_duplicates()
    )

    raise ValueError(
        "Missing donor-subsampling "
        "robustness annotation for:\n"
        f"{missing.to_string(index=False)}"
    )


# %% DONOR-LEVEL SUMMARIES: METADATA QC


metadata_qc_summary_records = []


for (
    compartment,
    trajectory,
), group in (
    donor_metadata_qc
    .groupby(
        [
            "compartment",
            "trajectory",
        ],
        observed=True,
    )
):

    record = {
        "compartment":
            compartment,

        "trajectory":
            trajectory,

        "n_donors":
            int(
                len(group)
            ),

        "n_donors_all_metadata_consistent":
            int(
                group[
                    "all_metadata_consistent"
                ]
                .sum()
            ),

        "n_donors_any_metadata_inconsistent":
            int(
                (
                    ~group[
                        "all_metadata_consistent"
                    ]
                )
                .sum()
            ),
    }

    for metadata_col in (
        DONOR_SUMMARY_METADATA_COLS
    ):

        consistency_col = f"{metadata_col}_consistent"

        if (
            consistency_col
            in group.columns
        ):
            record[
                f"n_{metadata_col}_consistent"
            ] = int(
                group[
                    consistency_col
                ]
                .sum()
            )

            record[
                f"n_{metadata_col}_inconsistent"
            ] = int(
                (
                    ~group[
                        consistency_col
                    ]
                )
                .sum()
            )

    metadata_qc_summary_records.append(record)


donor_metadata_qc_summary = (
    pd.DataFrame(metadata_qc_summary_records)
)


# %% DONOR-LEVEL SUMMARIES: LINEAGE COVERAGE SUMMARY


lineage_coverage_records = []


for (
    compartment,
    trajectory,
    lineage,
), group in (
    donor_trajectory_summary
    .groupby(
        [
            "compartment",
            "trajectory",
            "lineage",
        ],
        observed=True,
    )
):

    finite_weighted_pt = np.isfinite(
        pd.to_numeric(
            group[
                "weighted_mean_pseudotime"
            ],
            errors="coerce",
        )
    )

    positive_support = (
        pd.to_numeric(
            group[
                "lineage_weight_sum"
            ],
            errors="coerce",
        )
        > DONOR_SUMMARY_MIN_WEIGHT
    )

    lineage_coverage_records.append(
        {
            "compartment":
                compartment,

            "trajectory":
                trajectory,

            "lineage":
                lineage,

            "n_donors":
                int(
                    len(group)
                ),

            "n_donors_finite_weighted_pseudotime":
                int(
                    finite_weighted_pt.sum()
                ),

            "fraction_donors_finite_weighted_pseudotime":
                float(
                    finite_weighted_pt.mean()
                ),

            "n_donors_positive_lineage_support":
                int(
                    positive_support.sum()
                ),

            "fraction_donors_positive_lineage_support":
                float(
                    positive_support.mean()
                ),

            "median_metacells_per_donor":
                float(
                    np.median(
                        group[
                            "n_metacells"
                        ]
                    )
                ),

            "median_lineage_weight_sum":
                float(
                    np.nanmedian(
                        pd.to_numeric(
                            group[
                                "lineage_weight_sum"
                            ],
                            errors="coerce",
                        )
                    )
                ),

            "median_lineage_effective_metacells":
                float(
                    np.nanmedian(
                        pd.to_numeric(
                            group[
                                "lineage_effective_metacells"
                            ],
                            errors="coerce",
                        )
                    )
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


donor_lineage_coverage = (
    pd.DataFrame(lineage_coverage_records)
)


# %% DONOR-LEVEL SUMMARIES: SAVE


donor_trajectory_summary = (
    donor_trajectory_summary
    .sort_values(
        [
            "compartment",
            "trajectory",
            "lineage",
            DONOR_COL,
        ]
    )
    .reset_index(
        drop=True
    )
)


donor_metadata_qc = (
    donor_metadata_qc
    .sort_values(
        [
            "compartment",
            "trajectory",
            DONOR_COL,
        ]
    )
    .reset_index(
        drop=True
    )
)


donor_metadata_qc_summary = (
    donor_metadata_qc_summary
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


donor_lineage_coverage = (
    donor_lineage_coverage
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


donor_trajectory_summary.to_csv(
    TRAJECTORY_DIR
    / (
        "trajectory_donor_level_"
        "summary.csv"
    ),
    index=False,
)


donor_metadata_qc.to_csv(
    TRAJECTORY_DIR
    / (
        "trajectory_donor_metadata_"
        "qc.csv"
    ),
    index=False,
)


donor_metadata_qc_summary.to_csv(
    TRAJECTORY_DIR
    / (
        "trajectory_donor_metadata_"
        "qc_summary.csv"
    ),
    index=False,
)


donor_lineage_coverage.to_csv(
    TRAJECTORY_DIR
    / (
        "trajectory_donor_lineage_"
        "coverage.csv"
    ),
    index=False,
)


# %% DONOR-LEVEL TRAJECTORY SUMMARIES: CHECKPOINT


print(
    "\n"
    + "=" * 100
)

print(
    "04c DONOR-LEVEL TRAJECTORY "
    "SUMMARIES COMPLETE"
)

print(
    "=" * 100
)


print(
    "\nDONOR × LINEAGE COVERAGE"
)

print(
    donor_lineage_coverage[
        [
            "compartment",
            "trajectory",
            "lineage",
            "n_donors",
            "n_donors_finite_weighted_pseudotime",
            "fraction_donors_finite_weighted_pseudotime",
            "median_metacells_per_donor",
            "median_lineage_weight_sum",
            "median_lineage_effective_metacells",
            "robustness_match_fraction",
            "robustness_spearman_median",
            "robustness_spearman_q05",
        ]
    ].to_string(
        index=False
    )
)


print(
    "\nDONOR METADATA CONSISTENCY"
)

print(
    donor_metadata_qc_summary
    .to_string(
        index=False
    )
)


inconsistent_metadata = (
    donor_metadata_qc[
        ~donor_metadata_qc[
            "all_metadata_consistent"
        ]
    ]
)


print(
    "\nINCONSISTENT DONOR METADATA"
)

if (
    inconsistent_metadata.empty
):
    print(
        "None."
    )

else:
    display_cols = [
        "compartment",
        "trajectory",
        DONOR_COL,
    ]

    for metadata_col in (
        DONOR_SUMMARY_METADATA_COLS
    ):
        for suffix in [
            "n_unique",
            "consistent",
        ]:
            col = (
                f"{metadata_col}_"
                f"{suffix}"
            )

            if (
                col
                in inconsistent_metadata.columns
            ):
                display_cols.append(col)

    print(
        inconsistent_metadata[
            display_cols
        ]
        .to_string(
            index=False
        )
    )


print(
    "\nDONOR-LEVEL TABLE"
)

print(
    f"rows="
    f"{len(donor_trajectory_summary):,}; "
    f"donors="
    f"{donor_trajectory_summary[DONOR_COL].nunique():,}; "
    f"trajectories="
    f"{donor_trajectory_summary[['compartment', 'trajectory']].drop_duplicates().shape[0]}; "
    f"lineages="
    f"{donor_trajectory_summary[['compartment', 'trajectory', 'lineage']].drop_duplicates().shape[0]}"
)


print(
    "\nSaved:"
)

for filename in [
    "trajectory_donor_level_summary.csv",
    "trajectory_donor_metadata_qc.csv",
    "trajectory_donor_metadata_qc_summary.csv",
    "trajectory_donor_lineage_coverage.csv",
]:
    print(
        " ",
        TRAJECTORY_DIR
        / filename,
    )

