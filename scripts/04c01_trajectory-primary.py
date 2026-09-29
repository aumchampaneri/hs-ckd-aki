# %% TRAJECTORY
#
# Topology-first trajectory inference and primary Slingshot pseudotime calculation.
# Establishes graph structure before running Slingshot lineage fitting.

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


# %% TOPOLOGY HELPERS
def numeric_rep(adata, key):
    x = np.asarray(adata.obsm[key], dtype=float)
    if x.ndim != 2 or x.shape[0] != adata.n_obs:
        raise ValueError(f"{key} has invalid shape: {x.shape}")
    if not np.isfinite(x).all():
        raise ValueError(f"{key} contains non-finite values")
    return x


def filter_trajectory_input(adata):
    n_input = adata.n_obs

    if not ZERO_COUNT_FILTER:
        adata.obs["trajectory_input"] = True
        adata.uns["trajectory_input_qc"] = {
            "n_input_metacells": int(n_input),
            "n_excluded_zero_count": 0,
            "n_trajectory_metacells": int(n_input),
        }
        return adata

    if "counts" not in adata.layers:
        raise KeyError("04c requires the raw 'counts' layer.")

    counts = adata.layers["counts"]
    total_counts = np.asarray(counts.sum(axis=1)).ravel()
    keep = total_counts > 0

    adata.obs["trajectory_input"] = keep
    adata.uns["trajectory_input_qc"] = {
        "n_input_metacells": int(n_input),
        "n_excluded_zero_count": int((~keep).sum()),
        "n_trajectory_metacells": int(keep.sum()),
    }

    return adata[keep].copy()


def add_group_summary(adata, group_col, compartment):
    if group_col not in adata.obs:
        return pd.DataFrame()

    x = adata.obsm[PRIMARY_REP]

    out = (
        adata.obs.assign(
            v1=x[:, 0],
            v2=x[:, 1],
        )
        .groupby(group_col, observed=True)
        .agg(
            n_metacells=("v1", "size"),
            n_donors=(DONOR_COL, "nunique"),
            mean_v1=("v1", "mean"),
            mean_v2=("v2", "mean"),
            median_v1=("v1", "median"),
            median_v2=("v2", "median"),
        )
        .reset_index()
        .sort_values("n_metacells", ascending=False)
    )

    out.insert(0, "compartment", compartment)
    out.insert(1, "group_type", group_col)
    return out


def topology_summary(adata, compartment, cluster_key):
    labels = adata.obs[cluster_key].astype(str)
    out = (
        adata.obs.assign(cluster=labels)
        .groupby("cluster", observed=True)
        .agg(
            n_metacells=(cluster_key, "size"),
            n_donors=(DONOR_COL, "nunique"),
        )
        .reset_index()
    )

    for col in [STATE_COL, SUBCLASS_COL, DISEASE_COL, ASSAY_COL]:
        if col not in adata.obs:
            continue

        tab = (
            adata.obs.assign(cluster=labels)
            .groupby(["cluster", col], observed=True)
            .size()
            .rename("n")
            .reset_index()
        )
        tab["fraction"] = tab["n"] / tab.groupby("cluster")["n"].transform("sum")

        dominant = (
            tab.sort_values(["cluster", "fraction", "n"], ascending=[True, False, False])
            .drop_duplicates("cluster")
            .set_index("cluster")
        )

        out[f"dominant_{col}"] = out["cluster"].map(dominant[col])
        out[f"dominant_{col}_fraction"] = out["cluster"].map(dominant["fraction"])

    out.insert(0, "compartment", compartment)
    out.insert(1, "cluster_key", cluster_key)
    return out


def connected_components(adata, cluster_key):
    graph = adata.uns["paga"]["connectivities"]
    graph = np.asarray(graph.todense() if hasattr(graph, "todense") else graph)

    categories = list(adata.obs[cluster_key].cat.categories)
    parent = list(range(len(categories)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        i, j = find(i), find(j)
        if i != j:
            parent[j] = i

    for i in range(len(categories)):
        for j in range(i + 1, len(categories)):
            if graph[i, j] >= PAGA_THRESHOLD:
                union(i, j)

    groups = {}
    for i, category in enumerate(categories):
        groups.setdefault(find(i), []).append(category)

    component_map = {
        category: component
        for component, cats in enumerate(groups.values())
        for category in cats
    }

    return component_map


def candidate_roots(summary):
    if summary.empty:
        return summary

    out = summary.copy()

    if "dominant_dominant_state" in out:
        out["reference_fraction"] = np.where(
            out["dominant_dominant_state"].eq("reference"),
            out["dominant_dominant_state_fraction"],
            0.0,
        )

    if "dominant_dominant_subclass" in out:
        native = out["dominant_dominant_subclass"].astype(str).str.match(
            r"^(PT-S[123]|C-FIB|OM-FIB|IM-FIB|POD|PEC|EC-GC|MC|resMAC|Naive Th|B|PL)$"
        )
        out["native_identity_candidate"] = native

    out["root_candidate_score"] = 0.0
    if "reference_fraction" in out:
        out["root_candidate_score"] += out["reference_fraction"]

    if "native_identity_candidate" in out:
        out["root_candidate_score"] += out["native_identity_candidate"].astype(float) * 0.5

    return out.sort_values(
        ["root_candidate_score", "n_metacells"],
        ascending=False,
    )


# %% SLINGSHOT HELPERS
def safe_name(s):
    return str(s).replace("-", "_").replace(".", "_").replace(" ", "_")


def scale_pseudotime(x):
    x = np.asarray(x, dtype=float).copy()
    out = np.full(x.shape, np.nan, dtype=float)
    finite = np.isfinite(x)

    if finite.sum() == 0:
        return out

    lo = np.nanmin(x[finite])
    hi = np.nanmax(x[finite])

    if hi <= lo:
        out[finite] = 0.0
        return out

    out[finite] = (x[finite] - lo) / (hi - lo)
    return out


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


def fit_primary_trajectory(
    adata,
    compartment,
    trajectory_name,
    spec,
):
    original_cluster_col = (
        f"trajectory_leiden_"
        f"{str(PRIMARY_RESOLUTION).replace('.', 'p')}"
    )

    subclasses = adata.obs[SUBCLASS_COL].astype(str)
    mask = subclasses.isin(spec["subclasses"])
    sub_adata = adata[mask].copy()

    if sub_adata.n_obs == 0:
        raise ValueError(
            f"{compartment}/{trajectory_name}: trajectory subset is empty."
        )

    if sub_adata.n_obs <= N_NEIGHBORS:
        raise ValueError(
            f"{compartment}/{trajectory_name}: "
            f"only {sub_adata.n_obs} metacells."
        )

    trajectory_safe = safe_name(trajectory_name)
    neighbors_key = f"{trajectory_safe}_neighbors"
    trajectory_cluster_col = f"{trajectory_safe}_leiden"

    sc.pp.neighbors(
        sub_adata,
        n_neighbors=min(N_NEIGHBORS, sub_adata.n_obs - 1),
        use_rep=PRIMARY_REP,
        metric="euclidean",
        random_state=SEED,
        key_added=neighbors_key,
    )

    sc.tl.leiden(
        sub_adata,
        resolution=PRIMARY_RESOLUTION,
        key_added=trajectory_cluster_col,
        random_state=SEED,
        flavor="igraph",
        n_iterations=2,
        neighbors_key=neighbors_key,
    )

    cluster_sizes = (
        sub_adata.obs[trajectory_cluster_col]
        .astype(str)
        .value_counts()
        .sort_index()
    )
    singleton_clusters = cluster_sizes[cluster_sizes < 2].index.tolist()

    if singleton_clusters:
        raise ValueError(
            f"{compartment}/{trajectory_name}: "
            "trajectory-specific Leiden produced singleton clusters "
            f"{singleton_clusters}."
        )

    (
        root_cluster,
        n_original_root,
        n_root_mapped,
        root_overlap_fraction,
    ) = map_root_cluster(
        sub_adata=sub_adata,
        original_cluster_col=original_cluster_col,
        trajectory_cluster_col=trajectory_cluster_col,
        original_root_cluster=spec["root"]["cluster"],
    )

    print(
        f"{compartment}/{trajectory_name}: "
        f"{sub_adata.n_obs} metacells, "
        f"{sub_adata.obs[DONOR_COL].nunique()} donors, "
        f"{sub_adata.obs[trajectory_cluster_col].nunique()} "
        "trajectory-specific clusters"
    )
    print(
        f"audited root {spec['root']['subclass']} "
        f"(original cluster {spec['root']['cluster']}) "
        f"-> trajectory cluster {root_cluster}; "
        f"{n_root_mapped}/{n_original_root} root metacells "
        f"({root_overlap_fraction:.1%}) map there"
    )

    rd = numeric_rep(sub_adata, PRIMARY_REP)
    cluster_labels = (
        sub_adata.obs[trajectory_cluster_col]
        .astype(str)
        .to_numpy()
    )

    result_df, lineage_df = run_slingshot(
        rd=rd,
        cluster_labels=cluster_labels,
        start_clus=root_cluster,
        approx_points=SLINGSHOT_APPROX_POINTS,
    )

    pt_cols = [c for c in result_df.columns if c.startswith("pt__")]
    weight_cols = [c for c in result_df.columns if c.startswith("weight__")]

    if not pt_cols:
        raise ValueError(
            f"{compartment}/{trajectory_name}: "
            "Slingshot returned no lineages."
        )

    if len(pt_cols) != len(weight_cols):
        raise ValueError(
            f"{compartment}/{trajectory_name}: "
            "pseudotime/weight lineage count mismatch."
        )

    membership_col = f"trajectory_{trajectory_safe}_member"
    cluster_out_col = f"trajectory_{trajectory_safe}_cluster"

    adata.obs[membership_col] = False
    adata.obs.loc[sub_adata.obs_names, membership_col] = True

    adata.obs[cluster_out_col] = pd.NA
    adata.obs.loc[
        sub_adata.obs_names,
        cluster_out_col,
    ] = sub_adata.obs[trajectory_cluster_col].astype(str).to_numpy()

    trajectory_table = sub_adata.obs[
        [
            DONOR_COL,
            DISEASE_COL,
            STATE_COL,
            SUBCLASS_COL,
            ASSAY_COL,
            original_cluster_col,
        ]
    ].copy()

    trajectory_table.insert(
        0,
        "obs_id",
        sub_adata.obs_names.astype(str),
    )
    trajectory_table.insert(0, "compartment", compartment)
    trajectory_table.insert(1, "trajectory", trajectory_name)
    trajectory_table["trajectory_cluster"] = (
        sub_adata.obs[trajectory_cluster_col]
        .astype(str)
        .to_numpy()
    )

    for pt_col in pt_cols:
        lineage = pt_col.replace("pt__", "")
        weight_col = f"weight__{lineage}"

        if weight_col not in result_df.columns:
            raise ValueError(
                f"{compartment}/{trajectory_name}: "
                f"missing curve weights for {lineage}."
            )

        raw = result_df[pt_col].to_numpy(dtype=float)
        scaled = scale_pseudotime(raw)
        weights = result_df[weight_col].to_numpy(dtype=float)

        raw_col = f"pt_{trajectory_safe}_{lineage}"
        scaled_col = f"{raw_col}_scaled"
        weight_out_col = f"weight_{trajectory_safe}_{lineage}"

        adata.obs[raw_col] = np.nan
        adata.obs[scaled_col] = np.nan
        adata.obs[weight_out_col] = 0.0

        adata.obs.loc[sub_adata.obs_names, raw_col] = raw
        adata.obs.loc[sub_adata.obs_names, scaled_col] = scaled
        adata.obs.loc[sub_adata.obs_names, weight_out_col] = weights

        trajectory_table[f"{lineage}_pseudotime"] = raw
        trajectory_table[f"{lineage}_pseudotime01"] = scaled
        trajectory_table[f"{lineage}_weight"] = weights

    metadata = {
        "representation": PRIMARY_REP,
        "original_cluster_col": original_cluster_col,
        "trajectory_cluster_col": trajectory_cluster_col,
        "root_subclass": spec["root"]["subclass"],
        "original_root_cluster": str(spec["root"]["cluster"]),
        "root_cluster": root_cluster,
        "root_mapping_n": n_original_root,
        "root_mapping_selected_n": n_root_mapped,
        "root_mapping_fraction": root_overlap_fraction,
        "subclasses": list(spec["subclasses"]),
        "n_metacells": int(sub_adata.n_obs),
        "n_donors": int(sub_adata.obs[DONOR_COL].nunique()),
        "n_clusters": int(
            sub_adata.obs[trajectory_cluster_col].nunique()
        ),
        "n_lineages": int(len(pt_cols)),
    }

    return adata, trajectory_table, lineage_df, metadata


# %% LOAD / INPUT QC
all_cluster_summaries = []
all_group_summaries = []
all_paga_edges = []

for compartment in COMPARTMENTS:
    print("\n" + "=" * 100)
    print(compartment)
    print("=" * 100)

    h5ad = DECIPHER_DIR / f"{compartment}_decipher.h5ad"
    if not h5ad.exists():
        raise FileNotFoundError(h5ad)

    adata = sc.read_h5ad(h5ad)
    adata = filter_trajectory_input(adata)

    if PRIMARY_REP not in adata.obsm:
        raise KeyError(f"{h5ad} is missing {PRIMARY_REP}")

    numeric_rep(adata, PRIMARY_REP)

    for key in SENSITIVITY_REPS:
        if key in adata.obsm:
            numeric_rep(adata, key)

    qc = adata.uns["trajectory_input_qc"]
    print(
        f"metacells: {qc['n_input_metacells']} -> "
        f"{qc['n_trajectory_metacells']} "
        f"(excluded zero-count: {qc['n_excluded_zero_count']})"
    )

    # %% PRIMARY GRAPH
    print(f"\nKNN graph: {PRIMARY_REP}")

    sc.pp.neighbors(
        adata,
        n_neighbors=N_NEIGHBORS,
        use_rep=PRIMARY_REP,
        metric="euclidean",
        random_state=SEED,
        key_added="trajectory_neighbors",
    )

    adata.uns["trajectory_config"] = {
        "primary_representation": PRIMARY_REP,
        "sensitivity_representations": SENSITIVITY_REPS,
        "n_neighbors": N_NEIGHBORS,
        "leiden_resolutions": LEIDEN_RESOLUTIONS,
        "primary_resolution": PRIMARY_RESOLUTION,
        "paga_threshold": PAGA_THRESHOLD,
        "seed": SEED,
        "zero_count_filter": ZERO_COUNT_FILTER,
    }

    # %% CLUSTER / PAGA
    primary_key = f"trajectory_leiden_{str(PRIMARY_RESOLUTION).replace('.', 'p')}"

    for resolution in LEIDEN_RESOLUTIONS:
        key = f"trajectory_leiden_{str(resolution).replace('.', 'p')}"

        sc.tl.leiden(
            adata,
            resolution=resolution,
            key_added=key,
            random_state=SEED,
            flavor="igraph",
            n_iterations=2,
            neighbors_key="trajectory_neighbors",
        )

        summary = topology_summary(adata, compartment, key)
        all_cluster_summaries.append(summary)

        print(
            f"\n{key}: "
            f"{adata.obs[key].nunique()} clusters; "
            f"min={summary.n_metacells.min()} "
            f"median={summary.n_metacells.median():.0f} "
            f"max={summary.n_metacells.max()}"
        )

    sc.tl.paga(
        adata,
        groups=primary_key,
        neighbors_key="trajectory_neighbors",
    )

    adata.uns["paga"]["threshold"] = PAGA_THRESHOLD

    component_map = connected_components(adata, primary_key)
    adata.obs["trajectory_component"] = (
        adata.obs[primary_key].map(component_map).astype("category")
    )

    n_components = adata.obs["trajectory_component"].nunique()

    print(f"PAGA components: {n_components}")

    paga_conn = adata.uns["paga"]["connectivities"]
    categories = list(adata.obs[primary_key].cat.categories)

    if hasattr(paga_conn, "tocoo"):
        coo = paga_conn.tocoo()
        edges = pd.DataFrame(
            {
                "source": [categories[i] for i in coo.row],
                "target": [categories[i] for i in coo.col],
                "connectivity": coo.data,
            }
        )
        edges = edges[
            (edges["source"] < edges["target"])
            & (edges["connectivity"] >= PAGA_THRESHOLD)
        ].copy()
    else:
        edges = pd.DataFrame(
            columns=["source", "target", "connectivity"]
        )

    edges.insert(0, "compartment", compartment)
    all_paga_edges.append(edges)

    summary = topology_summary(adata, compartment, primary_key)
    summary["component"] = summary["cluster"].map(component_map)

    root_candidates = candidate_roots(summary)
    root_candidates.to_csv(
        TRAJECTORY_DIR / f"{compartment}_root_candidates.csv",
        index=False,
    )

    print("\nROOT CANDIDATES")
    cols = [
        "cluster",
        "component",
        "n_metacells",
        "n_donors",
        "dominant_dominant_state",
        "dominant_dominant_state_fraction",
        "dominant_dominant_subclass",
        "dominant_dominant_subclass_fraction",
        "root_candidate_score",
    ]
    cols = [c for c in cols if c in root_candidates]
    print(root_candidates[cols].head(15).to_string(index=False))

    # %% GROUP SUMMARIES
    for group_col in [STATE_COL, SUBCLASS_COL]:
        summary = add_group_summary(adata, group_col, compartment)
        if not summary.empty:
            all_group_summaries.append(summary)

    # %% PAGA OUTPUT
    paga_file = TRAJECTORY_DIR / f"{compartment}_paga_connectivities.csv"
    edges.to_csv(paga_file, index=False)

    sc.pl.paga(
        adata,
        threshold=PAGA_THRESHOLD,
        show=False,
        title=f"{compartment} PAGA — {PRIMARY_REP}",
    )
    plt.savefig(
        PLOT_DIR / f"{compartment}_paga.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close()

    # Save topology-ready AnnData
    adata.uns["trajectory_root_candidates"] = root_candidates.to_dict(
        orient="list"
    )

    out_h5ad = TRAJECTORY_DIR / f"{compartment}_trajectory_topology.h5ad"
    adata.write_h5ad(out_h5ad)

    print(f"Saved: {out_h5ad}")


# %% SAVE AUDITS
cluster_summary = pd.concat(
    all_cluster_summaries,
    ignore_index=True,
)
cluster_summary.to_csv(
    TRAJECTORY_DIR / "trajectory_cluster_summary.csv",
    index=False,
)

group_summary = pd.concat(
    all_group_summaries,
    ignore_index=True,
)
group_summary.to_csv(
    TRAJECTORY_DIR / "trajectory_group_summary.csv",
    index=False,
)

paga_edges = pd.concat(
    all_paga_edges,
    ignore_index=True,
)
paga_edges.to_csv(
    TRAJECTORY_DIR / "trajectory_paga_edges.csv",
    index=False,
)


# %% TOPOLOGY SUMMARY
summary = []

for compartment in COMPARTMENTS:
    adata = sc.read_h5ad(
        TRAJECTORY_DIR / f"{compartment}_trajectory_topology.h5ad",
        backed="r",
    )

    summary.append(
        {
            "compartment": compartment,
            "n_metacells": adata.n_obs,
            "n_donors": adata.obs[DONOR_COL].nunique(),
            "n_clusters": adata.obs[
                f"trajectory_leiden_{str(PRIMARY_RESOLUTION).replace('.', 'p')}"
            ].nunique(),
            "n_paga_components": adata.obs[
                "trajectory_component"
            ].nunique(),
        }
    )

pd.DataFrame(summary).to_csv(
    TRAJECTORY_DIR / "trajectory_topology_summary.csv",
    index=False,
)

print("\n" + "=" * 100)
print("04c TOPOLOGY COMPLETE")
print("=" * 100)
print(pd.DataFrame(summary).to_string(index=False))


# %% PRIMARY SLINGSHOT
check_slingshot()

all_primary_tables = []
all_lineage_tables = []
all_primary_summary = []

for compartment, specs in TRAJECTORY_SPECS.items():
    print("\n" + "=" * 100)
    print(f"{compartment} PRIMARY SLINGSHOT")
    print("=" * 100)

    h5ad = (
        TRAJECTORY_DIR
        / f"{compartment}_trajectory_topology.h5ad"
    )

    adata = sc.read_h5ad(h5ad)

    adata.uns.setdefault(
        "primary_trajectories",
        {},
    )

    for trajectory_name, spec in specs.items():
        print(
            f"\n{trajectory_name}: "
            f"root={spec['root']['subclass']} "
            f"(cluster {spec['root']['cluster']})"
        )

        (
            adata,
            trajectory_table,
            lineages,
            metadata,
        ) = fit_primary_trajectory(
            adata=adata,
            compartment=compartment,
            trajectory_name=trajectory_name,
            spec=spec,
        )

        trajectory_table.to_csv(
            TRAJECTORY_DIR
            / (
                f"{compartment}_"
                f"{trajectory_name}_"
                "primary_pseudotime.csv"
            ),
            index=False,
        )

        lineages.insert(
            0,
            "trajectory",
            trajectory_name,
        )

        lineages.insert(
            0,
            "compartment",
            compartment,
        )

        lineages.to_csv(
            TRAJECTORY_DIR
            / (
                f"{compartment}_"
                f"{trajectory_name}_"
                "slingshot_lineages.csv"
            ),
            index=False,
        )

        all_primary_tables.append(trajectory_table)
        all_lineage_tables.append(lineages)

        all_primary_summary.append(
            {
                "compartment": compartment,
                "trajectory": trajectory_name,
                **metadata,
            }
        )

        adata.uns["primary_trajectories"][trajectory_name] = metadata

        print(
            f"metacells={metadata['n_metacells']}; "
            f"donors={metadata['n_donors']}; "
            f"lineages={metadata['n_lineages']}"
        )

        print(lineages.to_string(index=False))

    out_h5ad = (
        TRAJECTORY_DIR
        / f"{compartment}_trajectory_primary.h5ad"
    )

    adata.write_h5ad(out_h5ad)
    print(f"\nSaved: {out_h5ad}")


# %% SAVE PRIMARY SLINGSHOT OUTPUTS
primary_pseudotime = pd.concat(
    all_primary_tables,
    ignore_index=True,
)

primary_pseudotime.to_csv(
    TRAJECTORY_DIR / "trajectory_primary_pseudotime.csv",
    index=False,
)

primary_lineages = pd.concat(
    all_lineage_tables,
    ignore_index=True,
)

primary_lineages.to_csv(
    TRAJECTORY_DIR / "trajectory_primary_lineages.csv",
    index=False,
)

primary_summary = pd.DataFrame(all_primary_summary)

primary_summary.to_csv(
    TRAJECTORY_DIR / "trajectory_primary_summary.csv",
    index=False,
)

print("\n" + "=" * 100)
print("04c PRIMARY SLINGSHOT COMPLETE")
print("=" * 100)

print(
    primary_summary[
        [
            "compartment",
            "trajectory",
            "root_subclass",
            "original_root_cluster",
            "root_cluster",
            "root_mapping_fraction",
            "n_metacells",
            "n_donors",
            "n_clusters",
            "n_lineages",
        ]
    ].to_string(index=False)
)

