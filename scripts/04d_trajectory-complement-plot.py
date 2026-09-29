# %% 04d TRAJECTORY + COMPLEMENT FIGURES (reader-facing UMAP / pseudotime revision)
# Visualization / integration only. Uses frozen 04c trajectories and 04c08 outputs.
# No trajectory, tradeSeq, donor, or DE model is refit or retuned here.

from __future__ import annotations

from pathlib import Path
import math
import re
import textwrap
import warnings

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch
import scanpy as sc
import scipy.sparse as sp
from scipy.interpolate import make_interp_spline, PchipInterpolator
from scipy.stats import t as student_t

# %% PATHS
PROJECT_ROOT = Path.cwd().resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
TRAJECTORY_DIR = DATA_DIR / "trajectory"
COMPLEMENT_DIR = TRAJECTORY_DIR / "complement"
TRADESEQ_DIR = TRAJECTORY_DIR / "tradeseq"
METACELL_DIR = DATA_DIR / "metacells"

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "trajectory"
MAIN_DIR = OUTPUT_DIR / "main"
COMPONENT_DIR = OUTPUT_DIR / "components"
SUPPLEMENT_DIR = OUTPUT_DIR / "supplement"
SOURCE_DIR = OUTPUT_DIR / "source_data"
for d in [OUTPUT_DIR, MAIN_DIR, COMPONENT_DIR, SUPPLEMENT_DIR, SOURCE_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# 04d owns ONLY its explicitly named outputs. Never delete other scripts' results.
# An existing old 04d file with the same name may be overwritten on save.
INTEGRATED_PATH = COMPLEMENT_DIR / "complement_trajectory_integrated.csv"
CURVE_PATH = COMPLEMENT_DIR / "complement_pseudotime_binned_expression.csv"
ANNOTATION_PATH = COMPLEMENT_DIR / "canonical_complement_gene_annotation.csv"
INPUT_MANIFEST_PATH = TRADESEQ_DIR / "trajectory_tradeseq_input_manifest.csv"

FULL_COUNT_PATHS = {
    "PT": METACELL_DIR / "PT_metacells.h5ad",
    "FIB": METACELL_DIR / "FIB_metacells.h5ad",
    "Glomerular": METACELL_DIR / "Glomerular_metacells.h5ad",
}
TRAJECTORY_H5AD_PATHS = {
    "PT": TRAJECTORY_DIR / "PT_trajectory_primary.h5ad",
    "FIB": TRAJECTORY_DIR / "FIB_trajectory_primary.h5ad",
    "Glomerular": TRAJECTORY_DIR / "Glomerular_trajectory_primary.h5ad",
}

# %% CONFIG
FDR_ALPHA = 0.05
N_SHOWCASE = 8
PATH_WEIGHT_FLOOR = 0.05
N_PATH_BINS = 30
MIN_PATH_BIN_POINTS = 4
DISPLAY_POINT_SIZE = 8
N_CURVE_BINS = 12
EARLY_Q = 0.33
LATE_Q = 0.67
MIN_GROUP_POINTS_PER_BIN = 2
SMOOTH_CURVE_POINTS = 200
RIBBON_ALPHA = 0.14
MIN_DONOR_METACELLS_BIN = 2
MIN_DONORS_RIBBON = 5
# Ribbon = donor-level t interval of mean within-bin donor summaries;
# no ribbon when fewer than MIN_DONORS_RIBBON independent donors contribute.
# Visual strokes are designed to retain ~0.25–1 pt after reduction to double-column width.
FOCAL_PATH_LW = 1.75
BACKGROUND_PATH_LW = 0.76
ARROW_LW = 1.45
GENE_CMAP = "cividis"   # perceptually ordered, color-vision-deficiency friendly
PSEUDOTIME_CMAP = "viridis"

TRAJECTORY_ORDER = [
    ("PT", "PT"),
    ("FIB", "contractile"),
    ("FIB", "outer_medullary"),
    ("Glomerular", "podocyte_PEC"),
    ("Glomerular", "endothelial_mesangial"),
]
MODULE_ORDER = ["classical", "lectin", "alternative", "terminal", "receptor", "regulator", "cfhr"]

CAUTION_LINEAGES = {
    ("PT", "PT"): {"Lineage4", "Lineage11"},
    ("FIB", "contractile"): {"Lineage2", "Lineage4", "Lineage6"},
    ("FIB", "outer_medullary"): set(),
    ("Glomerular", "podocyte_PEC"): {"Lineage3"},
    ("Glomerular", "endothelial_mesangial"): {"Lineage2"},
}

# Accessible qualitative palette inspired by the Okabe–Ito palette.
# Color is never the sole key: CKD and AKI have different line styles.
MODULE_COLORS = {
    "classical": "#0072B2",      # blue
    "lectin": "#CC79A7",        # reddish purple
    "alternative": "#E69F00",   # orange
    "terminal": "#D55E00",      # vermilion
    "receptor": "#009E73",      # bluish green
    "regulator": "#56B4E9",     # sky blue
    "cfhr": "#333333",          # neutral charcoal, descriptive family
}
MODULE_LINESTYLES = {
    "classical": "-", "lectin": "--", "alternative": "-.",
    "terminal": ":", "receptor": "-", "regulator": "--", "cfhr": ":",
}
TRAJECTORY_COLORS = {
    ("PT", "PT"): "#0072B2",
    ("FIB", "contractile"): "#D55E00",
    ("FIB", "outer_medullary"): "#E69F00",
    ("Glomerular", "podocyte_PEC"): "#CC79A7",
    ("Glomerular", "endothelial_mesangial"): "#009E73",
}
DISEASE_COLORS = {
    "CKD": "#0072B2",           # blue
    "AKI": "#D55E00",           # vermilion, NOT red/green
    "Control/Other": "#666666",
    "Unknown": "#BFBFBF",
}
DISEASE_LINESTYLES = {"All": "-", "CKD": "-", "AKI": "--", "Control/Other": "-."}

mpl.rcParams.update({
    "font.family": "Arial",
    "font.size": 8.5,
    "axes.titlesize": 10,
    "axes.labelsize": 8.5,
    "xtick.labelsize": 7.6,
    "ytick.labelsize": 7.6,
    "legend.fontsize": 7.4,
    "axes.linewidth": 0.7,
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.transparent": True,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.04,
})

# %% EXPLANATORY TEXT
FIGURE_TEXT = {
    "trajectory_umap_atlas": (
        "Trajectory UMAP atlas. Each panel shows one frozen renal trajectory projected onto a "
        "reader-facing UMAP-style layout for visualization only. Dots are metacells; thin gray curves "
        "are other display trace through frozen pseudotime/weightss. One illustrative lineage is highlighted per panel (chosen by weighted "
        "metacell coverage, not significance). Its clear arrow points from early toward late inferred pseudotime. "
        "The black star marks the inferred root/early end of the highlighted path, and the open circle marks "
        "its late end; neither represents a measured longitudinal time point."
    ),
    "trajectory_disease_atlas": (
        "Disease-context UMAP atlas. The same display-only trajectory layouts are shown, but metacells are colored by "
        "their dominant disease label. One illustrative lineage is highlighted, with other paths faint. "
        "The black star denotes its inferred root/early end and the open circle its late end. "
        "Colors describe state occupancy, not disease prevalence or donor-level enrichment."
    ),
    "complement_gene_maps": (
        "Showcase complement-gene maps. Each panel highlights one selected complement gene on the trajectory UMAP. "
        "Dot color shows descriptive metacell expression as log1p(counts per 10,000), while the emphasized arrowed "
        "path marks the lineage used for interpretation (star = inferred root/early end; open circle = late end). "
        "Gray lines show other paths as context. These panels answer where along the cell-state manifold the "
        "highlighted complement signal appears."
    ),
    "complement_gene_pseudotime_showcase": (
        "Showcase complement-gene pseudotime trends. Each panel plots smoothed descriptive expression across scaled "
        "pseudotime for one gene-lineage pair. The black curve shows all supported metacells, blue shows CKD, red shows AKI, "
        "and translucent ribbons show approximate 95% metacell-level intervals from weighted observations within pseudotime bins. "
        "They are descriptive and do not account for donor clustering or establish a CKD-versus-AKI difference. "
        "Faint points show the binned observed means that anchor the trend. These panels answer how complement expression changes "
        "from early to late pseudotime and whether CKD and AKI follow similar or distinct trajectories."
    ),
    "complement_module_pseudotime": (
        "Complement-module pseudotime summary. For each frozen trajectory, the lineage with the strongest overall complement signal is shown. "
        "Colored curves summarize the descriptive mean expression of each complement module across scaled pseudotime. This figure is useful for seeing "
        "whether the classical, lectin, alternative, terminal, receptor, or regulator arms dominate different trajectory contexts."
    ),
}

# %% HELPERS

def sanitize(x: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(x))


def save_figure(fig, stem: Path, dpi: int = 300):
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".svg"))
    fig.savefig(stem.with_suffix(".pdf"))
    fig.savefig(stem.with_suffix(".png"), dpi=dpi, facecolor="white", transparent=False)
    plt.close(fig)


def require_columns(df: pd.DataFrame, cols: list[str], name: str):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise RuntimeError(f"{name} missing required columns: {missing}")


def lineage_number(x: str) -> int:
    m = re.search(r"(\d+)$", str(x))
    return int(m.group(1)) if m else 999


def format_traj(comp: str, traj: str) -> str:
    names = {
        ("PT", "PT"): "Proximal tubule",
        ("FIB", "contractile"): "Contractile fibroblast",
        ("FIB", "outer_medullary"): "Outer-medullary fibroblast",
        ("Glomerular", "podocyte_PEC"): "Podocyte–PEC",
        ("Glomerular", "endothelial_mesangial"): "Endothelial–mesangial",
    }
    return names.get((comp, traj), f"{comp} · {traj.replace('_', ' ')}")


def gene_symbol_series(var: pd.DataFrame) -> pd.Series:
    for col in ["gene_symbol", "feature_name", "gene_name", "symbol"]:
        if col in var.columns:
            return var[col].astype(str)
    return pd.Series(var.index.astype(str), index=var.index)


def choose_obs_column(obs: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in obs.columns:
            return c
    return None


def scale01(x: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    x = np.asarray(x, float)
    out = np.full_like(x, np.nan, dtype=float)
    ok = np.isfinite(x) if mask is None else (np.isfinite(x) & mask)
    if not ok.any():
        return out
    lo, hi = np.nanmin(x[ok]), np.nanmax(x[ok])
    out[ok] = (x[ok] - lo) / (hi - lo) if hi > lo else 0.0
    return out


def robustness_label(row) -> str:
    key = (str(row["compartment"]), str(row["trajectory"]))
    return "caution" if str(row["lineage"]) in CAUTION_LINEAGES.get(key, set()) else "strong"


def concordant_direction(row) -> bool:
    a = pd.to_numeric(pd.Series([row.get("start_end_logFC")]), errors="coerce").iloc[0]
    b = pd.to_numeric(pd.Series([row.get("late_minus_early")]), errors="coerce").iloc[0]
    return np.isfinite(a) and np.isfinite(b) and a != 0 and b != 0 and np.sign(a) == np.sign(b)


def categorical_palette(values: pd.Series) -> dict[str, tuple]:
    cats = [str(x) for x in pd.unique(values.dropna().astype(str))]
    cmap = mpl.colormaps["tab20"]
    return {c: cmap((i % 20) / 20) for i, c in enumerate(cats)}


def clean_axes(ax, equal: bool = True):
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xlabel("")
    ax.set_ylabel("")
    if equal:
        ax.set_aspect("equal", adjustable="datalim")


def add_figure_note(fig, text: str, bottom: float = 0.02, fontsize: float = 7.0):
    lines = textwrap.wrap(text, width=165, break_long_words=False)
    fig.text(0.02, bottom, "\n".join(lines), va="bottom", ha="left", fontsize=fontsize, color="#303030", linespacing=1.35)


def simplify_disease(x: str) -> str:
    s = str(x).strip().lower()
    if s in {"", "nan", "none"}:
        return "Unknown"
    if "aki" in s or "acute" in s:
        return "AKI"
    if "ckd" in s or "chronic" in s:
        return "CKD"
    if "control" in s or "normal" in s or "healthy" in s or "reference" in s:
        return "Control/Other"
    return "Control/Other"


def smooth_series(x: np.ndarray, y: np.ndarray, n_points: int = SMOOTH_CURVE_POINTS) -> tuple[np.ndarray, np.ndarray]:
    """Shape-preserving display interpolation; never extrapolates outside observed bins."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if not len(x): return np.array([]), np.array([])
    order = np.argsort(x); x, y = x[order], y[order]
    if len(np.unique(x)) != len(x):
        tmp = pd.DataFrame({"x": x, "y": y}).groupby("x", as_index=False)["y"].mean()
        x, y = tmp.x.to_numpy(float), tmp.y.to_numpy(float)
    if len(x) < 3: return x, y
    xs = np.linspace(float(x.min()), float(x.max()), n_points)
    return xs, PchipInterpolator(x, y)(xs)


def smooth_path(points: np.ndarray, n_points: int = 250) -> np.ndarray:
    points = np.asarray(points, float)
    if len(points) < 3:
        return points
    diffs = np.diff(points, axis=0)
    seg = np.sqrt((diffs ** 2).sum(axis=1))
    t = np.concatenate([[0.0], np.cumsum(seg)])
    if t[-1] == 0:
        return points
    t /= t[-1]
    k = min(3, len(points) - 1)
    try:
        tt = np.linspace(0, 1, n_points)
        sx = make_interp_spline(t, points[:, 0], k=k)
        sy = make_interp_spline(t, points[:, 1], k=k)
        out = np.c_[sx(tt), sy(tt)]
        out[0] = points[0]
        out[-1] = points[-1]
        return out
    except Exception:
        return points


def add_curve_arrow(ax, path_xy: np.ndarray, color: str, lw: float = ARROW_LW,
                    frac: float = 0.74, size: float = 12):
    """Draw a clearly directed, short shaft and triangular head along the path.

    Use a substantial arclength interval rather than adjacent spline samples;
    adjacent samples can produce an almost head-only, ambiguous arrow.
    """
    if path_xy is None or len(path_xy) < 5:
        return
    distances = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(path_xy, axis=0), axis=1))]
    if not np.isfinite(distances[-1]) or distances[-1] <= 0:
        return
    distances /= distances[-1]
    start_t, end_t = max(0.01, frac - 0.095), min(0.99, frac + 0.015)
    p0 = np.array([np.interp(start_t, distances, path_xy[:, d]) for d in (0, 1)])
    p1 = np.array([np.interp(end_t, distances, path_xy[:, d]) for d in (0, 1)])
    if np.linalg.norm(p1 - p0) <= 1e-9:
        return
    arrow = FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=size,
                            linewidth=lw, color=color, shrinkA=0, shrinkB=0,
                            connectionstyle="arc3,rad=0", zorder=9,
                            path_effects=[pe.Stroke(linewidth=lw + 1.2, foreground="white"), pe.Normal()])
    ax.add_patch(arrow)


def weighted_mean(x: np.ndarray, w: np.ndarray) -> float:
    x = np.asarray(x, float)
    w = np.asarray(w, float)
    ok = np.isfinite(x) & np.isfinite(w) & (w > 0)
    if not ok.any():
        return np.nan
    return float(np.average(x[ok], weights=w[ok]))


def infer_plot_umap(ad: sc.AnnData, preferred_keys: list[str]) -> np.ndarray:
    # If a UMAP already exists, use it. Otherwise compute a display-only UMAP.
    for key in ["X_umap", "umap"]:
        if key in ad.obsm:
            arr = np.asarray(ad.obsm[key])
            if arr.shape[1] >= 2:
                return arr[:, :2]

    rep_key = None
    rep = None
    for key in preferred_keys:
        if key in ad.obsm:
            arr = np.asarray(ad.obsm[key])
            if arr.ndim == 2 and arr.shape[1] >= 2:
                rep_key = key
                rep = arr
                break
    if rep is None:
        raise RuntimeError("Could not find a display representation to compute a UMAP.")

    tmp = sc.AnnData(np.zeros((ad.n_obs, 1), dtype=np.float32))
    tmp.obs_names = ad.obs_names.copy()
    tmp.obsm[rep_key] = rep.copy()
    sc.pp.neighbors(tmp, use_rep=rep_key, n_neighbors=min(30, max(5, tmp.n_obs - 1)), metric="euclidean", random_state=0)
    sc.tl.umap(tmp, min_dist=0.45, spread=1.0, random_state=0)
    return np.asarray(tmp.obsm["X_umap"])[:, :2]


def lineage_path(xy: np.ndarray, pseudotime: np.ndarray, weight: np.ndarray,
                 n_bins: int = N_PATH_BINS) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    p = np.asarray(pseudotime, float)
    w = np.asarray(weight, float)
    ok = np.isfinite(p) & np.isfinite(w) & (w > PATH_WEIGHT_FLOOR)
    if ok.sum() < 8:
        return np.empty((0, 2)), np.array([]), np.array([])
    ps = scale01(p, ok)
    edges = np.linspace(0, 1, n_bins + 1)
    pts, mids, counts = [], [], []
    for a, b in zip(edges[:-1], edges[1:]):
        sel = ok & np.isfinite(ps) & (ps >= a) & (ps < b if b < 1 else ps <= b)
        if sel.sum() < MIN_PATH_BIN_POINTS:
            continue
        ww = w[sel]
        pts.append([
            np.average(xy[sel, 0], weights=ww),
            np.average(xy[sel, 1], weights=ww),
        ])
        mids.append((a + b) / 2)
        counts.append(sel.sum())
    if len(pts) < 3:
        return np.empty((0, 2)), np.array([]), np.array([])
    pts = np.asarray(pts, float)
    sm = smooth_path(pts, n_points=250)
    return sm, np.asarray(mids), np.asarray(counts)


def add_root_and_end_markers(ax, path_xy: np.ndarray, color: str):
    """Root marker indicates early *inferred* pseudotime, not a measured time point."""
    if path_xy is None or len(path_xy) < 2:
        return
    ax.scatter(path_xy[0, 0], path_xy[0, 1], marker="*", s=65,
               color="#151515", edgecolor="white", linewidths=0.5, zorder=11)
    ax.scatter(path_xy[-1, 0], path_xy[-1, 1], marker="o", s=28,
               facecolor="white", edgecolor=color, linewidths=1.0, zorder=11)


def draw_paths(ax, td, highlight: str | None = None, label_endpoints: bool = False,
               show_root_only_for_highlight: bool = True):
    """One readable focal path, with nonfocal branches deliberately de-emphasized."""
    for lin in sorted(td["lineages"], key=lineage_number):
        path = td["paths"].get(lin)
        if path is None or len(path) < 3:
            continue
        strong = lin not in CAUTION_LINEAGES.get((td["compartment"], td["trajectory"]), set())
        focal = lin == highlight if highlight is not None else strong
        if highlight is None:
            col = TRAJECTORY_COLORS[(td["compartment"], td["trajectory"])] if strong else "#A4A4A4"
            alpha, lw = (0.70, 1.06) if strong else (0.36, 0.66)
        elif focal:
            col = TRAJECTORY_COLORS[(td["compartment"], td["trajectory"])]
            alpha, lw = 0.98, FOCAL_PATH_LW
        else:
            col, alpha, lw = "#BBBBBB", 0.27, BACKGROUND_PATH_LW
        ax.plot(path[:, 0], path[:, 1], color=col, lw=lw, alpha=alpha,
                solid_capstyle="round", zorder=6 if focal else 4)
        # Do not put arrows/stars/endpoint labels on all overlaid paths: this
        # produces a dense nest in high-branch trajectories such as PT.
        if highlight is not None and focal:
            add_curve_arrow(ax, path, col, lw=ARROW_LW, frac=0.73, size=12)
            add_root_and_end_markers(ax, path, col)
        elif highlight is None and len(td["lineages"]) <= 3 and strong:
            add_curve_arrow(ax, path, col, lw=1.1, frac=0.73, size=10.5)
        if label_endpoints and highlight is not None and focal:
            ax.text(path[-1, 0], path[-1, 1], lin.replace("Lineage", "L"),
                    fontsize=7.0, ha="left", va="center", color=col, zorder=12,
                    path_effects=[pe.withStroke(linewidth=2.0, foreground="white")])


def draw_disease_map(ax, td, title: str | None = None):
    vals = td["disease_group"]
    for cat in [c for c in ["Control/Other", "CKD", "AKI", "Unknown"] if c in set(vals.astype(str))]:
        sel = vals.to_numpy() == cat
        ax.scatter(td["plot_xy"][sel, 0], td["plot_xy"][sel, 1], s=DISPLAY_POINT_SIZE, color=DISEASE_COLORS.get(cat, "#BBBBBB"), alpha=0.72, linewidths=0, zorder=2)
    draw_paths(ax, td, highlight=td["display_lineage"], label_endpoints=True)
    clean_axes(ax)
    if title:
        ax.set_title(title, loc="left", fontweight="bold")


def draw_gene_map(ax, td, values: np.ndarray, lineage: str, title: str, add_colorbar: bool = False):
    vals = np.asarray(values, float)
    ax.scatter(td["plot_xy"][:, 0], td["plot_xy"][:, 1], s=DISPLAY_POINT_SIZE, color="#D9D9D9", alpha=0.24, linewidths=0, zorder=1)
    ok = np.isfinite(vals)
    if ok.any():
        lo, hi = np.nanquantile(vals[ok], [0.02, 0.98])
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            lo, hi = np.nanmin(vals[ok]), np.nanmax(vals[ok]) + 1e-9
        order = np.argsort(vals[ok])
        idx = np.flatnonzero(ok)[order]
        sca = ax.scatter(td["plot_xy"][idx, 0], td["plot_xy"][idx, 1], c=vals[idx], s=DISPLAY_POINT_SIZE + 3,
                         cmap=GENE_CMAP, norm=Normalize(lo, hi), alpha=0.92, linewidths=0, zorder=3)
        if add_colorbar:
            cb = plt.colorbar(sca, ax=ax, fraction=0.045, pad=0.015)
            cb.set_label("Expression\nlog1p(CP10K)")
    draw_paths(ax, td, highlight=lineage, label_endpoints=False)
    clean_axes(ax)
    ax.set_title(title, loc="left", fontweight="bold")


def draw_pseudotime_map(ax, td, lineage: str, title: str, add_colorbar: bool = False):
    j = td["lineage_to_j"][lineage]
    ps = td["pt_scaled"][:, j]
    w = td["wt"][:, j]
    ax.scatter(td["plot_xy"][:, 0], td["plot_xy"][:, 1], s=DISPLAY_POINT_SIZE, color="#D9D9D9", alpha=0.20, linewidths=0, zorder=1)
    ok = np.isfinite(ps) & np.isfinite(w) & (w > 0)
    sca = ax.scatter(td["plot_xy"][ok, 0], td["plot_xy"][ok, 1], c=ps[ok], s=DISPLAY_POINT_SIZE + 3,
                     cmap=PSEUDOTIME_CMAP, vmin=0, vmax=1, alpha=np.clip(0.25 + 0.75 * w[ok], 0.25, 1.0), linewidths=0, zorder=3)
    draw_paths(ax, td, highlight=lineage, label_endpoints=False)
    clean_axes(ax)
    ax.set_title(title, loc="left", fontweight="bold")
    if add_colorbar:
        cb = plt.colorbar(sca, ax=ax, fraction=0.045, pad=0.015)
        cb.set_label("Scaled\npseudotime")
        cb.set_ticks([0, 1])
        cb.set_ticklabels(["Early", "Late"])


def make_group_curve(expr: np.ndarray, ps: np.ndarray, wt: np.ndarray,
                     group_mask: np.ndarray, donors: np.ndarray) -> tuple[pd.DataFrame, dict]:
    """Donor-aware descriptive bin means and t intervals. Donors, not metacells, are replicates.

    The plot uses equal-donor-weighted means when donor support exists; original pooled
    metacell mean is retained separately for audit and as a fallback with NO ribbon.
    No new association test or disease contrast is performed.
    """
    expr, ps, wt, donors = (np.asarray(v) for v in (expr, ps, wt, donors))
    assert len(expr) == len(ps) == len(wt) == len(group_mask) == len(donors)
    edges = np.linspace(0, 1, N_CURVE_BINS + 1)
    supported = np.isfinite(expr) & np.isfinite(ps) & np.isfinite(wt) & (wt > 0)
    base = supported & group_mask
    records=[]
    for b in range(N_CURVE_BINS):
        lo, hi = edges[b], edges[b+1]
        mask = base & (ps >= lo) & ((ps < hi) if b < N_CURVE_BINS - 1 else (ps <= hi))
        pooled = weighted_mean(expr[mask], wt[mask]) if mask.any() else np.nan
        donor_values=[]
        for d in pd.unique(donors[mask]):
            dm = mask & (donors == d)
            if dm.sum() < MIN_DONOR_METACELLS_BIN: continue
            donor_values.append(weighted_mean(expr[dm], wt[dm]))
        donor_values = np.array([v for v in donor_values if np.isfinite(v)],float)
        nd = len(donor_values)
        donor_mean = float(np.mean(donor_values)) if nd else np.nan
        # Keep a donor-centered plotted trend only when donor support permits.
        mean = donor_mean if nd >= MIN_DONORS_RIBBON else pooled
        if nd >= MIN_DONORS_RIBBON:
            sd = float(np.std(donor_values, ddof=1))
            sem = sd / np.sqrt(nd)
            half = float(student_t.ppf(0.975, nd-1) * sem)
            low, high = donor_mean-half, donor_mean+half
        else:
            sd = sem = low = high = np.nan
        records.append(dict(pseudotime_bin=b+1,pseudotime_midpoint=(lo+hi)/2,
            weighted_mean_expression=mean, pooled_metacell_mean=pooled,
            donor_mean_expression=donor_mean, donor_sd=sd, donor_sem=sem,
            ci95_low=low,ci95_high=high,n_donors=nd,
            n_metacells=int(mask.sum()), sum_weight=float(np.sum(wt[mask]))))
    out=pd.DataFrame(records)
    early=base & (ps <= EARLY_Q); late=base & (ps >= LATE_Q)
    e=weighted_mean(expr[early],wt[early]); l=weighted_mean(expr[late],wt[late])
    return out, dict(n_supported_metacells=int(base.sum()), early_mean=e,late_mean=l,
                     late_minus_early=l-e if np.isfinite(e) and np.isfinite(l) else np.nan)


def draw_pseudotime_gene_panel(ax, curve_df: pd.DataFrame, title: str, subtitle: str | None = None):
    curve_df = curve_df.copy()
    styles = {
        "All": {"color": "#111111", "lw": 2.3, "alpha": 1.0, "z": 6, "ribbon_alpha": 0.08},
        "CKD": {"color": DISEASE_COLORS["CKD"], "lw": 2.0, "alpha": 0.98, "z": 5, "ribbon_alpha": RIBBON_ALPHA},
        "AKI": {"color": DISEASE_COLORS["AKI"], "lw": 2.0, "alpha": 0.98, "z": 5, "ribbon_alpha": RIBBON_ALPHA},
        "Control/Other": {"color": DISEASE_COLORS["Control/Other"], "lw": 1.2, "alpha": 0.75, "z": 4, "ribbon_alpha": 0.08},
    }
    groups_to_draw = ["All", "CKD", "AKI", "Control/Other"]
    ylim_vals: list[float] = []

    for grp in groups_to_draw:
        sub = curve_df[curve_df["group"] == grp].copy()
        if sub.empty:
            continue
        sub = sub.sort_values("pseudotime_midpoint")
        st = styles[grp]
        ax.scatter(
            sub["pseudotime_midpoint"],
            sub["weighted_mean_expression"],
            s=11 if grp == "All" else 8,
            color=st["color"],
            alpha=0.28 if grp == "All" else 0.18,
            linewidths=0,
            zorder=2,
        )

        # Draw only adjacent supported bins: never bridge missing-donor regions.
        ribbon = sub[["pseudotime_bin", "pseudotime_midpoint", "ci95_low", "ci95_high"]].dropna()
        if len(ribbon) >= 2 and grp in {"All", "CKD", "AKI"}:
            ribbon = ribbon.sort_values("pseudotime_bin")
            groups = (ribbon["pseudotime_bin"].diff().fillna(1) != 1).cumsum()
            for _, seg in ribbon.groupby(groups):
                if len(seg) < 2: continue
                xx=seg["pseudotime_midpoint"].to_numpy(float)
                rx1, ry1 = smooth_series(xx, seg["ci95_low"].to_numpy(float))
                rx2, ry2 = smooth_series(xx, seg["ci95_high"].to_numpy(float))
                if len(rx1) and len(rx2):
                    xs=np.linspace(max(rx1.min(),rx2.min()), min(rx1.max(),rx2.max()), SMOOTH_CURVE_POINTS)
                    if len(xs)<2: continue
                    low=np.interp(xs,rx1,ry1); high=np.interp(xs,rx2,ry2)
                    low2,high2=np.minimum(low,high),np.maximum(low,high)
                    ax.fill_between(xs,low2,high2,color=st["color"],alpha=st["ribbon_alpha"],zorder=1)
                    ylim_vals.extend(low2.tolist());ylim_vals.extend(high2.tolist())

        xs, ys = smooth_series(
            sub["pseudotime_midpoint"].to_numpy(float),
            sub["weighted_mean_expression"].to_numpy(float),
        )
        if len(xs):
            ax.plot(xs, ys, color=st["color"], lw=st["lw"], ls=DISEASE_LINESTYLES[grp], alpha=st["alpha"], zorder=st["z"])
            ylim_vals.extend(np.asarray(ys, float).tolist())
            if grp in {"CKD", "AKI"}:
                ax.text(
                    xs[-1] + 0.012,
                    ys[-1],
                    grp,
                    fontsize=6.8,
                    color=st["color"],
                    va="center",
                    path_effects=[pe.withStroke(linewidth=2.2, foreground="white")],
                )

    ax.set_xlim(-0.02, 1.07)
    yvals = np.asarray([v for v in ylim_vals if np.isfinite(v)], float)
    if len(yvals):
        lo, hi = float(np.nanmin(yvals)), float(np.nanmax(yvals))
        pad = max(0.08, 0.10 * (hi - lo if hi > lo else 1.0))
        ax.set_ylim(lo - pad, hi + pad)
    ax.axvspan(0.0, EARLY_Q, color="#F5F5F5", alpha=0.9, zorder=0)
    ax.axvspan(LATE_Q, 1.0, color="#F5F5F5", alpha=0.9, zorder=0)
    ax.axvline(EARLY_Q, color="#C7C7C7", lw=0.8, ls="--", zorder=1)
    ax.axvline(LATE_Q, color="#C7C7C7", lw=0.8, ls="--", zorder=1)
    ax.text(0.01, 0.98, "Early pseudotime", transform=ax.transAxes, ha="left", va="top", fontsize=6.7, color="#555555")
    ax.text(0.99, 0.98, "Late pseudotime", transform=ax.transAxes, ha="right", va="top", fontsize=6.7, color="#555555")
    ax.set_xlabel("Scaled pseudotime (early → late)")
    ax.set_ylabel("Expression\nlog1p(CP10K)")
    ax.grid(axis="y", color="#E6E6E6", lw=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_title(title, loc="left", fontweight="bold")
    if subtitle:
        ax.text(0.0, 1.01, subtitle, transform=ax.transAxes, ha="left", va="bottom", fontsize=6.9, color="#444444")
def write_figure_summary(path: Path, lines: list[str]):
    with path.open("w") as f:
        f.write("04d trajectory / complement figure guide\n")
        f.write("=" * 52 + "\n\n")
        for section in lines:
            f.write(section.rstrip() + "\n\n")

# %% LOAD INPUTS
for p in [INTEGRATED_PATH, CURVE_PATH, ANNOTATION_PATH, INPUT_MANIFEST_PATH]:
    if not p.exists() or p.stat().st_size == 0:
        raise FileNotFoundError(p)

integrated = pd.read_csv(INTEGRATED_PATH)
curve_bins = pd.read_csv(CURVE_PATH)
annotation = pd.read_csv(ANNOTATION_PATH)
manifest = pd.read_csv(INPUT_MANIFEST_PATH)

require_columns(integrated, [
    "compartment", "trajectory", "lineage", "gene", "module",
    "association_q", "start_end_q", "start_end_logFC", "late_minus_early",
    "donor_supported_fdr_0_05",
], "integrated")

annotation["gene"] = annotation["gene"].astype(str)
annotation["module"] = annotation["module"].astype(str).str.lower()
integrated["trajectory_robustness"] = integrated.apply(robustness_label, axis=1)
integrated["direction_concordant"] = integrated.apply(concordant_direction, axis=1)
integrated["association_sig"] = pd.to_numeric(integrated["association_q"], errors="coerce") < FDR_ALPHA
integrated["start_end_sig"] = pd.to_numeric(integrated["start_end_q"], errors="coerce") < FDR_ALPHA
integrated["donor_sig"] = integrated["donor_supported_fdr_0_05"].fillna(False).astype(bool)

print("PROJECT ROOT", PROJECT_ROOT)
print("OUTPUT", OUTPUT_DIR)
print(f"Integrated rows: {len(integrated):,}")
print(f"Complement genes: {annotation['gene'].nunique()}")
print(f"Lineages: {integrated[['compartment','trajectory','lineage']].drop_duplicates().shape[0]}")

# %% SHOWCASE SELECTION
cand = integrated.copy()
cand["abs_start_end_logFC"] = pd.to_numeric(cand["start_end_logFC"], errors="coerce").abs()
cand["minus_log10_association_q"] = -np.log10(pd.to_numeric(cand["association_q"], errors="coerce").clip(lower=1e-300))
cand["minus_log10_start_end_q"] = -np.log10(pd.to_numeric(cand["start_end_q"], errors="coerce").clip(lower=1e-300))
cand["eligible_showcase"] = (
    cand["association_sig"] & cand["start_end_sig"] &
    (cand["trajectory_robustness"] == "strong") & cand["direction_concordant"] &
    cand["start_end_logFC"].notna()
)
cand["display_score"] = (
    3.0 * cand["donor_sig"].astype(int) +
    1.0 * cand["direction_concordant"].astype(int) +
    np.minimum(cand["minus_log10_association_q"].fillna(0), 10) / 10 +
    np.minimum(cand["minus_log10_start_end_q"].fillna(0), 10) / 10 +
    np.minimum(cand["abs_start_end_logFC"].fillna(0), 6) / 6
)
cand = cand.sort_values(["eligible_showcase", "display_score", "abs_start_end_logFC"], ascending=[False, False, False])

selected_rows, traj_counts, used_genes, used_modules = [], {}, set(), set()
eligible = cand[cand["eligible_showcase"]].copy()
for _, r in eligible.iterrows():
    key = (r["compartment"], r["trajectory"])
    if len(selected_rows) >= N_SHOWCASE:
        break
    if r["gene"] in used_genes or r["module"] in used_modules or traj_counts.get(key, 0) >= 2:
        continue
    selected_rows.append(r)
    used_genes.add(r["gene"])
    used_modules.add(r["module"])
    traj_counts[key] = traj_counts.get(key, 0) + 1
for _, r in eligible.iterrows():
    key = (r["compartment"], r["trajectory"])
    if len(selected_rows) >= N_SHOWCASE:
        break
    if r["gene"] in used_genes or traj_counts.get(key, 0) >= 2:
        continue
    selected_rows.append(r)
    used_genes.add(r["gene"])
    traj_counts[key] = traj_counts.get(key, 0) + 1
selected = pd.DataFrame(selected_rows)
if selected.empty:
    raise RuntimeError("No showcase candidates passed the display criteria.")
selected["selection_reason"] = "association+start/end FDR<0.05; robust lineage; model/observed concordance; diversity-aware display ranking"
selected.to_csv(SOURCE_DIR / "04d_showcase_selection.csv", index=False)

print("\nSHOWCASE SELECTION")
print(selected[["compartment", "trajectory", "lineage", "gene", "module", "start_end_logFC", "donor_sig", "display_score"]].to_string(index=False))

# %% LOAD FROZEN TRAJECTORIES + COMPUTE READER-FACING UMAPS
trajectory_data: dict[tuple[str, str], dict] = {}
state_rows = []
path_rows = []

for comp, traj in TRAJECTORY_ORDER:
    mm = manifest[(manifest["compartment"].astype(str) == comp) & (manifest["trajectory"].astype(str) == traj)]
    if len(mm) != 1:
        raise RuntimeError(f"Expected one manifest row for {comp}/{traj}; found {len(mm)}")
    mr = mm.iloc[0]
    pt_df = pd.read_csv(mr["pseudotime_path"], index_col=0)
    wt_df = pd.read_csv(mr["cellweights_path"], index_col=0)
    pt_df.index = pt_df.index.astype(str)
    wt_df.index = wt_df.index.astype(str)
    if pt_df.index.tolist() != wt_df.index.tolist():
        raise RuntimeError(f"Pseudotime / weight order mismatch for {comp}/{traj}")

    lineages_df = pd.read_csv(mr["lineages_path"])
    lineages = lineages_df["lineage"].astype(str).tolist() if "lineage" in lineages_df.columns else lineages_df.iloc[:, 0].astype(str).tolist()
    if len(lineages) != pt_df.shape[1]:
        lineages = [f"Lineage{i+1}" for i in range(pt_df.shape[1])]

    ad_traj = sc.read_h5ad(TRAJECTORY_H5AD_PATHS[comp])
    ad_traj.obs_names = ad_traj.obs_names.astype(str)
    loc = pd.Index(ad_traj.obs_names).get_indexer(pt_df.index)
    if (loc < 0).any():
        raise RuntimeError(f"Missing trajectory metacells in {TRAJECTORY_H5AD_PATHS[comp]}: {int((loc < 0).sum())}")
    sub_ad = ad_traj[pt_df.index].copy()

    # display UMAP; uses frozen representation only for visualization
    plot_xy = infer_plot_umap(sub_ad, ["X_decipher_z", "X_scVI", "X_decipher_v"])

    obs = sub_ad.obs.copy()
    obs.index = pt_df.index

    identity_col = choose_obs_column(obs, [
        "dominant_subclass", "dominant_SubclassLevel3", "SubclassLevel3",
        "dominant_SubclassLevel2", "SubclassLevel2",
        "dominant_state", "state",
    ])
    identity = obs[identity_col].astype(str).fillna("Unknown") if identity_col else pd.Series([traj] * len(obs), index=obs.index)

    disease_col = choose_obs_column(obs, ["dominant_disease", "disease", "Disease", "disease_state"])
    disease_raw = obs[disease_col].astype(str).fillna("Unknown") if disease_col else pd.Series(["Unknown"] * len(obs), index=obs.index)
    disease_group = disease_raw.map(simplify_disease)

    pt = pt_df.apply(pd.to_numeric, errors="coerce").to_numpy(float)
    wt = wt_df.apply(pd.to_numeric, errors="coerce").to_numpy(float)
    pt_scaled = np.column_stack([scale01(pt[:, j], np.isfinite(wt[:, j]) & (wt[:, j] > 0)) for j in range(pt.shape[1])])

    lin_to_j = {lin: j for j, lin in enumerate(lineages)}
    # Choose an illustrative path by maximum weighted metacell support among
    # non-caution branches. This is a DISPLAY CHOICE, not trajectory inference.
    eligible_display = [lin for lin in lineages if lin not in CAUTION_LINEAGES.get((comp, traj), set())]
    eligible_display = eligible_display or lineages
    display_lineage = max(eligible_display, key=lambda lin: float(np.nansum(np.clip(wt[:, lin_to_j[lin]], 0, None))))
    paths = {}
    for lin, j in lin_to_j.items():
        path, mids, counts = lineage_path(plot_xy, pt[:, j], wt[:, j])
        paths[lin] = path
        for k, point in enumerate(path):
            path_rows.append({
                "compartment": comp,
                "trajectory": traj,
                "lineage": lin,
                "path_index": k,
                "umap1": point[0],
                "umap2": point[1],
                "trajectory_robustness": "caution" if lin in CAUTION_LINEAGES.get((comp, traj), set()) else "strong",
            })

    td = {
        "compartment": comp,
        "trajectory": traj,
        "plot_xy": plot_xy,
        "obs": obs,
        "metacells": pt_df.index.to_numpy(),
        "pt": pt,
        "pt_scaled": pt_scaled,
        "wt": wt,
        "lineages": lineages,
        "lineage_to_j": lin_to_j,
        "display_lineage": display_lineage,
        "paths": paths,
        "identity": identity.reset_index(drop=True),
        "identity_col": identity_col,
        "identity_palette": categorical_palette(identity),
        "disease_raw": disease_raw.reset_index(drop=True),
        "disease_group": disease_group.reset_index(drop=True),
        "disease_col": disease_col,
    }
    trajectory_data[(comp, traj)] = td

    for i, mc in enumerate(td["metacells"]):
        state_rows.append({
            "compartment": comp,
            "trajectory": traj,
            "metacell": mc,
            "umap1": plot_xy[i, 0],
            "umap2": plot_xy[i, 1],
            "identity": str(td["identity"].iloc[i]),
            "disease_group": str(td["disease_group"].iloc[i]),
        })

pd.DataFrame(state_rows).to_csv(SOURCE_DIR / "04d_source_umap_coordinates.csv", index=False)
pd.DataFrame(path_rows).to_csv(SOURCE_DIR / "04d_source_umap_lineage_paths.csv", index=False)

# %% LOAD EXPRESSION + DISEASE-STRATIFIED PSEUDOTIME CURVES
module_genes = {m: annotation.loc[annotation["module"] == m, "gene"].astype(str).tolist() for m in MODULE_ORDER}
needed_genes = sorted(set(annotation["gene"].astype(str)) | set(selected["gene"].astype(str)))
expr_source_rows = []
curve_source_rows = []
module_curve_rows = []
module_shape_rows = []

# choose one summary lineage per trajectory for module curves: most association-significant complement genes
module_lineage_lookup = {}
for key in TRAJECTORY_ORDER:
    comp, traj = key
    sub = integrated[(integrated["compartment"] == comp) & (integrated["trajectory"] == traj)].copy()
    line_sum = (
        sub.groupby("lineage", as_index=False)
        .agg(n_assoc_sig=("association_sig", "sum"), n_start_end_sig=("start_end_sig", "sum"), mean_abs_fc=("start_end_logFC", lambda x: pd.to_numeric(x, errors="coerce").abs().mean()))
        .sort_values(["n_assoc_sig", "n_start_end_sig", "mean_abs_fc"], ascending=[False, False, False])
    )
    module_lineage_lookup[key] = str(line_sum.iloc[0]["lineage"]) if len(line_sum) else trajectory_data[key]["lineages"][0]

for comp in sorted({x[0] for x in TRAJECTORY_ORDER}):
    ad = sc.read_h5ad(FULL_COUNT_PATHS[comp])
    ad.obs_names = ad.obs_names.astype(str)
    symbols = gene_symbol_series(ad.var)
    symbol_to_idx = {}
    for i, g in enumerate(symbols.astype(str)):
        symbol_to_idx.setdefault(g, i)
    avail = [g for g in needed_genes if g in symbol_to_idx]

    for key, td in trajectory_data.items():
        if key[0] != comp:
            continue
        idx = pd.Index(ad.obs_names).get_indexer(td["metacells"])
        if (idx < 0).any():
            raise RuntimeError(f"Missing full-count metacells for {key}: {int((idx < 0).sum())}")
        sub = ad[idx]
        if "donor_id" not in sub.obs.columns:
            raise RuntimeError(f"donor_id unavailable for donor-aware intervals in {key}")
        donors = sub.obs["donor_id"].astype(str).to_numpy()
        if np.any(np.isin(donors, ["nan", "None", "", "NA"])):
            raise RuntimeError(f"Missing donor_id in {key}; cannot construct donor-level intervals")
        counts = sub.layers["counts"] if "counts" in sub.layers else sub.X
        if not sp.issparse(counts):
            counts = sp.csr_matrix(counts)
        counts = counts.tocsr()
        lib = np.asarray(counts.sum(axis=1)).ravel()
        lib_safe = np.where(lib > 0, lib, 1.0)

        gene_expr = {}
        for gene in avail:
            raw = np.asarray(counts[:, symbol_to_idx[gene]].toarray()).ravel()
            gene_expr[gene] = np.log1p(raw / lib_safe * 1e4)
        td["gene_expr"] = gene_expr

        module_scores = {}
        for mod, genes in module_genes.items():
            gg = [g for g in genes if g in gene_expr]
            if not gg:
                module_scores[mod] = np.full(len(td["metacells"]), np.nan)
            else:
                module_scores[mod] = np.nanmean(np.vstack([gene_expr[g] for g in gg]), axis=0)
        td["module_scores"] = module_scores

        showcase_here = selected[(selected["compartment"] == key[0]) & (selected["trajectory"] == key[1])]
        disease_group = td["disease_group"].to_numpy(str)
        group_masks = {
            "All": np.ones(len(td["metacells"]), dtype=bool),
            "CKD": disease_group == "CKD",
            "AKI": disease_group == "AKI",
            "Control/Other": disease_group == "Control/Other",
        }

        # gene-level source and curves for showcase panels
        for _, r in showcase_here.iterrows():
            gene = str(r["gene"])
            lin = str(r["lineage"])
            if gene not in gene_expr:
                continue
            expr = gene_expr[gene]
            j = td["lineage_to_j"][lin]
            ps = td["pt_scaled"][:, j]
            wt = td["wt"][:, j]

            for i, mc in enumerate(td["metacells"]):
                expr_source_rows.append({
                    "compartment": key[0], "trajectory": key[1], "lineage": lin, "gene": gene,
                    "metacell": mc, "umap1": td["plot_xy"][i, 0], "umap2": td["plot_xy"][i, 1],
                    "pseudotime": ps[i], "lineage_weight": wt[i],
                    "disease_group": disease_group[i], "donor_id": donors[i], "expression_log1p_cp10k": expr[i],
                })

            for grp, gm in group_masks.items():
                cdf, stats = make_group_curve(expr, ps, wt, gm, donors)
                cdf["compartment"] = key[0]
                cdf["trajectory"] = key[1]
                cdf["lineage"] = lin
                cdf["gene"] = gene
                cdf["group"] = grp
                cdf["module"] = str(r["module"])
                cdf["start_end_logFC"] = r["start_end_logFC"]
                cdf["association_q"] = r["association_q"]
                cdf["start_end_q"] = r["start_end_q"]
                curve_source_rows.append(cdf)

        # module-level curves for one summary lineage per trajectory
        summary_lin = module_lineage_lookup[key]
        j = td["lineage_to_j"][summary_lin]
        ps = td["pt_scaled"][:, j]
        wt = td["wt"][:, j]
        for mod in MODULE_ORDER:
            expr = td["module_scores"][mod]
            cdf, stats = make_group_curve(expr, ps, wt, group_masks["All"], donors)
            cdf["compartment"] = key[0]
            cdf["trajectory"] = key[1]
            cdf["lineage"] = summary_lin
            cdf["module"] = mod
            module_curve_rows.append(cdf)
            module_shape_rows.append({
                "compartment": key[0],
                "trajectory": key[1],
                "lineage": summary_lin,
                "module": mod,
                **stats,
            })

pd.DataFrame(expr_source_rows).to_csv(SOURCE_DIR / "04d_source_showcase_expression_umap.csv", index=False)
curve_source = pd.concat(curve_source_rows, ignore_index=True) if curve_source_rows else pd.DataFrame()
curve_source.to_csv(SOURCE_DIR / "04d_source_showcase_pseudotime_curves.csv", index=False)
module_curve_source = pd.concat(module_curve_rows, ignore_index=True) if module_curve_rows else pd.DataFrame()
module_curve_source.to_csv(SOURCE_DIR / "04d_source_module_pseudotime_curves.csv", index=False)
pd.DataFrame(module_shape_rows).to_csv(SOURCE_DIR / "04d_source_module_curve_summary.csv", index=False)

# %% FIGURE 1 — TRAJECTORY UMAP ATLAS
fig, axes = plt.subplots(1, len(TRAJECTORY_ORDER), figsize=(16.8, 3.6), squeeze=False)
for ax, key in zip(axes.ravel(), TRAJECTORY_ORDER):
    td = trajectory_data[key]
    ax.scatter(td["plot_xy"][:, 0], td["plot_xy"][:, 1], s=DISPLAY_POINT_SIZE, color="#D5D5D5", alpha=0.42, linewidths=0, zorder=1)
    draw_paths(ax, td, highlight=td["display_lineage"], label_endpoints=True)
    clean_axes(ax)
    ax.set_title(f"{format_traj(*key)} · {td['display_lineage'].replace('Lineage', 'L')} highlighted", loc="left", fontweight="bold")
legend_handles = [
    Line2D([0], [0], marker="o", color="none", markerfacecolor="#A9A9A9", markeredgecolor="none", markersize=5.5, label="Metacell"),
    Line2D([0], [0], color="#285F9E", lw=1.8, label="Highlighted frozen lineage"),
    Line2D([0], [0], color="#BBBBBB", lw=0.9, label="Other lineages (context)"),
    Line2D([0], [0], marker="*", color="#111111", markerfacecolor="#111111", lw=0, markersize=8, label="Star: inferred root / early end"),
    Line2D([0], [0], marker="o", color=TRAJECTORY_COLORS[("PT", "PT")], markerfacecolor="white", lw=0, markersize=6.3, label="Open circle: late end"),
    Line2D([0], [0], marker=">", color=TRAJECTORY_COLORS[("PT", "PT")], markerfacecolor=TRAJECTORY_COLORS[("PT", "PT")], lw=0, markersize=7, label="Arrow: early → late pseudotime"),
]
fig.legend(handles=legend_handles, loc="lower center", bbox_to_anchor=(0.5, 0.11), ncol=3, frameon=False, title="How to read the map")
fig.suptitle("Trajectory architecture projected onto UMAP-style state-space maps", y=1.02, fontsize=13, fontweight="bold")
fig.tight_layout(rect=(0, 0.20, 1, 0.95))
add_figure_note(fig, FIGURE_TEXT["trajectory_umap_atlas"], bottom=0.018)
save_figure(fig, MAIN_DIR / "trajectory_umap_atlas")

# %% FIGURE 2 — DISEASE UMAP ATLAS
fig, axes = plt.subplots(1, len(TRAJECTORY_ORDER), figsize=(16.8, 3.6), squeeze=False)
for ax, key in zip(axes.ravel(), TRAJECTORY_ORDER):
    draw_disease_map(ax, trajectory_data[key], format_traj(*key))
disease_handles = [
    Line2D([0], [0], marker="o", color="none", markerfacecolor=DISEASE_COLORS[grp], markeredgecolor="none", markersize=6, label=grp)
    for grp in ["CKD", "AKI", "Control/Other", "Unknown"]
]
fig.legend(handles=disease_handles, loc="lower center", bbox_to_anchor=(0.5, 0.11), ncol=4, frameon=False, title="Metacell disease label")
fig.suptitle("CKD and AKI occupancy along the same frozen trajectories", y=1.02, fontsize=13, fontweight="bold")
fig.tight_layout(rect=(0, 0.20, 1, 0.95))
add_figure_note(fig, FIGURE_TEXT["trajectory_disease_atlas"], bottom=0.018)
save_figure(fig, MAIN_DIR / "trajectory_disease_atlas")

# %% FIGURE 3 — SHOWCASE COMPLEMENT GENE MAPS
ncols = 4
nrows = math.ceil(len(selected) / ncols)
fig, axes = plt.subplots(nrows, ncols, figsize=(14.0, 3.5 * nrows), squeeze=False)
for ax in axes.ravel():
    ax.set_visible(False)
for ax, (_, r) in zip(axes.ravel(), selected.iterrows()):
    ax.set_visible(True)
    key = (str(r["compartment"]), str(r["trajectory"]))
    td = trajectory_data[key]
    gene = str(r["gene"])
    lin = str(r["lineage"])
    vals = td["gene_expr"].get(gene, np.full(len(td["metacells"]), np.nan))
    title = f"{gene} · {format_traj(*key)}"
    draw_gene_map(ax, td, vals, lin, title, add_colorbar=True)
    aq = pd.to_numeric(pd.Series([r["association_q"]]), errors="coerce").iloc[0]
    sq = pd.to_numeric(pd.Series([r["start_end_q"]]), errors="coerce").iloc[0]
    lfc = pd.to_numeric(pd.Series([r["start_end_logFC"]]), errors="coerce").iloc[0]
    donor = "donor-supported" if bool(r["donor_sig"]) else "trajectory-only"
    ax.text(
        0.02, 0.02,
        f"{lin.replace('Lineage', 'L')}  ·  module: {r['module']}\nassociation q={aq:.2g}  ·  start/end q={sq:.2g}  ·  Δ={lfc:+.2f}\n{donor}",
        transform=ax.transAxes, ha="left", va="bottom", fontsize=6.3,
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="none", alpha=0.86),
    )
fig.suptitle("Where key complement genes sit along the renal trajectories", y=1.01, fontsize=13, fontweight="bold")
fig.tight_layout(rect=(0, 0.12, 1, 0.97))
add_figure_note(fig, FIGURE_TEXT["complement_gene_maps"], bottom=0.013)
save_figure(fig, MAIN_DIR / "complement_gene_maps")

# %% FIGURE 4 — SHOWCASE GENE PSEUDOTIME TRENDS WITH CKD / AKI
ncols = 2
nrows = math.ceil(len(selected) / ncols)
fig, axes = plt.subplots(nrows, ncols, figsize=(12.0, 2.8 * nrows), squeeze=False)
for ax in axes.ravel():
    ax.set_visible(False)
for ax, (_, r) in zip(axes.ravel(), selected.iterrows()):
    ax.set_visible(True)
    key = (str(r["compartment"]), str(r["trajectory"]))
    gene = str(r["gene"])
    lin = str(r["lineage"])
    sub = curve_source[(curve_source["compartment"] == key[0]) & (curve_source["trajectory"] == key[1]) & (curve_source["lineage"] == lin) & (curve_source["gene"] == gene)].copy()
    subtitle = f"{format_traj(*key)} · {lin.replace('Lineage', 'L')} · module: {r['module']}"
    draw_pseudotime_gene_panel(ax, sub, gene, subtitle=subtitle)
    donor = "donor-supported" if bool(r["donor_sig"]) else "no donor confirmation"
    ax.text(0.995, 0.03, f"Δ start→end = {float(r['start_end_logFC']):+.2f}\n{donor}", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=6.4, color="#333333")
legend_handles = [
    Line2D([0], [0], color="#111111", lw=2.3, ls="-", label="All donors / available metacells"),
    Line2D([0], [0], color=DISEASE_COLORS["CKD"], lw=2.0, ls="-", label="CKD: solid blue"),
    Line2D([0], [0], color=DISEASE_COLORS["AKI"], lw=2.0, ls="--", label="AKI: dashed vermilion"),
    Line2D([0], [0], color=DISEASE_COLORS["Control/Other"], lw=1.2, ls="-.", label="Control / other: dash-dot"),
    mpl.patches.Patch(facecolor=DISEASE_COLORS["CKD"], alpha=RIBBON_ALPHA, edgecolor="none", label="Ribbon = 95% donor-level t interval (≥5 donors)"),
]
fig.legend(handles=legend_handles, loc="lower center", bbox_to_anchor=(0.5, 0.11), ncol=3, frameon=False, title="How to read the curves")
fig.suptitle("How complement expression changes across pseudotime", y=1.02, fontsize=13, fontweight="bold")
fig.tight_layout(rect=(0, 0.16, 1, 0.97))
add_figure_note(fig, FIGURE_TEXT["complement_gene_pseudotime_showcase"], bottom=0.015)
save_figure(fig, MAIN_DIR / "complement_gene_pseudotime_showcase")

# %% FIGURE 5 — MODULE-LEVEL PSEUDOTIME SUMMARY
fig, axes = plt.subplots(len(TRAJECTORY_ORDER), 1, figsize=(9.0, 2.1 * len(TRAJECTORY_ORDER)), squeeze=False)
for ax, key in zip(axes.ravel(), TRAJECTORY_ORDER):
    sub = module_curve_source[(module_curve_source["compartment"] == key[0]) & (module_curve_source["trajectory"] == key[1])].copy()
    chosen_lineage = module_lineage_lookup[key]
    for mod in MODULE_ORDER:
        ss = sub[sub["module"] == mod].copy()
        if ss.empty:
            continue
        xs, ys = smooth_series(ss["pseudotime_midpoint"].to_numpy(float), ss["weighted_mean_expression"].to_numpy(float))
        ax.scatter(ss["pseudotime_midpoint"], ss["weighted_mean_expression"], s=8, color=MODULE_COLORS[mod], alpha=0.22, linewidths=0)
        if len(xs):
            ax.plot(xs, ys, color=MODULE_COLORS[mod], ls=MODULE_LINESTYLES[mod], lw=2.0 if mod != "cfhr" else 1.6, alpha=0.95, label=mod.capitalize())
            ax.text(xs[-1] + 0.01, ys[-1], mod.capitalize(), color=MODULE_COLORS[mod], fontsize=6.3, va="center",
                    path_effects=[pe.withStroke(linewidth=2.0, foreground="white")])
    ax.set_xlim(-0.02, 1.06)
    ax.axvspan(0.0, EARLY_Q, color="#F7F7F7", alpha=0.9, zorder=0)
    ax.axvspan(LATE_Q, 1.0, color="#F7F7F7", alpha=0.9, zorder=0)
    ax.axvline(EARLY_Q, color="#D1D1D1", lw=0.8, ls="--")
    ax.axvline(LATE_Q, color="#D1D1D1", lw=0.8, ls="--")
    ax.grid(axis="y", color="#E7E7E7", lw=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_ylabel("Mean module\nexpression")
    ax.set_xlabel("Pseudotime (early → late)")
    ax.set_title(f"{format_traj(*key)} · summary lineage {chosen_lineage.replace('Lineage', 'L')}", loc="left", fontweight="bold")
fig.suptitle("Module-level complement programs across pseudotime", y=1.01, fontsize=13, fontweight="bold")
fig.tight_layout(rect=(0, 0.10, 1, 0.98))
add_figure_note(fig, FIGURE_TEXT["complement_module_pseudotime"], bottom=0.012)
save_figure(fig, MAIN_DIR / "complement_module_pseudotime")

# %% SUPPLEMENT / COMPONENTS
# selected lineage pseudotime maps
for _, r in selected.iterrows():
    key = (str(r["compartment"]), str(r["trajectory"]))
    td = trajectory_data[key]
    lin = str(r["lineage"])
    fig, ax = plt.subplots(figsize=(4.2, 3.7))
    draw_pseudotime_map(ax, td, lin, f"{format_traj(*key)} · {lin.replace('Lineage', 'L')}", add_colorbar=True)
    fig.text(0.02, 0.02, "Dots = metacells; color = scaled pseudotime; opacity = lineage weight", fontsize=6.5)
    save_figure(fig, SUPPLEMENT_DIR / f"pseudotime_map_{sanitize(key[0])}_{sanitize(key[1])}_{sanitize(lin)}")

# standalone legends for inkscape assembly
fig, ax = plt.subplots(figsize=(7.6, 1.7))
ax.axis("off")
ax.legend(handles=legend_handles, loc="center", frameon=False, ncol=4, title="Pseudotime trend legend")
save_figure(fig, COMPONENT_DIR / "legend_pseudotime_trends")

fig, ax = plt.subplots(figsize=(9.0, 1.8))
ax.axis("off")
ax.legend(handles=legend_handles + disease_handles[:0], loc="center", frameon=False, ncol=4)
save_figure(fig, COMPONENT_DIR / "legend_general")

# more complete figure guide and interpretation text
summary_lines = [
    "trajectory_umap_atlas: " + FIGURE_TEXT["trajectory_umap_atlas"] + "\nInterpretation: use this figure to orient the reader. It shows where each trajectory lives in the cellular landscape, which way pseudotime runs, and how many branches are present.",
    "trajectory_disease_atlas: " + FIGURE_TEXT["trajectory_disease_atlas"] + "\nInterpretation: compare CKD and AKI occupancy along the same lineage scaffold. Colors describe which metacells occur in each region; donor-level association tests are shown separately.",
    "complement_gene_maps: " + FIGURE_TEXT["complement_gene_maps"] + "\nInterpretation: these panels answer where the strongest complement signals are located. Compare expression within each panel using its color scale; formal longitudinal change is not inferred from these maps.",
    "complement_gene_pseudotime_showcase: " + FIGURE_TEXT["complement_gene_pseudotime_showcase"] + "\nInterpretation: rising curves indicate increasing expression over pseudotime, falling curves indicate decreasing expression, and differences between CKD and AKI are descriptive and should not be claimed significant without a donor-aware contrast. Ribbons are between-donor descriptive intervals, not a formal CKD-versus-AKI test; gaps indicate insufficient donor support.",
    "complement_module_pseudotime: " + FIGURE_TEXT["complement_module_pseudotime"] + "\nInterpretation: use this figure to summarize whether entire complement programs move together across the trajectory. For example, late rises indicate higher normalized RNA expression, not direct evidence of complement protein activation.",
]
summary_lines.extend([
    "complement_integrated_evidence: The complete canonical complement universe across the frozen lineages. Each colored cell has both tradeSeq association and start/end FDR < 0.05; the hue is the observed, not modeled, late-minus-early log1p(CP10K) difference. Open ring means separate donor early/late confirmation. Gray is no joint support or unavailable, not absence of expression.",
    "complement_disease_de_context: Selected genes in existing donor-pseudobulk CKD/AKI contrasts from 03a as integrated by 04c08. This is a separate disease test, not a disease-specific pseudotime test. Figure is skipped with an audit record if source data or contrast columns are absent.",
    "disease_association_frozen: Original 04c05 donor-level pairwise disease comparisons using its reported endpoints and corrected p-values. It is not inferred from metacell color or plotted trajectory geometry.",
    "trajectory_robustness_donor_coverage: Original lineage-matched root/representation/DPT Spearman metrics, donor subsampling summaries, and donor coverage from 04c02–04c04. Panels are labeled with their original metric and are never pooled into a single score.",
    "RIBBON METHODS: Within each fixed pseudotime bin, first summarize metacells separately for each donor using frozen lineage weights. For >=5 contributing donors (each >=2 metacells), plot equal-donor mean and a descriptive Student-t 95% interval across donor means. If fewer donors contribute, plot pooled metacell mean with NO interval. Pseudotime is inferred and cross-sectional; ribbon overlap is not a disease interaction test. No new inferential model has been fitted.",
])
write_figure_summary(OUTPUT_DIR / "04d_figure_summary_and_interpretation.txt", summary_lines)

# exact source-data exports for figure reproduction
selected[["compartment", "trajectory", "lineage", "gene", "module", "start_end_logFC", "association_q", "start_end_q", "donor_sig", "display_score"]].to_csv(SOURCE_DIR / "04d_source_showcase_metadata.csv", index=False)
integrated.to_csv(SOURCE_DIR / "04d_source_integrated_table_copy.csv", index=False)
curve_bins.to_csv(SOURCE_DIR / "04d_source_curve_bins_copy.csv", index=False)
annotation.to_csv(SOURCE_DIR / "04d_source_annotation_copy.csv", index=False)


# %% INTEGRATED FROZEN EVIDENCE: 04c02–04c05 + 04c08 + existing 03a DE
# Optional source tables are audited individually. Absence = explicitly omitted,
# never interpreted as a negative result and never replaced by guessed values.
EVIDENCE_INPUTS = {
    "root_sensitivity": TRAJECTORY_DIR / "trajectory_root_sensitivity_lineages.csv",
    "representation_sensitivity": TRAJECTORY_DIR / "trajectory_representation_sensitivity_lineages.csv",
    "dpt_comparison": TRAJECTORY_DIR / "trajectory_dpt_lineage_comparison.csv",
    "donor_subsampling": TRAJECTORY_DIR / "trajectory_donor_subsampling_lineage_stability.csv",
    "donor_lineage_coverage": TRAJECTORY_DIR / "trajectory_donor_lineage_coverage.csv",
    "disease_pairwise": TRAJECTORY_DIR / "trajectory_disease_pairwise.csv",
    "disease_group_summary": TRAJECTORY_DIR / "trajectory_disease_group_summary.csv",
    "disease_assay_adjusted": TRAJECTORY_DIR / "trajectory_disease_assay_adjusted.csv",
    "disease_assay_concordance": TRAJECTORY_DIR / "trajectory_disease_assay_concordance.csv",
    "existing_disease_de": COMPLEMENT_DIR / "complement_existing_disease_de.csv",
    "fit_test_audit": COMPLEMENT_DIR / "complement_fit_test_audit.csv",
    "test_availability": COMPLEMENT_DIR / "complement_tradeseq_test_availability.csv",
}
evidence={}; audit=[]
for name,path in EVIDENCE_INPUTS.items():
    if path.exists() and path.stat().st_size:
        try:
            df=pd.read_csv(path)
            evidence[name]=df
            status="loaded" if len(df) else "empty"
            if len(df): df.to_csv(SOURCE_DIR/f"04d_evidence_{name}.csv",index=False)
        except Exception as e:
            status=f"read_error: {type(e).__name__}: {e}"
            evidence[name]=pd.DataFrame()
    else:
        status="missing"; evidence[name]=pd.DataFrame()
    audit.append(dict(source=name,path=str(path),status=status,rows=len(evidence[name]),
                      columns=";".join(evidence[name].columns.astype(str))))
pd.DataFrame(audit).to_csv(SOURCE_DIR/"04d_evidence_input_audit.csv",index=False)

# Figure 6: actual frozen tradeSeq evidence. Gray cells = no joint FDR support,
# not zero effect. Dot = separate donor early/late confirmation from 04c08.
lineage_keys=integrated[["compartment","trajectory","lineage"]].drop_duplicates().sort_values(
    ["compartment","trajectory","lineage"]).reset_index(drop=True)
col_ids=[(str(r.compartment),str(r.trajectory),str(r.lineage)) for r in lineage_keys.itertuples(index=False)]
genes=annotation["gene"].astype(str).tolist()
values=np.full((len(genes),len(col_ids)),np.nan)
supported=np.zeros_like(values,dtype=bool); donor_sig=np.zeros_like(values,dtype=bool)
for i,g in enumerate(genes):
    for j,(c,t,l) in enumerate(col_ids):
        sub=integrated[(integrated.gene==g)&(integrated.compartment==c)&(integrated.trajectory==t)&(integrated.lineage==l)]
        if len(sub)!=1:continue
        r=sub.iloc[0]
        val=pd.to_numeric(pd.Series([r["late_minus_early"]]),errors="coerce").iloc[0]
        values[i,j]=val
        supported[i,j]=bool(r["association_sig"] and r["start_end_sig"])
        donor_sig[i,j]=bool(r["donor_sig"])
show=np.where(supported,values,np.nan)
lim=max(0.05,float(np.nanquantile(np.abs(show[np.isfinite(show)]),0.95))) if np.isfinite(show).any() else 1.
fig,ax=plt.subplots(figsize=(18,13))
cmap=mpl.colormaps["PuOr"].copy(); cmap.set_bad("#E7E7E7")
im=ax.imshow(show,aspect="auto",cmap=cmap,vmin=-lim,vmax=lim,interpolation="nearest")
for i,j in zip(*np.where(donor_sig & supported)):
    ax.scatter(j,i,facecolor="none",edgecolor="#101010",marker="o",s=19,linewidths=0.75)
ax.set_xticks(np.arange(len(col_ids)),[f"{c}/{t}\n{l.replace('Lineage','L')}" for c,t,l in col_ids],rotation=90,fontsize=6.5)
ax.set_yticks(np.arange(len(genes)),genes,fontsize=7)
ax.set_title("Complement expression change by lineage: frozen evidence",fontweight="bold",loc="left")
cb=fig.colorbar(im,ax=ax,fraction=0.022,pad=0.012);cb.set_label("Observed late − early mean log1p(CP10K)\n(only if association and start/end q < 0.05)")
fig.subplots_adjust(left=0.08,right=0.92,bottom=0.20,top=0.94)
fig.text(0.09,0.02,"Gray = no joint FDR support or unavailable (not zero expression). Open circle = donor early/late FDR < 0.05. "
         "Each lineage is an independent scaled pseudotime axis; displayed values are descriptive, not tradeSeq logFC.",fontsize=8)
save_figure(fig,MAIN_DIR/"complement_integrated_evidence")

# Figure 7: cross-context links are separate observational tests, never a combined p-value.
de=evidence["existing_disease_de"].copy()
if not de.empty and "gene" in de.columns:
    de=de[de["gene"].isin(set(selected["gene"]))].copy()
    de.to_csv(SOURCE_DIR/"04d_selected_existing_disease_de.csv",index=False)
    effect_col=next((x for x in ["log2FoldChange_shrunk","log2FoldChange"] if x in de),None)
    q_col=next((x for x in ["padj","deseq2_q_value"] if x in de),None)
    comp_col=next((x for x in ["comparison_id_join","comparison"] if x in de),None)
    if effect_col and comp_col:
        de["effect"] = pd.to_numeric(de[effect_col],errors="coerce")
        de["q"] = pd.to_numeric(de[q_col],errors="coerce") if q_col else np.nan
        fig,ax=plt.subplots(figsize=(9, max(3,0.42*len(de)+1.4)))
        ypos=np.arange(len(de))
        cols=["#0072B2" if x=="ckd_vs_normal" else "#D55E00" if x=="aki_vs_normal" else "#666666" for x in de[comp_col].astype(str)]
        ax.barh(ypos,de["effect"],color=cols,alpha=0.85,height=0.7)
        ax.axvline(0,color="#222222",lw=0.8)
        for idx,(_,r) in enumerate(de.iterrows()):
            if np.isfinite(r["q"]) and r["q"]<0.05:ax.scatter(r["effect"],idx,marker="*",s=48,color="#111111",zorder=5)
        ax.set_yticks(ypos,[f"{r['gene']} · {str(r[comp_col]).replace('_',' ')}" for _,r in de.iterrows()])
        ax.invert_yaxis();ax.set_xlabel("Existing donor-pseudobulk DE log2 fold change")
        ax.set_title("Disease-associated expression in the existing 03a contrasts",loc="left",fontweight="bold")
        ax.spines[["top","right"]].set_visible(False)
        fig.text(0.02,0.01,"Star: original DE FDR < 0.05. Disease DE and pseudotime association are separate tests; no joint significance is inferred.",fontsize=7)
        fig.tight_layout(rect=(0,0.035,1,1))
        save_figure(fig,MAIN_DIR/"complement_disease_de_context")

# Figure 8: frozen disease association. Use actual pairwise endpoints, not UMAP colors.
dp=evidence["disease_pairwise"].copy()
if not dp.empty and {"compartment","trajectory","lineage","endpoint","disease_1","disease_2","median_difference"}.issubset(dp):
    dp["contrast"]=dp.disease_1.astype(str)+" vs "+dp.disease_2.astype(str)
    dp["median_difference"]=pd.to_numeric(dp["median_difference"],errors="coerce")
    qname=next((c for c in ["pvalue_bh","pvalue"] if c in dp),None)
    endpoints=sorted(dp["endpoint"].dropna().astype(str).unique())
    fig,axes=plt.subplots(len(endpoints),1,figsize=(11,max(4,2.0*len(endpoints))),squeeze=False)
    for ax,endpoint in zip(axes.ravel(),endpoints):
        d=dp[dp.endpoint.astype(str)==endpoint].copy()
        # Small-multiples of all frozen contrasts, not filtering by significance.
        d=d.sort_values(["compartment","trajectory","lineage","contrast"])
        yy=np.arange(len(d));eff=d.median_difference.to_numpy(float)
        ax.scatter(eff,yy,c=[DISEASE_COLORS["CKD"] if "ckd" in x.lower() else DISEASE_COLORS["AKI"] if "aki" in x.lower() else "#777777" for x in d.contrast],s=17)
        if qname:
            q=pd.to_numeric(d[qname],errors="coerce").to_numpy(float)
            sig=np.isfinite(q)&(q<.05)
            ax.scatter(eff[sig],yy[sig],marker="o",facecolor="none",edgecolor="black",s=39,lw=.8)
        ax.axvline(0,c="#444444",lw=.8)
        ax.set_yticks(yy,[f"{r.compartment}/{r.trajectory} {r.lineage.replace('Lineage','L')} · {r.contrast}" for r in d.itertuples(index=False)],fontsize=5.8)
        ax.invert_yaxis();ax.set_xlabel("Frozen donor-level median difference (disease 1 − disease 2)")
        ax.set_title(endpoint.replace("_"," "),loc="left",fontweight="bold")
        ax.spines[["top","right"]].set_visible(False)
    fig.suptitle("Donor-level disease comparisons along frozen trajectories",fontweight="bold")
    fig.text(.01,.003,f"Open ring: {qname} < 0.05. Endpoints retain original 04c05 definitions; differences are observational.",fontsize=7)
    fig.tight_layout(rect=(0,.025,1,.97))
    save_figure(fig,SUPPLEMENT_DIR/"disease_association_frozen")

# Supplement robustness: show each metric with its real name and provenance.
metrics=[("Root sensitivity","root_sensitivity","primary_lineage","spearman_rho"),
         ("Representation sensitivity","representation_sensitivity","primary_lineage","spearman_rho"),
         ("DPT comparison","dpt_comparison","lineage","spearman_rho"),
         ("Donor subsampling","donor_subsampling","primary_lineage","spearman_median"),
         ("Donor lineage coverage","donor_lineage_coverage","lineage","fraction_donors_positive_lineage_support")]
valid=[]
for title,name,lc,metric in metrics:
    df=evidence[name]
    if {"compartment","trajectory",lc,metric}.issubset(df.columns) and len(df):
        d=df.copy();d[metric]=pd.to_numeric(d[metric],errors="coerce")
        d["label"]=d.compartment.astype(str)+"/"+d.trajectory.astype(str)+" · "+d[lc].astype(str)
        valid.append((title,name,metric,d.dropna(subset=[metric])))
if valid:
    fig,axes=plt.subplots(len(valid),1,figsize=(12,2.7*len(valid)),squeeze=False)
    for ax,(title,name,metric,d) in zip(axes.ravel(),valid):
        order=d.groupby("label")[metric].median().sort_values().index
        for j,label in enumerate(order):
            xx=d.loc[d.label==label,metric].to_numpy(float)
            ax.scatter(xx,np.full(len(xx),j),s=15,alpha=.42,c="#0072B2")
            ax.scatter(np.median(xx),j,marker="|",s=70,c="#111111",lw=1.5)
        ax.set_yticks(range(len(order)),order,fontsize=6)
        ax.set_xlabel(metric.replace("_"," "));ax.set_title(title,loc="left",fontweight="bold")
        ax.grid(axis="x",alpha=.2);ax.spines[["top","right"]].set_visible(False)
    fig.suptitle("Frozen trajectory sensitivity and donor coverage (separate metrics)",fontweight="bold")
    fig.tight_layout(rect=(0,0,1,.98))
    save_figure(fig,SUPPLEMENT_DIR/"trajectory_robustness_donor_coverage")

# Explicit audit separates omitted source data from negative evidence.
print("\nEVIDENCE AUDIT")
for rec in audit: print(f"  {rec['source']:<27} {rec['status']:<12} rows={rec['rows']}")


# %% CREATIVE FIGURES — COMPLEMENT PROGRAM RIVER / PULSE MAP / CKD–AKI DIVERGENCE
CREATIVE_N_BINS = 12
DIVERGENCE_TOP_GENES = 12

def simple_binned_curve(expr: np.ndarray, ps: np.ndarray, wt: np.ndarray,
                        group_mask: np.ndarray, n_bins: int = CREATIVE_N_BINS) -> pd.DataFrame:
    expr = np.asarray(expr, float)
    ps = np.asarray(ps, float)
    wt = np.asarray(wt, float)
    group_mask = np.asarray(group_mask, bool)
    edges = np.linspace(0, 1, n_bins + 1)
    supported = np.isfinite(expr) & np.isfinite(ps) & np.isfinite(wt) & (wt > 0) & group_mask
    records = []
    for b in range(n_bins):
        lo, hi = edges[b], edges[b + 1]
        mask = supported & (ps >= lo) & ((ps < hi) if b < n_bins - 1 else (ps <= hi))
        records.append({
            'pseudotime_bin': b + 1,
            'pseudotime_midpoint': (lo + hi) / 2,
            'weighted_mean_expression': weighted_mean(expr[mask], wt[mask]) if mask.any() else np.nan,
            'n_metacells': int(mask.sum()),
            'sum_lineage_weight': float(np.nansum(wt[mask])) if mask.any() else 0.0,
        })
    return pd.DataFrame(records)


def focus_lineage_for_key(key: tuple[str, str]) -> str:
    return str(module_lineage_lookup.get(key, trajectory_data[key]['display_lineage']))


def order_genes_for_pulse(key: tuple[str, str], lineage: str, restrict_to: list[str] | None = None) -> list[str]:
    sub = integrated[(integrated['compartment'] == key[0]) & (integrated['trajectory'] == key[1]) & (integrated['lineage'] == lineage)].copy()
    available = set(curve_bins[(curve_bins['compartment'] == key[0]) & (curve_bins['trajectory'] == key[1]) & (curve_bins['lineage'] == lineage)]['gene'].astype(str))
    if restrict_to is not None:
        available &= set(map(str, restrict_to))
    sub = sub[sub['gene'].astype(str).isin(available)].copy()
    if sub.empty:
        return []
    sub['module_rank'] = sub['module'].map({m: i for i, m in enumerate(MODULE_ORDER)}).fillna(99)
    sub['peak_bin_num'] = pd.to_numeric(sub['peak_bin'], errors='coerce').fillna(999)
    sub['abs_effect'] = pd.to_numeric(sub['late_minus_early'], errors='coerce').abs().fillna(0)
    sub['association_sig'] = sub['association_sig'].fillna(False)
    sub['start_end_sig'] = sub['start_end_sig'].fillna(False)
    sub['donor_sig'] = sub['donor_sig'].fillna(False)
    sub = sub.sort_values(
        ['module_rank', 'peak_bin_num', 'association_sig', 'start_end_sig', 'donor_sig', 'abs_effect', 'gene'],
        ascending=[True, True, False, False, False, False, True],
    )
    return sub['gene'].astype(str).tolist()


def top_divergence_genes(key: tuple[str, str], lineage: str, n: int = DIVERGENCE_TOP_GENES) -> list[str]:
    sub = integrated[(integrated['compartment'] == key[0]) & (integrated['trajectory'] == key[1]) & (integrated['lineage'] == lineage)].copy()
    if sub.empty:
        return []
    sub['abs_effect'] = pd.to_numeric(sub['late_minus_early'], errors='coerce').abs().fillna(0)
    sub['rank_score'] = (
        4.0 * sub['donor_sig'].fillna(False).astype(int) +
        2.0 * sub['association_sig'].fillna(False).astype(int) +
        2.0 * sub['start_end_sig'].fillna(False).astype(int) +
        np.minimum(sub['abs_effect'], 4.0)
    )
    sub = sub.sort_values(['rank_score', 'abs_effect', 'gene'], ascending=[False, False, True])
    out = []
    for gene in sub['gene'].astype(str):
        if gene in trajectory_data[key]['gene_expr'] and gene not in out:
            out.append(gene)
        if len(out) >= n:
            break
    return out


def draw_module_river_panel(ax, panel_df: pd.DataFrame, title: str, group: str):
    panel_df = panel_df[(panel_df['group'] == group)].copy()
    yvals = []
    for mod in MODULE_ORDER:
        ss = panel_df[panel_df['module'] == mod].copy().sort_values('pseudotime_midpoint')
        if ss['weighted_mean_expression'].notna().sum() < 2:
            continue
        ax.scatter(ss['pseudotime_midpoint'], ss['weighted_mean_expression'], s=8,
                   color=MODULE_COLORS[mod], alpha=0.18, linewidths=0, zorder=2)
        xs, ys = smooth_series(ss['pseudotime_midpoint'].to_numpy(float), ss['weighted_mean_expression'].to_numpy(float))
        if len(xs):
            ax.plot(xs, ys, color=MODULE_COLORS[mod], lw=1.85, ls=MODULE_LINESTYLES.get(mod, '-'), alpha=0.98, zorder=5)
            ax.fill_between(xs, ys, np.nanmin(ys) - 0.02, color=MODULE_COLORS[mod], alpha=0.03, zorder=1)
            ax.text(xs[-1] + 0.012, ys[-1], mod.capitalize(), fontsize=5.9, va='center', color=MODULE_COLORS[mod],
                    path_effects=[pe.withStroke(linewidth=2.2, foreground='white')])
            yvals.extend(np.asarray(ys, float).tolist())
    ax.set_xlim(-0.02, 1.10)
    if yvals:
        lo, hi = float(np.nanmin(yvals)), float(np.nanmax(yvals))
        pad = max(0.07, 0.10 * (hi - lo if hi > lo else 1.0))
        ax.set_ylim(lo - pad, hi + pad)
    ax.axvspan(0.0, EARLY_Q, color='#F7F7F7', alpha=0.9, zorder=0)
    ax.axvspan(LATE_Q, 1.0, color='#F7F7F7', alpha=0.9, zorder=0)
    ax.axvline(EARLY_Q, color='#CFCFCF', lw=0.8, ls='--', zorder=1)
    ax.axvline(LATE_Q, color='#CFCFCF', lw=0.8, ls='--', zorder=1)
    ax.text(0.01, 0.98, group, transform=ax.transAxes, ha='left', va='top', fontsize=7.2,
            color=DISEASE_COLORS.get(group, '#444444'), fontweight='bold')
    ax.set_xlabel('Scaled pseudotime (early → late)')
    ax.set_ylabel('Mean module\nexpression')
    ax.grid(axis='y', color='#E6E6E6', lw=0.7)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_title(title, loc='left', fontweight='bold')


def build_heatmap_matrix(sub_df: pd.DataFrame, genes_order: list[str], value_col: str) -> tuple[np.ndarray, list[str], list[int]]:
    if sub_df.empty or not genes_order:
        return np.zeros((0, 0)), [], []
    bins = sorted(pd.unique(pd.to_numeric(sub_df['pseudotime_bin'], errors='coerce').dropna().astype(int)))
    matrix = np.full((len(genes_order), len(bins)), np.nan)
    modules = []
    for i, gene in enumerate(genes_order):
        sg = sub_df[sub_df['gene'].astype(str) == gene].copy()
        modules.append(str(sg['module'].iloc[0]) if len(sg) and 'module' in sg.columns else 'other')
        vals = (sg.set_index(pd.to_numeric(sg['pseudotime_bin'], errors='coerce').astype('Int64'))
                  .reindex(bins)[value_col].to_numpy(float))
        matrix[i, :] = vals
    return matrix, modules, bins


def draw_pulse_heatmap(ax, matrix: np.ndarray, genes_order: list[str], modules: list[str], bins: list[int], title: str):
    if matrix.size == 0:
        ax.axis('off')
        return None
    scaled = np.full_like(matrix, np.nan, dtype=float)
    for i in range(matrix.shape[0]):
        row = matrix[i, :]
        ok = np.isfinite(row)
        if ok.any():
            lo, hi = np.nanmin(row[ok]), np.nanmax(row[ok])
            scaled[i, ok] = (row[ok] - lo) / (hi - lo) if hi > lo else 0.5
    cmap = mpl.colormaps['magma'].copy()
    cmap.set_bad('#F0F0F0')
    im = ax.imshow(scaled, aspect='auto', interpolation='nearest', cmap=cmap, vmin=0, vmax=1)
    ax.set_xticks(range(len(bins)), ['E' if i == 0 else 'L' if i == len(bins) - 1 else '' for i in range(len(bins))])
    ax.set_xlabel('Pseudotime bins (E = early, L = late)')
    ax.set_yticks(range(len(genes_order)), genes_order, fontsize=5.9)
    ax.set_title(title, loc='left', fontweight='bold')
    module_boundaries = []
    module_labels = []
    start = 0
    for i in range(1, len(modules) + 1):
        if i == len(modules) or modules[i] != modules[start]:
            module_boundaries.append(i - 0.5)
            module_labels.append((modules[start], (start + i - 1) / 2))
            start = i
    for y in module_boundaries[:-1]:
        ax.axhline(y, color='white', lw=0.8)
    for mod, ypos in module_labels:
        ax.text(1.01, ypos, str(mod).capitalize(), transform=ax.get_yaxis_transform(), ha='left', va='center',
                fontsize=5.8, color=MODULE_COLORS.get(str(mod), '#444444'))
    ax.tick_params(axis='both', length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    return im


def draw_divergence_heatmap(ax, matrix: np.ndarray, genes_order: list[str], modules: list[str], bins: list[int], title: str, limit: float):
    if matrix.size == 0:
        ax.axis('off')
        return None
    cmap = mpl.colormaps['PuOr'].copy()
    cmap.set_bad('#E7E7E7')
    im = ax.imshow(matrix, aspect='auto', interpolation='nearest', cmap=cmap, vmin=-limit, vmax=limit)
    ax.set_xticks(range(len(bins)), ['E' if i == 0 else 'L' if i == len(bins) - 1 else '' for i in range(len(bins))])
    ax.set_xlabel('Pseudotime bins (E = early, L = late)')
    ax.set_yticks(range(len(genes_order)), genes_order, fontsize=6.0)
    ax.set_title(title, loc='left', fontweight='bold')
    start = 0
    for i in range(1, len(modules) + 1):
        if i == len(modules) or modules[i] != modules[start]:
            if i < len(modules):
                ax.axhline(i - 0.5, color='white', lw=0.8)
            mod = modules[start]
            ypos = (start + i - 1) / 2
            ax.text(1.01, ypos, str(mod).capitalize(), transform=ax.get_yaxis_transform(), ha='left', va='center',
                    fontsize=5.9, color=MODULE_COLORS.get(str(mod), '#444444'))
            start = i
    ax.tick_params(axis='both', length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    return im


# Build source tables for the three new figure types.
river_records = []
divergence_records = []
pulse_subset_records = []
creative_focus_rows = []

for key in TRAJECTORY_ORDER:
    td = trajectory_data[key]
    lineage = focus_lineage_for_key(key)
    creative_focus_rows.append({'compartment': key[0], 'trajectory': key[1], 'lineage': lineage, 'figure_role': 'summary'})
    j = td['lineage_to_j'][lineage]
    ps = td['pt_scaled'][:, j]
    wt = td['wt'][:, j]
    disease = td['disease_group'].to_numpy(str)
    disease_masks = {'CKD': disease == 'CKD', 'AKI': disease == 'AKI'}

    for group, gm in disease_masks.items():
        for mod in MODULE_ORDER:
            expr = np.asarray(td['module_scores'][mod], float)
            cdf = simple_binned_curve(expr, ps, wt, gm, n_bins=CREATIVE_N_BINS)
            cdf['compartment'] = key[0]
            cdf['trajectory'] = key[1]
            cdf['lineage'] = lineage
            cdf['group'] = group
            cdf['module'] = mod
            river_records.append(cdf)

    # pulse map uses all genes available in the 04c08 descriptive curve table for the same lineage.
    pulse_genes = order_genes_for_pulse(key, lineage)
    creative_focus_rows.extend([
        {'compartment': key[0], 'trajectory': key[1], 'lineage': lineage, 'figure_role': 'pulse_gene', 'gene': g}
        for g in pulse_genes
    ])
    pulse_sub = curve_bins[(curve_bins['compartment'] == key[0]) & (curve_bins['trajectory'] == key[1]) & (curve_bins['lineage'] == lineage)].copy()
    pulse_sub = pulse_sub.merge(annotation[['gene', 'module']], on='gene', how='left')
    pulse_subset_records.append(pulse_sub)

    # divergence map uses a curated smaller set of genes.
    div_genes = top_divergence_genes(key, lineage)
    creative_focus_rows.extend([
        {'compartment': key[0], 'trajectory': key[1], 'lineage': lineage, 'figure_role': 'divergence_gene', 'gene': g}
        for g in div_genes
    ])
    for gene in div_genes:
        expr = td['gene_expr'].get(gene)
        if expr is None:
            continue
        ckd = simple_binned_curve(expr, ps, wt, disease_masks['CKD'], n_bins=CREATIVE_N_BINS)
        aki = simple_binned_curve(expr, ps, wt, disease_masks['AKI'], n_bins=CREATIVE_N_BINS)
        merged = ckd.merge(aki, on=['pseudotime_bin', 'pseudotime_midpoint'], how='outer', suffixes=('_ckd', '_aki'))
        merged['compartment'] = key[0]
        merged['trajectory'] = key[1]
        merged['lineage'] = lineage
        merged['gene'] = gene
        merged['module'] = str(annotation.set_index('gene').reindex([gene])['module'].iloc[0]) if gene in set(annotation['gene']) else np.nan
        merged['difference_ckd_minus_aki'] = pd.to_numeric(merged['weighted_mean_expression_ckd'], errors='coerce') - pd.to_numeric(merged['weighted_mean_expression_aki'], errors='coerce')
        divergence_records.append(merged)

river_df = pd.concat(river_records, ignore_index=True) if river_records else pd.DataFrame()
pulse_focus_df = pd.concat(pulse_subset_records, ignore_index=True) if pulse_subset_records else pd.DataFrame()
divergence_df = pd.concat(divergence_records, ignore_index=True) if divergence_records else pd.DataFrame()
creative_focus_df = pd.DataFrame(creative_focus_rows)

river_df.to_csv(SOURCE_DIR / '04d_source_complement_program_river.csv', index=False)
pulse_focus_df.to_csv(SOURCE_DIR / '04d_source_complement_pulse_map.csv', index=False)
divergence_df.to_csv(SOURCE_DIR / '04d_source_ckd_aki_divergence.csv', index=False)
creative_focus_df.to_csv(SOURCE_DIR / '04d_source_creative_figure_focus.csv', index=False)

# Figure 9: complement program river.
if not river_df.empty:
    fig, axes = plt.subplots(len(TRAJECTORY_ORDER), 2, figsize=(13.4, 2.2 * len(TRAJECTORY_ORDER)), squeeze=False)
    for row_idx, key in enumerate(TRAJECTORY_ORDER):
        lineage = focus_lineage_for_key(key)
        panel = river_df[(river_df['compartment'] == key[0]) & (river_df['trajectory'] == key[1]) & (river_df['lineage'] == lineage)].copy()
        draw_module_river_panel(axes[row_idx, 0], panel, f"{format_traj(*key)} · {lineage.replace('Lineage', 'L')}", 'CKD')
        draw_module_river_panel(axes[row_idx, 1], panel, f"{format_traj(*key)} · {lineage.replace('Lineage', 'L')}", 'AKI')
        if row_idx > 0:
            axes[row_idx, 0].set_title('')
            axes[row_idx, 1].set_title('')
    river_handles = [Line2D([0], [0], color=MODULE_COLORS[m], lw=1.9, ls=MODULE_LINESTYLES.get(m, '-'), label=m.capitalize()) for m in MODULE_ORDER]
    fig.legend(handles=river_handles, loc='lower center', bbox_to_anchor=(0.5, 0.045), ncol=4, frameon=False, title='Complement modules')
    fig.suptitle('Complement programs across pseudotime: CKD and AKI shown separately', y=0.997, fontsize=13, fontweight='bold')
    fig.text(0.02, 0.008,
             'Each row shows one representative lineage per frozen trajectory (selected from the strongest complement-associated lineage in that trajectory). '
             'Colored curves are descriptive module means across pseudotime. Compare CKD and AKI within the same row to see whether the timing or magnitude of module activation differs.',
             ha='left', va='bottom', fontsize=7.1)
    fig.tight_layout(rect=(0, 0.08, 1, 0.975))
    save_figure(fig, MAIN_DIR / 'complement_program_river')

# Figure 10: pulse map.
if not pulse_focus_df.empty:
    ncols = 2
    nrows = math.ceil(len(TRAJECTORY_ORDER) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(12.8, 3.1 * nrows), squeeze=False)
    axes_flat = axes.ravel()
    last_im = None
    for ax in axes_flat:
        ax.set_visible(False)
    for ax, key in zip(axes_flat, TRAJECTORY_ORDER):
        ax.set_visible(True)
        lineage = focus_lineage_for_key(key)
        sub = pulse_focus_df[(pulse_focus_df['compartment'] == key[0]) & (pulse_focus_df['trajectory'] == key[1]) & (pulse_focus_df['lineage'] == lineage)].copy()
        order = order_genes_for_pulse(key, lineage)
        mat, mods, bins = build_heatmap_matrix(sub, order, 'weighted_mean_log1p_cp10k')
        last_im = draw_pulse_heatmap(ax, mat, order, mods, bins, f"{format_traj(*key)} · {lineage.replace('Lineage', 'L')}")
    if last_im is not None:
        cbar = fig.colorbar(last_im, ax=[a for a in axes_flat if a.get_visible()], fraction=0.018, pad=0.01)
        cbar.set_label('Within-gene scaled expression\n0 = lower, 1 = higher')
    fig.suptitle('Complement pulse map: when each gene peaks along pseudotime', y=0.997, fontsize=13, fontweight='bold')
    fig.text(0.02, 0.012,
             'Rows are complement genes grouped by module. Color is row-scaled descriptive expression, so this figure emphasizes timing and shape rather than absolute magnitude. '
             'Early-hot rows peak near the beginning of pseudotime; late-hot rows peak near the end; mid-trajectory bands suggest transient activation.',
             ha='left', va='bottom', fontsize=7.1)
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    save_figure(fig, MAIN_DIR / 'complement_pulse_map')

# Figure 11: CKD–AKI divergence map.
if not divergence_df.empty:
    ncols = 2
    nrows = math.ceil(len(TRAJECTORY_ORDER) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(12.8, 2.65 * nrows), squeeze=False)
    axes_flat = axes.ravel()
    for ax in axes_flat:
        ax.set_visible(False)
    finite = pd.to_numeric(divergence_df['difference_ckd_minus_aki'], errors='coerce')
    limit = float(np.nanquantile(np.abs(finite[np.isfinite(finite)]), 0.95)) if np.isfinite(finite).any() else 1.0
    limit = max(limit, 0.05)
    last_im = None
    for ax, key in zip(axes_flat, TRAJECTORY_ORDER):
        ax.set_visible(True)
        lineage = focus_lineage_for_key(key)
        genes = top_divergence_genes(key, lineage)
        sub = divergence_df[(divergence_df['compartment'] == key[0]) & (divergence_df['trajectory'] == key[1]) & (divergence_df['lineage'] == lineage)].copy()
        sub = sub[sub['gene'].astype(str).isin(genes)].copy()
        sub['module'] = sub['module'].astype(str)
        gene_to_module = {g: str(sub.loc[sub['gene'].astype(str) == g, 'module'].dropna().iloc[0]) if (sub['gene'].astype(str) == g).any() else 'other' for g in genes}
        genes = sorted(genes, key=lambda g: ({m: i for i, m in enumerate(MODULE_ORDER)}.get(gene_to_module.get(g, 'other'), 99), genes.index(g)))
        mat, mods, bins = build_heatmap_matrix(sub, genes, 'difference_ckd_minus_aki')
        last_im = draw_divergence_heatmap(ax, mat, genes, mods, bins, f"{format_traj(*key)} · {lineage.replace('Lineage', 'L')}", limit=limit)
    if last_im is not None:
        cbar = fig.colorbar(last_im, ax=[a for a in axes_flat if a.get_visible()], fraction=0.018, pad=0.01)
        cbar.set_label('CKD − AKI descriptive expression\n(log1p(CP10K))')
    fig.suptitle('Where CKD and AKI diverge along pseudotime', y=0.997, fontsize=13, fontweight='bold')
    fig.text(0.02, 0.012,
             'Orange cells indicate higher descriptive expression in CKD than AKI at that pseudotime bin; purple cells indicate higher expression in AKI. '
             'Gray cells mean one or both disease groups lacked usable support in that bin. This is a descriptive visualization, not a formal interaction test.',
             ha='left', va='bottom', fontsize=7.1)
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    save_figure(fig, MAIN_DIR / 'ckd_aki_divergence_map')

summary_lines.extend([
    'complement_program_river: Pathway-level summary of complement behavior along pseudotime. Each row is one representative lineage per trajectory; left and right columns separate CKD and AKI. Interpretation: compare module timing and amplitude within a row to see whether the overall complement program differs between disease contexts.',
    'complement_pulse_map: Gene-by-pseudotime heatmap using within-gene scaling. Interpretation: this figure answers when each complement gene is highest along the lineage. It is especially useful for identifying early, late, and transient waves of complement expression.',
    'ckd_aki_divergence_map: Descriptive CKD-minus-AKI heatmap for top informative genes in each representative lineage. Interpretation: positive colors mean higher expression in CKD at that pseudotime position, negative colors mean higher expression in AKI. These panels suggest where disease-context differences are concentrated, but they do not by themselves establish a formal differential pseudotime interaction.',
])
write_figure_summary(OUTPUT_DIR / '04d_figure_summary_and_interpretation.txt', summary_lines)


# %% CREATIVE FIGURES II — LINEAGE FINGERPRINTS / CONSTELLATION / EARLY→LATE DUMBBELLS

def module_lineage_fingerprint_table(integrated_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (comp, traj, lin, mod), sub in integrated_df.groupby(['compartment', 'trajectory', 'lineage', 'module'], observed=True):
        sig = sub['association_sig'].fillna(False) & sub['start_end_sig'].fillna(False)
        vals = pd.to_numeric(sub.loc[sig, 'late_minus_early'], errors='coerce').dropna()
        donor = sub.loc[sig, 'donor_sig'].fillna(False)
        rows.append({
            'compartment': comp,
            'trajectory': traj,
            'lineage': lin,
            'module': mod,
            'n_genes_total': int(len(sub)),
            'n_genes_joint_fdr': int(sig.sum()),
            'fraction_joint_fdr': float(sig.mean()) if len(sub) else np.nan,
            'median_late_minus_early': float(vals.median()) if len(vals) else np.nan,
            'mean_abs_late_minus_early': float(vals.abs().mean()) if len(vals) else np.nan,
            'n_donor_supported': int(donor.sum()) if len(donor) else 0,
        })
    return pd.DataFrame(rows)


def draw_fingerprint_panel(ax, df: pd.DataFrame, key: tuple[str, str], limit: float):
    sub = df[(df['compartment'] == key[0]) & (df['trajectory'] == key[1])].copy()
    if sub.empty:
        ax.axis('off')
        return None
    lineages = sorted(sub['lineage'].astype(str).unique(), key=lineage_number)
    matrix = np.full((len(lineages), len(MODULE_ORDER)), np.nan)
    frac = np.zeros_like(matrix)
    donors = np.zeros_like(matrix)
    for i, lin in enumerate(lineages):
        for j, mod in enumerate(MODULE_ORDER):
            r = sub[(sub['lineage'].astype(str) == lin) & (sub['module'].astype(str) == mod)]
            if r.empty:
                continue
            matrix[i, j] = pd.to_numeric(r['median_late_minus_early'], errors='coerce').iloc[0]
            frac[i, j] = pd.to_numeric(r['fraction_joint_fdr'], errors='coerce').fillna(0).iloc[0]
            donors[i, j] = pd.to_numeric(r['n_donor_supported'], errors='coerce').fillna(0).iloc[0]

    cmap = mpl.colormaps['PuOr_r'].copy()
    cmap.set_bad('#EEEEEE')
    # Draw each tile manually so support fraction can control saturation/alpha.
    for i in range(len(lineages)):
        for j in range(len(MODULE_ORDER)):
            val = matrix[i, j]
            if not np.isfinite(val):
                face = '#EEEEEE'; alpha = 1.0
            else:
                face = cmap(Normalize(-limit, limit)(np.clip(val, -limit, limit)))
                alpha = 0.25 + 0.75 * np.clip(frac[i, j], 0, 1)
            rect = mpl.patches.FancyBboxPatch(
                (j - 0.43, i - 0.34), 0.86, 0.68,
                boxstyle='round,pad=0.02,rounding_size=0.08',
                linewidth=0.6, edgecolor='#FFFFFF', facecolor=face, alpha=alpha,
            )
            ax.add_patch(rect)
            if donors[i, j] > 0:
                ax.scatter(j, i, marker='o', s=22, facecolor='none', edgecolor='#111111', linewidths=0.7, zorder=5)
            if frac[i, j] >= 0.50:
                ax.scatter(j + 0.31, i - 0.23, marker='s', s=8, color='#111111', linewidths=0, zorder=6)

    ax.set_xlim(-0.6, len(MODULE_ORDER) - 0.4)
    ax.set_ylim(len(lineages) - 0.5, -0.5)
    ax.set_xticks(range(len(MODULE_ORDER)), [m.capitalize() for m in MODULE_ORDER], rotation=35, ha='right', fontsize=6.2)
    ax.set_yticks(range(len(lineages)), [l.replace('Lineage', 'L') for l in lineages], fontsize=6.6)
    ax.set_title(format_traj(*key), loc='left', fontweight='bold')
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    return matrix


def build_constellation_edges(integrated_df: pd.DataFrame, max_edges: int = 46) -> pd.DataFrame:
    d = integrated_df.copy()
    d = d[d['association_sig'].fillna(False) & d['start_end_sig'].fillna(False)].copy()
    d['abs_effect'] = pd.to_numeric(d['late_minus_early'], errors='coerce').abs()
    d['support_score'] = (
        4.0 * d['donor_sig'].fillna(False).astype(int) +
        np.minimum(d['abs_effect'].fillna(0), 3.0) +
        np.minimum(-np.log10(pd.to_numeric(d['association_q'], errors='coerce').clip(lower=1e-300)), 8) / 8
    )
    d = d.sort_values(['support_score', 'abs_effect'], ascending=[False, False])
    # Preserve gene and trajectory diversity before filling remaining slots.
    chosen = []
    seen_gene = set(); seen_ctx = set()
    for idx, r in d.iterrows():
        ctx = (str(r['compartment']), str(r['trajectory']))
        g = str(r['gene'])
        if g not in seen_gene or ctx not in seen_ctx:
            chosen.append(idx); seen_gene.add(g); seen_ctx.add(ctx)
        if len(chosen) >= max_edges:
            break
    if len(chosen) < max_edges:
        for idx in d.index:
            if idx not in chosen:
                chosen.append(idx)
            if len(chosen) >= max_edges:
                break
    return d.loc[chosen].copy()


def draw_constellation(ax, edges: pd.DataFrame):
    if edges.empty:
        ax.axis('off'); return
    lineages = edges[['compartment', 'trajectory', 'lineage']].drop_duplicates().copy()
    lineages['label'] = lineages.apply(lambda r: f"{r['compartment']}\n{r['lineage'].replace('Lineage', 'L')}", axis=1)
    genes = sorted(edges['gene'].astype(str).unique())

    # Inner ring: lineage nodes, grouped by trajectory context.
    lin_rows = []
    start_angle = np.deg2rad(90)
    ctxs = TRAJECTORY_ORDER
    total_lin = max(len(lineages), 1)
    current = 0
    for key in ctxs:
        subset = lineages[(lineages['compartment'] == key[0]) & (lineages['trajectory'] == key[1])].copy()
        subset = subset.sort_values('lineage', key=lambda s: s.map(lineage_number))
        for _, r in subset.iterrows():
            angle = start_angle - 2 * np.pi * (current / total_lin)
            lin_rows.append((str(r['compartment']), str(r['trajectory']), str(r['lineage']), angle))
            current += 1
    lin_pos = {(c, t, l): (1.05 * np.cos(a), 1.05 * np.sin(a), a) for c, t, l, a in lin_rows}

    # Outer ring: genes, ordered by module then alphabetically.
    gene_meta = annotation.set_index('gene')['module'].to_dict()
    genes = sorted(genes, key=lambda g: (MODULE_ORDER.index(gene_meta.get(g, 'cfhr')) if gene_meta.get(g, 'cfhr') in MODULE_ORDER else 99, g))
    gene_pos = {}
    for i, g in enumerate(genes):
        a = start_angle - 2 * np.pi * (i / max(len(genes), 1))
        gene_pos[g] = (2.08 * np.cos(a), 2.08 * np.sin(a), a)

    max_effect = max(0.2, float(pd.to_numeric(edges['late_minus_early'], errors='coerce').abs().quantile(0.95)))
    for _, r in edges.iterrows():
        lk = (str(r['compartment']), str(r['trajectory']), str(r['lineage']))
        g = str(r['gene'])
        if lk not in lin_pos or g not in gene_pos:
            continue
        x0, y0, _ = lin_pos[lk]; x1, y1, _ = gene_pos[g]
        effect = float(pd.to_numeric(pd.Series([r['late_minus_early']]), errors='coerce').iloc[0])
        lw = 0.35 + 2.2 * min(abs(effect) / max_effect, 1.0)
        alpha = 0.26 + (0.42 if bool(r['donor_sig']) else 0.0)
        col = MODULE_COLORS.get(str(r['module']), '#777777')
        ax.plot([x0, x1], [y0, y1], color=col, lw=lw, alpha=min(alpha, 0.82), zorder=1)

    # lineage nodes
    for (c, t, l), (x, y, a) in lin_pos.items():
        col = TRAJECTORY_COLORS.get((c, t), '#666666')
        ax.scatter(x, y, s=86, color=col, edgecolor='white', linewidths=0.8, zorder=4)
        ax.text(x * 0.86, y * 0.86, l.replace('Lineage', 'L'), ha='center', va='center', fontsize=5.7,
                color='#111111', path_effects=[pe.withStroke(linewidth=2.5, foreground='white')], zorder=5)

    # gene nodes and labels
    for g, (x, y, a) in gene_pos.items():
        mod = gene_meta.get(g, 'cfhr')
        col = MODULE_COLORS.get(mod, '#777777')
        ax.scatter(x, y, s=43, color=col, edgecolor='white', linewidths=0.55, zorder=4)
        ha = 'left' if np.cos(a) >= 0 else 'right'
        ax.text(x + (0.065 if ha == 'left' else -0.065), y, g, ha=ha, va='center', fontsize=5.8, color=col,
                path_effects=[pe.withStroke(linewidth=2.2, foreground='white')], zorder=5)

    # central key
    ax.text(0, 0.15, 'Complement', ha='center', va='center', fontsize=11, fontweight='bold')
    ax.text(0, -0.05, 'gene ↔ lineage\nconstellation', ha='center', va='center', fontsize=7, color='#555555')
    ax.set_xlim(-2.65, 2.65); ax.set_ylim(-2.65, 2.65); ax.set_aspect('equal')
    ax.axis('off')


def draw_dumbbell_panel(ax, rows: pd.DataFrame, title: str):
    if rows.empty:
        ax.axis('off'); return
    rows = rows.copy()
    rows['early'] = pd.to_numeric(rows['early_mean_log1p_cp10k'], errors='coerce')
    rows['late'] = pd.to_numeric(rows['late_mean_log1p_cp10k'], errors='coerce')
    rows['delta'] = rows['late'] - rows['early']
    rows = rows.sort_values('delta')
    y = np.arange(len(rows))
    for i, (_, r) in enumerate(rows.iterrows()):
        col = MODULE_COLORS.get(str(r['module']), '#666666')
        ax.plot([r['early'], r['late']], [i, i], color=col, lw=2.0 if bool(r['donor_sig']) else 1.35,
                alpha=0.85, zorder=1)
        ax.scatter(r['early'], i, s=34, facecolor='white', edgecolor=col, linewidths=1.15, zorder=3)
        ax.scatter(r['late'], i, s=36, facecolor=col, edgecolor='white', linewidths=0.55, zorder=4)
        if bool(r['donor_sig']):
            ax.scatter(r['late'], i, s=67, facecolor='none', edgecolor='#111111', linewidths=0.8, zorder=5)
        # small arrowhead at late endpoint makes direction immediate
        direction = 1 if r['late'] >= r['early'] else -1
        ax.scatter(r['late'], i, marker='>' if direction > 0 else '<', s=17, color='#111111', zorder=6)
    labels = [f"{r.gene} · {r.lineage.replace('Lineage','L')}" for r in rows.itertuples(index=False)]
    ax.set_yticks(y, labels, fontsize=6.6)
    ax.axvline(0, color='#E0E0E0', lw=0.7, zorder=0)
    ax.set_xlabel('Expression log1p(CP10K)')
    ax.set_title(title, loc='left', fontweight='bold')
    ax.grid(axis='x', color='#ECECEC', lw=0.7)
    ax.spines[['top', 'right', 'left']].set_visible(False)
    ax.tick_params(axis='y', length=0)


# Figure 12: lineage complement fingerprints.
fingerprint_df = module_lineage_fingerprint_table(integrated)
fingerprint_df.to_csv(SOURCE_DIR / '04d_source_lineage_complement_fingerprint.csv', index=False)
finite_fp = pd.to_numeric(fingerprint_df['median_late_minus_early'], errors='coerce')
fp_limit = float(np.nanquantile(np.abs(finite_fp[np.isfinite(finite_fp)]), 0.95)) if np.isfinite(finite_fp).any() else 1.0
fp_limit = max(fp_limit, 0.05)
fig, axes = plt.subplots(2, 3, figsize=(13.6, 7.7), squeeze=False)
axes_flat = axes.ravel()
for ax in axes_flat:
    ax.set_visible(False)
for ax, key in zip(axes_flat, TRAJECTORY_ORDER):
    ax.set_visible(True)
    draw_fingerprint_panel(ax, fingerprint_df, key, fp_limit)
# shared colorbar
sm = mpl.cm.ScalarMappable(norm=Normalize(-fp_limit, fp_limit), cmap='PuOr_r')
cbar = fig.colorbar(sm, ax=[a for a in axes_flat if a.get_visible()], fraction=0.018, pad=0.018)
cbar.set_label('Median late − early expression\namong joint-FDR genes')
legend_fp = [
    Line2D([0], [0], marker='o', color='none', markerfacecolor='none', markeredgecolor='#111111', markersize=6, label='≥1 donor-supported gene'),
    Line2D([0], [0], marker='s', color='#111111', lw=0, markersize=4, label='≥50% of module genes jointly FDR-supported'),
]
fig.legend(handles=legend_fp, loc='lower center', bbox_to_anchor=(0.5, 0.035), ncol=2, frameon=False)
fig.suptitle('Complement fingerprints across all 26 frozen lineages', y=0.995, fontsize=13, fontweight='bold')
fig.text(0.02, 0.006,
         'Each rounded tile summarizes one complement module within one lineage. Hue indicates the median observed late-minus-early expression change among genes with both association and start/end FDR < 0.05; tile opacity reflects the fraction of module genes with joint support. Gray means no jointly supported genes.',
         fontsize=7.1, ha='left', va='bottom')
fig.tight_layout(rect=(0, 0.07, 1, 0.97))
save_figure(fig, MAIN_DIR / 'complement_lineage_fingerprints')

# Figure 13: complement–trajectory constellation.
constellation_edges = build_constellation_edges(integrated, max_edges=46)
constellation_edges.to_csv(SOURCE_DIR / '04d_source_complement_constellation_edges.csv', index=False)
fig, ax = plt.subplots(figsize=(10.5, 10.5))
draw_constellation(ax, constellation_edges)
module_handles = [Line2D([0], [0], marker='o', color='none', markerfacecolor=MODULE_COLORS[m], markeredgecolor='white', markersize=7, label=m.capitalize()) for m in MODULE_ORDER]
fig.legend(handles=module_handles, loc='lower center', bbox_to_anchor=(0.5, 0.015), ncol=4, frameon=False, title='Gene module / edge color')
fig.suptitle('Complement–trajectory constellation: strongest supported relationships', y=0.985, fontsize=13, fontweight='bold')
fig.text(0.04, 0.055,
         'Outer nodes are complement genes; inner nodes are frozen lineages. Edges are shown only for selected strongest gene–lineage relationships with both association and start/end FDR < 0.05. Edge width scales with absolute observed late-minus-early change; donor-supported relationships are more opaque. This display is filtered for readability and is not a complete network.',
         fontsize=7.1, ha='left', va='bottom', wrap=True)
fig.tight_layout(rect=(0, 0.09, 1, 0.96))
save_figure(fig, MAIN_DIR / 'complement_trajectory_constellation')

# Figure 14: early → late dumbbells for showcase findings.
dumbbell_df = selected.merge(
    integrated[['compartment','trajectory','lineage','gene','early_mean_log1p_cp10k','late_mean_log1p_cp10k','donor_sig']],
    on=['compartment','trajectory','lineage','gene'], how='left', suffixes=('', '_int')
)
if 'donor_sig_int' in dumbbell_df.columns:
    dumbbell_df['donor_sig'] = dumbbell_df['donor_sig'].fillna(dumbbell_df['donor_sig_int'])
dumbbell_df.to_csv(SOURCE_DIR / '04d_source_showcase_early_late_dumbbells.csv', index=False)
fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.2), squeeze=False)
left = dumbbell_df[dumbbell_df['compartment'].astype(str).isin(['PT','FIB'])].copy()
right = dumbbell_df[dumbbell_df['compartment'].astype(str).eq('Glomerular')].copy()
draw_dumbbell_panel(axes[0,0], left, 'Tubulointerstitial trajectories')
draw_dumbbell_panel(axes[0,1], right, 'Glomerular trajectories')
legend_db = [
    Line2D([0], [0], marker='o', color='#555555', markerfacecolor='white', markersize=6, lw=0, label='Early mean'),
    Line2D([0], [0], marker='o', color='#555555', markerfacecolor='#555555', markersize=6, lw=0, label='Late mean'),
    Line2D([0], [0], marker='o', color='#111111', markerfacecolor='none', markersize=8, lw=0, label='Outer ring = donor-supported'),
]
fig.legend(handles=legend_db, loc='lower center', bbox_to_anchor=(0.5, 0.035), ncol=3, frameon=False)
fig.suptitle('Early-to-late complement shifts in the showcase findings', y=0.99, fontsize=13, fontweight='bold')
fig.text(0.02, 0.008,
         'Open circles are early pseudotime means and filled circles are late pseudotime means. Connecting segments and arrowheads show direction of change. Color denotes complement module; an outer ring at the late endpoint indicates separate donor-level early/late FDR support.',
         fontsize=7.1, ha='left', va='bottom')
fig.tight_layout(rect=(0, 0.08, 1, 0.95))
save_figure(fig, MAIN_DIR / 'complement_showcase_early_late_dumbbells')

summary_lines.extend([
    'complement_lineage_fingerprints: Compact 26-lineage summary across complement modules. Hue encodes median observed late-minus-early change among genes with both trajectory association and start/end FDR support; opacity encodes the fraction of module genes jointly supported; open rings mark modules containing donor-supported genes.',
    'complement_trajectory_constellation: Radial, readability-filtered network of the strongest jointly FDR-supported gene-lineage relationships. Gene nodes are colored by complement module, lineage nodes by trajectory context, edge width by absolute descriptive late-minus-early change, and edge opacity by donor support. This is intentionally not exhaustive.',
    'complement_showcase_early_late_dumbbells: Early-versus-late descriptive expression for the selected showcase gene-lineage pairs. Open endpoint = early, filled endpoint = late, module color = biological program, outer ring = separate donor confirmation.',
])
write_figure_summary(OUTPUT_DIR / '04d_figure_summary_and_interpretation.txt', summary_lines)

# %% FINAL AUDIT
outputs = sorted(OUTPUT_DIR.rglob("*"))
fig_files = [p for p in outputs if p.suffix in {".svg", ".pdf", ".png"}]
source_files = [p for p in outputs if p.suffix == ".csv"]

print("\n" + "=" * 100)
print("04d TRAJECTORY + COMPLEMENT FIGURES COMPLETE")
print("=" * 100)
print(f"Trajectory UMAP panels: {len(TRAJECTORY_ORDER)}")
print(f"Showcase gene UMAP panels: {len(selected)}")
print(f"Showcase pseudotime panels: {len(selected)}")
print(f"Module-summary panels: {len(TRAJECTORY_ORDER)}")
print(f"Figure files: {len(fig_files)}")
print(f"Source-data CSVs: {len(source_files)}")
print("\nMain outputs:")
for p in [
    MAIN_DIR / "complement_integrated_evidence.svg",
    MAIN_DIR / "complement_disease_de_context.svg",
    MAIN_DIR / "trajectory_umap_atlas.svg",
    MAIN_DIR / "trajectory_disease_atlas.svg",
    MAIN_DIR / "complement_gene_maps.svg",
    MAIN_DIR / "complement_gene_pseudotime_showcase.svg",
    MAIN_DIR / "complement_module_pseudotime.svg",
    MAIN_DIR / "complement_program_river.svg",
    MAIN_DIR / "complement_pulse_map.svg",
    MAIN_DIR / "ckd_aki_divergence_map.svg",
    MAIN_DIR / "complement_lineage_fingerprints.svg",
    MAIN_DIR / "complement_trajectory_constellation.svg",
    MAIN_DIR / "complement_showcase_early_late_dumbbells.svg",
]:
    print(" ", p)
print("\nNotes:")
print(" - The displayed UMAP is a visualization layer only; frozen trajectories and tradeSeq results are unchanged.")
print(" - Pseudotime curves are descriptive summaries anchored to the frozen metacell expression and lineage weights.")
print(" - No trajectory, GAM, tradeSeq test, donor test, or DE model was refit or retuned.")
