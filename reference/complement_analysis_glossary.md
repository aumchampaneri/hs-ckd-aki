# Complement Cascade Analysis — Metric & Output Glossary

This glossary covers every score, metric, flag, and output file produced by the *Exploring the Role of the Complement Cascade in CKD and AKI* notebook. Entries are grouped by the section of the pipeline that produces them.

---

## 1. Input & Cell Metadata

| Field | Description |
|-------|-------------|
| `nFeature_RNA` | Number of unique genes detected per cell. A quality-control indicator: very low values suggest a dying/empty droplet; very high values can indicate doublets. |
| `nCount_RNA` | Total RNA counts (UMIs) per cell. Scales with sequencing depth and cell size. |
| `percent.mt` | Percentage of counts mapping to mitochondrial genes. High values (typically >20–30 %) indicate damaged or apoptotic cells and are used as a QC filter. |
| `donor_id` | Anonymised identifier for the human kidney donor/patient sample. |
| `disease` | Disease state of the donor: `normal`, `chronic kidney disease`, or `acute kidney failure`. |
| `library` | Sequencing library identifier, used together with `donor_id` to define a processing batch. |
| `batch` | Concatenation of `donor_id` and `library` (`donor_library`). Passed to scVI as the batch correction key so that technical variation between runs is removed during model training. |

---

## 2. Cell Annotation Layers

The notebook evaluates complement activity across eight nested annotation levels, from the broadest biological class down to individual cell states.

| Layer | Granularity | Example values |
|-------|-------------|----------------|
| `Class` | Broadest — major compartment | Epithelial, Immune, Stromal |
| `SubclassLevel1` | Major cell lineage | PT, TAL, EC, Myeloid, FIB |
| `SubclassLevel2` | Cell subtype | PT_S1, EC_arterial, Macro_resident |
| `SubclassLevel3` | Fine subtype | PC_CNT, IC_A, FIB_myofib |
| `SubclassLevel3_FullName` | Human-readable version of SubclassLevel3 | — |
| `cell_type` | Ontology-mapped cell type from CZ CELLxGENE | macrophage, podocyte, … |
| `CellStateLevel1` | Broad functional state | reference, injured, activated |
| `CellStateLevel2` | Fine-grained functional state | — |

---

## 3. eGFR-Derived Metrics

eGFR (estimated Glomerular Filtration Rate) quantifies how well the kidneys are filtering blood. The notebook converts categorical eGFR bins into two numeric values.

| Metric | Formula / Logic | Interpretation |
|--------|-----------------|----------------|
| `egfr_midpoint` | Mid-point of the reported eGFR bin (ml/min/1.73 m²); ">60" bins are assigned 65.0 | Numeric approximation of kidney filtration capacity. Higher = better kidney function. |
| `kidney_severity` | `max(0, (130 − egfr_midpoint) / 10)` | A continuous severity score derived from eGFR. **Higher = worse kidney function.** A value of 0 corresponds to ~normal eGFR (≥130); values around 6–13 correspond to severe CKD stages. Used throughout the analysis as the primary disease-severity axis. |

---

## 4. Complement Pathway Scores

Scores are computed with `sc.tl.score_genes` on the raw count matrix. Each score reflects the mean expression of a curated gene set relative to a background of randomly sampled control genes (Seurat-style gene-set scoring). **Positive values indicate above-background expression; negative values indicate below-background expression.**

| Score | Pathway | Key genes scored | Biological meaning |
|-------|---------|------------------|--------------------|
| `classical_score` | Classical pathway | C1QA, C1QB, C1QC, C1R, C1S, C2, C4A, C4B, C4BPA, C4BPB | Activation triggered by antibody–antigen complexes binding C1q. Relevant to autoimmune-mediated kidney injury. |
| `lectin_score` | Lectin pathway | MBL2, FCN1–3, MASP1–3 | Activation triggered by pattern recognition of carbohydrate structures on pathogens or damaged host cells. |
| `alternative_score` | Alternative pathway | C3, CFB, CFD, CFP, C3AR1 | Constitutively active at a low level; amplifies all complement activation. Major effector in CKD/AKI amplification loops. |
| `terminal_score` | Terminal / lytic pathway | C5, C5AR1/2, C6, C7, C8A/B/G, C9 | Formation of the Membrane Attack Complex (MAC), which lyses target cells. High terminal scores suggest direct cytolytic complement activity. |
| `receptor_score` | Complement receptors | C3AR1, C5AR1, C5AR2, CR1, CR2, ITGAM, ITGAX, VSIG4 | Responsiveness to complement fragments (C3a, C5a, iC3b). High receptor score = cell is positioned to *respond to* complement activation, not necessarily to produce it. |
| `regulator_score` | Complement regulators | CFH, CFHR1–5, CFI, CD46, CD55, CD59, SERPING1 | Proteins that dampen or limit complement activation. High regulator score = strong self-protective or suppressive complement tone. |

---

## 5. Derived Complement Summary Metrics

These are computed per-cell from the six pathway scores above.

| Metric | Calculation | Interpretation |
|--------|-------------|----------------|
| `production_score` | Mean of classical, lectin, alternative, terminal, and regulator scores | Reflects the overall complement *production* capacity of the cell, averaging activation and regulatory arms. |
| `response_score` | Equals `receptor_score` | How strongly a cell expresses complement receptors — i.e., how "listening" or reactive it is to complement fragments in the local environment. |
| `activation_index` | Mean(classical, lectin, alternative, terminal, receptor) − regulator_score | **The central complement activity metric.** Positive = complement activation programmes dominate regulatory programmes (net activated state). Negative = regulatory programmes dominate (net suppressed/protected state). |
| `net_complement_load` | Sum of all six pathway scores | Total complement transcriptional burden on the cell, regardless of direction. High values in either direction indicate heavy complement involvement. |
| `dominant_pathway` | `idxmax` across the six pathway scores | The single complement pathway with the highest score in that cell. Useful for characterising which arm of the cascade is most transcriptionally active. |

---

## 6. Population-Level Ranking Metrics

Computed by `build_layer_ranking()` for every cell-type population within each annotation layer. Each row in the ranking CSVs represents one population (e.g., "PT_S1 cells from all donors").

### Composition

| Column | Description |
|--------|-------------|
| `annotation_layer` | Which of the eight annotation layers this row belongs to. |
| `population` | Name of the cell population within that layer. |
| `n_cells` | Total cells in this population across all donors. |
| `n_donors` | Number of unique donors contributing cells to this population. |
| `cell_fraction` | Fraction of all cells in the dataset that belong to this population. |
| `mean_cells_per_donor` | Average number of cells contributed per donor. |
| `severity_bins_observed` | Number of distinct `kidney_severity` values seen across donors in this population — a proxy for how well the population samples the disease spectrum. |
| `egfr_midpoint_mean` | Mean eGFR midpoint across all donors in this population. |
| `kidney_severity_mean` | Mean kidney severity score across all donors. |

### Complement Activity

| Column | Description |
|--------|-------------|
| `<score>_mean` | Donor-averaged mean of each complement score (e.g., `activation_index_mean`, `terminal_score_mean`). |
| `<score>_sd` | Standard deviation of each complement score across donors. |
| `dominant_pathway` | The complement pathway with the highest mean score for this population. |
| `activation_direction` | `high_activation` if `activation_index_mean > 0`; `low_activation` if < 0; `neutral` otherwise. |

### Disease Association

| Column | Description |
|--------|-------------|
| `severity_rho` | Spearman correlation between donor-level `activation_index` and `kidney_severity`. Computed across donors within the population. Positive = complement activation rises as kidney function worsens. |
| `abs_severity_rho` | Absolute value of `severity_rho`. Used for ranking regardless of direction. |
| `severity_pval` | Two-sided p-value for `severity_rho`. |
| `severity_association_direction` | `positive` (activation rises with severity), `negative` (activation falls with severity), or `none_or_unknown`. |
| `severity_association_type` | Combined classification: `hyperactivation` (positive rho AND high activation), `positive_dysregulation` (positive rho but activation not elevated), `suppression` (negative rho), or `none_or_unknown`. |
| `ckd_vs_normal_delta` | Mean `activation_index` in CKD donors minus mean in normal donors. Positive = complement more activated in CKD. |
| `aki_vs_normal_delta` | Mean `activation_index` in AKI donors minus mean in normal donors. |
| `disease_dynamic_range` | Max minus min of mean `activation_index` across the three disease groups (normal, CKD, AKI). Reflects how much complement activity shifts across disease states regardless of direction. |

### Quality Filters

| Column | Description |
|--------|-------------|
| `passes_min_cells` | `True` if `n_cells ≥ 500`. Ensures sufficient statistical power. |
| `passes_min_donors` | `True` if `n_donors ≥ 5`. Ensures findings are not driven by a single individual. |
| `passes_severity_range` | `True` if `severity_bins_observed ≥ 2`. Ensures the population spans at least two disease-severity categories, making the severity correlation meaningful. |
| `passes_candidate_filters` | `True` only if all three filters above are satisfied. This is the primary inclusion criterion for downstream analysis. |

### Priority Scoring

| Column | Description |
|--------|-------------|
| `dysregulation_priority_score` | Composite z-score-based ranking metric. Computed as the sum of five z-scored components: (1) magnitude of `activation_index_mean`, (2) `abs_severity_rho`, (3) `disease_dynamic_range`, (4) log(1 + n_donors), (5) log(1 + n_cells). **Higher = stronger, better-supported complement dysregulation signal.** |
| `trajectory_priority_score` | Exact alias of `dysregulation_priority_score` retained for backward compatibility. |

---

## 7. Trajectory & Pseudotime Metrics

These are computed per cell inside `run_expanded_trajectory()` and `run_publication_decipher()`, then aggregated into group and donor summaries.

| Metric | Description |
|--------|-------------|
| `dpt_pseudotime` | Diffusion pseudotime (DPT) computed via `sc.tl.dpt`. A continuous value [0, 1] ordering cells along an inferred developmental or disease-progression trajectory, with 0 anchored at the homeostatic root cell. **This is not real time** — it reflects transcriptional distance from the root state. |
| `pseudotime_bin` | `dpt_pseudotime` quantile-binned into 5 equal-frequency bins (1 = most root-like, 5 = most progressed). Used to enable donor-level averaging that is more robust than using the continuous value directly. |
| `decipher_dpt_oriented` | The DPT value after orientation correction in the DECIPHER runs. The orientation is flipped if the Spearman correlation between raw DPT and the expected metric has the wrong sign, ensuring bin 1 consistently represents the biologically "earlier" state. |
| `decipher_bin` | Quantile bin (1–5) of `decipher_dpt_oriented`. Equivalent to `pseudotime_bin` but for the DECIPHER trajectories. |
| `trajectory_group` | The annotation-layer label assigned to each cell for grouping along the trajectory (e.g., `PT_S1`). Derived from the `group_col` field of each trajectory specification. |

### Trajectory Correlation Outputs (`*_correlations.csv`)

| Column | Description |
|--------|-------------|
| `cell_dpt_rho` | Spearman ρ between `dpt_pseudotime` and a given metric, computed across all individual cells in the trajectory. Sensitive but noisy. |
| `cell_dpt_pval` | p-value for `cell_dpt_rho`. |
| `donor_bin_rho` | Spearman ρ between `pseudotime_bin` and a given metric, computed across *donor × bin* averages. **This is the primary trend metric** — donor-averaging removes within-donor cell-sampling noise and pseudoreplication bias. |
| `donor_bin_pval` | p-value for `donor_bin_rho`. |
| `n_cells` / `n_donor_bins` / `n_donors` | Sample sizes used for the respective correlations. |
| `strongest_donor_metric` | (In scorecard) The metric with the largest absolute `donor_bin_rho` for that trajectory. |
| `strongest_abs_donor_rho` | The value of that largest absolute `donor_bin_rho`. |

---

## 8. DECIPHER-Specific Metrics

DECIPHER is a variational model (from scvi-tools) that disentangles a low-dimensional *program* space (`v`) from a residual cell-state space (`z`), enabling better separation of biological programmes from noise.

| Metric | Description |
|--------|-------------|
| `X_decipher_v` | Low-dimensional DECIPHER program embedding (dim = `dim_v`, default 3). Captures the major axes of transcriptional variation used to build the trajectory neighbourhood graph. |
| `X_decipher_z` | Residual latent embedding (dim = `dim_z`, default 16). Captures remaining cell-to-cell variation not explained by the programme axes. |
| `orientation_multiplier` | +1 or −1. Records whether the pseudotime axis was flipped during orientation correction, so that bin 1 always corresponds to the expected biological start (homeostatic/normal). |
| `expected_metric` | The complement metric pre-specified for each DECIPHER subset as the primary biological readout (e.g., `response_score` for PEC/POD, `terminal_score` for collecting duct PC). |
| `expected_direction` | +1 (metric expected to rise along the trajectory) or −1 (expected to fall). Used to guide orientation correction. |

---

## 9. Output Files Reference

### Main Ranking Tables (`outputs/scVI-tm/`)

| File | Contents |
|------|----------|
| `complement_activity_rankings_all_layers.csv` | All populations across all 8 annotation layers, sorted by filter status then priority score. |
| `complement_activity_rankings_<layer>.csv` | Same rankings filtered to a single annotation layer. |
| `complement_activity_donor_summaries_all_layers.csv` | Per-population, per-donor mean complement scores — the intermediate table used to compute population-level statistics and severity correlations. |
| `top_trajectory_candidates.csv` | Top 50 populations by `dysregulation_priority_score` that pass all quality filters. |
| `top_hyperactivation_candidates.csv` | Top 50 passing populations classified as `hyperactivation` or `positive_dysregulation`. |
| `top_suppression_candidates.csv` | Top 50 passing populations classified as `suppression`. |

### Expanded Trajectory Suite (`outputs/scVI-tm/expanded_trajectory_suite/`)

| File | Contents |
|------|----------|
| `expanded_trajectory_specs.csv` | Specification table listing all trajectory axes run (lineage-based and top ranked cell-type axes). |
| `expanded_trajectory_run_summary.csv` | One row per trajectory: cells available, cells used, number of groups, root cell ID. |
| `<name>_cells.csv` | Per-cell table with pseudotime, bin, group, complement scores, and metadata for one trajectory. |
| `<name>_groups.csv` | Per-group (e.g., per cell subtype) summary: mean pseudotime and mean complement scores. Groups sorted by pseudotime. |
| `<name>_donor_bins.csv` | Per-donor, per-pseudotime-bin averages of complement scores. The table used to compute `donor_bin_rho`. |
| `<name>_correlations.csv` | Cell-level and donor-bin-level Spearman correlations of each complement metric with pseudotime for one trajectory. |
| `expanded_trajectory_correlations.csv` | All correlation rows concatenated across every trajectory. |
| `expanded_trajectory_group_summaries.csv` | All group summaries concatenated across every trajectory. |
| `expanded_trajectory_scorecard.csv` | Pivoted summary table: trajectories × metrics, showing `donor_bin_rho` values. Sorted by strongest absolute correlation. |
| `expanded_trajectory_donor_trend_heatmap.png` | Heatmap of `donor_bin_rho` values across all trajectories and summary metrics. Diverging colour scale (blue = negative trend, red = positive trend). |
| `<name>_umap.png` | UMAP coloured by group label, disease, DPT pseudotime, and `activation_index` for one trajectory. |

### DECIPHER Publication Subsets (`outputs/scVI-tm/decipher_publication_subsets/`)

| File | Contents |
|------|----------|
| `publication_decipher_specs.csv` | Specification table for the six focal DECIPHER runs. |
| `publication_decipher_run_summary.csv` | Run metadata for all six DECIPHER models. |
| `<name>_cells.csv` | Per-cell table including DECIPHER latent coordinates (`decipher_v1–v3`), oriented pseudotime, and complement scores. |
| `<name>_group_summary.csv` | Per-group mean complement scores sorted by oriented pseudotime. |
| `<name>_disease_bin_summary.csv` | Mean complement scores per disease group × pseudotime bin (5 bins). The primary table for disease-stratified trend plots. |
| `<name>_donor_disease_bins.csv` | Donor-level averages per disease × pseudotime bin — used to compute donor-aware trend correlations. |
| `<name>_disease_trends.csv` | Spearman `donor_bin_rho` of key metrics vs. pseudotime bin, computed separately for each disease group and pooled ("all"). |
| `publication_decipher_disease_trends.csv` | All disease trend rows concatenated. |
| `publication_decipher_disease_bin_summary.csv` | All disease × bin summaries concatenated. |
| `publication_expected_metric_disease_trends.csv` | Filtered to only the `expected_metric` for each DECIPHER subset — the clearest summary of the primary biological finding per cell type. |
| `publication_expected_metric_disease_trend_heatmap.png/.pdf` | Annotated heatmap: rows = cell type subsets, columns = disease groups, values = `donor_bin_rho` of the expected metric. |
| `<name>_umap_overview.png/.pdf` | UMAP coloured by group, disease, oriented pseudotime, and expected metric. |
| `<name>_disease_metric_trends.png/.pdf` | Line plots of mean score vs. pseudotime bin, stratified by disease group, one panel per key metric. |
| `<name>_expected_metric_by_disease.png/.pdf` | Box + strip plot comparing the expected metric distribution across the three disease groups (ignoring pseudotime). |
| `<name>_model/` | Saved DECIPHER model weights for reproducibility and inference on new data. |

---

## 10. Quick Interpretation Guide

**What does a high `activation_index` mean?**
The cell is expressing more complement *activation* genes (classical, lectin, alternative, terminal, receptor pathways) than complement *regulatory* genes. This indicates the cell is in a pro-inflammatory, complement-active state.

**What does a positive `severity_rho` mean?**
As kidney function worsens (higher `kidney_severity`), complement activation in that cell population also increases. This is the signature of complement-driven disease progression.

**What is `severity_association_type = "hyperactivation"`?**
The population both has an elevated `activation_index` *and* that activation correlates positively with worsening kidney disease — the strongest indicator that complement dysregulation in that population is clinically relevant.

**What does `donor_bin_rho` measure in trajectory outputs?**
It captures how strongly a complement metric changes as cells progress along the pseudotime trajectory, after collapsing individual cells to donor-level bin averages. A large positive value means the metric rises consistently across donors as disease/differentiation progresses; a large negative value means it falls.

**How should I use `dysregulation_priority_score` to prioritise candidates?**
Sort by this score (descending) within `passes_candidate_filters == True`. Higher scores reflect cell populations where complement dysregulation is (a) large in magnitude, (b) strongly correlated with disease severity, (c) changes broadly across disease states, and (d) is well-supported by cell and donor numbers.
