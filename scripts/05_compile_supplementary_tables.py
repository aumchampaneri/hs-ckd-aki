# %% Supplementary Table Compilation
# > Packages existing analysis outputs into Supporting Tables S1-S6.
# > This script is packaging-only: it does not recompute statistics,
# > modify analysis outputs, or select results based on significance.
#
# > Run from the scripts/ directory, consistent with the rest of the pipeline:
# >     pixi run python 05_compile_supplementary_tables.py
#
# > Oversized data tables are written as compressed CSV files beside the workbook.
# > Requires xlsxwriter in the Pixi environment.

# %% PATH SETUP
from pathlib import Path

SCRIPT_DIR = Path.cwd()
PROJECT_DIR = SCRIPT_DIR.parent

OUTPUT_DIR = PROJECT_DIR / "supplementary_tables"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PSEUDOBULK_DIR = PROJECT_DIR / "outputs" / "pseudobulk_de"
SCVI_DIR = PROJECT_DIR / "outputs" / "scvi"
SCVI_VALIDATION_DIR = SCVI_DIR / "validation"
QC_DIR = PROJECT_DIR / "outputs" / "qc"

TRAJECTORY_DIR = PROJECT_DIR / "data" / "trajectory"
TRADESEQ_TEST_DIR = TRAJECTORY_DIR / "tradeseq" / "results" / "tests"
COMPLEMENT_TRAJECTORY_DIR = TRAJECTORY_DIR / "complement"

# %% IMPORTS
import pandas as pd

try:
    import xlsxwriter  # noqa: F401
except ImportError as exc:
    raise ImportError(
        "xlsxwriter is required to compile the supplementary workbooks. "
        "Add it to the Pixi environment and rerun this script."
    ) from exc


# %%
# CONFIGURATION & CONTROLS
# > Excel formatting and workbook-level metadata only.
# > No statistical-analysis parameters are defined in this script.
####
MAX_COLUMN_WIDTH = 45
MAX_README_WIDTH = 70
EXCEL_MAX_ROWS = 1_048_576

GLOBAL_PSEUDOBULK_COMPARISONS = [
    "ckd_vs_normal",
    "aki_vs_normal",
    "aki_vs_ckd",
]


# %%
# FILE HELPERS
# > Required files fail loudly. Optional files are included when present.
####
def require_file(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Required supplementary-table input is missing: {path}"
        )
    if path.stat().st_size == 0:
        raise RuntimeError(
            f"Required supplementary-table input is empty: {path}"
        )
    return path


def optional_file(path):
    path = Path(path)
    if path.exists() and path.stat().st_size > 0:
        return path
    return None


def read_csv(path, source_file=False):
    path = require_file(path)
    df = pd.read_csv(path, low_memory=False)

    if source_file:
        df.insert(0, "source_file", path.name)

    return df


def concat_csvs(paths, source_file=True):
    paths = sorted(
        {
            Path(path)
            for path in paths
            if Path(path).exists()
            and Path(path).stat().st_size > 0
        }
    )

    if not paths:
        return pd.DataFrame()

    frames = [
        read_csv(
            path,
            source_file=source_file,
        )
        for path in paths
    ]

    return pd.concat(
        frames,
        ignore_index=True,
        sort=False,
    )


def add_comparison_from_filename(df):
    if df.empty or "source_file" not in df.columns:
        return df

    out = df.copy()

    comparison_from_filename = (
        out["source_file"]
        .astype(str)
        .str.replace(
            r"^pseudobulk_de_",
            "",
            regex=True,
        )
        .str.replace(
            r"\.csv$",
            "",
            regex=True,
        )
    )

    # Current pseudobulk outputs already contain a comparison column.
    # Preserve it when present rather than inserting a duplicate. For older
    # outputs without that column, recover the comparison from the filename.
    if "comparison" in out.columns:
        existing = out["comparison"].astype("string")
        missing = existing.isna() | existing.str.strip().eq("")

        if missing.any():
            out.loc[missing, "comparison"] = comparison_from_filename.loc[missing]

        # Keep provenance-derived comparison adjacent to source_file while
        # retaining the values already written by the analysis script.
        comparison_values = out.pop("comparison")
        out.insert(
            1,
            "comparison",
            comparison_values,
        )
    else:
        out.insert(
            1,
            "comparison",
            comparison_from_filename,
        )

    return out


def safe_sheet_name(name):
    name = str(name)

    for char in "[]:*?/\\":
        name = name.replace(
            char,
            "_",
        )

    return name[:31]


def relative_source(path):
    path = Path(path)

    try:
        return str(
            path.relative_to(PROJECT_DIR)
        )
    except ValueError:
        return str(path)


# %%
# EXCEL FORMATTING HELPERS
####
def value_display_width(series, header):
    if len(series) == 0:
        return min(
            max(
                len(str(header)) + 2,
                10,
            ),
            MAX_COLUMN_WIDTH,
        )

    sample = (
        series
        .astype(str)
        .replace("nan", "")
        .head(5000)
    )

    width = (
        max(
            [len(str(header))]
            + [len(value) for value in sample]
        )
        + 2
    )

    return min(
        max(width, 10),
        MAX_COLUMN_WIDTH,
    )


def write_readme_sheet(
    writer,
    title,
    legend,
    sheet_records,
):
    workbook = writer.book
    worksheet = workbook.add_worksheet("README")
    writer.sheets["README"] = worksheet

    title_format = workbook.add_format(
        {
            "bold": True,
            "font_size": 14,
        }
    )

    section_format = workbook.add_format(
        {
            "bold": True,
            "font_size": 11,
        }
    )

    header_format = workbook.add_format(
        {
            "bold": True,
            "bg_color": "#D9EAF7",
            "border": 1,
            "valign": "top",
        }
    )

    wrap_format = workbook.add_format(
        {
            "text_wrap": True,
            "valign": "top",
        }
    )

    worksheet.write(
        0,
        0,
        title,
        title_format,
    )

    worksheet.write(
        2,
        0,
        "Legend",
        section_format,
    )

    worksheet.write(
        3,
        0,
        legend,
        wrap_format,
    )

    start_row = 6
    headers = [
        "sheet",
        "description",
        "unit_of_observation",
        "source",
    ]

    for column_index, header in enumerate(headers):
        worksheet.write(
            start_row,
            column_index,
            header,
            header_format,
        )

    for row_index, record in enumerate(
        sheet_records,
        start=start_row + 1,
    ):
        worksheet.write(
            row_index,
            0,
            record["sheet"],
            wrap_format,
        )
        worksheet.write(
            row_index,
            1,
            record["description"],
            wrap_format,
        )
        worksheet.write(
            row_index,
            2,
            record["unit"],
            wrap_format,
        )
        worksheet.write(
            row_index,
            3,
            record["source"],
            wrap_format,
        )

    worksheet.set_column(0, 0, 24)
    worksheet.set_column(1, 1, MAX_README_WIDTH)
    worksheet.set_column(2, 2, 28)
    worksheet.set_column(3, 3, MAX_README_WIDTH)
    worksheet.freeze_panes(
        start_row + 1,
        0,
    )


def write_dataframe_sheet(
    writer,
    sheet_name,
    df,
):
    sheet_name = safe_sheet_name(sheet_name)

    if len(df) + 1 > EXCEL_MAX_ROWS:
        raise RuntimeError(
            f"{sheet_name} has {len(df):,} rows and exceeds Excel's "
            f"{EXCEL_MAX_ROWS:,}-row limit."
        )

    df.to_excel(
        writer,
        sheet_name=sheet_name,
        index=False,
    )

    worksheet = writer.sheets[sheet_name]
    workbook = writer.book

    header_format = workbook.add_format(
        {
            "bold": True,
            "bg_color": "#D9EAF7",
            "border": 1,
            "text_wrap": True,
            "valign": "top",
        }
    )

    text_format = workbook.add_format(
        {
            "valign": "top",
        }
    )

    wrap_format = workbook.add_format(
        {
            "text_wrap": True,
            "valign": "top",
        }
    )

    scientific_format = workbook.add_format(
        {
            "num_format": "0.000E+00",
            "valign": "top",
        }
    )

    float_format = workbook.add_format(
        {
            "num_format": "0.0000",
            "valign": "top",
        }
    )

    for column_index, column in enumerate(df.columns):
        worksheet.write(
            0,
            column_index,
            column,
            header_format,
        )

        width = value_display_width(
            df[column],
            column,
        )

        lower = str(column).lower()
        column_format = text_format

        if any(
            token in lower
            for token in [
                "p_value",
                "pvalue",
                "q_value",
                "padj",
            ]
        ):
            column_format = scientific_format
            width = max(width, 14)

        elif pd.api.types.is_float_dtype(df[column]):
            column_format = float_format
            width = max(width, 12)

        elif width >= 35:
            column_format = wrap_format

        worksheet.set_column(
            column_index,
            column_index,
            width,
            column_format,
        )

    worksheet.freeze_panes(
        1,
        0,
    )

    if len(df.columns) > 0:
        worksheet.autofilter(
            0,
            0,
            max(len(df), 1),
            len(df.columns) - 1,
        )


def write_external_csv(
    output_path,
    sheet_name,
    df,
):
    output_path = Path(output_path)
    sheet_name = safe_sheet_name(sheet_name)

    csv_path = (
        output_path.parent
        / f"{output_path.stem}_{sheet_name}.csv.gz"
    )

    df.to_csv(
        csv_path,
        index=False,
        compression="gzip",
    )

    print(
        f"  {sheet_name}: {len(df):,} rows exceed Excel limit; "
        f"wrote {csv_path.name}"
    )

    return csv_path


def write_workbook(
    output_path,
    title,
    legend,
    sheets,
):
    output_path = Path(output_path)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    sheet_records = []

    with pd.ExcelWriter(
        output_path,
        engine="xlsxwriter",
        engine_kwargs={
            "options": {
                "strings_to_formulas": False,
                "nan_inf_to_errors": False,
            }
        },
    ) as writer:

        for (
            sheet_name,
            df,
            description,
            unit,
            source,
        ) in sheets:

            if df is None:
                continue

            if df.empty and len(df.columns) == 0:
                continue

            if len(df) + 1 > EXCEL_MAX_ROWS:
                csv_path = write_external_csv(
                    output_path,
                    sheet_name,
                    df,
                )

                sheet_records.append(
                    {
                        "sheet": f"External CSV: {csv_path.name}",
                        "description": (
                            description
                            + " Stored as a compressed CSV because the complete "
                            + "result exceeds Excel's row limit."
                        ),
                        "unit": unit,
                        "source": source,
                    }
                )
                continue

            write_dataframe_sheet(
                writer,
                sheet_name,
                df,
            )

            sheet_records.append(
                {
                    "sheet": safe_sheet_name(sheet_name),
                    "description": description,
                    "unit": unit,
                    "source": source,
                }
            )

        write_readme_sheet(
            writer,
            title,
            legend,
            sheet_records,
        )

    print(
        f"Wrote {output_path.name} "
        f"({len(sheet_records)} data sheets)"
    )


# %%
# S1 TABLE: COHORT, SOURCE METADATA, AND QC
####
def compile_s1():
    print("\n" + "=" * 80)
    print("S1 TABLE: COHORT, SOURCE METADATA, AND QC")
    print("=" * 80)

    sheets = [
        (
            "Donor_metadata",
            read_csv(
                PSEUDOBULK_DIR
                / "donor_cohort_table.csv"
            ),
            "One row per donor with disease assignment, available donor-level covariates, and total analyzed nuclei/cells.",
            "donor",
            "outputs/pseudobulk_de/donor_cohort_table.csv",
        ),
        (
            "Disease_summary",
            read_csv(
                PSEUDOBULK_DIR
                / "donor_cohort_summary_by_disease.csv"
            ),
            "Disease-group donor and cell/nucleus counts with covariate missingness summaries.",
            "disease group",
            "outputs/pseudobulk_de/donor_cohort_summary_by_disease.csv",
        ),
        (
            "Covariate_audit",
            read_csv(
                PSEUDOBULK_DIR
                / "covariate_audit.csv"
            ),
            "Audit of candidate donor-level adjustment variables and disease-group overlap.",
            "candidate covariate",
            "outputs/pseudobulk_de/covariate_audit.csv",
        ),
        (
            "Count_QC",
            read_csv(
                QC_DIR
                / "raw_count_summary.csv"
            ),
            "Raw count-matrix summary metrics.",
            "QC metric",
            "outputs/qc/raw_count_summary.csv",
        ),
    ]

    mad_path = optional_file(
        QC_DIR
        / "mad_threshold_suggestions.csv"
    )

    if mad_path is not None:
        sheets.append(
            (
                "MAD_thresholds",
                read_csv(mad_path),
                "Exploratory median-absolute-deviation QC threshold summaries.",
                "QC metric",
                relative_source(mad_path),
            )
        )

    legend = (
        "Source experiments, donor characteristics, clinical metadata, and "
        "quality-control summaries. Donor-level metadata and candidate "
        "adjustment variables are reported with disease-group sample sizes "
        "and missingness. Candidate covariates were audited for donor-level "
        "consistency and disease-group overlap before inferential modeling."
    )

    write_workbook(
        OUTPUT_DIR / "S1_Table.xlsx",
        "S1 Table",
        legend,
        sheets,
    )


# %%
# S2 TABLE: COMPLETE PSEUDOBULK DIFFERENTIAL EXPRESSION
####
def compile_s2():
    print("\n" + "=" * 80)
    print("S2 TABLE: COMPLETE PSEUDOBULK DIFFERENTIAL EXPRESSION")
    print("=" * 80)

    global_files = [
        require_file(
            PSEUDOBULK_DIR
            / f"pseudobulk_de_{comparison}.csv"
        )
        for comparison in GLOBAL_PSEUDOBULK_COMPARISONS
    ]

    global_de = add_comparison_from_filename(
        concat_csvs(global_files)
    )

    targeted_files = sorted(
        PSEUDOBULK_DIR.glob(
            "pseudobulk_de_targeted_SubclassLevel1_*.csv"
        )
    )

    targeted_de = add_comparison_from_filename(
        concat_csvs(targeted_files)
    )

    concordance_summary = concat_csvs(
        PSEUDOBULK_DIR.glob(
            "concordance_summary_*.csv"
        )
    )

    sheets = [
        (
            "Global_DE",
            global_de,
            "Complete global donor-level pseudobulk differential-expression results for the three prespecified disease contrasts.",
            "gene x comparison",
            "outputs/pseudobulk_de/pseudobulk_de_{comparison}.csv",
        ),
    ]

    if not targeted_de.empty:
        sheets.append(
            (
                "Targeted_DE",
                targeted_de,
                "Complete targeted SubclassLevel1 donor-level pseudobulk differential-expression results.",
                "gene x population x comparison",
                "outputs/pseudobulk_de/pseudobulk_de_targeted_SubclassLevel1_*.csv",
            )
        )

    if not concordance_summary.empty:
        sheets.append(
            (
                "Concordance_summary",
                concordance_summary,
                "Cross-contrast concordance statistics for eligible global differential-expression comparisons.",
                "comparison pair",
                "outputs/pseudobulk_de/concordance_summary_*.csv",
            )
        )

    optional_specs = [
        (
            "subclasslevel1_targeted_pseudobulk_specs.csv",
            "Targeted_specs",
            "Targeted DE contrasts that passed preflight eligibility.",
        ),
        (
            "subclasslevel1_targeted_pseudobulk_skipped_specs.csv",
            "Skipped_specs",
            "Targeted DE contrasts excluded during preflight and their reasons.",
        ),
        (
            "subclasslevel1_targeted_pseudobulk_run_records.csv",
            "Run_records",
            "Execution status for targeted pseudobulk contrasts.",
        ),
    ]

    for filename, sheet_name, description in optional_specs:
        path = optional_file(
            PSEUDOBULK_DIR
            / filename
        )

        if path is None:
            continue

        sheets.append(
            (
                sheet_name,
                read_csv(path),
                description,
                "analysis specification",
                relative_source(path),
            )
        )

    legend = (
        "Complete donor-level pseudobulk differential-expression results. "
        "Adjusted PyDESeq2 p-values are reported as false-discovery-rate "
        "q-values. Complete tested results are retained irrespective of "
        "statistical significance."
    )

    write_workbook(
        OUTPUT_DIR / "S2_Table.xlsx",
        "S2 Table",
        legend,
        sheets,
    )


# %%
# S3 TABLE: COMPLEMENT DEFINITIONS AND MODULE-LEVEL ANALYSES
####
def compile_s3():
    print("\n" + "=" * 80)
    print("S3 TABLE: COMPLEMENT DEFINITIONS AND MODULE-LEVEL ANALYSES")
    print("=" * 80)

    pathway_tests = read_csv(
        PSEUDOBULK_DIR
        / "pathway_level_group_tests.csv"
    )

    if "gene_set" in pathway_tests.columns:
        pathway_tests = pathway_tests.loc[
            pathway_tests["gene_set"]
            .astype(str)
            .eq("complement")
        ].copy()

    sheets = [
        (
            "Module_definitions",
            read_csv(
                COMPLEMENT_TRAJECTORY_DIR
                / "canonical_complement_gene_annotation.csv"
            ),
            "Canonical complement-gene annotation used for the final complement-focused analyses.",
            "gene",
            "data/trajectory/complement/canonical_complement_gene_annotation.csv",
        ),
        (
            "Gene_resolution",
            read_csv(
                SCVI_DIR
                / "complement_module_gene_resolution.csv"
            ),
            "Resolution of intended complement-module genes against the analyzed expression matrix.",
            "module x gene",
            "outputs/scvi/complement_module_gene_resolution.csv",
        ),
        (
            "Module_group_tests",
            pathway_tests,
            "Donor-level complement module/group tests from the pseudobulk workflow.",
            "module x comparison",
            "outputs/pseudobulk_de/pathway_level_group_tests.csv (complement rows)",
        ),
        (
            "Trajectory_tests",
            read_csv(
                COMPLEMENT_TRAJECTORY_DIR
                / "complement_tradeseq_tests_long.csv"
            ),
            "Complement-focused tradeSeq test results in long format.",
            "gene x trajectory x test",
            "data/trajectory/complement/complement_tradeseq_tests_long.csv",
        ),
        (
            "Trajectory_summary",
            read_csv(
                COMPLEMENT_TRAJECTORY_DIR
                / "complement_trajectory_summary.csv"
            ),
            "Summary of complement-associated transcription along trajectories.",
            "trajectory summary",
            "data/trajectory/complement/complement_trajectory_summary.csv",
        ),
        (
            "Integrated_results",
            read_csv(
                COMPLEMENT_TRAJECTORY_DIR
                / "complement_trajectory_integrated.csv"
            ),
            "Integrated complement trajectory, disease-DE, and trajectory-test results.",
            "gene x trajectory",
            "data/trajectory/complement/complement_trajectory_integrated.csv",
        ),
    ]

    optional_outputs = [
        (
            "complement_trajectory_priority_table.csv",
            "Priority_table",
            "Compact prioritized complement trajectory findings.",
        ),
        (
            "complement_fit_test_audit.csv",
            "Fit_test_audit",
            "Audit of complement genes against available fitted tradeSeq tests.",
        ),
        (
            "complement_tradeseq_test_availability.csv",
            "Test_availability",
            "Availability of tradeSeq tests for complement genes.",
        ),
        (
            "complement_pseudotime_shape_summary.csv",
            "Shape_summary",
            "Summary of complement expression shapes across pseudotime.",
        ),
    ]

    for filename, sheet_name, description in optional_outputs:
        path = optional_file(
            COMPLEMENT_TRAJECTORY_DIR
            / filename
        )

        if path is None:
            continue

        sheets.append(
            (
                sheet_name,
                read_csv(path),
                description,
                "analysis result",
                relative_source(path),
            )
        )

    legend = (
        "Canonical non-overlapping complement transcriptional module "
        "definitions and module-level analyses. CFHR-family genes are "
        "treated separately as descriptive genes rather than included in "
        "the directional regulator module. Transcript-level complement "
        "scores do not directly measure biochemical complement activation."
    )

    write_workbook(
        OUTPUT_DIR / "S3_Table.xlsx",
        "S3 Table",
        legend,
        sheets,
    )


# %%
# S4 TABLE: DONOR-LEVEL TRAJECTORY AND DISEASE ASSOCIATIONS
####
def compile_s4():
    print("\n" + "=" * 80)
    print("S4 TABLE: DONOR-LEVEL TRAJECTORY AND DISEASE ASSOCIATIONS")
    print("=" * 80)

    required_outputs = [
        (
            "Donor_trajectory",
            "trajectory_donor_level_summary.csv",
            "Donor-level weighted trajectory-position summaries.",
            "donor x trajectory x lineage",
        ),
        (
            "Lineage_coverage",
            "trajectory_donor_lineage_coverage.csv",
            "Donor support and coverage for inferred lineages.",
            "donor x lineage",
        ),
        (
            "Disease_omnibus",
            "trajectory_disease_omnibus.csv",
            "Donor-level omnibus disease-association tests.",
            "trajectory x lineage",
        ),
        (
            "Disease_pairwise",
            "trajectory_disease_pairwise.csv",
            "Donor-level pairwise disease-association tests.",
            "trajectory x lineage x contrast",
        ),
        (
            "Disease_group_summary",
            "trajectory_disease_group_summary.csv",
            "Disease-group summaries of donor-level trajectory quantities.",
            "trajectory x lineage x disease group",
        ),
        (
            "Assay_adjusted",
            "trajectory_disease_assay_adjusted.csv",
            "Assay-adjusted donor-level disease-association results where estimable.",
            "trajectory x lineage x model",
        ),
        (
            "Assay_concordance",
            "trajectory_disease_assay_concordance.csv",
            "Concordance and sensitivity summaries for disease estimates across assay structure.",
            "trajectory x lineage",
        ),
    ]

    sheets = []

    for (
        sheet_name,
        filename,
        description,
        unit,
    ) in required_outputs:
        path = require_file(
            TRAJECTORY_DIR
            / filename
        )

        sheets.append(
            (
                sheet_name,
                read_csv(path),
                description,
                unit,
                relative_source(path),
            )
        )

    optional_outputs = [
        (
            "Metadata_QC",
            "trajectory_donor_metadata_qc.csv",
            "QC of donor-level metadata consistency used for trajectory summaries.",
            "donor metadata field",
        ),
        (
            "Metadata_QC_summary",
            "trajectory_donor_metadata_qc_summary.csv",
            "Summary of donor metadata QC checks.",
            "QC summary",
        ),
        (
            "Assay_overlap",
            "trajectory_disease_assay_overlap.csv",
            "Disease-by-assay overlap audit for trajectory analyses.",
            "trajectory x disease x assay",
        ),
        (
            "Lineage_design",
            "trajectory_disease_lineage_design.csv",
            "Lineage-specific design eligibility for disease-association testing.",
            "trajectory x lineage",
        ),
    ]

    for (
        sheet_name,
        filename,
        description,
        unit,
    ) in optional_outputs:
        path = optional_file(
            TRAJECTORY_DIR
            / filename
        )

        if path is None:
            continue

        sheets.append(
            (
                sheet_name,
                read_csv(path),
                description,
                unit,
                relative_source(path),
            )
        )

    legend = (
        "Donor-level trajectory summaries and disease-association analyses. "
        "Donors, rather than individual metacells, are the biological "
        "inferential units for disease-association testing."
    )

    write_workbook(
        OUTPUT_DIR / "S4_Table.xlsx",
        "S4 Table",
        legend,
        sheets,
    )


# %%
# S5 TABLE: COMPLETE TRADESEQ RESULTS
####
def collect_tradeseq_test(test_name):
    paths = sorted(
        path
        for path in TRADESEQ_TEST_DIR.glob(
            f"*/{test_name}_combined.csv"
        )
        if path.is_file()
        and path.stat().st_size > 0
    )

    if not paths:
        raise FileNotFoundError(
            f"No {test_name}_combined.csv files found under "
            f"{TRADESEQ_TEST_DIR}"
        )

    frames = []

    for path in paths:
        df = read_csv(path)

        # Bundle directory names are generated from compartment/trajectory.
        # Preserve this provenance explicitly even if equivalent columns are
        # already present in the combined result.
        df.insert(
            0,
            "source_bundle",
            path.parent.name,
        )

        frames.append(df)

    return pd.concat(
        frames,
        ignore_index=True,
        sort=False,
    )


def compile_s5():
    print("\n" + "=" * 80)
    print("S5 TABLE: COMPLETE TRADESEQ RESULTS")
    print("=" * 80)

    test_specs = [
        (
            "association",
            "Association",
            "Primary tradeSeq associationTest results for expression changes along pseudotime.",
        ),
        (
            "start_vs_end",
            "Start_vs_end",
            "tradeSeq startVsEndTest results comparing fitted trajectory endpoints.",
        ),
        (
            "diff_end",
            "Differential_end",
            "Secondary tradeSeq diffEndTest branch-end comparisons.",
        ),
        (
            "pattern",
            "Pattern",
            "Secondary tradeSeq patternTest branch-pattern comparisons.",
        ),
    ]

    sheets = []

    for test_name, sheet_name, description in test_specs:
        sheets.append(
            (
                sheet_name,
                collect_tradeseq_test(test_name),
                description,
                "gene x trajectory x test contrast",
                f"data/trajectory/tradeseq/results/tests/*/{test_name}_combined.csv",
            )
        )

    summary_outputs = [
        (
            "trajectory_tradeseq_test_summary.csv",
            "Test_summary",
            "Per-trajectory tradeSeq fit/test summary.",
        ),
        (
            "trajectory_tradeseq_fdr_family_manifest.csv",
            "FDR_families",
            "Manifest of predefined Benjamini-Hochberg testing families.",
        ),
    ]

    for filename, sheet_name, description in summary_outputs:
        path = require_file(
            TRADESEQ_TEST_DIR
            / filename
        )

        sheets.append(
            (
                sheet_name,
                read_csv(path),
                description,
                "trajectory/test summary",
                relative_source(path),
            )
        )

    legend = (
        "Complete tradeSeq trajectory-associated expression results. "
        "associationTest is the primary test of expression variation with "
        "pseudotime; startVsEndTest, diffEndTest, and patternTest are "
        "secondary or directional analyses. Computational batches are "
        "combined before Benjamini-Hochberg correction within each "
        "predefined trajectory/test family."
    )

    write_workbook(
        OUTPUT_DIR / "S5_Table.xlsx",
        "S5 Table",
        legend,
        sheets,
    )


# %%
# S6 TABLE: SENSITIVITY AND ROBUSTNESS ANALYSES
####
def compile_s6():
    print("\n" + "=" * 80)
    print("S6 TABLE: SENSITIVITY AND ROBUSTNESS ANALYSES")
    print("=" * 80)

    output_specs = [
        (
            "ULM_AUCell",
            SCVI_VALIDATION_DIR
            / "ulm_vs_aucell_agreement.csv",
            "Agreement between primary ULM complement-module scores and AUCell sensitivity scores.",
            "module",
        ),
        (
            "Leave_one_gene_out",
            SCVI_VALIDATION_DIR
            / "leave_one_gene_out_sensitivity.csv",
            "Leave-one-gene-out complement module sensitivity analysis.",
            "module x omitted gene",
        ),
        (
            "Continuous_confounders",
            SCVI_VALIDATION_DIR
            / "confound_correlations_continuous.csv",
            "Associations between complement scores and continuous technical/QC covariates.",
            "module x covariate",
        ),
        (
            "Categorical_confounders",
            SCVI_VALIDATION_DIR
            / "confound_associations_categorical.csv",
            "Associations between complement scores and categorical technical covariates.",
            "module x covariate",
        ),
        (
            "Variance_partitioning",
            SCVI_VALIDATION_DIR
            / "confound_mixed_model_variance_partitioning.csv",
            "Mixed-model donor/experiment/residual variance partitioning for complement scores.",
            "module",
        ),
        (
            "Root_sensitivity",
            TRAJECTORY_DIR
            / "trajectory_root_sensitivity_summary.csv",
            "Trajectory sensitivity to alternative biologically plausible roots.",
            "trajectory",
        ),
        (
            "Representation_sensitivity",
            TRAJECTORY_DIR
            / "trajectory_representation_sensitivity_summary.csv",
            "Trajectory sensitivity across DECIPHER-v, DECIPHER-z, and scVI representations.",
            "trajectory x representation",
        ),
        (
            "Representation_lineages",
            TRAJECTORY_DIR
            / "trajectory_representation_sensitivity_lineages.csv",
            "Lineage-level correspondence across trajectory representations.",
            "trajectory x representation x lineage",
        ),
        (
            "DPT_sensitivity",
            TRAJECTORY_DIR
            / "trajectory_dpt_sensitivity_summary.csv",
            "Sensitivity comparison with diffusion pseudotime where available.",
            "trajectory",
        ),
        (
            "Subsampling_summary",
            TRAJECTORY_DIR
            / "trajectory_donor_subsampling_summary.csv",
            "Summary of repeated donor-subsampling trajectory reconstruction robustness.",
            "trajectory",
        ),
        (
            "Lineage_stability",
            TRAJECTORY_DIR
            / "trajectory_donor_subsampling_lineage_stability.csv",
            "Primary-lineage matching and pseudotime concordance across donor-subsampling runs.",
            "trajectory x lineage",
        ),
        (
            "Subsampling_failures",
            TRAJECTORY_DIR
            / "trajectory_donor_subsampling_failures.csv",
            "Failure-stage counts from donor-subsampling robustness runs.",
            "trajectory x failure stage",
        ),
    ]

    sheets = []

    for (
        sheet_name,
        path,
        description,
        unit,
    ) in output_specs:
        path = require_file(path)

        sheets.append(
            (
                sheet_name,
                read_csv(path),
                description,
                unit,
                relative_source(path),
            )
        )

    legend = (
        "Sensitivity and robustness analyses for complement-module scoring "
        "and trajectory inference. Complement robustness includes independent "
        "AUCell scoring, leave-one-gene-out analysis, confound assessment, "
        "and variance partitioning. Trajectory robustness includes alternative "
        "roots and representations, diffusion-pseudotime comparison, and "
        "repeated donor subsampling."
    )

    write_workbook(
        OUTPUT_DIR / "S6_Table.xlsx",
        "S6 Table",
        legend,
        sheets,
    )


# %%
# EXECUTE COMPILATION
####
print("PROJECT DIR")
print(PROJECT_DIR)

print("\nSUPPLEMENTARY TABLE OUTPUT")
print(OUTPUT_DIR)

print(
    "\nPackaging existing CSV outputs only; "
    "no statistical analyses are recomputed."
)

compile_s1()
compile_s2()
compile_s3()
compile_s4()
compile_s5()
compile_s6()


# %% FINAL CHECKPOINT
expected_outputs = [
    OUTPUT_DIR / f"S{i}_Table.xlsx"
    for i in range(1, 7)
]

missing_outputs = [
    path
    for path in expected_outputs
    if not path.exists()
    or path.stat().st_size == 0
]

if missing_outputs:
    raise RuntimeError(
        "Supplementary workbook compilation finished with missing/empty outputs: "
        f"{missing_outputs}"
    )

print("\n" + "=" * 80)
print("SUPPLEMENTARY TABLE COMPILATION COMPLETE")
print("=" * 80)

for path in expected_outputs:
    print(
        f"{path.name:16s} "
        f"{path.stat().st_size / (1024 ** 2):8.2f} MB"
    )
