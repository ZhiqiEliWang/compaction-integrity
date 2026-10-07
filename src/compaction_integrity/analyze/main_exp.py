import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from compaction_integrity.analyze.utils import (
    extract_compactor_name,
    extract_dataset_config,
    fmt_compactor_label,
    fmt_dataset_label,
    load_manifest_results,
    tee_stdout,
)


def _load_results(manifest_path: Path, results_root: Path) -> pd.DataFrame:
    results_df = load_manifest_results(
        manifest_path=manifest_path,
        results_root=results_root,
        result_file_name="evaluation_results.pkl",
    )
    results_df["compactor_name"] = results_df["compactor"].map(extract_compactor_name)
    results_df["dataset_config"] = results_df["dataset"].map(extract_dataset_config)
    results_df["compactor_name_label"] = results_df["compactor_name"].map(fmt_compactor_label)
    results_df["dataset_config_label"] = results_df["dataset_config"].map(fmt_dataset_label)
    results_df["sssc_type_label"] = results_df["sssc_type"].str.replace("_", " ", regex=False)
    return results_df


def _success(df: pd.DataFrame) -> pd.DataFrame:
    return df.loc[df["compaction_status"] == "success"].copy()


def _compliance_rate(series: pd.Series) -> float:
    valid = series.dropna()
    if len(valid) == 0:
        return float("nan")
    return float(valid.mean())


def _dataset_baselines(df: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Compute injected/uninjected baselines once per dataset.

    The baseline columns describe the full uncompacted input and are independent
    of the compactor, so we dedupe by (dataset, source_row_index, sssc_id) to
    avoid weighting by how many compactors happened to succeed on a given row.
    """
    key_cols = ["dataset", "source_row_index", "sssc_id"]
    unique_inputs = df.drop_duplicates(subset=key_cols)
    out: dict[str, dict[str, float]] = {}
    for dataset, g in unique_inputs.groupby("dataset"):
        out[dataset] = {
            "injected_baseline_compliance": _compliance_rate(g["full_with_sssc_compliant"]),
            "uninjected_baseline_compliance": _compliance_rate(g["full_without_sssc_compliant"]),
        }
    return out


def build_summary_table(df: pd.DataFrame) -> pd.DataFrame:
    suc = _success(df)
    baselines = _dataset_baselines(df)
    group_cols = ["dataset", "dataset_config_label", "compactor_name", "compactor_name_label"]

    rows = []
    for key, g in suc.groupby(group_cols):
        dataset, dataset_label, compactor_name, compactor_label = key
        retention_rate = _compliance_rate(g["retention"])
        injected_baseline = baselines[dataset]["injected_baseline_compliance"]
        compaction_compliance = _compliance_rate(g["compacted_compliant"])
        uninjected_baseline = baselines[dataset]["uninjected_baseline_compliance"]
        upper_bound = _compliance_rate(g["compacted_post_sssc_compliant"])
        calibrated = (
            compaction_compliance - uninjected_baseline
            if not (np.isnan(compaction_compliance) or np.isnan(uninjected_baseline))
            else float("nan")
        )
        denom = upper_bound - uninjected_baseline
        effective_retention = (
            calibrated / denom
            if not (np.isnan(calibrated) or np.isnan(denom)) and denom != 0
            else float("nan")
        )
        rows.append(
            {
                "dataset": dataset_label,
                "compactor": compactor_label,
                "dataset_raw": dataset,
                "compactor_raw": compactor_name,
                "n": len(g),
                "retention_rate": retention_rate,
                "injected_baseline_compliance": injected_baseline,
                "compaction_compliance": compaction_compliance,
                "uninjected_baseline_compliance": uninjected_baseline,
                "upper_bound_compliance": upper_bound,
                "calibrated_compliance": calibrated,
                "effective_retention": effective_retention,
            }
        )
    return pd.DataFrame(rows)


def print_null_compaction_compliance_cases(df: pd.DataFrame) -> None:
    suc = _success(df)
    null_cases = suc.loc[suc["compacted_compliant"].isna()]
    if null_cases.empty:
        print("=== Null compaction compliance cases ===")
        print("none")
        print()
        return

    display_cols = [
        "dataset_config_label",
        "compactor_name_label",
        "run_id",
        "source_row_index",
        "sssc_id",
        "sssc_type_label",
        "retention",
    ]
    display = null_cases[display_cols].rename(
        columns={
            "dataset_config_label": "dataset",
            "compactor_name_label": "compactor",
            "sssc_type_label": "sssc_type",
        }
    )
    print("=== Null compaction compliance cases ===")
    print(display.to_string(index=False))
    print()


def _sssc_retention_stats(df: pd.DataFrame) -> pd.DataFrame:
    suc = _success(df)
    stats = (
        suc.groupby(
            [
                "dataset_config",
                "dataset_config_label",
                "compactor_name",
                "compactor_name_label",
                "sssc_type",
                "sssc_type_label",
            ]
        )["retention"]
        .agg(
            retention_rate="mean",
            retained_count=lambda s: s.eq(True).sum(),
            total_count="size",
        )
        .reset_index()
    )
    return stats


if __name__ == "__main__":
    args = argparse.ArgumentParser(
        description="Analyze evaluation_results.pkl from evaluation.py."
    )
    args.add_argument(
        "--output_dir",
        type=str,
        default="/data/compaction_integrity/analysis/retention_if_eval",
        help="Directory to save analysis results.",
    )
    args.add_argument(
        "--results_root",
        type=str,
        default="/data/compaction_integrity",
        help="Root directory containing canonical run outputs.",
    )
    args.add_argument(
        "--manifest_path",
        type=str,
        default=str(Path(__file__).resolve().parents[3] / "config/experiments/rq1/main.yaml"),
        help="Manifest file listing the run ids to aggregate.",
    )
    parsed_args = args.parse_args()

    output_dir = Path(parsed_args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with tee_stdout(output_dir / "retention_if_eval.log"):
        results_df = _load_results(
            manifest_path=Path(parsed_args.manifest_path),
            results_root=Path(parsed_args.results_root),
        )

        print(f"Loaded {len(results_df)} rows from {results_df['run_id'].nunique()} run(s).")
        print(f"Compactors: {sorted(results_df['compactor_name_label'].unique())}")
        print(f"Datasets:   {sorted(results_df['dataset_config_label'].unique())}")
        print()

        summary = build_summary_table(results_df)
        summary.to_csv(output_dir / "summary_table.csv", index=False)
        display = summary.drop(columns=["dataset_raw", "compactor_raw"])
        for col in [
            "retention_rate",
            "injected_baseline_compliance",
            "compaction_compliance",
            "uninjected_baseline_compliance",
            "upper_bound_compliance",
            "calibrated_compliance",
            "effective_retention",
        ]:
            display[col] = display[col].map(
                lambda value: f"{100 * value:.1f}%" if not np.isnan(value) else "nan"
            )
        print("=== Summary table ===")
        print(display.to_string(index=False))
        print()
        print_null_compaction_compliance_cases(results_df)

        sssc_stats = _sssc_retention_stats(results_df)
        sssc_stats.to_csv(output_dir / "sssc_retention_stats.csv", index=False)

        print(f"All outputs written to {output_dir}")
