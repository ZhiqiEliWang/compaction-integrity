# visualizations and stats for results from evaluating SC repeat counts

import argparse
import ast
from pathlib import Path
import re
import sys
from typing import Any

from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd
import seaborn as sns

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MANIFEST_PATH = REPO_ROOT / "config/experiments/rq2/repeat.yaml"

from compaction_integrity.analyze.utils import (
    extract_compactor_name,
    extract_dataset_config,
    fmt_compactor_label,
    fmt_dataset_label,
    load_manifest_results,
    tee_stdout,
)
from compaction_integrity.viz_config import save_fig, set_paper_style, set_talk_style


def _extract_repeat(sssc_attrs: Any) -> int:
    if isinstance(sssc_attrs, dict):
        return int(sssc_attrs["repeat"])
    if isinstance(sssc_attrs, str) and sssc_attrs.startswith("{"):
        return int(ast.literal_eval(sssc_attrs)["repeat"])
    raise ValueError(f"Cannot extract repeat from sssc_attrs: {sssc_attrs!r}")


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_")


def _load_results(manifest_path: Path, results_root: Path) -> pd.DataFrame:
    results_df = load_manifest_results(
        manifest_path=manifest_path,
        results_root=results_root,
        result_file_name="evaluation_results.pkl",
    )
    results_df["repeat"] = results_df["sssc_attrs"].map(_extract_repeat)
    results_df["dataset_config"] = results_df["dataset"].map(extract_dataset_config)
    results_df["compactor_name"] = results_df["compactor"].map(extract_compactor_name)
    results_df["dataset_config_label"] = results_df["dataset_config"].map(fmt_dataset_label)
    results_df["compactor_name_label"] = results_df["compactor_name"].map(fmt_compactor_label)
    return results_df


def _build_stats(results_df: pd.DataFrame) -> pd.DataFrame:
    success_df = results_df.loc[results_df["compaction_status"] == "success"].copy()
    rows = []
    group_cols = [
        "dataset",
        "dataset_config",
        "dataset_config_label",
        "compactor_name",
        "compactor_name_label",
        "repeat",
    ]
    for key, group in success_df.groupby(group_cols):
        (
            dataset,
            dataset_config,
            dataset_config_label,
            compactor_name,
            compactor_name_label,
            repeat,
        ) = key
        retention = group["retention"].dropna()
        compaction_compliance = group["compacted_compliant"].dropna()
        uninjected_baseline = group["full_without_sssc_compliant"].dropna()
        upper_bound = group["compacted_post_sssc_compliant"].dropna()
        retention_rate = float(retention.mean()) if len(retention) else float("nan")
        compaction_rate = (
            float(compaction_compliance.mean()) if len(compaction_compliance) else float("nan")
        )
        uninjected_rate = (
            float(uninjected_baseline.mean()) if len(uninjected_baseline) else float("nan")
        )
        upper_bound_rate = (
            float(upper_bound.mean()) if len(upper_bound) else float("nan")
        )
        calibrated = (
            compaction_rate - uninjected_rate
            if not (np.isnan(compaction_rate) or np.isnan(uninjected_rate))
            else float("nan")
        )
        denom = upper_bound_rate - uninjected_rate
        effect_retention = (
            calibrated / denom
            if not (np.isnan(calibrated) or np.isnan(denom)) and denom != 0
            else float("nan")
        )
        rows.append(
            {
                "dataset": dataset,
                "dataset_config": dataset_config,
                "dataset_config_label": dataset_config_label,
                "compactor_name": compactor_name,
                "compactor_name_label": compactor_name_label,
                "repeat": int(repeat),
                "n": len(group),
                "retention_rate": retention_rate,
                "calibrated_compliance": calibrated,
                "effect_retention": effect_retention,
            }
        )
    stats_df = pd.DataFrame(rows).sort_values(
        ["dataset_config", "compactor_name_label", "repeat"]
    ).reset_index(drop=True)
    return stats_df


def _plot_metrics_by_repeat(
    stats_df: pd.DataFrame,
    output_dir: Path,
    filename_prefix: str,
) -> None:
    repeat_values = sorted(stats_df["repeat"].unique())
    metrics = [
        ("retention_rate", "Average retention", "-", "o"),
        ("effect_retention", "Effect retention", "--", "s"),
    ]
    for dataset_config in sorted(stats_df["dataset_config"].unique()):
        sub = stats_df.loc[stats_df["dataset_config"] == dataset_config]
        fig, ax = plt.subplots()
        for metric, label, linestyle, marker in metrics:
            sns.lineplot(
                data=sub,
                x="repeat",
                y=metric,
                marker=marker,
                linestyle=linestyle,
                color="gray",
                legend=False,
                ax=ax,
            )
        ax.set_xlabel("SC Repetition")
        ax.set_ylabel("Rate")
        max_repeat = max(repeat_values)
        major_ticks = sorted({v for v in repeat_values if v == 1 or v % 5 == 0 or v == max_repeat})
        ax.set_xticks(major_ticks)
        ax.set_xticks(repeat_values, minor=True)
        ax.yaxis.set_major_formatter(PercentFormatter(1.0))
        metric_handles = [
            Line2D(
                [0],
                [0],
                color="gray",
                linestyle=linestyle,
                marker=marker,
                linewidth=2,
                label=label,
            )
            for _, label, linestyle, marker in metrics
        ]
        ax.legend(
            handles=metric_handles,
            loc="lower right",
            frameon=True,
            fontsize="small",
            handlelength=3.5,
        )
        save_fig(output_dir / f"{filename_prefix}__{_safe_name(dataset_config)}.pdf")
        plt.close(fig)


if __name__ == "__main__":
    args = argparse.ArgumentParser(
        description="Analyze the impact of SC repeat count on retention and compliance."
    )
    args.add_argument(
        "--output_dir",
        type=str,
        default="/data/compaction_integrity/analysis/repeat",
        help="Directory to save analysis results and visualizations.",
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
        default=str(DEFAULT_MANIFEST_PATH),
        help="Manifest file listing the run ids to aggregate.",
    )
    args.add_argument(
        "--talk_style",
        action="store_true",
        help="Use talk plotting style instead of paper plotting style.",
    )
    parsed_args = args.parse_args()

    output_dir = Path(parsed_args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with tee_stdout(output_dir / "repeat.log"):
        if parsed_args.talk_style:
            set_talk_style(use_latex=False)
        else:
            set_paper_style(use_latex=False)

        results_df = _load_results(
            manifest_path=Path(parsed_args.manifest_path),
            results_root=Path(parsed_args.results_root),
        )
        stats_df = _build_stats(results_df)
        stats_df.to_csv(output_dir / "repeat_stats.csv", index=False)
        print(stats_df.to_string(index=False))

        _plot_metrics_by_repeat(
            stats_df,
            output_dir=output_dir,
            filename_prefix="metrics_by_repeat",
        )

        print(f"All outputs written to {output_dir}")
