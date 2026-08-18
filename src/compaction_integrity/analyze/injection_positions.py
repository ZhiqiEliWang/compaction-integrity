# visualizations and stats for results from evaluating SC injection positions

import argparse
import ast
from pathlib import Path
import re
import sys
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import pandas as pd
import seaborn as sns

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MANIFEST_PATH = REPO_ROOT / "config/experiments/rq2/injection_positions.yaml"

from compaction_integrity.analyze.utils import (
    COMPACTOR_NAME_ORDER,
    extract_compactor_name,
    extract_context_length,
    extract_dataset_config,
    fmt_compactor_label,
    fmt_dataset_label,
    load_manifest_results,
    ordered_compactor_labels,
    ordered_values,
    tee_stdout,
)
from compaction_integrity.viz_config import save_fig, set_paper_style, set_talk_style


POSITION_ORDER = ["top", "middle", "bottom"]
POSITION_DISPLAY_LABELS = {position: position.capitalize() for position in POSITION_ORDER}
DATASET_CONFIG_ORDER = ["wildchat_cat", "hermes_cat"]
CONTEXT_LENGTH_ORDER = ["10k", "50k", "100k", "300k"]


def _extract_position(sssc_attrs: Any) -> str:
    if isinstance(sssc_attrs, dict):
        return str(sssc_attrs["position"])
    if isinstance(sssc_attrs, str) and sssc_attrs.startswith("{"):
        return str(ast.literal_eval(sssc_attrs)["position"])
    raise ValueError(f"Cannot extract position from sssc_attrs: {sssc_attrs!r}")


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_")


def _load_results(manifest_path: Path, results_root: Path) -> pd.DataFrame:
    results_df = load_manifest_results(
        manifest_path=manifest_path,
        results_root=results_root,
        result_file_name="evaluation_results.pkl",
    )
    results_df["position"] = results_df["sssc_attrs"].map(_extract_position)
    results_df["context_length"] = results_df["dataset"].map(extract_context_length)
    results_df["dataset_config"] = results_df["dataset"].map(extract_dataset_config)
    results_df["compactor_name"] = results_df["compactor"].map(extract_compactor_name)
    results_df["dataset_config_label"] = results_df["dataset_config"].map(fmt_dataset_label)
    results_df["compactor_name_label"] = results_df["compactor_name"].map(fmt_compactor_label)
    return results_df


def _build_retention_stats(results_df: pd.DataFrame) -> pd.DataFrame:
    success_df = results_df.loc[results_df["compaction_status"] == "success"].copy()
    stats_df = (
        success_df.groupby(
            [
                "context_length",
                "dataset_config",
                "dataset_config_label",
                "compactor_name",
                "compactor_name_label",
                "position",
            ],
            as_index=False,
        ).agg(
            retention_rate=("retention", "mean"),
            retained_count=("retention", "sum"),
            total_count=("retention", "count"),
        )
    )
    stats_df["position"] = pd.Categorical(
        stats_df["position"], categories=POSITION_ORDER, ordered=True
    )
    context_length_order = ordered_values(stats_df["context_length"], CONTEXT_LENGTH_ORDER)
    dataset_config_order = ordered_values(stats_df["dataset_config"], DATASET_CONFIG_ORDER)
    compactor_name_order = ordered_values(stats_df["compactor_name"], COMPACTOR_NAME_ORDER)
    stats_df["context_length"] = pd.Categorical(
        stats_df["context_length"], categories=context_length_order, ordered=True
    )
    stats_df["dataset_config"] = pd.Categorical(
        stats_df["dataset_config"], categories=dataset_config_order, ordered=True
    )
    stats_df["compactor_name"] = pd.Categorical(
        stats_df["compactor_name"], categories=compactor_name_order, ordered=True
    )
    stats_df = stats_df.sort_values(
        ["context_length", "dataset_config", "compactor_name", "position"]
    ).reset_index(drop=True)
    return stats_df


def _build_overall_retention_table(results_df: pd.DataFrame) -> pd.DataFrame:
    success_df = results_df.loc[results_df["compaction_status"] == "success"].copy()
    table_df = (
        success_df.groupby(["context_length", "position"], as_index=False)
        .agg(retention_rate=("retention", "mean"))
    )
    context_length_order = ordered_values(table_df["context_length"], CONTEXT_LENGTH_ORDER)
    table_df["context_length"] = pd.Categorical(
        table_df["context_length"], categories=context_length_order, ordered=True
    )
    table_df["position"] = pd.Categorical(
        table_df["position"], categories=POSITION_ORDER, ordered=True
    )
    return table_df.sort_values(["context_length", "position"]).reset_index(drop=True)


def _plot_individual_retention_by_position(
    stats_df: pd.DataFrame,
    output_dir: Path,
) -> None:
    for context_length in ordered_values(stats_df["context_length"], CONTEXT_LENGTH_ORDER):
        length_df = stats_df.loc[stats_df["context_length"] == context_length]
        compactor_order = ordered_compactor_labels(length_df)
        for dataset_config in ordered_values(length_df["dataset_config"], DATASET_CONFIG_ORDER):
            sub = length_df.loc[length_df["dataset_config"] == dataset_config]
            fig, ax = plt.subplots()
            sns.barplot(
                data=sub,
                x="position",
                y="retention_rate",
                hue="compactor_name_label",
                hue_order=compactor_order,
                order=POSITION_ORDER,
                ax=ax,
            )
            ax.set_xlabel("SC Injection Location")
            ax.set_ylabel("Retention Rate")
            ax.set_ylim(0, 1)
            ax.yaxis.set_major_formatter(PercentFormatter(1.0))
            ax.legend(title="Compactor")
            save_fig(
                output_dir
                / (
                    f"retention_rate_by_position__{_safe_name(dataset_config)}"
                    f"__{_safe_name(str(context_length))}.pdf"
                )
            )
            plt.close(fig)


def _abbreviate_row_label(dataset_label: str, compactor_label: str) -> str:
    dataset_abbr = "".join(ch for ch in dataset_label if ch.isupper()) or dataset_label[:1].upper()
    paren_match = re.search(r"\(([^)]+)\)", compactor_label)
    compactor_base = re.sub(r"\s*\([^)]*\)\s*", "", compactor_label).strip()
    compactor_abbr = (
        compactor_base.replace("gpt-oss", "GPT")
        .replace("qwen3", "QWEN")
        .replace("Llmlingua2 T500", "Lingua")
        .replace("Recent 5", "R5")
    )
    if paren_match:
        variant = paren_match.group(1)
        variant_abbr = "P" if variant.startswith("pi") else variant[:1].upper()
        compactor_abbr = f"{compactor_abbr}-{variant_abbr}"
    return f"{dataset_abbr}/{compactor_abbr}"


def _plot_all_retention_by_position(
    stats_df: pd.DataFrame,
    output_dir: Path,
) -> None:
    for context_length in ordered_values(stats_df["context_length"], CONTEXT_LENGTH_ORDER):
        plot_df = stats_df.loc[stats_df["context_length"] == context_length].copy()
        plot_df["dataset_compactor_label"] = [
            _abbreviate_row_label(d, c)
            for d, c in zip(plot_df["dataset_config_label"], plot_df["compactor_name_label"])
        ]
        row_order = []
        for dataset_config in ordered_values(plot_df["dataset_config"], DATASET_CONFIG_ORDER):
            dataset_rows = plot_df.loc[plot_df["dataset_config"] == dataset_config]
            for compactor_label in ordered_compactor_labels(dataset_rows):
                row_order.extend(
                    dataset_rows.loc[
                        dataset_rows["compactor_name_label"] == compactor_label,
                        "dataset_compactor_label",
                    ].unique()
                )
        heatmap_df = (
            plot_df.pivot(
                index="dataset_compactor_label",
                columns="position",
                values="retention_rate",
            )
            .reindex(index=row_order, columns=POSITION_ORDER)
            .rename(columns=POSITION_DISPLAY_LABELS)
        )
        n_rows = len(heatmap_df.index)
        heatmap_width = plt.rcParams["figure.figsize"][0]
        heatmap_height = max(1.2, 0.22 * n_rows + 0.6)
        fig, ax = plt.subplots(figsize=(heatmap_width, heatmap_height))
        sns.heatmap(
            heatmap_df,
            ax=ax,
            cmap="rocket_r",
            vmin=0.0,
            vmax=1.0,
            annot=True,
            fmt=".0%",
            annot_kws={"fontsize": 7},
            cbar_kws={"label": "Retention Rate", "format": PercentFormatter(1.0)},
            linewidths=0.4,
            linecolor="white",
        )
        ax.set_xlabel("SC Injection Location")
        ax.set_ylabel("")
        ax.tick_params(axis="y", rotation=0)
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
        save_fig(
            output_dir / f"retention_rate_by_position_all__{_safe_name(str(context_length))}.pdf"
        )
        plt.close(fig)


if __name__ == "__main__":
    args = argparse.ArgumentParser(
        description="Analyze the impact of SC injection position on retention."
    )
    args.add_argument(
        "--output_dir",
        type=str,
        default="/data/compaction_integrity/analysis/position",
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

    with tee_stdout(output_dir / "position.log"):
        if parsed_args.talk_style:
            set_talk_style(use_latex=False)
        else:
            set_paper_style(use_latex=False)

        results_df = _load_results(
            manifest_path=Path(parsed_args.manifest_path),
            results_root=Path(parsed_args.results_root),
        )
        stats_df = _build_retention_stats(results_df)
        stats_df.to_csv(output_dir / "retention_rate_by_position.csv", index=False)
        overall_table_df = _build_overall_retention_table(results_df)
        overall_table_df.to_csv(
            output_dir / "retention_rate_by_position_all.csv", index=False
        )
        print("Retention rate by position, all runs:")
        print(overall_table_df.to_string(index=False))
        print()
        print("Retention rate by position, split by dataset and compactor:")
        print(stats_df.to_string(index=False))

        _plot_individual_retention_by_position(stats_df, output_dir)
        _plot_all_retention_by_position(stats_df, output_dir)

        print(f"All outputs written to {output_dir}")
