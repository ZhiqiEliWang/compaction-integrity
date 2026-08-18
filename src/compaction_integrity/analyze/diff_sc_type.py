# visualizations and stats for results from run_compactor_ablation
# focused on differences across SSSC types, aggregating across context_length

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd
import seaborn as sns

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from compaction_integrity.viz_config import set_paper_style, save_fig, set_talk_style
from compaction_integrity.analyze.utils import (
    extract_compactor_name,
    extract_dataset_config,
    fmt_compactor_label,
    fmt_dataset_label,
    load_manifest_results,
    ordered_compactor_labels,
    tee_stdout,
)


"""
evaluation_results.pkl has the following columns:
- dataset, dataset_path, source_row_index
- sssc_id, sssc_type, sssc_message, sssc_probe, sssc_attrs
- probe, compactor, evaluator
- full_with_sssc_compliant, full_without_sssc_compliant
- compacted_context, compaction_status, compaction_error
- compacted_compliant, retention
"""


def _load_results(manifest_path: Path, results_root: Path) -> pd.DataFrame:
    results_df = load_manifest_results(
        manifest_path=manifest_path,
        results_root=results_root,
        result_file_name="evaluation_results.pkl",
    )
    results_df["dataset_config"] = results_df["dataset"].map(extract_dataset_config)
    results_df["compactor_name"] = results_df["compactor"].map(extract_compactor_name)
    results_df["dataset_config_label"] = results_df["dataset_config"].map(fmt_dataset_label)
    results_df["compactor_name_label"] = results_df["compactor_name"].map(fmt_compactor_label)
    results_df["sssc_type_label"] = results_df["sssc_type"].str.replace("_", " ", regex=False)
    return results_df


def _aggregate_retention(
    results_df: pd.DataFrame,
    group_cols: list[str],
) -> pd.DataFrame:
    """Aggregate retention rate over the given grouping (collapses context_length)."""
    filtered_df = results_df.loc[results_df["compaction_status"] == "success"].copy()
    filtered_df["retention_numeric"] = filtered_df["retention"].astype(float)
    stats_df = (
        filtered_df.groupby(group_cols, as_index=False).agg(
            retention_rate=("retention_numeric", "mean"),
            retained_count=("retention_numeric", "sum"),
            total_count=("retention_numeric", "count"),
        )
    )
    return stats_df


def _ordered_sssc_types(stats_df: pd.DataFrame) -> list[str]:
    """Order SSSC types by overall mean retention rate (descending) for stable axes."""
    return (
        stats_df.groupby("sssc_type_label")["retention_rate"]
        .mean()
        .sort_values(ascending=False)
        .index.tolist()
    )


def _ordered_compactors(stats_df: pd.DataFrame) -> list[str]:
    """Canonical compactor display order (see utils.COMPACTOR_NAME_ORDER)."""
    return ordered_compactor_labels(stats_df)


def _draw_styled_heatmap(
    pivot: pd.DataFrame,
    xlabel: str,
    ylabel: str,
    output_path: Path,
) -> None:
    """Draw a heatmap with the canonical "all_datasets" style (bold labels, rotated x-ticks)."""
    n_cols = pivot.shape[1]
    n_rows = pivot.shape[0]
    fig, ax = plt.subplots(
        figsize=(0.7 * n_cols + 1.5, 0.38 * n_rows + 0.9)
    )
    sns.heatmap(
        pivot,
        annot=True,
        fmt=".0%",
        cmap="viridis",
        vmin=0,
        vmax=1.0,
        cbar_kws={"format": PercentFormatter(1.0), "label": "Retention Rate"},
        annot_kws={"weight": "bold"},
        ax=ax,
    )
    ax.set_xlabel(xlabel, fontweight="bold")
    ax.set_ylabel(ylabel, fontweight="bold")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=20, ha="right", fontweight="bold")
    ax.set_yticklabels(ax.get_yticklabels(), fontweight="bold")
    cbar = ax.collections[0].colorbar
    cbar.ax.set_ylabel("Retention Rate", fontweight="bold")
    for label in cbar.ax.get_yticklabels():
        label.set_fontweight("bold")
    save_fig(output_path)
    plt.close(fig)


def _heatmap_per_dataset(
    stats_df: pd.DataFrame,
    output_dir: Path,
) -> None:
    """One heatmap per dataset: rows=compactor, cols=SSSC type."""
    sssc_order = _ordered_sssc_types(stats_df)
    compactor_order = _ordered_compactors(stats_df)
    for dataset_config in sorted(stats_df["dataset_config"].unique()):
        plot_df = stats_df.loc[stats_df["dataset_config"] == dataset_config]
        pivot = (
            plot_df.pivot(index="compactor_name_label", columns="sssc_type_label", values="retention_rate")
            .reindex(index=compactor_order, columns=sssc_order)
            .astype(float)
        )
        file_name = f"heatmap_retention_by_compactor_sssc__{dataset_config}.pdf"
        _draw_styled_heatmap(pivot, "SC Type", "Compactor", output_dir / file_name)


def _collapse_dataset_retention(stats_df: pd.DataFrame) -> pd.DataFrame:
    return (
        stats_df.groupby(["compactor_name", "compactor_name_label", "sssc_type", "sssc_type_label"], as_index=False)
        .agg(
            retained_count=("retained_count", "sum"),
            total_count=("total_count", "sum"),
        )
        .assign(retention_rate=lambda df: df["retained_count"] / df["total_count"])
    )


def _heatmap_collapsed_dataset(
    stats_df: pd.DataFrame,
    output_dir: Path,
) -> pd.DataFrame:
    """One heatmap collapsed over dataset: rows=compactor, cols=SC type."""
    collapsed = _collapse_dataset_retention(stats_df)
    sssc_order = _ordered_sssc_types(collapsed)
    compactor_order = _ordered_compactors(collapsed)
    pivot = (
        collapsed.pivot(index="compactor_name_label", columns="sssc_type_label", values="retention_rate")
        .reindex(index=compactor_order, columns=sssc_order)
        .astype(float)
    )
    _draw_styled_heatmap(
        pivot,
        "SC Type",
        "Compactor",
        output_dir / "heatmap_retention_by_compactor_sc_type__all_datasets.pdf",
    )
    return collapsed


def _heatmap_per_compactor(
    stats_df: pd.DataFrame,
    output_dir: Path,
) -> None:
    """One heatmap per compactor: rows=dataset, cols=SSSC type. Only useful with >1 dataset."""
    if stats_df["dataset_config"].nunique() < 2:
        return
    sssc_order = _ordered_sssc_types(stats_df)
    dataset_label_order = sorted(stats_df["dataset_config_label"].unique())
    for compactor_name in stats_df["compactor_name"].drop_duplicates():
        plot_df = stats_df.loc[stats_df["compactor_name"] == compactor_name]
        pivot = (
            plot_df.pivot(index="dataset_config_label", columns="sssc_type_label", values="retention_rate")
            .reindex(index=dataset_label_order, columns=sssc_order)
            .astype(float)
        )
        safe_name = compactor_name.replace("/", "_")
        file_name = f"heatmap_retention_by_dataset_sssc__{safe_name}.pdf"
        _draw_styled_heatmap(pivot, "SC Type", "Dataset", output_dir / file_name)


def _grouped_bar_per_dataset(
    stats_df: pd.DataFrame,
    output_dir: Path,
) -> None:
    """One grouped bar plot per dataset: x=SSSC type, hue=compactor."""
    sssc_order = _ordered_sssc_types(stats_df)
    compactor_order = _ordered_compactors(stats_df)
    y_max = min(1.0, stats_df["retention_rate"].max() * 1.15 + 0.01)
    for dataset_config in sorted(stats_df["dataset_config"].unique()):
        plot_df = stats_df.loc[stats_df["dataset_config"] == dataset_config]
        dataset_label = plot_df["dataset_config_label"].iloc[0]
        fig, ax = plt.subplots(
            figsize=(0.6 * len(sssc_order) * max(1, len(compactor_order) / 3) + 2.0, 3.0)
        )
        sns.barplot(
            data=plot_df,
            x="sssc_type_label",
            y="retention_rate",
            hue="compactor_name_label",
            order=sssc_order,
            hue_order=compactor_order,
            ax=ax,
        )
        ax.set_xlabel("SC Type")
        ax.set_ylabel("Retention Rate")
        ax.set_ylim(0, y_max)
        ax.yaxis.set_major_formatter(PercentFormatter(1.0))
        ax.legend(title="Compactor", bbox_to_anchor=(1.02, 1.0), loc="upper left")
        file_name = f"bar_retention_by_compactor__{dataset_config}.pdf"
        save_fig(output_dir / file_name)
        plt.close(fig)


def _summary_heatmap_compactor_dataset(
    stats_df: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Aggregate over SSSC type: rows=compactor, cols=dataset. A high-level summary."""
    if stats_df["dataset_config"].nunique() < 2:
        return
    summary = (
        stats_df.groupby(["compactor_name", "compactor_name_label", "dataset_config", "dataset_config_label"], as_index=False)
        .apply(
            lambda g: pd.Series(
                {"retention_rate": g["retained_count"].sum() / max(g["total_count"].sum(), 1)}
            ),
            include_groups=False,
        )
        .reset_index(drop=True)
    )
    compactor_label_order = ordered_compactor_labels(summary)
    dataset_label_order = sorted(summary["dataset_config_label"].unique())
    pivot = (
        summary.pivot(index="compactor_name_label", columns="dataset_config_label", values="retention_rate")
        .reindex(index=compactor_label_order, columns=dataset_label_order)
        .astype(float)
    )
    _draw_styled_heatmap(
        pivot,
        "Dataset",
        "Compactor",
        output_dir / "heatmap_retention_compactor_by_dataset.pdf",
    )


def _scatter_all(
    stats_df: pd.DataFrame,
    output_dir: Path,
) -> None:
    """One scatter plot for all data: x=SSSC type, y=retention rate, color=compactor, marker=dataset.

    Compactors are spread horizontally within each SSSC-type tick using a small
    fixed offset so overlapping points are legible.
    """
    sssc_order = _ordered_sssc_types(stats_df)
    compactor_order = _ordered_compactors(stats_df)
    dataset_order = sorted(stats_df["dataset_config_label"].unique())

    sssc_pos = {label: i for i, label in enumerate(sssc_order)}
    n_compactors = len(compactor_order)
    compactor_offsets = np.linspace(-0.25, 0.25, n_compactors) if n_compactors > 1 else np.array([0.0])
    compactor_offset_map = dict(zip(compactor_order, compactor_offsets))

    palette = sns.color_palette("tab10", n_colors=n_compactors)
    compactor_colors = dict(zip(compactor_order, palette))

    markers = ["o", "s", "^", "D", "v", "P", "X"]
    dataset_markers = {ds: markers[i % len(markers)] for i, ds in enumerate(dataset_order)}

    fig, ax = plt.subplots(figsize=(1.1 * len(sssc_order) + 2.0, 3.5))

    for _, row in stats_df.iterrows():
        x = sssc_pos[row["sssc_type_label"]] + compactor_offset_map[row["compactor_name_label"]]
        ax.scatter(
            x,
            row["retention_rate"],
            color=compactor_colors[row["compactor_name_label"]],
            marker=dataset_markers[row["dataset_config_label"]],
            s=60,
            zorder=3,
        )

    ax.set_xticks(range(len(sssc_order)))
    ax.set_xticklabels(sssc_order, rotation=20, ha="right")
    ax.set_xlabel("SC Type")
    ax.set_ylabel("Retention Rate")
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_ylim(0, min(1.05, stats_df["retention_rate"].max() * 1.15 + 0.01))
    ax.set_xlim(-0.5, len(sssc_order) - 0.5)
    ax.grid(axis="y", linewidth=0.5, alpha=0.4)

    compactor_handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=compactor_colors[c], markersize=7, label=c)
        for c in compactor_order
    ]
    dataset_handles = [
        Line2D([0], [0], marker=dataset_markers[d], color="gray", markersize=7, linestyle="None", label=d)
        for d in dataset_order
    ]
    legend1 = ax.legend(handles=compactor_handles, title="Compactor", bbox_to_anchor=(1.02, 1.0), loc="upper left")
    ax.add_artist(legend1)
    ax.legend(handles=dataset_handles, title="Dataset", bbox_to_anchor=(1.02, 0.0), loc="lower left")

    save_fig(output_dir / "scatter_retention_all.pdf")
    plt.close(fig)


if __name__ == "__main__":
    args = argparse.ArgumentParser(
        description="Analyze retention differences across SSSC types, compactors, and datasets."
    )
    args.add_argument(
        "--output_dir",
        type=str,
        default="/data/compaction_integrity/analysis/diff_sc_type",
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
        default=str(Path(__file__).resolve().parents[3] / "config/experiments/rq2/diff_sc_type.yaml"),
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

    with tee_stdout(output_dir / "diff_sc_type.log"):
        if parsed_args.talk_style:
            set_talk_style(use_latex=False)
        else:
            set_paper_style(use_latex=False)

        results_df = _load_results(
            manifest_path=Path(parsed_args.manifest_path),
            results_root=Path(parsed_args.results_root),
        )

        stats_df = _aggregate_retention(
            results_df,
            group_cols=["dataset_config", "dataset_config_label", "compactor_name", "compactor_name_label", "sssc_type", "sssc_type_label"],
        )
        stats_df.to_csv(output_dir / "retention_by_dataset_compactor_sssc.csv", index=False)
        print(stats_df.to_string(index=False))

        collapsed_stats_df = _heatmap_collapsed_dataset(stats_df, output_dir)
        collapsed_stats_df.to_csv(output_dir / "retention_by_compactor_sssc_collapsed_dataset.csv", index=False)
        _heatmap_per_dataset(stats_df, output_dir)
        _heatmap_per_compactor(stats_df, output_dir)
        _grouped_bar_per_dataset(stats_df, output_dir)
        _summary_heatmap_compactor_dataset(stats_df, output_dir)
        _scatter_all(stats_df, output_dir)
