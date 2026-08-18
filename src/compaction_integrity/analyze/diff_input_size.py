# visualizations and stats for results from run_compactor_ablation

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter
from matplotlib.transforms import Bbox
import pandas as pd
import seaborn as sns

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MANIFEST_PATH = REPO_ROOT / "config/experiments/rq2/diff_input_size.yaml"

from compaction_integrity.analyze.utils import (
    extract_compactor_name,
    extract_context_length,
    extract_dataset_config,
    fmt_compactor_label,
    fmt_dataset_label,
    load_manifest_results,
    ordered_compactor_labels,
    tee_stdout,
)
from compaction_integrity.viz_config import set_paper_style, save_fig, set_talk_style


"""
evaluation_results.pkl has the following columns:
- dataset, dataset_path, source_row_index
- sssc_id, sssc_type, sssc_message, sssc_probe, sssc_attrs
- probe, compactor, evaluator
- full_with_sssc_compliant, full_without_sssc_compliant
- compacted_context, compaction_status, compaction_error
- compacted_compliant, retention
"""

def _context_length_to_int(context_length: str) -> int:
    return int(context_length[:-1]) * 1000


def _load_results(manifest_path: Path, results_root: Path) -> pd.DataFrame:
    results_df = load_manifest_results(
        manifest_path=manifest_path,
        results_root=results_root,
        result_file_name="evaluation_results.pkl",
    )
    results_df["context_length"] = results_df["dataset"].map(extract_context_length)
    results_df["dataset_config"] = results_df["dataset"].map(extract_dataset_config)
    results_df["compactor_name"] = results_df["compactor"].map(extract_compactor_name)
    return results_df


def _build_retention_stats(results_df: pd.DataFrame) -> pd.DataFrame:
    filtered_df = results_df.loc[results_df["compaction_status"] == "success"].copy()
    stats_df = (
        filtered_df.groupby(
            ["dataset_config", "compactor_name", "context_length"],
            as_index=False,
        ).agg(
            retention_rate=("retention", "mean"),
            retained_count=("retention", "sum"),
            total_count=("retention", "count"),
        )
    )
    stats_df["context_length_tokens"] = stats_df["context_length"].map(_context_length_to_int)
    stats_df["dataset_config_label"] = stats_df["dataset_config"].map(fmt_dataset_label)
    stats_df["compactor_name_label"] = stats_df["compactor_name"].map(fmt_compactor_label)
    stats_df["config_label"] = (
        stats_df["dataset_config"] + " | " + stats_df["compactor_name"]
    )
    stats_df = stats_df.sort_values(
        ["context_length_tokens", "dataset_config_label", "compactor_name_label"]
    ).reset_index(drop=True)
    return stats_df


def _plot_retention_stats(
    stats_df: pd.DataFrame,
    output_dir: Path,
    talk_style: bool,
) -> None:
    if talk_style:
        set_talk_style(use_latex=False)
    else:
        set_paper_style(use_latex=False)
    fig, ax = plt.subplots()

    context_order = (
        stats_df[["context_length", "context_length_tokens"]]
        .drop_duplicates()
        .sort_values("context_length_tokens")["context_length"]
        .tolist()
    )
    x_pos = {context_length: i for i, context_length in enumerate(context_order)}

    compactor_order = ordered_compactor_labels(stats_df)
    compactor_colors = dict(
        zip(compactor_order, sns.color_palette(n_colors=len(compactor_order)))
    )
    dataset_name_order = [
        name for name in ["hermes_cat", "wildchat_cat"] if name in set(stats_df["dataset_config"])
    ]
    dataset_name_order += sorted(set(stats_df["dataset_config"]) - set(dataset_name_order))
    dataset_styles = {
        "hermes_cat": {"linestyle": "-", "marker": "o"},
        "wildchat_cat": {"linestyle": "--", "marker": "s"},
    }

    for dataset_name in dataset_name_order:
        for compactor_label in compactor_order:
            plot_df = stats_df.loc[
                (stats_df["dataset_config"] == dataset_name)
                & (stats_df["compactor_name_label"] == compactor_label)
            ].sort_values("context_length_tokens")
            if plot_df.empty:
                continue
            style = dataset_styles.get(dataset_name, {"linestyle": ":", "marker": "^"})
            ax.plot(
                [x_pos[value] for value in plot_df["context_length"]],
                plot_df["retention_rate"],
                color=compactor_colors[compactor_label],
                linestyle=style["linestyle"],
                marker=style["marker"],
                linewidth=2,
                label=compactor_label,
            )

    ax.set_xticks(range(len(context_order)))
    ax.set_xticklabels(context_order)
    ax.set_xlabel("Context Length")
    ax.set_ylabel("Average Retention Rate")
    ax.set_ylim(0, 1)
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))

    compactor_handles = [
        Line2D([0], [0], color=compactor_colors[label], linewidth=2, label=label)
        for label in compactor_order
    ]
    dataset_handles = [
        Line2D(
            [0],
            [0],
            color="gray",
            linewidth=2,
            linestyle=dataset_styles.get(name, {"linestyle": ":"})["linestyle"],
            marker=dataset_styles.get(name, {"marker": "^"})["marker"],
            label=fmt_dataset_label(name),
        )
        for name in dataset_name_order
    ]
    save_fig(output_dir / "avg_retention_rate_by_context_length.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(2.6, 2.2))
    ax.axis("off")
    compactor_legend = ax.legend(
        handles=compactor_handles,
        title="Compactor",
        loc="upper center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=1,
        frameon=False,
    )
    ax.add_artist(compactor_legend)
    fig.canvas.draw()
    compactor_bbox = compactor_legend.get_window_extent(
        fig.canvas.get_renderer()
    ).transformed(ax.transAxes.inverted())
    ax.legend(
        handles=dataset_handles,
        title="Dataset",
        loc="upper center",
        bbox_to_anchor=(0.5, compactor_bbox.y0 - 0.02),
        ncol=len(dataset_handles),
        frameon=False,
    )
    fig.canvas.draw()
    legend_bboxes = [
        legend.get_window_extent(fig.canvas.get_renderer()).transformed(fig.dpi_scale_trans.inverted())
        for legend in fig.legends + [compactor_legend, ax.get_legend()]
        if legend is not None
    ]
    legend_bbox = Bbox.union(legend_bboxes)
    fig.savefig(
        output_dir / "avg_retention_rate_by_context_length_legend.pdf",
        format="pdf",
        bbox_inches=legend_bbox.expanded(1.02, 1.08),
        pad_inches=0.0,
    )
    print(f"Saved figure: {output_dir / 'avg_retention_rate_by_context_length_legend.pdf'}")
    plt.close(fig)


if __name__ == "__main__":
    args = argparse.ArgumentParser(description="Analyze the impact of input context size on SSSC generation and retention.")
    args.add_argument(
        "--output_dir",
        type=str,
        default="/data/compaction_integrity/analysis/diff_input_size",
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

    with tee_stdout(output_dir / "diff_input_size.log"):
        results_df = _load_results(
            manifest_path=Path(parsed_args.manifest_path),
            results_root=Path(parsed_args.results_root),
        )
        stats_df = _build_retention_stats(results_df)
        stats_df.to_csv(output_dir / "avg_retention_rate_by_context_length.csv", index=False)
        print(stats_df.to_string(index=False))
        _plot_retention_stats(stats_df, output_dir, talk_style=parsed_args.talk_style)
