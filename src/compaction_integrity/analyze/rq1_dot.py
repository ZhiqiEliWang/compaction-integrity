"""RQ1 dot plot: retention rate and effective retention per compactor, one row of
panels per dataset.

Usage:
    # Main figure: three 100k datasets plus the GPT-5.4-mini 220k runs
    python -m compaction_integrity.analyze.rq1_dot \
        --supplement_manifest_path config/experiments/rq1/gpt-5.4-mini.yaml

    # SWE-Natural, separate figure
    python -m compaction_integrity.analyze.rq1_dot \
        --manifest_path config/experiments/rq1/swe_natural_sc_n126.yaml \
        --output_path /data/compaction_integrity/analysis/swe_natural_sc_n126/dot_retention_compliance.pdf
"""

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from compaction_integrity.analyze.main_exp import _load_results, build_summary_table
from compaction_integrity.analyze.utils import ordered_values
from compaction_integrity.viz_config import PALETTE, set_paper_style

REPO_ROOT = Path(__file__).resolve().parents[3]

# Table 2 row order: small compactors first.
RQ1_COMPACTOR_ORDER: list[str] = [
    "recent_5",
    "llmlingua2_t500",
    "gemma_4_anthropic_prompt",
    "qwen30b_anthropic_prompt",
    "gpt_oss_120b_anthropic_prompt",
    "gpt_oss_120b_pi_mono_prompt",
]

ROW_HEIGHT_IN = 0.092
GROUP_GAP = 0.7


def plot_retention_dot(
    summary: pd.DataFrame, supplement: pd.DataFrame | None, output_path: Path
) -> None:
    """Single-column figure. Rows: datasets. Columns: retention rate, effective retention.

    Supplement runs are listed after the main compactors, separated by a gap. They
    have no long-ctx baselines, so their effective retention is shown as n/a.
    """
    main_order = ordered_values(summary["compactor_raw"], RQ1_COMPACTOR_ORDER)
    supp_order = [] if supplement is None else sorted(supplement["compactor_raw"].unique())
    rows = pd.concat([summary] if supplement is None else [summary, supplement], ignore_index=True)
    compactor_order = main_order + supp_order
    y_pos = {
        name: i + (GROUP_GAP if i >= len(main_order) else 0)
        for i, name in enumerate(compactor_order)
    }
    label_by_name = rows.drop_duplicates("compactor_raw").set_index("compactor_raw")["compactor"]
    dataset_order = sorted(rows["dataset"].unique())
    color = PALETTE[0]
    value_color = "0.3"
    small, label_size, value_size, title_size, dataset_size = 6.5, 7, 6.5, 8, 8.5

    band_rows = max(y_pos.values()) + 1
    fig, axes = plt.subplots(
        len(dataset_order),
        2,
        sharex=True,
        sharey=True,
        squeeze=False,
        figsize=(
            plt.rcParams["figure.figsize"][0],
            0.35 + ROW_HEIGHT_IN * band_rows * len(dataset_order) + 0.05 * (len(dataset_order) - 1),
        ),
    )
    fig.get_layout_engine().set(w_pad=0.01, h_pad=0.02, wspace=0.04, hspace=0.03)
    for (ax_ret, ax_eff), dataset in zip(axes, dataset_order):
        sub = rows.loc[rows["dataset"] == dataset]
        for row in sub.itertuples():
            y = y_pos[row.compactor_raw]
            for ax, x in ((ax_ret, row.retention_rate), (ax_eff, row.effective_retention)):
                if np.isnan(x):
                    ax.text(0.0, y, "n/a", ha="left", va="center", fontsize=value_size, color=value_color)
                    continue
                ax.scatter(x, y, s=14, color=color, edgecolor="white", linewidth=0.6, zorder=3)
                # Values near 100% go left of the dot to stay inside the panel.
                left = x > 0.8
                ax.annotate(
                    f"{100 * x:.1f}",
                    (x, y),
                    xytext=(-4 if left else 4, 0),
                    textcoords="offset points",
                    ha="right" if left else "left",
                    va="center",
                    fontsize=value_size,
                    color=value_color,
                )
        for ax in (ax_ret, ax_eff):
            ax.grid(axis="y", visible=False)
            ax.grid(axis="x", which="minor", alpha=0.3)
            ax.tick_params(axis="x", which="both", length=0, labelsize=small, pad=2)
            ax.tick_params(axis="y", length=0, labelsize=label_size, pad=2)
            ax.xaxis.set_major_formatter(mticker.PercentFormatter(1.0, decimals=0))
        ax_eff.yaxis.set_label_position("right")
        ax_eff.set_ylabel(dataset, rotation=-90, va="bottom", fontsize=dataset_size, labelpad=3)
    axes[0, 0].set_yticks(list(y_pos.values()), [label_by_name[n] for n in compactor_order])
    axes[0, 0].set_ylim(max(y_pos.values()) + 0.5, -0.6)
    axes[0, 0].set_xlim(-0.05, 1.05)
    axes[0, 0].xaxis.set_major_locator(mticker.MultipleLocator(0.5))
    axes[0, 0].xaxis.set_minor_locator(mticker.MultipleLocator(0.25))
    axes[0, 0].set_title("Retention Rate", fontsize=title_size, pad=2)
    axes[0, 1].set_title("Effective Retention", fontsize=title_size, pad=2)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    args = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    args.add_argument(
        "--manifest_path",
        type=str,
        default=str(REPO_ROOT / "config/experiments/rq1/main.yaml"),
        help="Manifest of the runs plotted as the main compactor rows.",
    )
    args.add_argument(
        "--supplement_manifest_path",
        type=str,
        default=None,
        help="Optional manifest of runs listed below the main rows (e.g. GPT-5.4-mini at 220k).",
    )
    args.add_argument(
        "--results_root",
        type=str,
        default="/data/compaction_integrity",
        help="Root directory containing canonical run outputs.",
    )
    args.add_argument(
        "--output_path",
        type=str,
        default="/data/compaction_integrity/analysis/retention_if_eval/dot_retention_compliance.pdf",
        help="Where to write the figure.",
    )
    parsed_args = args.parse_args()

    set_paper_style(use_latex=False)
    results_root = Path(parsed_args.results_root)
    summary = build_summary_table(_load_results(Path(parsed_args.manifest_path), results_root))
    supplement = (
        None
        if parsed_args.supplement_manifest_path is None
        else build_summary_table(_load_results(Path(parsed_args.supplement_manifest_path), results_root))
    )
    plot_retention_dot(summary, supplement, Path(parsed_args.output_path))
    print(f"Saved {parsed_args.output_path}")
