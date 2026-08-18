# Visualizations and stats for results from the rq3/diff_prefix experiment.
#
# Each run is parametrised by two binary "prefix" flags (explicit on/off,
# hard on/off), giving a 2x2 quadrant per compactor. For each compactor we
# plot the retention rate and the effect-retention metric across the quadrant.

import argparse
from pathlib import Path
import re
import sys

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MANIFEST_PATH = REPO_ROOT / "config/experiments/rq3/diff_prefix.yaml"

from compaction_integrity.analyze.utils import (
    extract_compactor_name,
    extract_dataset_config,
    fmt_compactor_label,
    fmt_dataset_label,
    load_manifest_results,
    ordered_compactor_labels,
    tee_stdout,
)
from compaction_integrity.viz_config import set_paper_style, set_talk_style


PREFIX_SUFFIXES: list[tuple[str, bool, bool]] = [
    # Order matters: longer/more-specific suffixes must come first.
    ("_non_explicit_non_hard", False, False),
    ("_non_explicit_hard", False, True),
    ("_explicit_hard", True, True),
    ("_explicit", True, False),
]

EXPLICIT_ORDER = [False, True]
HARD_ORDER = [False, True]
EXPLICIT_LABELS = {False: "non-explicit", True: "explicit"}
HARD_LABELS = {False: "non-hard", True: "hard"}

METRIC_TITLES = {
    "retention_rate": "Retention Rate",
    "effect_retention": "Effect Retention",
}


def _parse_prefix_flags(label: str) -> tuple[bool, bool]:
    for suffix, explicit, hard in PREFIX_SUFFIXES:
        if label.endswith(suffix):
            return explicit, hard
    raise ValueError(f"Could not parse prefix flags from label: {label!r}")


def _safe_name(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", s).strip("_")


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

    flags = results_df["run_label"].map(_parse_prefix_flags)
    results_df["explicit"] = flags.map(lambda x: x[0])
    results_df["hard"] = flags.map(lambda x: x[1])
    return results_df


def _success(df: pd.DataFrame) -> pd.DataFrame:
    return df.loc[df["compaction_status"] == "success"].copy()


def _rate(series: pd.Series) -> float:
    valid = series.dropna()
    if len(valid) == 0:
        return float("nan")
    return float(valid.mean())


def build_summary_table(df: pd.DataFrame) -> pd.DataFrame:
    suc = _success(df)
    group_cols = [
        "dataset_config",
        "dataset_config_label",
        "compactor_name",
        "compactor_name_label",
        "explicit",
        "hard",
    ]

    rows = []
    for key, g in suc.groupby(group_cols):
        (
            dataset_config,
            dataset_label,
            compactor_name,
            compactor_label,
            explicit,
            hard,
        ) = key
        retention_rate = _rate(g["retention"])
        compaction_compliance = _rate(g["compacted_compliant"])
        uninjected_baseline = _rate(g["full_without_sssc_compliant"])
        upper_bound = _rate(g["compacted_post_sssc_compliant"])
        calibrated = (
            compaction_compliance - uninjected_baseline
            if not (np.isnan(compaction_compliance) or np.isnan(uninjected_baseline))
            else float("nan")
        )
        denom = upper_bound - uninjected_baseline
        effect_retention = (
            calibrated / denom
            if not (np.isnan(calibrated) or np.isnan(denom)) and denom != 0
            else float("nan")
        )
        rows.append(
            {
                "dataset_config": dataset_config,
                "dataset": dataset_label,
                "compactor_name": compactor_name,
                "compactor": compactor_label,
                "explicit": bool(explicit),
                "hard": bool(hard),
                "n": len(g),
                "retention_rate": retention_rate,
                "compaction_compliance": compaction_compliance,
                "uninjected_baseline_compliance": uninjected_baseline,
                "upper_bound_compliance": upper_bound,
                "calibrated_compliance": calibrated,
                "effect_retention": effect_retention,
            }
        )
    return pd.DataFrame(rows)


def _quadrant_matrix(g: pd.DataFrame, metric: str) -> pd.DataFrame:
    pivot = g.pivot_table(
        index="hard",
        columns="explicit",
        values=metric,
        aggfunc="mean",
    ).reindex(index=HARD_ORDER, columns=EXPLICIT_ORDER)
    pivot.index = [HARD_LABELS[v] for v in pivot.index]
    pivot.columns = [EXPLICIT_LABELS[v] for v in pivot.columns]
    pivot.index.name = "Prefix: Hard"
    pivot.columns.name = "Prefix: Explicit"
    return pivot


def _plot_single_quadrant(
    pivot: pd.DataFrame,
    title: str,
    metric: str,
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(3.2, 2.8))
    if metric == "retention_rate":
        vmin, vmax, cmap = 0.0, 1.0, "viridis"
    else:
        vmin, vmax, cmap = -0.2, 1.0, "magma"
    sns.heatmap(
        pivot,
        annot=True,
        fmt=".0%",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        cbar_kws={"format": mticker.PercentFormatter(1.0), "label": METRIC_TITLES[metric]},
        ax=ax,
    )
    ax.set_title(title)
    ax.set_xlabel(pivot.columns.name)
    ax.set_ylabel(pivot.index.name)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    print(f"Saved figure: {out_path}")
    plt.close(fig)


def _plot_grid(
    summary: pd.DataFrame,
    metric: str,
    out_path: Path,
) -> None:
    compactor_order = ordered_compactor_labels(
        summary.rename(columns={"compactor": "compactor_name_label"})
    )
    n = len(compactor_order)
    n_cols = min(n, 5)
    n_rows = int(np.ceil(n / n_cols))
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(2.6 * n_cols + 0.6, 2.6 * n_rows + 0.4),
        squeeze=False,
    )
    if metric == "retention_rate":
        vmin, vmax, cmap = 0.0, 1.0, "viridis"
    else:
        vmin, vmax, cmap = -0.2, 1.0, "magma"

    for idx, compactor_label in enumerate(compactor_order):
        r, c = divmod(idx, n_cols)
        ax = axes[r][c]
        g = summary.loc[summary["compactor"] == compactor_label]
        pivot = _quadrant_matrix(g, metric)
        sns.heatmap(
            pivot,
            annot=True,
            fmt=".0%",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            cbar=False,
            ax=ax,
        )
        ax.set_title(compactor_label, fontsize=10)
        ax.set_xlabel(pivot.columns.name if r == n_rows - 1 else "")
        ax.set_ylabel(pivot.index.name if c == 0 else "")

    for idx in range(n, n_rows * n_cols):
        r, c = divmod(idx, n_cols)
        axes[r][c].axis("off")

    fig.suptitle(METRIC_TITLES[metric], y=1.02)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    print(f"Saved figure: {out_path}")
    plt.close(fig)


# Framing combinations ordered by expected retention (weakest -> strongest).
# Each entry: (hard, explicit, label)
FRAMING_ORDER: list[tuple[bool, bool, str]] = [
    (False, False, "soft+contextualized"),
    (True, False, "hard+contextualized"),
    (False, True, "soft+direct"),
    (True, True, "hard+direct"),
]


def _framing_pivot(g: pd.DataFrame, metric: str) -> pd.DataFrame:
    pivot = g.pivot_table(
        index="compactor",
        columns=["hard", "explicit"],
        values=metric,
        aggfunc="mean",
    )
    ordered_cols = [(h, e) for h, e, _ in FRAMING_ORDER if (h, e) in pivot.columns]
    pivot = pivot.reindex(columns=ordered_cols)
    pivot.columns = [lab for h, e, lab in FRAMING_ORDER if (h, e) in ordered_cols]
    compactor_order = ordered_compactor_labels(
        g.rename(columns={"compactor": "compactor_name_label"})
    )
    pivot = pivot.reindex(index=[c for c in compactor_order if c in pivot.index])
    return pivot


def _plot_framing_grouped_bar(
    pivot: pd.DataFrame,
    metric: str,
    out_path: Path,
) -> None:
    framings = list(pivot.columns)
    compactors = list(pivot.index)
    n_framings = len(framings)
    n_compactors = len(compactors)

    x = np.arange(n_framings)
    total_width = 0.8
    bar_width = total_width / max(n_compactors, 1)

    fig, ax = plt.subplots(figsize=(6.3, 2.6))
    cmap = plt.get_cmap("tab10")
    for i, compactor in enumerate(compactors):
        offsets = x - total_width / 2 + bar_width * (i + 0.5)
        values = pivot.loc[compactor].values
        bars = ax.bar(
            offsets,
            values,
            bar_width,
            label=compactor,
            color=cmap(i % 10),
        )
        ax.bar_label(
            bars,
            labels=[
                f"{100 * v:.0f}%" if np.isfinite(v) else "" for v in values
            ],
            padding=2,
            fontsize=9,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(framings, rotation=0)
    ax.set_xlabel("Framing")
    ax.set_ylabel(METRIC_TITLES[metric])
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    if metric == "retention_rate":
        max_val = np.nanmax(pivot.values)
        upper = 1.5 * max_val if np.isfinite(max_val) and max_val > 0 else 1.0
        ax.set_ylim(0.0, upper)
    else:
        ax.axhline(0.0, color="black", linewidth=0.6, alpha=0.5)
    ax.legend(loc="best", fontsize=8, ncol=min(n_compactors, 3))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    print(f"Saved figure: {out_path}")
    plt.close(fig)


def _plot_framing_slope(
    pivot: pd.DataFrame,
    metric: str,
    out_path: Path,
) -> None:
    framings = list(pivot.columns)
    compactors = list(pivot.index)
    x = np.arange(len(framings))

    fig, ax = plt.subplots(figsize=(6.3, 2.6))
    cmap = plt.get_cmap("tab10")
    for i, compactor in enumerate(compactors):
        values = pivot.loc[compactor].values
        ax.plot(
            x,
            values,
            marker="o",
            linewidth=1.8,
            color=cmap(i % 10),
            label=compactor,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(framings, rotation=0)
    ax.set_xlabel("Framing")
    ax.set_ylabel(METRIC_TITLES[metric])
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    if metric == "retention_rate":
        ax.set_ylim(0.0, 1.0)
    else:
        ax.axhline(0.0, color="black", linewidth=0.6, alpha=0.5)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="best", fontsize=8, ncol=min(len(compactors), 3))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    print(f"Saved figure: {out_path}")
    plt.close(fig)


def plot_framing_comparisons(summary: pd.DataFrame, output_dir: Path) -> None:
    for metric in ("retention_rate", "effect_retention"):
        for dataset_config, g in summary.groupby("dataset_config"):
            pivot = _framing_pivot(g, metric)
            if pivot.empty:
                continue
            table_path = (
                output_dir
                / f"framing_table__{metric}__{_safe_name(dataset_config)}.csv"
            )
            pivot.to_csv(table_path)
            print(f"Saved table: {table_path}")
            display = pivot.map(
                lambda v: f"{100 * v:.1f}%" if np.isfinite(v) else "nan"
            )
            print(f"=== Framing {metric} — {dataset_config} ===")
            print(display.to_string())
            print()
            bar_path = (
                output_dir
                / f"framing_grouped_bar__{metric}__{_safe_name(dataset_config)}.pdf"
            )
            _plot_framing_grouped_bar(
                pivot,
                metric=metric,
                out_path=bar_path,
            )
            slope_path = (
                output_dir
                / f"framing_slope__{metric}__{_safe_name(dataset_config)}.pdf"
            )
            _plot_framing_slope(
                pivot,
                metric=metric,
                out_path=slope_path,
            )


def plot_all_quadrants(summary: pd.DataFrame, output_dir: Path) -> None:
    for metric in ("retention_rate", "effect_retention"):
        for (dataset_config, compactor_name, compactor_label), g in summary.groupby(
            ["dataset_config", "compactor_name", "compactor"]
        ):
            pivot = _quadrant_matrix(g, metric)
            fname = (
                f"quadrant__{metric}__{_safe_name(dataset_config)}__"
                f"{_safe_name(compactor_name)}.pdf"
            )
            _plot_single_quadrant(
                pivot,
                title=compactor_label,
                metric=metric,
                out_path=output_dir / fname,
            )

        for dataset_config, g in summary.groupby("dataset_config"):
            fname = f"quadrant_grid__{metric}__{_safe_name(dataset_config)}.pdf"
            _plot_grid(g, metric, out_path=output_dir / fname)


if __name__ == "__main__":
    args = argparse.ArgumentParser(
        description="Analyze the impact of explicit/hard prefix flags on retention."
    )
    args.add_argument(
        "--output_dir",
        type=str,
        default="/data/compaction_integrity/analysis/diff_prefix",
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

    with tee_stdout(output_dir / "diff_prefix.log"):
        if parsed_args.talk_style:
            set_talk_style(use_latex=False)
        else:
            set_paper_style(use_latex=False)

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

        display = summary.copy()
        for col in [
            "retention_rate",
            "compaction_compliance",
            "uninjected_baseline_compliance",
            "upper_bound_compliance",
            "calibrated_compliance",
            "effect_retention",
        ]:
            display[col] = display[col].map(
                lambda value: f"{100 * value:.1f}%" if not np.isnan(value) else "nan"
            )
        print("=== Summary table ===")
        print(display.to_string(index=False))
        print()

        plot_all_quadrants(summary, output_dir)
        plot_framing_comparisons(summary, output_dir)

        print(f"All outputs written to {output_dir}")
