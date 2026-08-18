# Analyze per-row evaluation stability for OpenResearcher position runs.

import argparse
import ast
from pathlib import Path
import sys
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

REPO_ROOT = Path(__file__).resolve().parents[3]

DEFAULT_RUNS = [
    {
        "run_id": "openresearcher_cat_100k__gpt_oss_120b_anthropic_prompt__gpt_oss_120b__llm_gpt_5_4__n50__full__c77d9f760a",
        "label": "openresearcher_cat_100k_gpt_oss_120b_anthropic_prompt_top",
    },
    {
        "run_id": "openresearcher_cat_100k__gpt_oss_120b_anthropic_prompt__gpt_oss_120b__llm_gpt_5_4__n50__full__84ce838ce5",
        "label": "openresearcher_cat_100k_gpt_oss_120b_anthropic_prompt_equivalent",
    },
    {
        "run_id": "openresearcher_cat_100k__gpt_oss_120b_pi_mono_prompt__gpt_oss_120b__llm_gpt_5_4__n50__full__5699f86b8e",
        "label": "openresearcher_cat_100k_gpt_oss_120b_pi_mono_prompt_top",
    },
    {
        "run_id": "openresearcher_cat_100k__gpt_oss_120b_pi_mono_prompt__gpt_oss_120b__llm_gpt_5_4__n50__full__e1185d3362",
        "label": "openresearcher_cat_100k_gpt_oss_120b_pi_mono_prompt_equivalent",
    },
    {
        "run_id": "openresearcher_cat_100k__qwen30b_anthropic_prompt__gpt_oss_120b__llm_gpt_5_4__n50__full__60ae758def",
        "label": "openresearcher_cat_100k_qwen30b_anthropic_prompt_top",
    },
    {
        "run_id": "openresearcher_cat_100k__qwen30b_anthropic_prompt__gpt_oss_120b__llm_gpt_5_4__n50__full__73621022a0",
        "label": "openresearcher_cat_100k_qwen30b_anthropic_prompt_equivalent",
    },
    {
        "run_id": "openresearcher_cat_100k__recent_5__gpt_oss_120b__llm_gpt_5_4__n50__full__15d0890680",
        "label": "openresearcher_cat_100k_recent_5_top",
    },
    {
        "run_id": "openresearcher_cat_100k__recent_5__gpt_oss_120b__llm_gpt_5_4__n50__full__49a17b1bf2",
        "label": "openresearcher_cat_100k_recent_5_equivalent",
    },
    {
        "run_id": "openresearcher_cat_100k__llmlingua2_t500__gpt_oss_120b__llm_gpt_5_4__n50__full__a5026f1c40",
        "label": "openresearcher_cat_100k_llmlingua2_t500_top",
    },
    {
        "run_id": "openresearcher_cat_100k__llmlingua2_t500__gpt_oss_120b__llm_gpt_5_4__n50__full__1fe08ec2bc",
        "label": "openresearcher_cat_100k_llmlingua2_t500_equivalent",
    },
]

from compaction_integrity.analyze.utils import (  # noqa: E402
    extract_compactor_name,
    fmt_compactor_label,
    load_manifest_results,
    ordered_compactor_labels,
    tee_stdout,
)
from compaction_integrity.viz_config import save_fig, set_paper_style, set_talk_style  # noqa: E402


METRICS = [
    "full_with_sssc_compliant",
    "full_without_sssc_compliant",
    "compacted_compliant",
    "compacted_post_sssc_compliant",
    "retention",
]

METRIC_LABELS = {
    "full_with_sssc_compliant": "Full + SSSC",
    "full_without_sssc_compliant": "Full no SSSC",
    "compacted_compliant": "Compacted",
    "compacted_post_sssc_compliant": "Compacted + post SSSC",
    "retention": "Retention",
}


def _extract_actual_position(sssc_attrs: Any) -> str:
    if isinstance(sssc_attrs, dict):
        return str(sssc_attrs["position"])
    if isinstance(sssc_attrs, str) and sssc_attrs.startswith("{"):
        return str(ast.literal_eval(sssc_attrs)["position"])
    raise ValueError(f"Cannot extract position from sssc_attrs: {sssc_attrs!r}")


def _extract_declared_position(run_label: str) -> str:
    return run_label.rsplit("_", 1)[-1]


def _bool_to_float(value: Any) -> float:
    if pd.isna(value):
        return float("nan")
    return float(bool(value))


def _load_default_results(results_root: Path) -> pd.DataFrame:
    frames = []
    for run in DEFAULT_RUNS:
        run_id = run["run_id"]
        run_dir = results_root / "runs" / run_id
        df = pd.read_pickle(run_dir / "evaluation_results.pkl")
        df["_run_dir"] = str(run_dir)
        df["run_id"] = run_id
        df["experiment"] = "openresearcher_stability"
        df["run_label"] = run["label"]
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def _load_openresearcher_results(
    manifest_path: Path | None,
    results_root: Path,
) -> pd.DataFrame:
    if manifest_path is None:
        df = _load_default_results(results_root)
    else:
        df = load_manifest_results(
            manifest_path=manifest_path,
            results_root=results_root,
            result_file_name="evaluation_results.pkl",
        )
    df = df.loc[df["dataset"] == "openresearcher_cat_100k"].copy()
    df["declared_position"] = df["run_label"].map(_extract_declared_position)
    df["actual_position"] = df["sssc_attrs"].map(_extract_actual_position)
    df["compactor_name"] = df["compactor"].map(extract_compactor_name)
    df["compactor_name_label"] = df["compactor_name"].map(fmt_compactor_label)
    return df


def _dedupe_manifest_aliases(df: pd.DataFrame) -> pd.DataFrame:
    key_cols = [
        "run_id",
        "source_row_index",
        "sssc_id",
        "compactor_name",
        "declared_position",
    ]
    return df.drop_duplicates(subset=key_cols).copy()


def _build_row_stability(df: pd.DataFrame) -> pd.DataFrame:
    key_cols = ["compactor_name", "source_row_index", "sssc_id"]
    rows: list[dict[str, Any]] = []
    df = _dedupe_manifest_aliases(df)
    for metric in METRICS:
        metric_df = df[key_cols + ["declared_position", "actual_position", "run_id", metric]].copy()
        pivot = metric_df.pivot_table(
            index=key_cols,
            columns="declared_position",
            values=metric,
            aggfunc="first",
            dropna=False,
        )
        metadata = metric_df.pivot_table(
            index=key_cols,
            columns="declared_position",
            values=["actual_position", "run_id"],
            aggfunc="first",
            dropna=False,
        )
        if "top" not in pivot.columns:
            raise ValueError("OpenResearcher stability requires a declared top run.")

        compare_positions = [
            str(position) for position in pivot.columns if str(position) != "top"
        ]
        for compare_position in sorted(compare_positions):
            if compare_position not in pivot.columns:
                continue
            for key, values in pivot.iterrows():
                top_value = values["top"]
                compare_value = values[compare_position]
                top_float = _bool_to_float(top_value)
                compare_float = _bool_to_float(compare_value)
                comparable = not (pd.isna(top_float) or pd.isna(compare_float))
                deviation = abs(top_float - compare_float) if comparable else float("nan")
                compactor_name, source_row_index, sssc_id = key
                rows.append(
                    {
                        "compactor_name": compactor_name,
                        "compactor_name_label": fmt_compactor_label(str(compactor_name)),
                        "source_row_index": int(source_row_index),
                        "sssc_id": int(sssc_id),
                        "metric": metric,
                        "comparison": f"top_vs_{compare_position}",
                        "top_run_id": metadata.loc[key, ("run_id", "top")],
                        "compare_run_id": metadata.loc[key, ("run_id", compare_position)],
                        "top_actual_position": metadata.loc[key, ("actual_position", "top")],
                        "compare_declared_position": compare_position,
                        "compare_actual_position": metadata.loc[
                            key, ("actual_position", compare_position)
                        ],
                        "top_value": top_value,
                        "compare_value": compare_value,
                        "comparable": comparable,
                        "match": bool(deviation == 0.0) if comparable else None,
                        "abs_deviation": deviation,
                    }
                )
    return pd.DataFrame(rows)


def _build_summary(row_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    group_cols = ["compactor_name", "compactor_name_label", "metric", "comparison"]
    for key, group in row_df.groupby(group_cols):
        compactor_name, compactor_name_label, metric, comparison = key
        comparable = group.loc[group["comparable"]].copy()
        n_pairs = len(group)
        n_comparable = len(comparable)
        disagreement_count = int((comparable["abs_deviation"] > 0).sum())
        mean_abs_deviation = float(comparable["abs_deviation"].mean()) if n_comparable else float("nan")
        top_rate = float(comparable["top_value"].map(_bool_to_float).mean()) if n_comparable else float("nan")
        compare_rate = (
            float(comparable["compare_value"].map(_bool_to_float).mean())
            if n_comparable
            else float("nan")
        )
        rows.append(
            {
                "compactor_name": compactor_name,
                "compactor_name_label": compactor_name_label,
                "metric": metric,
                "comparison": comparison,
                "n_pairs": n_pairs,
                "n_comparable": n_comparable,
                "disagreement_count": disagreement_count,
                "agreement_rate": 1.0 - mean_abs_deviation if n_comparable else float("nan"),
                "mean_abs_deviation": mean_abs_deviation,
                "top_rate": top_rate,
                "compare_rate": compare_rate,
                "rate_delta": compare_rate - top_rate,
                "consistent": bool(disagreement_count == 0 and n_comparable == n_pairs),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["metric", "compactor_name_label", "comparison"]
    ).reset_index(drop=True)


def _plot_disagreement_heatmap(summary_df: pd.DataFrame, output_dir: Path) -> None:
    plot_df = summary_df.loc[summary_df["n_comparable"] > 0].copy()
    plot_df["metric_label"] = plot_df["metric"].map(METRIC_LABELS)
    pivot = plot_df.pivot(
        index="metric_label",
        columns="compactor_name_label",
        values="mean_abs_deviation",
    )
    metric_order = [METRIC_LABELS[m] for m in METRICS if METRIC_LABELS[m] in pivot.index]
    compactor_order = ordered_compactor_labels(plot_df)
    pivot = pivot.loc[metric_order, compactor_order]

    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    sns.heatmap(
        pivot,
        annot=True,
        fmt=".3f",
        cmap="rocket_r",
        vmin=0,
        vmax=max(0.25, float(plot_df["mean_abs_deviation"].max())),
        linewidths=0.5,
        cbar_kws={"label": "Per-Row Disagreement Rate"},
        ax=ax,
    )
    ax.set_xlabel("Compactor")
    ax.set_ylabel("Metric")
    ax.set_title("OpenResearcher top vs equivalent stability")
    save_fig(output_dir / "openresearcher_stability_disagreement_heatmap.pdf")
    plt.close(fig)


def _plot_retention_disagreement(summary_df: pd.DataFrame, output_dir: Path) -> None:
    plot_df = summary_df.loc[
        (summary_df["metric"] == "retention") & (summary_df["n_comparable"] > 0)
    ].copy()
    compactor_order = ordered_compactor_labels(plot_df)

    fig, ax = plt.subplots(figsize=(5.4, 2.8))
    sns.barplot(
        data=plot_df,
        x="mean_abs_deviation",
        y="compactor_name_label",
        order=compactor_order,
        color="#4C78A8",
        ax=ax,
    )
    ax.set_xlabel("Retention Disagreement Rate")
    ax.set_ylabel("Compactor")
    ax.set_xlim(0, max(0.2, float(plot_df["mean_abs_deviation"].max()) * 1.15))
    ax.set_title("Retention stability")
    save_fig(output_dir / "openresearcher_retention_disagreement.pdf")
    plt.close(fig)


def _plot_rate_delta(summary_df: pd.DataFrame, output_dir: Path) -> None:
    plot_df = summary_df.loc[summary_df["n_comparable"] > 0].copy()
    plot_df["metric_label"] = plot_df["metric"].map(METRIC_LABELS)
    plot_df["rate_delta_abs"] = plot_df["rate_delta"].abs()
    pivot = plot_df.pivot(
        index="metric_label",
        columns="compactor_name_label",
        values="rate_delta_abs",
    )
    metric_order = [METRIC_LABELS[m] for m in METRICS if METRIC_LABELS[m] in pivot.index]
    compactor_order = ordered_compactor_labels(plot_df)
    pivot = pivot.loc[metric_order, compactor_order]

    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    sns.heatmap(
        pivot,
        annot=True,
        fmt=".3f",
        cmap="mako_r",
        vmin=0,
        vmax=max(0.05, float(plot_df["rate_delta_abs"].max())),
        linewidths=0.5,
        cbar_kws={"label": "Absolute Aggregate-Rate Delta"},
        ax=ax,
    )
    ax.set_xlabel("Compactor")
    ax.set_ylabel("Metric")
    ax.set_title("Aggregate-rate drift")
    save_fig(output_dir / "openresearcher_stability_rate_delta_heatmap.pdf")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Measure per-row stability for equivalent OpenResearcher injections."
    )
    parser.add_argument(
        "--manifest_path",
        type=str,
        default=None,
        help="Optional manifest containing injection-position run ids. Defaults to the hard-coded five-compactor OpenResearcher comparison set.",
    )
    parser.add_argument(
        "--results_root",
        type=str,
        default="/data/compaction_integrity",
        help="Root directory containing canonical run outputs.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/data/compaction_integrity/analysis/openresearcher_stability",
        help="Directory for CSV and log outputs.",
    )
    parser.add_argument(
        "--talk_style",
        action="store_true",
        help="Use talk plotting style instead of paper plotting style.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with tee_stdout(output_dir / "openresearcher_stability.log"):
        if args.talk_style:
            set_talk_style(use_latex=False)
        else:
            set_paper_style(use_latex=False)

        results_df = _load_openresearcher_results(
            manifest_path=Path(args.manifest_path) if args.manifest_path else None,
            results_root=Path(args.results_root),
        )
        row_df = _build_row_stability(results_df)
        summary_df = _build_summary(row_df)

        row_path = output_dir / "openresearcher_stability_by_row.csv"
        summary_path = output_dir / "openresearcher_stability_summary.csv"
        row_df.to_csv(row_path, index=False)
        summary_df.to_csv(summary_path, index=False)

        print(summary_df.to_string(index=False))
        _plot_disagreement_heatmap(summary_df, output_dir)
        _plot_retention_disagreement(summary_df, output_dir)
        _plot_rate_delta(summary_df, output_dir)
        print(f"Row-level output written to {row_path}")
        print(f"Summary output written to {summary_path}")
