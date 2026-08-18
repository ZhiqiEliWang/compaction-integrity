"""Analyze the GPT-OSS constraint-targeted Anthropic prompt ablation.

Pairs the regular Anthropic prompt and the Anthropic + SC-targeted prompt on
the same ``(source_row_index, sssc_id)`` rows. Confidence intervals resample
source conversations within each dataset so the 15 SSSCs attached to a source
conversation are kept together.

Usage:
  python -m compaction_integrity.analyze.prompt_targeting
"""

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import binomtest

from compaction_integrity.tokenization import count_tokens_messages_batch


DATASETS = [
    "wildchat_cat_100k",
    "hermes_cat_100k",
    "openresearcher_cat_100k",
]
DATASET_LABELS = {
    "wildchat_cat_100k": "WildChat",
    "hermes_cat_100k": "Hermes",
    "openresearcher_cat_100k": "OpenResearcher",
    "all": "Overall",
}
BASELINE_COMPACTOR = "gpt_oss_120b_anthropic_prompt"
TARGETED_COMPACTOR = "gpt_oss_120b_anthropic_sc_targeted_prompt"
METRICS = {
    "retention": "Retention",
    "compacted_compliant": "Compacted compliance",
    "compacted_post_sssc_compliant": "Post-SC upper bound",
}
PAIR_KEYS = ["source_row_index", "sssc_id"]
RUN_MATCH = {
    "num_rows": 50,
    "global_seed": 42,
    "probe_name": "gpt_oss_120b",
    "sssc_attrs": {
        "position": "top",
        "repeat": 1,
        "explicitness": True,
        "hard": False,
    },
}


def _matching_run(
    runs_dir: Path,
    dataset: str,
    compactor: str,
) -> tuple[str, Path, pd.DataFrame]:
    matches = []
    for run_dir in runs_dir.iterdir():
        metadata_path = run_dir / "metadata.json"
        results_path = run_dir / "evaluation_results.pkl"
        if not (metadata_path.exists() and results_path.exists()):
            continue
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        spec = metadata["run_spec"]
        if (
            spec["dataset"]["name"] == dataset
            and spec["compactor"]["name"] == compactor
            and spec["dataset"]["num_rows"] == RUN_MATCH["num_rows"]
            and spec["global_seed"] == RUN_MATCH["global_seed"]
            and spec["probe"]["name"] == RUN_MATCH["probe_name"]
            and spec["sssc_attrs"] == RUN_MATCH["sssc_attrs"]
        ):
            matches.append((str(metadata["run_id"]), results_path))

    run_id, results_path = matches[0]
    return run_id, results_path, pd.read_pickle(results_path)


def _binary(value: Any) -> float:
    if isinstance(value, (bool, np.bool_)):
        return float(value)
    return float("nan")


def _summary_tokens(contexts: pd.Series) -> list[float]:
    present = contexts.notna()
    lengths = np.full(len(contexts), np.nan)
    lengths[present.to_numpy()] = count_tokens_messages_batch(contexts.loc[present].tolist())
    return lengths.tolist()


def _paired_dataset(
    dataset: str,
    baseline_df: pd.DataFrame,
    targeted_df: pd.DataFrame,
) -> pd.DataFrame:
    identity = PAIR_KEYS + ["sssc_type"]
    columns = identity + ["compaction_status", "compacted_context", *METRICS]
    baseline = baseline_df[columns].copy()
    targeted = targeted_df[columns].copy()
    baseline["summary_tokens"] = _summary_tokens(baseline["compacted_context"])
    targeted["summary_tokens"] = _summary_tokens(targeted["compacted_context"])
    baseline = baseline.drop(columns="compacted_context").add_prefix("baseline_")
    targeted = targeted.drop(columns=["sssc_type", "compacted_context"]).add_prefix("targeted_")
    paired = baseline.merge(
        targeted,
        left_on=[f"baseline_{key}" for key in PAIR_KEYS],
        right_on=[f"targeted_{key}" for key in PAIR_KEYS],
        how="inner",
    )
    paired.insert(0, "dataset", dataset)
    paired = paired.rename(
        columns={
            "baseline_source_row_index": "source_row_index",
            "baseline_sssc_id": "sssc_id",
            "baseline_sssc_type": "sssc_type",
        }
    ).drop(columns=["targeted_source_row_index", "targeted_sssc_id"])
    for metric in METRICS:
        paired[f"baseline_{metric}"] = paired[f"baseline_{metric}"].map(_binary)
        paired[f"targeted_{metric}"] = paired[f"targeted_{metric}"].map(_binary)
    return paired


def _clustered_delta_ci(
    frame: pd.DataFrame,
    baseline_column: str,
    targeted_column: str,
    rng: np.random.Generator,
    n_boot: int,
) -> tuple[float, float]:
    valid = frame.dropna(subset=[baseline_column, targeted_column]).copy()
    valid["difference"] = valid[targeted_column] - valid[baseline_column]
    cluster_stats = (
        valid.groupby(["dataset", "source_row_index"])["difference"]
        .agg(["sum", "count"])
        .reset_index()
    )
    by_dataset = [
        group[["sum", "count"]].to_numpy(dtype=float)
        for _, group in cluster_stats.groupby("dataset", sort=False)
    ]
    bootstrap = np.empty(n_boot)
    for index in range(n_boot):
        total = 0.0
        count = 0.0
        for clusters in by_dataset:
            sampled = clusters[rng.integers(0, len(clusters), size=len(clusters))]
            total += sampled[:, 0].sum()
            count += sampled[:, 1].sum()
        bootstrap[index] = total / count
    low, high = np.quantile(bootstrap, [0.025, 0.975])
    return float(low), float(high)


def _metric_summary(
    paired: pd.DataFrame,
    group_columns: list[str],
    rng: np.random.Generator,
    n_boot: int,
) -> pd.DataFrame:
    rows = []
    grouped = [((), paired)] if not group_columns else paired.groupby(group_columns, sort=False)
    for group_key, group in grouped:
        group_key = group_key if isinstance(group_key, tuple) else (group_key,)
        group_values = dict(zip(group_columns, group_key))
        for metric, metric_label in METRICS.items():
            baseline_column = f"baseline_{metric}"
            targeted_column = f"targeted_{metric}"
            valid = group.dropna(subset=[baseline_column, targeted_column])
            baseline = valid[baseline_column].to_numpy(dtype=float)
            targeted = valid[targeted_column].to_numpy(dtype=float)
            losses = int(((baseline == 1) & (targeted == 0)).sum())
            gains = int(((baseline == 0) & (targeted == 1)).sum())
            discordant = gains + losses
            low, high = _clustered_delta_ci(
                valid, baseline_column, targeted_column, rng, n_boot
            )
            rows.append(
                {
                    **group_values,
                    "metric": metric,
                    "metric_label": metric_label,
                    "n_total_pairs": len(group),
                    "n_paired_decisive": len(valid),
                    "n_excluded_null": len(group) - len(valid),
                    "baseline_true": int(baseline.sum()),
                    "targeted_true": int(targeted.sum()),
                    "baseline_rate": float(baseline.mean()),
                    "targeted_rate": float(targeted.mean()),
                    "targeted_minus_baseline": float((targeted - baseline).mean()),
                    "difference_ci_low": low,
                    "difference_ci_high": high,
                    "targeted_gain_count": gains,
                    "targeted_loss_count": losses,
                    "unchanged_true_count": int(((baseline == 1) & (targeted == 1)).sum()),
                    "unchanged_false_count": int(((baseline == 0) & (targeted == 0)).sum()),
                    "mcnemar_exact_p": float(
                        binomtest(min(gains, losses), discordant, 0.5).pvalue
                        if discordant
                        else 1.0
                    ),
                }
            )
    return pd.DataFrame(rows)


def _run_summary(
    sources: pd.DataFrame,
    runs: list[tuple[str, str, str, pd.DataFrame]],
) -> pd.DataFrame:
    rows = []
    for dataset, prompt, run_id, frame in runs:
        row = {
            "dataset": dataset,
            "prompt": prompt,
            "run_id": run_id,
            "n_rows": len(frame),
            "compaction_success_count": int(frame["compaction_status"].eq("success").sum()),
        }
        for metric in METRICS:
            row[f"{metric}_non_null"] = int(frame[metric].notna().sum())
        rows.append(row)
    return pd.DataFrame(rows).merge(sources, on=["dataset", "prompt", "run_id"])


def _summary_length(paired: pd.DataFrame) -> pd.DataFrame:
    rows = []
    groups = [("all", paired), *list(paired.groupby("dataset", sort=False))]
    for dataset, group in groups:
        valid = group.dropna(subset=["baseline_summary_tokens", "targeted_summary_tokens"])
        rows.append(
            {
                "dataset": dataset,
                "n_paired": len(valid),
                "baseline_mean_tokens": float(valid["baseline_summary_tokens"].mean()),
                "targeted_mean_tokens": float(valid["targeted_summary_tokens"].mean()),
                "targeted_minus_baseline_tokens": float(
                    (valid["targeted_summary_tokens"] - valid["baseline_summary_tokens"]).mean()
                ),
            }
        )
    return pd.DataFrame(rows)


def _plot_deltas(summary: pd.DataFrame, output_path: Path) -> None:
    dataset_order = ["all", *DATASETS]
    metric_order = list(METRICS)
    fig, axes = plt.subplots(1, len(metric_order), figsize=(10.5, 3.3), sharey=True)
    for axis, metric in zip(axes, metric_order):
        frame = (
            summary.loc[summary["metric"] == metric]
            .set_index("dataset")
            .loc[dataset_order]
            .reset_index()
        )
        estimate = frame["targeted_minus_baseline"].to_numpy()
        low = frame["difference_ci_low"].to_numpy()
        high = frame["difference_ci_high"].to_numpy()
        positions = np.arange(len(frame))
        axis.errorbar(
            estimate,
            positions,
            xerr=np.vstack([estimate - low, high - estimate]),
            fmt="o",
            capsize=3,
            color="#0072B2",
        )
        axis.axvline(0, color="black", linewidth=0.8, linestyle="--")
        axis.set_title(METRICS[metric])
        axis.set_xlabel("Targeted - baseline")
        axis.set_yticks(positions, [DATASET_LABELS[name] for name in frame["dataset"]])
        axis.xaxis.set_major_formatter(lambda value, _: f"{value:+.0%}")
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def _pct(value: float) -> str:
    return f"{value:.1%}"


def _write_report(
    path: Path,
    summary: pd.DataFrame,
    length_summary: pd.DataFrame,
    sources: pd.DataFrame,
) -> None:
    overall = summary.loc[summary["dataset"] == "all"].set_index("metric")
    lines = [
        "# GPT-OSS targeted Anthropic prompt analysis",
        "",
        "The analysis pairs the regular `anthropic` prompt with the additive "
        "`anthropic-sc-targeted` prompt on `(source_row_index, sssc_id)`.",
        "",
        "## Overall paired results",
        "",
        "| Metric | Paired denominator | Baseline | Targeted | Difference | 95% cluster CI | Gains / losses | Exact p |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for metric in METRICS:
        row = overall.loc[metric]
        lines.append(
            f"| {METRICS[metric]} | {int(row['n_paired_decisive'])} | "
            f"{_pct(row['baseline_rate'])} | {_pct(row['targeted_rate'])} | "
            f"{_pct(row['targeted_minus_baseline'])} | "
            f"[{_pct(row['difference_ci_low'])}, {_pct(row['difference_ci_high'])}] | "
            f"{int(row['targeted_gain_count'])} / {int(row['targeted_loss_count'])} | "
            f"{row['mcnemar_exact_p']:.3g} |"
        )

    lines.extend([
        "",
        "## By dataset",
        "",
        "| Dataset | Metric | n | Baseline | Targeted | Difference | 95% cluster CI |",
        "|---|---|---:|---:|---:|---:|---:|",
    ])
    for _, row in summary.loc[summary["dataset"] != "all"].iterrows():
        lines.append(
            f"| {DATASET_LABELS[row['dataset']]} | {row['metric_label']} | "
            f"{int(row['n_paired_decisive'])} | {_pct(row['baseline_rate'])} | "
            f"{_pct(row['targeted_rate'])} | {_pct(row['targeted_minus_baseline'])} | "
            f"[{_pct(row['difference_ci_low'])}, {_pct(row['difference_ci_high'])}] |"
        )

    overall_length = length_summary.loc[length_summary["dataset"] == "all"].iloc[0]
    lines.extend([
        "",
        "## Summary length",
        "",
        f"Across {int(overall_length['n_paired'])} paired summaries, the regular prompt "
        f"averaged {overall_length['baseline_mean_tokens']:.1f} tokens and the targeted "
        f"prompt averaged {overall_length['targeted_mean_tokens']:.1f} tokens "
        f"(difference {overall_length['targeted_minus_baseline_tokens']:+.1f}).",
        "",
        "## Denominators and uncertainty",
        "",
        "Each metric uses only pairs where both prompts have a non-null binary verdict; "
        "`n_excluded_null` in `metric_summary.csv` records excluded pairs. The 95% "
        "confidence intervals use a paired cluster bootstrap over source conversations, "
        "stratified by dataset. Exact p-values use the two-sided McNemar/binomial test "
        "over discordant paired outcomes.",
        "",
        "## Input runs",
        "",
    ])
    for _, row in sources.iterrows():
        lines.append(f"- `{row['dataset']}` / `{row['prompt']}`: `{row['run_id']}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-root", type=Path, default=Path("/data/compaction_integrity")
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("/data/compaction_integrity/analysis/prompt_targeting_gpt_oss"),
    )
    parser.add_argument("--n-boot", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    runs_dir = args.results_root / "runs"
    paired_frames = []
    source_rows = []
    loaded_runs = []
    for dataset in DATASETS:
        baseline_id, baseline_path, baseline = _matching_run(
            runs_dir, dataset, BASELINE_COMPACTOR
        )
        targeted_id, targeted_path, targeted = _matching_run(
            runs_dir, dataset, TARGETED_COMPACTOR
        )
        paired_frames.append(_paired_dataset(dataset, baseline, targeted))
        for prompt, run_id, result_path, frame in [
            ("anthropic", baseline_id, baseline_path, baseline),
            ("anthropic-sc-targeted", targeted_id, targeted_path, targeted),
        ]:
            source_rows.append(
                {
                    "dataset": dataset,
                    "prompt": prompt,
                    "run_id": run_id,
                    "evaluation_results_path": str(result_path),
                }
            )
            loaded_runs.append((dataset, prompt, run_id, frame))

    paired = pd.concat(paired_frames, ignore_index=True)
    rng = np.random.default_rng(args.seed)
    overall = _metric_summary(paired, [], rng, args.n_boot)
    overall.insert(0, "dataset", "all")
    by_dataset = _metric_summary(paired, ["dataset"], rng, args.n_boot)
    metric_summary = pd.concat([overall, by_dataset], ignore_index=True)
    by_sssc_type = _metric_summary(
        paired, ["dataset", "sssc_type"], rng, args.n_boot
    )
    sources = pd.DataFrame(source_rows)
    run_summary = _run_summary(sources, loaded_runs)
    length_summary = _summary_length(paired)

    paired.to_csv(args.out_dir / "paired_rows.csv", index=False)
    sources.to_csv(args.out_dir / "input_runs.csv", index=False)
    run_summary.to_csv(args.out_dir / "run_summary.csv", index=False)
    metric_summary.to_csv(args.out_dir / "metric_summary.csv", index=False)
    by_sssc_type.to_csv(args.out_dir / "by_sssc_type.csv", index=False)
    length_summary.to_csv(args.out_dir / "summary_length.csv", index=False)
    _plot_deltas(metric_summary, args.out_dir / "metric_deltas.pdf")
    _write_report(args.out_dir / "report.md", metric_summary, length_summary, sources)

    print(metric_summary.to_string(index=False))
    print(f"\nAnalysis saved to {args.out_dir}")


if __name__ == "__main__":
    main()
