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


def _pct(ax: plt.Axes) -> None:
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _safe_name(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", s).strip("_")


def _compactor_label_order(df: pd.DataFrame) -> list[str]:
    return ordered_compactor_labels(df)


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
        effect_retention = (
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
                "effect_retention": effect_retention,
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


def plot_sssc_retention_bar(stats: pd.DataFrame, output_dir: Path) -> None:
    sssc_order = (
        stats.groupby("sssc_type_label")["retention_rate"]
        .mean()
        .sort_values(ascending=False)
        .index.tolist()
    )
    compactor_label_order = _compactor_label_order(stats)
    for dataset_config in sorted(stats["dataset_config"].unique()):
        sub = stats.loc[stats["dataset_config"] == dataset_config]
        n_compactors = sub["compactor_name"].nunique()
        fig, ax = plt.subplots(
            figsize=(max(5, 0.9 * len(sssc_order) * max(1, n_compactors / 2) + 1.5), 3.5)
        )
        sns.barplot(
            data=sub,
            x="sssc_type_label",
            y="retention_rate",
            hue="compactor_name_label" if n_compactors > 1 else None,
            hue_order=compactor_label_order if n_compactors > 1 else None,
            order=sssc_order,
            ax=ax,
        )
        _pct(ax)
        ax.set_ylim(0, 1.05)
        ax.set_xlabel("SC Type")
        ax.set_ylabel("Retention Rate")
        if n_compactors > 1:
            ax.legend(title="Compactor", bbox_to_anchor=(1.02, 1.0), loc="upper left")
        _save(fig, output_dir / f"bar_sssc_retention__{_safe_name(dataset_config)}.pdf")


def plot_sssc_retention_heatmap(stats: pd.DataFrame, output_dir: Path) -> None:
    sssc_order = (
        stats.groupby("sssc_type_label")["retention_rate"]
        .mean()
        .sort_values(ascending=False)
        .index.tolist()
    )
    compactor_label_order = _compactor_label_order(stats)
    for dataset_config in sorted(stats["dataset_config"].unique()):
        sub = stats.loc[stats["dataset_config"] == dataset_config]
        pivot = (
            sub.pivot_table(
                index="compactor_name_label",
                columns="sssc_type_label",
                values="retention_rate",
                aggfunc="mean",
            )
            .reindex(index=compactor_label_order, columns=sssc_order)
            .astype(float)
        )
        fig, ax = plt.subplots(
            figsize=(max(4, 0.9 * len(sssc_order) + 2.0), max(2, 0.5 * len(compactor_label_order) + 1.2))
        )
        sns.heatmap(
            pivot,
            annot=True,
            fmt=".0%",
            cmap="viridis",
            vmin=0,
            vmax=1.0,
            cbar_kws={"format": mticker.PercentFormatter(1.0), "label": "Retention Rate"},
            ax=ax,
        )
        ax.set_xlabel("SC Type")
        ax.set_ylabel("Compactor")
        _save(fig, output_dir / f"heatmap_sssc_retention__{_safe_name(dataset_config)}.pdf")


def plot_confusion_heatmap(df: pd.DataFrame, output_dir: Path) -> None:
    suc = _success(df)
    suc["retention_label"] = suc["retention"].map(
        {True: "retained", False: "not_retained"}
    )
    suc["compliance_label"] = (
        suc["compacted_compliant"]
        .map({True: "compliant", False: "not_compliant"})
        .fillna("null")
    )

    compliance_order = ["compliant", "not_compliant", "null"]
    retention_order = ["retained", "not_retained"]

    for (dataset_config, compactor_name), g in suc.groupby(["dataset_config", "compactor_name"]):
        matrix = pd.crosstab(
            g["compliance_label"],
            g["retention_label"],
        ).reindex(
            index=compliance_order,
            columns=retention_order,
            fill_value=0,
        )
        total = matrix.values.sum()
        annot = matrix.map(lambda v: f"{v}\n({100*v/total:.1f}%)")

        fig, ax = plt.subplots(figsize=(4.2, 3.5))
        sns.heatmap(
            matrix,
            annot=annot,
            fmt="",
            cmap="Blues",
            cbar_kws={"label": "Count"},
            ax=ax,
        )
        ax.set_xlabel("Retention")
        ax.set_ylabel("Compaction Compliance")
        fname = f"confusion__{_safe_name(dataset_config)}__{_safe_name(compactor_name)}.pdf"
        _save(fig, output_dir / fname)


def plot_not_retained_yet_complied(df: pd.DataFrame, output_dir: Path) -> None:
    suc = _success(df)
    nr_yc = suc.loc[
        (suc["retention"] == False) & (suc["compacted_compliant"] == True)
    ].copy()

    if nr_yc.empty:
        print("No not-retained-yet-complied rows found; skipping analysis 4.")
        return

    nr_yc["uninjected_label"] = (
        nr_yc["full_without_sssc_compliant"]
        .map({True: "also_uninjected_compliant", False: "not_uninjected_compliant"})
        .fillna("null")
    )

    for (dataset_config, compactor_name), g in nr_yc.groupby(["dataset_config", "compactor_name"]):
        n_total = len(g)
        counts = g["uninjected_label"].value_counts()
        labels = ["also_uninjected_compliant", "not_uninjected_compliant", "null"]
        values = [counts.get(l, 0) for l in labels]
        fractions = [v / n_total for v in values]

        fig, ax = plt.subplots()
        bars = ax.bar(
            labels,
            fractions,
            color=["#4878cf", "#d65f5f", "#aaaaaa"],
        )
        for bar, v, frac in zip(bars, values, fractions):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                frac + 0.01,
                f"{v} ({100*frac:.1f}%)",
                ha="center",
                va="bottom",
                fontsize=9,
            )
        _pct(ax)
        ax.set_ylim(0, 1.1)
        ax.set_xlabel("Un-Injected Baseline Behaviour")
        ax.set_ylabel("Fraction Of Not-Retained-Yet-Complied")
        ax.set_xticks(
            range(len(labels)),
            ["Uninjected\ncompliant", "Uninjected\nnot compliant", "Null"],
            fontsize=9,
        )
        fname = f"nryc_composition__{_safe_name(dataset_config)}__{_safe_name(compactor_name)}.pdf"
        _save(fig, output_dir / fname)

    if nr_yc["compactor_name"].nunique() > 1:
        for dataset_config, g in nr_yc.groupby("dataset_config"):
            summary = (
                g.groupby("compactor_name_label")["uninjected_label"]
                .value_counts(normalize=True)
                .unstack(fill_value=0)
                .reindex(
                    columns=["also_uninjected_compliant", "not_uninjected_compliant", "null"],
                    fill_value=0,
                )
            )
            fig, ax = plt.subplots()
            summary.plot(
                kind="bar",
                stacked=True,
                color=["#4878cf", "#d65f5f", "#aaaaaa"],
                ax=ax,
            )
            _pct(ax)
            ax.set_ylim(0, 1.05)
            ax.set_xlabel("Compactor")
            ax.set_ylabel("Fraction")
            ax.legend(
                ["Uninjected compliant", "Uninjected not compliant", "Null"],
                bbox_to_anchor=(1.02, 1.0),
                loc="upper left",
            )
            fname = f"nryc_composition_stacked__{_safe_name(dataset_config)}.pdf"
            _save(fig, output_dir / fname)


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
    args.add_argument(
        "--talk_style",
        action="store_true",
        help="Use talk plotting style instead of paper plotting style.",
    )
    parsed_args = args.parse_args()

    output_dir = Path(parsed_args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with tee_stdout(output_dir / "retention_if_eval.log"):
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
        display = summary.drop(columns=["dataset_raw", "compactor_raw"])
        for col in [
            "retention_rate",
            "injected_baseline_compliance",
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
        print_null_compaction_compliance_cases(results_df)

        sssc_stats = _sssc_retention_stats(results_df)
        sssc_stats.to_csv(output_dir / "sssc_retention_stats.csv", index=False)
        plot_sssc_retention_bar(sssc_stats, output_dir)
        plot_sssc_retention_heatmap(sssc_stats, output_dir)
        plot_confusion_heatmap(results_df, output_dir)
        plot_not_retained_yet_complied(results_df, output_dir)

        print(f"All outputs written to {output_dir}")
