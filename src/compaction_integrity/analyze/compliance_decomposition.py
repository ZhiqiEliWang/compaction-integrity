"""Four-group compliance decomposition analysis.

Reads the same manifest as main_exp.py and produces two figures per dataset:

  Primary:    grouped bar chart of absolute compliance for the four groups
              {ub, inj, compact, uninj} across compactors, with bootstrap 95% CIs.
  Secondary:  single-quantity bar of paired compaction loss (C_inj - C_compact)
              per compactor, with bootstrap 95% CIs paired over SSSCs.

Group <-> column mapping (from evaluation.py):
  ub      = compacted_post_sssc_compliant   (SSSC re-presented post-compaction)
  inj     = full_with_sssc_compliant        (SSSC in full pre-compaction context)
  compact = compacted_compliant             (SSSC injected then compacted)
  uninj   = full_without_sssc_compliant     (no SSSC)
"""

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


GROUPS = ["ub", "inj", "compact", "uninj"]
COMPACTOR_SPECIFIC_GROUPS = ["ub", "compact"]
BASELINE_GROUPS = ["inj", "uninj"]
GROUP_COLUMN = {
    "ub": "compacted_post_sssc_compliant",
    "inj": "full_with_sssc_compliant",
    "compact": "compacted_compliant",
    "uninj": "full_without_sssc_compliant",
}
GROUP_LABEL = {
    "ub": "Upper-bound",
    "inj": "Long ctx w/ SC",
    "compact": "Compaction",
    "uninj": "Long ctx w/o SC",
}
ACL_WIDTH_INCH = 7.7 / 2.54
GROUPED_BAR_WIDTH_INCH = 3 * ACL_WIDTH_INCH
GROUPED_BAR_FIGSIZE = (GROUPED_BAR_WIDTH_INCH, GROUPED_BAR_WIDTH_INCH / 1.618 / 2)
LOSS_BAR_FIGSIZE = (6.0, 3.2)

DECOMP_TERMS = ["position_advantage", "compaction_loss", "residual"]
DECOMP_LABEL = {
    "position_advantage": r"Position advantage ($\overline{C}_{\text{ub}}-\overline{C}_{\text{inj}}$)",
    "compaction_loss":    r"Compaction loss ($\overline{C}_{\text{inj}}-\overline{C}_{\text{compact}}$)",
    "residual":           r"Residual compliance ($\overline{C}_{\text{compact}}-\overline{C}_{\text{uninj}}$)",
}


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
    return results_df


def _success(df: pd.DataFrame) -> pd.DataFrame:
    return df.loc[df["compaction_status"] == "success"].copy()


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


def _bootstrap_mean_ci(
    values: np.ndarray,
    n_boot: int = 2000,
    alpha: float = 0.05,
    rng: np.random.Generator | None = None,
) -> tuple[float, float, float]:
    """Return (mean, low, high) of the mean. NaNs dropped before resampling."""
    values = values[~np.isnan(values)]
    if len(values) == 0:
        return float("nan"), float("nan"), float("nan")
    mean = float(values.mean())
    if len(values) == 1:
        return mean, mean, mean
    rng = rng if rng is not None else np.random.default_rng(0)
    idx = rng.integers(0, len(values), size=(n_boot, len(values)))
    boot_means = values[idx].mean(axis=1)
    low = float(np.quantile(boot_means, alpha / 2))
    high = float(np.quantile(boot_means, 1 - alpha / 2))
    return mean, low, high


def _per_pair_compliance(suc: pd.DataFrame) -> pd.DataFrame:
    """Long-form rows: one entry per (dataset, compactor, sample, group)."""
    rows = []
    for group in COMPACTOR_SPECIFIC_GROUPS:
        column = GROUP_COLUMN[group]
        sub = suc[[
            "dataset_config", "dataset_config_label",
            "compactor_name", "compactor_name_label",
            "source_row_index", "sssc_id", column,
        ]].copy()
        sub = sub.rename(columns={column: "compliant"})
        sub["group"] = group
        sub["compliant_float"] = sub["compliant"].map(
            {True: 1.0, False: 0.0}
        )
        rows.append(sub)
    return pd.concat(rows, ignore_index=True)


def _baseline_pair_compliance(df: pd.DataFrame) -> pd.DataFrame:
    """Long-form compactor-independent baseline rows, one per dataset/sample/group."""
    rows = []
    for group in BASELINE_GROUPS:
        column = GROUP_COLUMN[group]
        sub = df[[
            "dataset_config", "dataset_config_label",
            "source_row_index", "sssc_id", column,
        ]].copy()
        sub = (
            sub.dropna(subset=[column])
            .drop_duplicates(
                ["dataset_config", "source_row_index", "sssc_id", column]
            )
        )
        sub = sub.drop_duplicates(
            ["dataset_config", "source_row_index", "sssc_id"],
            keep="first",
        )
        sub = sub.rename(columns={column: "compliant"})
        sub["group"] = group
        sub["compliant_float"] = sub["compliant"].map(
            {True: 1.0, False: 0.0}
        )
        rows.append(sub)
    return pd.concat(rows, ignore_index=True)


def build_group_summary(df: pd.DataFrame) -> pd.DataFrame:
    suc = _success(df)
    long = _per_pair_compliance(suc)
    rng = np.random.default_rng(0)
    out_rows = []
    grouped = long.groupby(
        ["dataset_config", "dataset_config_label",
         "compactor_name", "compactor_name_label", "group"]
    )
    for key, g in grouped:
        dataset_config, dataset_label, compactor_name, compactor_label, group = key
        mean, lo, hi = _bootstrap_mean_ci(g["compliant_float"].to_numpy(), rng=rng)
        out_rows.append({
            "dataset_config": dataset_config,
            "dataset_config_label": dataset_label,
            "compactor_name": compactor_name,
            "compactor_name_label": compactor_label,
            "group": group,
            "mean": mean,
            "ci_low": lo,
            "ci_high": hi,
            "n": int(g["compliant_float"].notna().sum()),
        })

    baseline_long = _baseline_pair_compliance(df)
    compactor_rows = (
        suc[[
            "dataset_config", "dataset_config_label",
            "compactor_name", "compactor_name_label",
        ]]
        .drop_duplicates()
    )
    baseline_summary_rows = []
    baseline_grouped = baseline_long.groupby(
        ["dataset_config", "dataset_config_label", "group"]
    )
    for key, g in baseline_grouped:
        dataset_config, dataset_label, group = key
        mean, lo, hi = _bootstrap_mean_ci(g["compliant_float"].to_numpy(), rng=rng)
        baseline_summary_rows.append({
            "dataset_config": dataset_config,
            "dataset_config_label": dataset_label,
            "group": group,
            "mean": mean,
            "ci_low": lo,
            "ci_high": hi,
            "n": int(g["compliant_float"].notna().sum()),
        })

    baseline_summary = pd.DataFrame(baseline_summary_rows)
    if not baseline_summary.empty:
        baseline_summary = baseline_summary.merge(
            compactor_rows,
            on=["dataset_config", "dataset_config_label"],
            how="left",
        )
        out_rows.extend(baseline_summary.to_dict(orient="records"))

    return pd.DataFrame(out_rows)


def build_decomposition(group_summary: pd.DataFrame) -> pd.DataFrame:
    pivot = group_summary.pivot_table(
        index=["dataset_config", "dataset_config_label",
               "compactor_name", "compactor_name_label"],
        columns="group",
        values="mean",
    ).reset_index()
    pivot["position_advantage"] = pivot["ub"] - pivot["inj"]
    pivot["compaction_loss"] = pivot["inj"] - pivot["compact"]
    pivot["residual"] = pivot["compact"] - pivot["uninj"]
    pivot["total_gap"] = pivot["ub"] - pivot["uninj"]
    pivot["identity_check"] = (
        pivot["position_advantage"] + pivot["compaction_loss"] + pivot["residual"]
        - pivot["total_gap"]
    )
    return pivot


def build_compaction_loss_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Paired compaction loss per compactor: mean of (inj - compact) over SSSCs.

    Bootstrap is over the paired per-SSSC differences, so the CI accounts for
    the within-SSSC correlation between the injected and compacted conditions.
    """
    suc = _success(df)
    inj_col = GROUP_COLUMN["inj"]
    cpt_col = GROUP_COLUMN["compact"]
    pair = suc[[
        "dataset_config", "dataset_config_label",
        "compactor_name", "compactor_name_label",
        "source_row_index", "sssc_id", inj_col, cpt_col,
    ]].copy()
    pair["diff"] = (
        pair[inj_col].map({True: 1.0, False: 0.0})
        - pair[cpt_col].map({True: 1.0, False: 0.0})
    )
    rng = np.random.default_rng(0)
    rows = []
    grouped = pair.groupby(
        ["dataset_config", "dataset_config_label",
         "compactor_name", "compactor_name_label"]
    )
    for key, g in grouped:
        dataset_config, dataset_label, compactor_name, compactor_label = key
        mean, lo, hi = _bootstrap_mean_ci(g["diff"].to_numpy(), rng=rng)
        rows.append({
            "dataset_config": dataset_config,
            "dataset_config_label": dataset_label,
            "compactor_name": compactor_name,
            "compactor_name_label": compactor_label,
            "mean": mean,
            "ci_low": lo,
            "ci_high": hi,
            "n": int(g["diff"].notna().sum()),
        })
    return pd.DataFrame(rows)


def plot_compaction_loss_bars(
    loss_summary: pd.DataFrame, output_dir: Path
) -> None:
    for dataset_config in sorted(loss_summary["dataset_config"].unique()):
        sub = loss_summary.loc[
            loss_summary["dataset_config"] == dataset_config
        ].copy()
        compactor_order = ordered_compactor_labels(sub)
        sub = (
            sub.set_index("compactor_name_label")
            .reindex(compactor_order)
            .reset_index()
        )
        err_low = (sub["mean"] - sub["ci_low"]).clip(lower=0).to_numpy()
        err_high = (sub["ci_high"] - sub["mean"]).clip(lower=0).to_numpy()
        fig, ax = plt.subplots(figsize=LOSS_BAR_FIGSIZE)
        x = np.arange(len(compactor_order))
        ax.bar(x, sub["mean"].to_numpy(), color=sns.color_palette()[0])
        ax.errorbar(
            x, sub["mean"].to_numpy(),
            yerr=[err_low, err_high],
            fmt="none", ecolor="black", capsize=2, elinewidth=0.8,
        )
        ax.axhline(0, color="black", linewidth=0.6)
        _pct(ax)
        ax.set_xticks(x)
        ax.set_xticklabels(compactor_order, rotation=20, ha="right")
        ax.set_xlabel("Compactor")
        ax.set_ylabel("Compaction Loss")
        _save(fig, output_dir / f"compaction_loss__{_safe_name(dataset_config)}.pdf")


def print_compaction_loss_summary(loss_summary: pd.DataFrame) -> None:
    display = loss_summary.copy()
    for col in ["mean", "ci_low", "ci_high"]:
        display[col] = display[col].map(
            lambda v: f"{100*v:+.1f}%" if not np.isnan(v) else "nan"
        )
    display = display[[
        "dataset_config_label", "compactor_name_label",
        "mean", "ci_low", "ci_high", "n",
    ]].rename(columns={
        "dataset_config_label": "dataset",
        "compactor_name_label": "compactor",
    })
    print("=== Paired compaction loss (inj - compact) with 95% bootstrap CI ===")
    print(display.to_string(index=False))
    print()


def plot_grouped_compliance_bars(
    group_summary: pd.DataFrame, output_dir: Path
) -> None:
    compactor_order = _compactor_label_order(group_summary)
    group_order_labels = [GROUP_LABEL[g] for g in GROUPS]
    plot_df = group_summary.copy()
    plot_df["group_label"] = plot_df["group"].map(GROUP_LABEL)
    plot_df["err_low"] = (plot_df["mean"] - plot_df["ci_low"]).clip(lower=0)
    plot_df["err_high"] = (plot_df["ci_high"] - plot_df["mean"]).clip(lower=0)

    for dataset_config in sorted(group_summary["dataset_config"].unique()):
        sub = plot_df.loc[plot_df["dataset_config"] == dataset_config]
        fig, ax = plt.subplots(figsize=GROUPED_BAR_FIGSIZE)
        sns.barplot(
            data=sub,
            x="compactor_name_label",
            y="mean",
            hue="group_label",
            order=compactor_order,
            hue_order=group_order_labels,
            ax=ax,
        )
        # Error bars positioned at each bar's center.
        n_groups = len(GROUPS)
        bar_width = 0.8 / n_groups
        for i, group in enumerate(GROUPS):
            offset = (i - (n_groups - 1) / 2) * bar_width
            for j, compactor_label in enumerate(compactor_order):
                row = sub[
                    (sub["compactor_name_label"] == compactor_label)
                    & (sub["group"] == group)
                ]
                if row.empty or np.isnan(row["mean"].iloc[0]):
                    continue
                ax.errorbar(
                    j + offset,
                    row["mean"].iloc[0],
                    yerr=[[row["err_low"].iloc[0]], [row["err_high"].iloc[0]]],
                    fmt="none",
                    ecolor="black",
                    capsize=2,
                    elinewidth=0.8,
                )
        _pct(ax)
        ax.set_ylim(0, 1.18)
        ax.set_xlabel("")
        ax.set_ylabel("Compliance Rate")
        plt.setp(ax.get_xticklabels(), rotation=25, ha="right")
        ax.legend(
            title=None,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.02),
            ncol=len(GROUPS),
            frameon=False,
            handlelength=1.2,
            columnspacing=1.0,
            borderaxespad=0.0,
        )
        _save(fig, output_dir / f"grouped_compliance__{_safe_name(dataset_config)}.pdf")


def print_group_summary(group_summary: pd.DataFrame) -> None:
    display = group_summary.copy()
    for col in ["mean", "ci_low", "ci_high"]:
        display[col] = display[col].map(
            lambda v: f"{100*v:.1f}%" if not np.isnan(v) else "nan"
        )
    display = display[[
        "dataset_config_label", "compactor_name_label", "group",
        "mean", "ci_low", "ci_high", "n",
    ]].rename(columns={
        "dataset_config_label": "dataset",
        "compactor_name_label": "compactor",
    })
    print("=== Per-group compliance (with 95% bootstrap CI) ===")
    print(display.to_string(index=False))
    print()


def print_compliance_table(decomp: pd.DataFrame, output_dir: Path) -> None:
    """Single multi-dataset table with the four compliance values per compactor.

    Dataset order is alphabetical; compactor order follows ``COMPACTOR_NAME_ORDER``.
    Also writes ``compliance_table.csv`` and ``compliance_table.tex`` (booktabs).
    """
    preferred_datasets = ["WildChat", "HermesAgent", "OpenResearcher"]
    present_datasets = set(decomp["dataset_config_label"].unique())
    dataset_order = [d for d in preferred_datasets if d in present_datasets]
    dataset_order += sorted(present_datasets - set(dataset_order))
    compactor_order = ordered_compactor_labels(decomp)
    table = (
        decomp[[
            "dataset_config_label", "compactor_name_label",
            "ub", "inj", "compact", "uninj",
        ]]
        .rename(columns={
            "dataset_config_label": "Dataset",
            "compactor_name_label": "Compactor",
            "ub": "Upper bound",
            "inj": "Injected baseline",
            "compact": "Compaction",
            "uninj": "Un-injected baseline",
        })
    )
    table["Dataset"] = pd.Categorical(table["Dataset"], categories=dataset_order, ordered=True)
    table["Compactor"] = pd.Categorical(table["Compactor"], categories=compactor_order, ordered=True)
    table = table.sort_values(["Dataset", "Compactor"]).reset_index(drop=True)

    value_cols = ["Injected baseline", "Un-injected baseline", "Compaction", "Upper bound"]
    table = table[["Dataset", "Compactor", *value_cols]]
    numeric = table.copy()
    table.to_csv(output_dir / "compliance_table.csv", index=False)

    for col in value_cols:
        table[col] = table[col].map(
            lambda v: f"{100*v:.1f}%" if not np.isnan(v) else "nan"
        )
    display = table.copy()
    display["Dataset"] = display["Dataset"].astype(str)
    display.loc[display["Dataset"].duplicated(), "Dataset"] = ""
    print("=== Compliance table (Comp_ub, Comp_inj, Comp_compact, Comp_uninj) ===")
    print(display.to_string(index=False))
    print()

    (output_dir / "compliance_table.tex").write_text(
        _to_latex_booktabs(numeric, value_cols), encoding="utf-8"
    )


_LATEX_ESCAPES = {
    "\\": r"\textbackslash{}",
    "&": r"\&",
    "%": r"\%",
    "$": r"\$",
    "#": r"\#",
    "_": r"\_",
    "{": r"\{",
    "}": r"\}",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
}


def _latex_escape(s: str) -> str:
    return "".join(_LATEX_ESCAPES.get(ch, ch) for ch in s)


def _to_latex_booktabs(table: pd.DataFrame, value_cols: list[str]) -> str:
    """Render the compliance table as a booktabs LaTeX table* with grouped Dataset rows."""
    lines: list[str] = []
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(r"\begin{tabular}{ll" + "r" * len(value_cols) + "}")
    lines.append(r"\toprule")
    header = ["Dataset", "Compactor", *value_cols]
    lines.append(" & ".join(_latex_escape(h) for h in header) + r" \\")
    lines.append(r"\midrule")
    last_dataset = None
    datasets = list(table["Dataset"].unique())
    for dataset in datasets:
        if last_dataset is not None:
            lines.append(r"\midrule")
        sub = table.loc[table["Dataset"] == dataset]
        first = True
        for _, row in sub.iterrows():
            label = _latex_escape(str(dataset)) if first else ""
            first = False
            cells = [label, _latex_escape(str(row["Compactor"]))]
            for col in value_cols:
                v = row[col]
                cells.append("nan" if np.isnan(v) else f"{100*v:.1f}\\%")
            lines.append(" & ".join(cells) + r" \\")
        last_dataset = dataset
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\caption{Compliance rates per (dataset, compactor).}")
    lines.append(r"\label{tab:compliance-decomposition}")
    lines.append(r"\end{table*}")
    return "\n".join(lines) + "\n"


def print_decomposition(decomp: pd.DataFrame) -> None:
    display = decomp.copy()
    cols = ["ub", "inj", "compact", "uninj",
            "position_advantage", "compaction_loss", "residual",
            "total_gap", "identity_check"]
    for col in cols:
        display[col] = display[col].map(
            lambda v: f"{100*v:+.1f}%" if not np.isnan(v) else "nan"
        )
    display = display[[
        "dataset_config_label", "compactor_name_label",
        "ub", "inj", "compact", "uninj",
        "position_advantage", "compaction_loss", "residual",
        "total_gap", "identity_check",
    ]].rename(columns={
        "dataset_config_label": "dataset",
        "compactor_name_label": "compactor",
    })
    print("=== Decomposition: total_gap = position_advantage + compaction_loss + residual ===")
    print(display.to_string(index=False))
    print()


if __name__ == "__main__":
    args = argparse.ArgumentParser(
        description="Four-group compliance decomposition on evaluation_results.pkl."
    )
    args.add_argument(
        "--output_dir",
        type=str,
        default="/data/compaction_integrity/analysis/retention_if_eval_decomposition",
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

    with tee_stdout(output_dir / "decomposition.log"):
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
        missing_ub = results_df["compacted_post_sssc_compliant"].isna().sum()
        print(f"Rows with null compacted_post_sssc_compliant (Comp_ub): {missing_ub}")
        print()

        group_summary = build_group_summary(results_df)
        group_summary.to_csv(output_dir / "group_summary.csv", index=False)
        print_group_summary(group_summary)

        decomp = build_decomposition(group_summary)
        decomp.to_csv(output_dir / "decomposition.csv", index=False)
        print_decomposition(decomp)
        print_compliance_table(decomp, output_dir)

        loss_summary = build_compaction_loss_summary(results_df)
        loss_summary.to_csv(output_dir / "compaction_loss_summary.csv", index=False)
        print_compaction_loss_summary(loss_summary)

        plot_grouped_compliance_bars(group_summary, output_dir)
        plot_compaction_loss_bars(loss_summary, output_dir)

        print(f"All outputs written to {output_dir}")
