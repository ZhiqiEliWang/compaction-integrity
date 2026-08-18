"""Compare free-generation action compliance with the original MCQ probes.

The generative reprobe pickle retains the original MCQ columns. This analysis
pairs the two verdicts row by row, treats ``not_enough_information`` as a
separate outcome, and optionally repeats the comparison for the MCQ prober-swap
runs that share the same contexts.

Usage:
  python -m compaction_integrity.analyze.generative_mcq
"""

import argparse
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import binomtest
import yaml


GENERATIVE_RUN_ID = (
    "hermes_cat_100k__gpt_oss_120b_anthropic_prompt__"
    "gpt_oss_120b_generative__llm_gpt_5_4__n50__full__e0e5c4bd4d"
)
PROBER_MANIFEST = Path("config/experiments/additional/prober_swap_hermes100k.yaml")
CASES = {
    "full_with_sssc": "full_with_sssc_compliant",
    "full_without_sssc": "full_without_sssc_compliant",
    "compacted": "compacted_compliant",
    "compacted_post_sssc": "compacted_post_sssc_compliant",
}
CASE_LABELS = {
    "full_with_sssc": "Long context + SC",
    "full_without_sssc": "Long context - SC",
    "compacted": "Compacted",
    "compacted_post_sssc": "Post-SC upper bound",
}
CONTRASTS = {
    "injected_effect": ("full_with_sssc", "full_without_sssc"),
    "compaction_loss": ("full_with_sssc", "compacted"),
    "post_sssc_gain": ("compacted_post_sssc", "compacted"),
}
KEYS = ["source_row_index", "sssc_id"]


def _binary(value: Any) -> float:
    if isinstance(value, (bool, np.bool_)):
        return float(value)
    return float("nan")


def _cohen_kappa(left: np.ndarray, right: np.ndarray) -> float:
    observed = float((left == right).mean())
    left_true = float(left.mean())
    right_true = float(right.mean())
    expected = left_true * right_true + (1 - left_true) * (1 - right_true)
    return 1.0 if expected == 1.0 else (observed - expected) / (1 - expected)


def _paired_difference_ci(
    left: np.ndarray,
    right: np.ndarray,
    rng: np.random.Generator,
    n_boot: int,
) -> tuple[float, float, float]:
    differences = left - right
    estimate = float(differences.mean())
    indices = rng.integers(0, len(differences), size=(n_boot, len(differences)))
    bootstrap = differences[indices].mean(axis=1)
    return estimate, float(np.quantile(bootstrap, 0.025)), float(np.quantile(bootstrap, 0.975))


def _tool_names(tool_calls: Any) -> list[str]:
    return [str(call["name"]) for call in tool_calls]


def build_row_level(generative_df: pd.DataFrame) -> pd.DataFrame:
    frames = []
    identity = KEYS + ["sssc_type", "sssc_message", "sssc_probe", "retention"]
    for case, mcq_column in CASES.items():
        frame = generative_df[identity].copy()
        frame["case"] = case
        frame["mcq"] = generative_df[mcq_column].map(_binary)
        frame["generative_raw"] = generative_df[
            f"{case}_generative_judge_verdict"
        ]
        frame["generative"] = frame["generative_raw"].map(_binary)
        frame["generative_decisive"] = frame["generative"].notna()
        frame["tool_names"] = generative_df[
            f"{case}_generative_tool_calls"
        ].map(_tool_names)
        frame["tool_call_count"] = frame["tool_names"].map(len)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def _agreement_summary(
    frame: pd.DataFrame,
    group_columns: list[str],
    mcq_column: str = "mcq",
    rng: np.random.Generator | None = None,
    n_boot: int = 5000,
) -> pd.DataFrame:
    rng = rng or np.random.default_rng(0)
    rows = []
    for key, group in frame.groupby(group_columns, sort=False):
        key = key if isinstance(key, tuple) else (key,)
        decisive = group.dropna(subset=[mcq_column, "generative"])
        mcq = decisive[mcq_column].to_numpy(dtype=float)
        generative = decisive["generative"].to_numpy(dtype=float)
        mcq_true_gen_true = int(((mcq == 1) & (generative == 1)).sum())
        mcq_true_gen_false = int(((mcq == 1) & (generative == 0)).sum())
        mcq_false_gen_true = int(((mcq == 0) & (generative == 1)).sum())
        mcq_false_gen_false = int(((mcq == 0) & (generative == 0)).sum())
        delta, ci_low, ci_high = _paired_difference_ci(
            generative, mcq, rng, n_boot
        )
        discordant = mcq_true_gen_false + mcq_false_gen_true
        mcnemar_p = (
            binomtest(
                min(mcq_true_gen_false, mcq_false_gen_true),
                discordant,
                0.5,
            ).pvalue
            if discordant
            else 1.0
        )
        rows.append({
            **dict(zip(group_columns, key)),
            "n_total": len(group),
            "n_generative_true": int((group["generative"] == 1).sum()),
            "n_generative_false": int((group["generative"] == 0).sum()),
            "n_not_enough_information": int(group["generative"].isna().sum()),
            "generative_decisive_rate": float(group["generative"].notna().mean()),
            "generative_rate_decisive": float(generative.mean()),
            "generative_rate_lower_bound": float((group["generative"] == 1).mean()),
            "generative_rate_upper_bound": float(
                ((group["generative"] == 1) | group["generative"].isna()).mean()
            ),
            "mcq_rate_all": float(group[mcq_column].mean()),
            "mcq_rate_decisive_rows": float(mcq.mean()),
            "generative_minus_mcq": delta,
            "difference_ci_low": ci_low,
            "difference_ci_high": ci_high,
            "agreement": float((mcq == generative).mean()),
            "cohen_kappa": _cohen_kappa(mcq, generative),
            "mcq_true_gen_true": mcq_true_gen_true,
            "mcq_true_gen_false": mcq_true_gen_false,
            "mcq_false_gen_true": mcq_false_gen_true,
            "mcq_false_gen_false": mcq_false_gen_false,
            "mcnemar_exact_p": float(mcnemar_p),
        })
    return pd.DataFrame(rows)


def build_prober_comparison(
    row_level: pd.DataFrame,
    manifest_path: Path,
    results_root: Path,
    rng: np.random.Generator,
    n_boot: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    joined = []
    for entry in manifest["runs"]:
        run_id = str(entry["run_id"])
        label = str(entry.get("label", run_id))
        mcq_df = pd.read_pickle(results_root / "runs" / run_id / "evaluation_results.pkl")
        for case, mcq_column in CASES.items():
            mcq = mcq_df[KEYS + [mcq_column]].copy()
            mcq["mcq_prober"] = label
            mcq["case"] = case
            mcq["prober_mcq"] = mcq[mcq_column].map(_binary)
            base = row_level.loc[row_level["case"] == case].drop(columns="mcq")
            joined.append(base.merge(mcq[KEYS + ["case", "mcq_prober", "prober_mcq"]], on=KEYS + ["case"]))
    paired = pd.concat(joined, ignore_index=True)
    summary = _agreement_summary(
        paired,
        ["mcq_prober", "case"],
        mcq_column="prober_mcq",
        rng=rng,
        n_boot=n_boot,
    )
    return paired, summary


def build_condition_contrasts(
    row_level: pd.DataFrame,
    prober_paired: pd.DataFrame,
    rng: np.random.Generator,
    n_boot: int,
) -> pd.DataFrame:
    metrics: list[tuple[str, str, pd.DataFrame]] = [
        ("Generative action judge", "generative", row_level)
    ]
    for prober, group in prober_paired.groupby("mcq_prober", sort=False):
        metrics.append((f"MCQ: {prober}", "prober_mcq", group))

    rows = []
    for metric_label, value_column, frame in metrics:
        wide = frame.pivot_table(index=KEYS, columns="case", values=value_column)
        for contrast, (left_case, right_case) in CONTRASTS.items():
            paired = wide[[left_case, right_case]].dropna()
            left = paired[left_case].to_numpy(dtype=float)
            right = paired[right_case].to_numpy(dtype=float)
            difference, ci_low, ci_high = _paired_difference_ci(
                left, right, rng, n_boot
            )
            rows.append({
                "metric": metric_label,
                "contrast": contrast,
                "left_case": left_case,
                "right_case": right_case,
                "n_paired": len(paired),
                "left_rate": float(left.mean()),
                "right_rate": float(right.mean()),
                "rate_difference": difference,
                "difference_ci_low": ci_low,
                "difference_ci_high": ci_high,
            })
    return pd.DataFrame(rows)


def build_tool_summary(row_level: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (case, sssc_id), group in row_level.groupby(["case", "sssc_id"], sort=False):
        counts = Counter(name for names in group["tool_names"] for name in names)
        rows.append({
            "case": case,
            "sssc_id": sssc_id,
            "n": len(group),
            "any_tool_call_rate": float((group["tool_call_count"] > 0).mean()),
            "mean_tool_calls": float(group["tool_call_count"].mean()),
            "tool_name_counts": "; ".join(
                f"{name}:{count}" for name, count in counts.most_common()
            ),
        })
    return pd.DataFrame(rows)


def build_retention_relationship(row_level: pd.DataFrame) -> pd.DataFrame:
    compacted = row_level.loc[row_level["case"] == "compacted"].copy()
    rows = []
    for retention, group in compacted.groupby("retention"):
        decisive = group.dropna(subset=["generative"])
        rows.append({
            "retention_judge": bool(retention),
            "n": len(group),
            "n_generative_decisive": len(decisive),
            "generative_decisive_rate": len(decisive) / len(group),
            "generative_compliance_rate": float(decisive["generative"].mean()),
            "mcq_compliance_rate": float(group["mcq"].mean()),
        })
    return pd.DataFrame(rows)


def plot_case_rates(case_summary: pd.DataFrame, out_path: Path) -> None:
    ordered = list(CASES)
    summary = case_summary.set_index("case").loc[ordered]
    x = np.arange(len(ordered))
    width = 0.36
    fig, ax = plt.subplots(figsize=(7.4, 3.5))
    ax.bar(x - width / 2, summary["mcq_rate_all"], width, label="MCQ")
    ax.bar(
        x + width / 2,
        summary["generative_rate_decisive"],
        width,
        label="Free generation",
    )
    ax.errorbar(
        x + width / 2,
        summary["generative_rate_decisive"],
        yerr=np.vstack([
            summary["generative_rate_decisive"] - summary["generative_rate_lower_bound"],
            summary["generative_rate_upper_bound"] - summary["generative_rate_decisive"],
        ]),
        fmt="none",
        color="black",
        capsize=3,
        label="NEI bounds",
    )
    ax.set_xticks(x, [CASE_LABELS[case] for case in ordered], rotation=15, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Compliance rate")
    ax.legend(frameon=False, ncol=3)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _pct(value: float) -> str:
    return f"{100 * value:.1f}%"


def write_report(
    out_path: Path,
    case_summary: pd.DataFrame,
    by_sssc: pd.DataFrame,
    prober_summary: pd.DataFrame,
    contrasts: pd.DataFrame,
    retention: pd.DataFrame,
) -> None:
    compacted = case_summary.set_index("case").loc["compacted"]
    upper = case_summary.set_index("case").loc["compacted_post_sssc"]
    compacted_probers = prober_summary.loc[prober_summary["case"] == "compacted"]
    compaction_losses = contrasts.loc[contrasts["contrast"] == "compaction_loss"]
    weakest = by_sssc.loc[by_sssc["case"] == "compacted"].sort_values("agreement").iloc[0]
    retained = retention.set_index("retention_judge")

    compacted_by_sssc = by_sssc.loc[by_sssc["case"] == "compacted"].sort_values(
        "sssc_id"
    )
    sssc_table = "\n".join(
        f"| {int(row.sssc_id)} | {int(row.n_generative_true)} | "
        f"{int(row.n_generative_false)} | {int(row.n_not_enough_information)} | "
        f"{_pct(row.generative_rate_decisive)} | {_pct(row.mcq_rate_all)} | "
        f"{_pct(row.agreement)} |"
        for row in compacted_by_sssc.itertuples()
    )

    prober_lines = "\n".join(
        f"- {row.mcq_prober}: agreement {_pct(row.agreement)}, "
        f"kappa {row.cohen_kappa:.3f}, MCQ rate {_pct(row.mcq_rate_all)}, "
        f"generative rate {_pct(row.generative_rate_decisive)}."
        for row in compacted_probers.itertuples()
    )
    loss_lines = "\n".join(
        f"- {row.metric}: {_pct(row.rate_difference)} "
        f"(95% bootstrap CI {_pct(row.difference_ci_low)} to {_pct(row.difference_ci_high)})."
        for row in compaction_losses.itertuples()
    )
    report = f"""# Generative action compliance versus MCQ compliance

## Main paired result

On the compacted condition, the free-generation judge was decisive for
{int(compacted.n_total - compacted.n_not_enough_information)}/{int(compacted.n_total)} rows
({_pct(compacted.generative_decisive_rate)}). Among decisive rows, generative
compliance was {_pct(compacted.generative_rate_decisive)} versus
{_pct(compacted.mcq_rate_decisive_rows)} for the original GPT-OSS MCQ decisions.
Their row-level agreement was {_pct(compacted.agreement)} (Cohen's kappa
{compacted.cohen_kappa:.3f}). Treating ambiguous generations as all failures or
all successes bounds generative compliance at
{_pct(compacted.generative_rate_lower_bound)}--{_pct(compacted.generative_rate_upper_bound)}.

The post-SC upper-bound condition reached
{_pct(upper.generative_rate_decisive)} generative compliance with
{_pct(upper.generative_decisive_rate)} decisive coverage. This checks that the
tool-use probe and judge can recover near-ceiling compliance when the constraint
is explicitly restored after compaction.

The weakest compacted agreement was SC {int(weakest.sssc_id)}:
{_pct(weakest.agreement)} agreement (kappa {weakest.cohen_kappa:.3f}). The
per-SC table should therefore be reported rather than relying only on the pooled
average.

| SC | Gen. true | Gen. false | Not enough info | Gen. rate | MCQ rate | Agreement |
|---:|---:|---:|---:|---:|---:|---:|
{sssc_table}

SCs 1--3 are especially diagnostic. After compaction, SC 1 produced compliant
direct command execution in all fifty generations while MCQ compliance was only
8%. SC 2 produced no compliant decisive generations while MCQ compliance was
100%. SC 3 produced 6% generative compliance while MCQ compliance was 98%.
These are action-versus-description inversions, not small calibration shifts.

## Compaction effect across evaluation modes

The paired long-context-with-SC minus compacted compliance differences are:

{loss_lines}

Agreement in the direction and magnitude of this contrast is more relevant to
the paper's compaction claim than raw agreement alone.

## Dependence on the downstream MCQ model

The same generative action verdicts compared with each MCQ prober give:

{prober_lines}

These comparisons directly quantify compactor-to-prober compatibility. They do
not justify calling any single MCQ model a model-independent semantic oracle.

## Relationship to semantic retention

For contexts where the separate retention judge said the SC was retained,
generative compacted compliance was
{_pct(retained.loc[True, 'generative_compliance_rate'])}; where it said the SC
was not retained, it was
{_pct(retained.loc[False, 'generative_compliance_rate'])}. The corresponding MCQ
rates were {_pct(retained.loc[True, 'mcq_compliance_rate'])} and
{_pct(retained.loc[False, 'mcq_compliance_rate'])}.

## What this does and does not answer

This experiment strengthens the behavioral evidence: the downstream model was
given native tool schemas, and the judge evaluated the complete generated
transcript, including attempted tool calls. It tests observable action selection
rather than only selecting an A/B description.

It is still not an executable agent-harness evaluation. The fake tools were not
executed, no tool results were returned, and the agent did not continue through
a multi-step trajectory. It also uses GPT-OSS-120B as the only free-generation
model; Qwen and Gemma vary only the MCQ prober. A precise paper claim is therefore
"free-generation structured action proxy with cross-model MCQ sensitivity,"
not "evaluation in multiple executable agent harnesses."

## Suggested reviewer response

We agree that binary MCQ compliance is an indirect proxy and have narrowed the
claim accordingly. We added a free-generation evaluation for six constraints
whose compliance is observable through structured tool-use behavior. GPT-OSS
was given native tool schemas, and an independent judge evaluated the complete
generated transcript, including attempted tool calls. Across 300 compacted
examples, the judge was decisive on {_pct(compacted.generative_decisive_rate)};
generative compliance was {_pct(compacted.generative_rate_decisive)}, compared
with {_pct(compacted.mcq_rate_decisive_rows)} for MCQ, with only
{_pct(compacted.agreement)} row-level agreement (kappa
{compacted.cohen_kappa:.3f}). The near-ceiling post-constraint condition
({_pct(upper.generative_rate_decisive)}) indicates that the action probes can
detect compliance when the constraint is explicitly available. We also compare
MCQ results across GPT-OSS-120B, Qwen3-30B, and Gemma-4-E4B while holding
contexts fixed; the estimated compaction effect changes materially across
probers. We therefore report the generative analysis as a behavioral
sensitivity check, explicitly acknowledge compactor-to-prober compatibility,
and avoid treating MCQ compliance as model-independent semantic preservation.
Our setup records structured action selection but does not execute tools or
continue after tool results, so we do not describe it as a full executable agent
harness.
"""
    out_path.write_text(report, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results_root", type=Path, default=Path("/data/compaction_integrity"))
    parser.add_argument("--generative_run_id", default=GENERATIVE_RUN_ID)
    parser.add_argument("--prober_manifest", type=Path, default=PROBER_MANIFEST)
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=Path("/data/compaction_integrity/analysis/generative_mcq"),
    )
    parser.add_argument("--n_boot", type=int, default=5000)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    input_path = (
        args.results_root / "runs" / args.generative_run_id / "generative_probe_results.pkl"
    )
    generative_df = pd.read_pickle(input_path)
    rng = np.random.default_rng(0)

    row_level = build_row_level(generative_df)
    case_summary = _agreement_summary(
        row_level, ["case"], rng=rng, n_boot=args.n_boot
    )
    by_sssc = _agreement_summary(
        row_level, ["case", "sssc_id"], rng=rng, n_boot=args.n_boot
    )
    prober_paired, prober_summary = build_prober_comparison(
        row_level,
        args.prober_manifest,
        args.results_root,
        rng,
        args.n_boot,
    )
    contrasts = build_condition_contrasts(
        row_level, prober_paired, rng, args.n_boot
    )
    tool_summary = build_tool_summary(row_level)
    retention = build_retention_relationship(row_level)

    row_level.drop(columns=["tool_names"]).to_csv(
        args.out_dir / "row_level_comparison.csv", index=False
    )
    case_summary.to_csv(args.out_dir / "case_summary.csv", index=False)
    by_sssc.to_csv(args.out_dir / "by_sssc.csv", index=False)
    prober_summary.to_csv(args.out_dir / "cross_prober_summary.csv", index=False)
    contrasts.to_csv(args.out_dir / "paired_condition_contrasts.csv", index=False)
    tool_summary.to_csv(args.out_dir / "tool_call_summary.csv", index=False)
    retention.to_csv(args.out_dir / "retention_relationship.csv", index=False)
    plot_case_rates(case_summary, args.out_dir / "case_rates.pdf")
    write_report(
        args.out_dir / "report.md",
        case_summary,
        by_sssc,
        prober_summary,
        contrasts,
        retention,
    )

    print(case_summary.to_string(index=False))
    print(f"\nAnalysis saved to {args.out_dir}")


if __name__ == "__main__":
    main()
