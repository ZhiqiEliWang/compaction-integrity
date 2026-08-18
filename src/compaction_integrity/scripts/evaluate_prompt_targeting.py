"""Compare the baseline and constraint-targeted Anthropic summarization prompts.

This script reads completed runs from ``scripts.evaluation`` and writes paired
retention, downstream-compliance, and summary-length comparisons. It never
recomputes compaction or probes; launch the evaluation config first.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import hydra
import pandas as pd
from omegaconf import DictConfig, OmegaConf
from tqdm.auto import tqdm

from compaction_integrity.tokenization import count_tokens_messages


@dataclass(frozen=True)
class PromptPair:
    model: str
    baseline: str
    targeted: str
    baseline_config: dict[str, Any]
    targeted_config: dict[str, Any]


def _to_container(value: Any) -> Any:
    return OmegaConf.to_container(value, resolve=True) if OmegaConf.is_config(value) else value


def _matching_run(
    runs_dir: Path,
    dataset: str,
    compactor_name: str,
    compactor_config: dict[str, Any],
    match: dict[str, Any],
) -> tuple[str, pd.DataFrame]:
    candidates: list[tuple[str, Path]] = []
    for run_dir in runs_dir.iterdir():
        metadata_path = run_dir / "metadata.json"
        results_path = run_dir / "evaluation_results.pkl"
        if not metadata_path.exists() or not results_path.exists():
            continue
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        spec = metadata["run_spec"]
        if spec["dataset"]["name"] != dataset:
            continue
        if spec["compactor"]["name"] != compactor_name:
            continue
        if spec["compactor"]["config"] != compactor_config:
            continue
        if spec["dataset"]["num_rows"] != match["num_rows"]:
            continue
        if spec["global_seed"] != match["global_seed"]:
            continue
        if spec["sssc_attrs"] != match["sssc_attrs"]:
            continue
        if spec["probe"]["name"] != match["probe_name"]:
            continue
        candidates.append((str(metadata["run_id"]), results_path))

    if len(candidates) != 1:
        raise ValueError(
            f"Expected one run for dataset={dataset}, compactor={compactor_name}; "
            f"found {len(candidates)}."
        )
    run_id, results_path = candidates[0]
    return run_id, pd.read_pickle(results_path)


def _rate(values: pd.Series) -> float:
    present = values.dropna()
    return float(present.astype(bool).mean()) if not present.empty else float("nan")


def _summary_tokens(context: object) -> float | None:
    if context is None:
        return None
    return float(count_tokens_messages(context))


def _pair_rows(
    baseline_df: pd.DataFrame,
    targeted_df: pd.DataFrame,
) -> pd.DataFrame:
    key = ["source_row_index", "sssc_id"]
    columns = [
        *key,
        "retention",
        "compacted_compliant",
        "compacted_post_sssc_compliant",
        "compacted_context",
    ]
    baseline = baseline_df[columns].copy()
    targeted = targeted_df[columns].copy()
    baseline["summary_tokens"] = baseline["compacted_context"].map(_summary_tokens)
    targeted["summary_tokens"] = targeted["compacted_context"].map(_summary_tokens)
    baseline = baseline.drop(columns="compacted_context").add_prefix("baseline_")
    targeted = targeted.drop(columns="compacted_context").add_prefix("targeted_")
    return baseline.merge(
        targeted,
        left_on=[f"baseline_{name}" for name in key],
        right_on=[f"targeted_{name}" for name in key],
        how="inner",
    )


def _metric_row(
    paired: pd.DataFrame,
    dataset: str,
    pair: PromptPair,
    baseline_run_id: str,
    targeted_run_id: str,
    baseline_rows: int,
    targeted_rows: int,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "dataset": dataset,
        "model": pair.model,
        "baseline_compactor": pair.baseline,
        "targeted_compactor": pair.targeted,
        "baseline_run_id": baseline_run_id,
        "targeted_run_id": targeted_run_id,
        "baseline_rows": baseline_rows,
        "targeted_rows": targeted_rows,
        "n_paired": len(paired),
    }
    for metric in ("retention", "compacted_compliant", "compacted_post_sssc_compliant"):
        baseline_rate = _rate(paired[f"baseline_{metric}"])
        targeted_rate = _rate(paired[f"targeted_{metric}"])
        row[f"baseline_{metric}_rate"] = baseline_rate
        row[f"targeted_{metric}_rate"] = targeted_rate
        row[f"delta_{metric}_rate"] = targeted_rate - baseline_rate
    for metric in ("summary_tokens",):
        baseline_mean = float(paired[f"baseline_{metric}"].mean())
        targeted_mean = float(paired[f"targeted_{metric}"].mean())
        row[f"baseline_{metric}_mean"] = baseline_mean
        row[f"targeted_{metric}_mean"] = targeted_mean
        row[f"delta_{metric}_mean"] = targeted_mean - baseline_mean
    return row


@hydra.main(
    version_base=None,
    config_path="../../../config/experiments/additional",
    config_name="prompt_targeting",
)
def main(cfg: DictConfig) -> None:
    config = _to_container(cfg)
    results_root = Path(config["results_root"])
    runs_dir = results_root / ("runs_test" if config["test"] else "runs")
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    match = dict(config["run_match"])
    match["sssc_attrs"] = dict(match["sssc_attrs"])
    pairs = [PromptPair(model=name, **pair_cfg) for name, pair_cfg in config["prompt_pairs"].items()]

    summary_rows: list[dict[str, Any]] = []
    paired_rows: list[pd.DataFrame] = []
    work = [(dataset, pair) for dataset in config["datasets"] for pair in pairs]
    for dataset, pair in tqdm(work, desc="Prompt-targeting comparisons", unit="run", dynamic_ncols=True):
        baseline_run_id, baseline_df = _matching_run(
            runs_dir, dataset, pair.baseline, pair.baseline_config, match
        )
        targeted_run_id, targeted_df = _matching_run(
            runs_dir, dataset, pair.targeted, pair.targeted_config, match
        )
        paired = _pair_rows(baseline_df, targeted_df)
        paired.insert(0, "dataset", dataset)
        paired.insert(1, "model", pair.model)
        paired_rows.append(paired)
        summary_rows.append(
            _metric_row(
                paired,
                dataset,
                pair,
                baseline_run_id,
                targeted_run_id,
                len(baseline_df),
                len(targeted_df),
            )
        )

    summary = pd.DataFrame(summary_rows)
    summary_path = output_dir / "prompt_targeting_comparison.csv"
    summary.to_csv(summary_path, index=False)
    paired_path = output_dir / "prompt_targeting_paired_rows.csv"
    pd.concat(paired_rows, ignore_index=True).to_csv(paired_path, index=False)
    print(f"Wrote prompt-targeting comparison -> {summary_path}")
    print(f"Wrote paired prompt-targeting rows -> {paired_path}")


if __name__ == "__main__":
    main()
