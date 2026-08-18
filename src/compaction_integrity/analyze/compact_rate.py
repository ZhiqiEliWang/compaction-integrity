# stats for results from run_compactor_ablation

import argparse
from pathlib import Path
import sys

import pandas as pd
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MANIFEST_PATH = REPO_ROOT / "config/experiments/rq2/diff_input_size.yaml"

from compaction_integrity.analyze.utils import (
    extract_compactor_name,
    extract_context_length,
    extract_dataset_config,
    load_manifest_results,
    tee_stdout,
)
from compaction_integrity.tokenization import count_tokens_messages_batch


"""
evaluation_results.pkl has the following columns:
- dataset, dataset_path, source_row_index
- sssc_id, sssc_type, sssc_message, sssc_probe, sssc_attrs
- probe, compactor, evaluator
- full_with_sssc_messages
- compacted_context, compaction_status, compaction_error
- compacted_compliant, retention
"""

SETTING_COLUMN = "context_length"
TOKEN_COUNT_BATCH_SIZE = 256


def _context_length_to_int(context_length: str) -> int:
    return int(context_length[:-1]) * 1000


def _count_tokens_messages_batch_with_progress(
    messages_batch: list[list[dict]],
    desc: str,
) -> list[int]:
    token_counts: list[int] = []
    for start in tqdm(
        range(0, len(messages_batch), TOKEN_COUNT_BATCH_SIZE),
        desc=desc,
        unit="batch",
        dynamic_ncols=True,
    ):
        token_counts.extend(
            count_tokens_messages_batch(
                messages_batch[start : start + TOKEN_COUNT_BATCH_SIZE]
            )
        )
    return token_counts


def _load_results(manifest_path: Path, results_root: Path) -> pd.DataFrame:
    results_df = load_manifest_results(
        manifest_path=manifest_path,
        results_root=results_root,
        result_file_name="evaluation_results.pkl",
    )
    results_df[SETTING_COLUMN] = results_df["dataset"].map(extract_context_length)
    results_df["dataset_config"] = results_df["dataset"].map(extract_dataset_config)
    results_df["compactor_name"] = results_df["compactor"].map(extract_compactor_name)
    results_df["tokens_before"] = _count_tokens_messages_batch_with_progress(
        results_df["full_with_sssc_messages"].tolist(),
        desc="Counting tokens before compaction",
    )
    results_df["turns_before"] = results_df["full_with_sssc_messages"].map(len)
    return results_df


def _build_compression_stats(results_df: pd.DataFrame) -> pd.DataFrame:
    filtered_df = results_df.loc[results_df["compaction_status"] == "success"].copy()
    filtered_df["tokens_after"] = _count_tokens_messages_batch_with_progress(
        filtered_df["compacted_context"].tolist(),
        desc="Counting tokens after compaction",
    )
    filtered_df["compression_rate"] = (
        filtered_df["tokens_after"] / filtered_df["tokens_before"]
    )

    stats_df = (
        filtered_df.groupby(
            ["dataset_config", "compactor_name", SETTING_COLUMN],
            as_index=False,
        )[
            ["tokens_before", "tokens_after", "compression_rate", "turns_before"]
        ]
        .mean()
        .rename(
            columns={
                "tokens_before": "avg_tokens_before",
                "tokens_after": "avg_tokens_after",
                "compression_rate": "avg_compression_rate",
                "turns_before": "avg_turns_before",
            }
        )
    )
    count_df = (
        filtered_df.groupby(["dataset_config", "compactor_name", SETTING_COLUMN])
        .size()
        .rename("total_count")
        .reset_index()
    )
    stats_df = stats_df.merge(
        count_df,
        on=["dataset_config", "compactor_name", SETTING_COLUMN],
    )
    stats_df["config_label"] = (
        stats_df["dataset_config"] + " | " + stats_df["compactor_name"]
    )
    stats_df["setting_order"] = stats_df[SETTING_COLUMN].map(_context_length_to_int)
    stats_df["avg_tokens_before"] = stats_df["avg_tokens_before"].round(1)
    stats_df["avg_tokens_after"] = stats_df["avg_tokens_after"].round(1)
    stats_df["avg_turns_before"] = stats_df["avg_turns_before"].round(1)
    stats_df = stats_df.sort_values(
        ["setting_order", "config_label"]
    ).reset_index(drop=True)
    return stats_df


if __name__ == "__main__":
    args = argparse.ArgumentParser(
        description="Analyze compression rate across dataset and compactor settings."
    )
    args.add_argument(
        "--output_dir",
        type=str,
        default="/data/compaction_integrity/analysis/compact_rate",
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
    parsed_args = args.parse_args()

    output_dir = Path(parsed_args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with tee_stdout(output_dir / "compact_rate.log"):
        results_df = _load_results(
            manifest_path=Path(parsed_args.manifest_path),
            results_root=Path(parsed_args.results_root),
        )
        stats_df = _build_compression_stats(results_df)
        stats_df.to_csv(output_dir / "avg_compression_rate_by_context_length.csv", index=False)
        print(stats_df.to_string(index=False))
