"""Print retention and compliance rates per run for an RQ1 experiment manifest.

Usage:
    python src/compaction_integrity/analyze/gpt_5.4_eval.py \
        [--manifest config/experiments/rq1/gpt-5.4-mini.yaml] \
        [--results-root /data/compaction_integrity]
"""

import argparse
from pathlib import Path

from compaction_integrity.analyze.utils import extract_compactor_name, load_manifest_results


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MANIFEST = REPO_ROOT / "config/experiments/rq1/gpt-5.4-mini.yaml"
DEFAULT_RESULTS_ROOT = Path("/data/compaction_integrity")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    args = parser.parse_args()

    df = load_manifest_results(
        manifest_path=args.manifest,
        results_root=args.results_root,
        result_file_name="evaluation_results.pkl",
    )

    success = df[df["compaction_status"] == "success"]
    stats = success.groupby("run_id", sort=False).agg(
        retention_rate=("retention", "mean"),
        retained=("retention", "sum"),
        retention_total=("retention", "count"),
        compliance_rate=("compacted_compliant", "mean"),
        compliant=("compacted_compliant", "sum"),
        compliance_total=("compacted_compliant", "count"),
    )

    run_config = (
        df.drop_duplicates("run_id")
        .set_index("run_id")[["dataset", "compactor"]]
        .assign(compactor=lambda x: x["compactor"].map(extract_compactor_name))
    )

    stats["dataset"] = run_config.loc[stats.index, "dataset"]
    stats["compactor"] = run_config.loc[stats.index, "compactor"]
    stats["retention_count"] = [
        f"{int(row.retained)}/{int(row.retention_total)}" for row in stats.itertuples()
    ]
    stats["compliance_count"] = [
        f"{int(row.compliant)}/{int(row.compliance_total)}" for row in stats.itertuples()
    ]
    display = stats.reset_index()[
        [
            "dataset",
            "compactor",
            "retention_rate",
            "retention_count",
            "compliance_rate",
            "compliance_count",
        ]
    ]
    display["retention_rate"] = display["retention_rate"].map("{:.4f}".format)
    display["compliance_rate"] = display["compliance_rate"].map("{:.4f}".format)
    print(display.to_string(index=False))


if __name__ == "__main__":
    main()
