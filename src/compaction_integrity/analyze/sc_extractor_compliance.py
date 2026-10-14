"""Retention, compliance and effective retention of the SC-aware extractor condition.

Reads scripts/eval_sc_extractor_compliance.py output for every RQ1 run in the
manifests and reports, per dataset x compactor, the compaction condition (case 3)
next to the extractor condition C(H) ⊕ S_t (case 5). ER follows Eq. 8 for both,
with the same no-SC baseline and upper bound:

    ER_g = (c_g - c_full) / (c_ub - c_full),   g in {comp, ext}

c_full is deduped per dataset as in main_exp.py. Retention of C(H) ⊕ S_t is the OR
of the two GPT-5.4 verdicts (summary, registry): the judge asks whether the SC is
present, and the concatenation contains both texts.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

from compaction_integrity.analyze.utils import (
    extract_dataset_config,
    fmt_compactor_label,
    fmt_dataset_label,
    ordered_compactor_names,
    tee_stdout,
)

REPO_ROOT = Path(__file__).resolve().parents[3]

_N_BOOT = 2000


def _rate(series: pd.Series) -> float:
    return float(series.dropna().astype(float).mean())


def _er(c: float, c_full: float, c_ub: float) -> float:
    return (c - c_full) / (c_ub - c_full)


def _load(manifests: list[str], results_root: Path, extractor_runs: dict[str, str]) -> pd.DataFrame:
    frames = []
    for manifest in manifests:
        for entry in OmegaConf.load(REPO_ROOT / manifest).runs:
            run_id = str(entry.run_id)
            dataset = run_id.split("__", 1)[0]
            path = (
                results_root / "sc_extractor_compliance_runs" / run_id
                / extractor_runs[dataset] / "results.pkl"
            )
            frames.append(pd.read_pickle(path).assign(run_id=run_id))
    return pd.concat(frames, ignore_index=True)


def _boot_er_ci(g: pd.DataFrame, col: str, rng: np.random.Generator) -> tuple[float, float]:
    """95% CI of ER over (row, SC) pairs resampled within one dataset x compactor."""
    arr = g[[col, "full_without_sssc_compliant", "compacted_post_sssc_compliant"]].to_numpy()
    ers = []
    for _ in range(_N_BOOT):
        s = arr[rng.integers(0, len(arr), len(arr))]
        c, c_full, c_ub = (_rate(pd.Series(s[:, i])) for i in range(3))
        ers.append(_er(c, c_full, c_ub))
    lo, hi = np.nanpercentile(ers, [2.5, 97.5])
    return float(lo), float(hi)


def build_table(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    baselines = (
        df.drop_duplicates(subset=["dataset", "source_row_index", "sssc_id"])
        .groupby("dataset")
        .agg(
            c_full_sc=("full_with_sssc_compliant", _rate),
            c_full=("full_without_sssc_compliant", _rate),
        )
    )
    suc = df.loc[df["compaction_status"] == "success"].copy()
    suc["retention_combined"] = suc["retention"].astype(bool) | suc["extractor_retention"].astype(bool)

    rows = []
    for (dataset, compactor), g in suc.groupby(["dataset", "compactor_name"]):
        c_full = baselines.loc[dataset, "c_full"]
        c_comp = _rate(g["compacted_compliant"])
        c_ext = _rate(g["compacted_extracted_compliant"])
        c_ub = _rate(g["compacted_post_sssc_compliant"])
        comp_lo, comp_hi = _boot_er_ci(g, "compacted_compliant", rng)
        ext_lo, ext_hi = _boot_er_ci(g, "compacted_extracted_compliant", rng)
        rows.append({
            "dataset": fmt_dataset_label(extract_dataset_config(dataset)),
            "compactor": fmt_compactor_label(compactor),
            "compactor_raw": compactor,
            "n_pairs": len(g),
            "registry_items_mean": float(g["registry"].apply(len).mean()),
            "retention_summary_pct": 100 * _rate(g["retention"]),
            "retention_registry_pct": 100 * _rate(g["extractor_retention"]),
            "retention_summary_or_registry_pct": 100 * _rate(g["retention_combined"]),
            "compliance_long_ctx_with_sc_pct": 100 * baselines.loc[dataset, "c_full_sc"],
            "compliance_long_ctx_without_sc_pct": 100 * c_full,
            "compliance_compaction_pct": 100 * c_comp,
            "compliance_compaction_extractor_pct": 100 * c_ext,
            "compliance_upper_bound_pct": 100 * c_ub,
            "er_compaction_pct": 100 * _er(c_comp, c_full, c_ub),
            "er_compaction_ci95_pct": f"[{100 * comp_lo:.1f}, {100 * comp_hi:.1f}]",
            "er_compaction_extractor_pct": 100 * _er(c_ext, c_full, c_ub),
            "er_compaction_extractor_ci95_pct": f"[{100 * ext_lo:.1f}, {100 * ext_hi:.1f}]",
            "unparsed_extractor_probes": int(g["compacted_extracted_compliant"].isna().sum()),
        })
    table = pd.DataFrame(rows)
    order = {name: i for i, name in enumerate(ordered_compactor_names(table["compactor_raw"]))}
    return table.sort_values(
        ["dataset", "compactor_raw"], key=lambda s: s.map(order) if s.name == "compactor_raw" else s
    ).reset_index(drop=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=str(REPO_ROOT / "config/tasks/sc_extractor/sc_extractor_compliance.yaml"),
        help="Config used for eval_sc_extractor_compliance (manifests, extractor runs).",
    )
    parser.add_argument(
        "--output_dir", default="/data/compaction_integrity/analysis/sc_extractor_compliance"
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with tee_stdout(output_dir / "sc_extractor_compliance.log"):
        df = _load(
            [str(m) for m in cfg.manifests],
            Path(str(cfg.results_root)),
            OmegaConf.to_container(cfg.extractor_runs),
        )
        print(f"Loaded {len(df)} rows from {df['run_id'].nunique()} run(s).\n")
        table = build_table(df, args.seed)
        table.to_csv(output_dir / "summary_table.csv", index=False)
        with pd.option_context("display.width", 250, "display.max_columns", None):
            print(table.drop(columns=["compactor_raw"]).round(1).to_string(index=False))
        print(f"\nAll outputs written to {output_dir}")
