"""Different-prober MCQ comparison table.

Aggregates a manifest of runs that share identical compacted contexts but differ
only in the downstream prober (the source gpt-oss-120b run plus the Qwen3-30B /
Gemma-4-E4B swaps from scripts/reprobe_mcq.py). Produces, per prober:

  - compliance rates for each probe case (uninjected base, injected full,
    compacted, post-SSSC upper bound),
  - the invalid (unparseable A/B) rate on compacted probes,
  - retention rate (prober-independent; printed as a shared reference),
  - agreement vs the reference prober on `compacted_compliant`,
  - pairwise compliance deltas, agreement, and Cohen's kappa for every probe
    case and every pair of probers.

The point: if compliance and its compactor-relative ordering are stable across
probers, the metric reflects semantic preservation rather than
compactor-to-prober format compatibility.

Usage:
  python -m compaction_integrity.analyze.prober_swap \
    --manifest_path config/experiments/additional/prober_swap_hermes100k.yaml \
    --results_root /data/compaction_integrity
"""

import argparse
import csv
from itertools import combinations
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


_PROBE_CASES = {
    "uninjected_base": "full_without_sssc_compliant",
    "injected_full": "full_with_sssc_compliant",
    "compacted": "compacted_compliant",
    "post_sssc_upper": "compacted_post_sssc_compliant",
}


def _compliance_rate(series: pd.Series) -> float:
    """Mean over rows with a non-null True/False verdict; NaN if none."""
    vals = [bool(v) for v in series.tolist() if v is not None and not pd.isna(v)]
    return sum(vals) / len(vals) if vals else float("nan")


def _invalid_rate(series: pd.Series) -> float:
    """Fraction of rows with a null verdict (unparseable A/B) among all rows."""
    n = len(series)
    if n == 0:
        return float("nan")
    null = sum(1 for v in series.tolist() if v is None or pd.isna(v))
    return null / n


def _cohen_kappa(a: list[bool], b: list[bool]) -> float:
    n = len(a)
    if n == 0:
        return float("nan")
    po = sum(1 for x, y in zip(a, b) if x == y) / n
    pa_t = sum(a) / n
    pb_t = sum(b) / n
    pe = pa_t * pb_t + (1 - pa_t) * (1 - pb_t)
    return 1.0 if pe == 1.0 else (po - pe) / (1 - pe)


def _row_key(row: dict[str, Any]) -> tuple[Any, Any]:
    return (row["source_row_index"], row["sssc_id"])


def _paired_agreement(
    ref: pd.DataFrame, other: pd.DataFrame, column: str
) -> dict[str, float]:
    """Agreement on `column` over rows both frames graded (non-null), keyed by
    (source_row_index, sssc_id)."""
    ref_by = {_row_key(r): r.get(column) for r in ref.to_dict(orient="records")}
    a: list[bool] = []
    b: list[bool] = []
    for r in other.to_dict(orient="records"):
        rv = ref_by.get(_row_key(r))
        ov = r.get(column)
        if rv is None or pd.isna(rv) or ov is None or pd.isna(ov):
            continue
        a.append(bool(rv))
        b.append(bool(ov))
    n = len(a)
    agree = sum(1 for x, y in zip(a, b) if x == y)
    return {
        "n_paired": n,
        "agree_rate": (agree / n) if n else float("nan"),
        "cohen_kappa": _cohen_kappa(a, b),
    }


def _paired_comparison(
    left: pd.DataFrame, right: pd.DataFrame, column: str
) -> dict[str, float]:
    left_by = {_row_key(r): r.get(column) for r in left.to_dict(orient="records")}
    left_values: list[bool] = []
    right_values: list[bool] = []
    for row in right.to_dict(orient="records"):
        left_value = left_by.get(_row_key(row))
        right_value = row.get(column)
        if (
            left_value is None
            or pd.isna(left_value)
            or right_value is None
            or pd.isna(right_value)
        ):
            continue
        left_values.append(bool(left_value))
        right_values.append(bool(right_value))

    n = len(left_values)
    left_rate = sum(left_values) / n
    right_rate = sum(right_values) / n
    agreement = sum(
        left_value == right_value
        for left_value, right_value in zip(left_values, right_values)
    ) / n
    return {
        "n_paired": n,
        "left_rate": left_rate,
        "right_rate": right_rate,
        "rate_delta": right_rate - left_rate,
        "agree_rate": agreement,
        "cohen_kappa": _cohen_kappa(left_values, right_values),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest_path", required=True, type=Path)
    parser.add_argument("--results_root", type=Path, default=Path("/data/compaction_integrity"))
    parser.add_argument("--out_dir", type=Path, default=None,
                        help="Where to write the CSV (default: manifest dir).")
    args = parser.parse_args()

    manifest = yaml.safe_load(args.manifest_path.read_text(encoding="utf-8"))
    runs = manifest["runs"]
    reference_run_id = str(manifest.get("reference_run_id", runs[0]["run_id"]))
    runs_dir = args.results_root / "runs"
    out_dir = args.out_dir or args.manifest_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load every run's success rows.
    frames: dict[str, dict[str, Any]] = {}
    for entry in runs:
        run_id = str(entry["run_id"])
        label = str(entry.get("label", run_id))
        df = pd.read_pickle(runs_dir / run_id / "evaluation_results.pkl")
        suc = df.loc[df["compaction_status"] == "success"].copy()
        frames[run_id] = {"label": label, "df": suc}

    ref_suc = frames[reference_run_id]["df"]

    rows_out: list[dict[str, Any]] = []
    for entry in runs:
        run_id = str(entry["run_id"])
        info = frames[run_id]
        suc = info["df"]
        agree = _paired_agreement(ref_suc, suc, "compacted_compliant")
        rows_out.append({
            "label": info["label"],
            "run_id": run_id,
            "n_success": len(suc),
            **{
                case: _compliance_rate(suc[column])
                for case, column in _PROBE_CASES.items()
            },
            "retention": _compliance_rate(suc["retention"]),
            "compacted_invalid_rate": _invalid_rate(suc["compacted_compliant"]),
            "agree_vs_ref": agree["agree_rate"],
            "kappa_vs_ref": agree["cohen_kappa"],
            "n_paired_vs_ref": agree["n_paired"],
        })

    pairwise_rows: list[dict[str, Any]] = []
    for left_entry, right_entry in combinations(runs, 2):
        left_run_id = str(left_entry["run_id"])
        right_run_id = str(right_entry["run_id"])
        left_info = frames[left_run_id]
        right_info = frames[right_run_id]
        for case, column in _PROBE_CASES.items():
            comparison = _paired_comparison(
                left_info["df"], right_info["df"], column
            )
            pairwise_rows.append({
                "case": case,
                "left_label": left_info["label"],
                "left_run_id": left_run_id,
                "right_label": right_info["label"],
                "right_run_id": right_run_id,
                **comparison,
            })

    # -- console table -----------------------------------------------------
    hdr = (f"{'prober':<22} {'n':>4} {'uninj':>6} {'inj':>6} {'compact':>7} "
           f"{'upper':>6} {'retain':>6} {'invalid':>7} {'agree':>6} {'kappa':>6}")
    print(f"\nDifferent-prober MCQ comparison (compacted contexts held fixed)")
    print(f"reference prober: {frames[reference_run_id]['label']}\n")
    print(hdr)
    print("-" * len(hdr))
    for r in rows_out:
        is_ref = r["run_id"] == reference_run_id
        agree_s = "  ref" if is_ref else f"{r['agree_vs_ref']:.3f}"
        kappa_s = "  ref" if is_ref else f"{r['kappa_vs_ref']:.3f}"
        print(f"{r['label']:<22} {r['n_success']:>4d} "
              f"{r['uninjected_base']:>6.3f} {r['injected_full']:>6.3f} "
              f"{r['compacted']:>7.3f} {r['post_sssc_upper']:>6.3f} "
              f"{r['retention']:>6.3f} {r['compacted_invalid_rate']:>7.3f} "
              f"{agree_s:>6} {kappa_s:>6}")

    # -- csv ---------------------------------------------------------------
    csv_path = out_dir / f"{args.manifest_path.stem}__prober_swap.csv"
    fieldnames = ["label", "run_id", "n_success", "uninjected_base", "injected_full",
                  "compacted", "post_sssc_upper", "retention", "compacted_invalid_rate",
                  "agree_vs_ref", "kappa_vs_ref", "n_paired_vs_ref"]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows_out:
            w.writerow(r)
    print(f"\nWrote {csv_path}")

    pairwise_csv_path = out_dir / f"{args.manifest_path.stem}__pairwise.csv"
    pairwise_fieldnames = [
        "case",
        "left_label",
        "left_run_id",
        "right_label",
        "right_run_id",
        "n_paired",
        "left_rate",
        "right_rate",
        "rate_delta",
        "agree_rate",
        "cohen_kappa",
    ]
    with pairwise_csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=pairwise_fieldnames)
        writer.writeheader()
        writer.writerows(pairwise_rows)
    print(f"Wrote {pairwise_csv_path}")


if __name__ == "__main__":
    main()
