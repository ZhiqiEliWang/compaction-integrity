"""Generate a blind retention test for a human to fill in by hand.

Retention is scored by a single LLM judge (GPT-5.4) deciding, per row, whether a
compacted summary still preserves the injected SSSC. To check that judge against
a human, this script draws a reproducible sample of judged rows and writes two
files:

    <out>/<tag>_blind.jsonl   one item per line, with an empty "label" you fill
    <out>/<tag>_key.jsonl     the held-aside LLM verdict + provenance per item

You edit ONLY the blind file: set each line's "label" to "yes" (SSSC still
present in the summary) or "no" (SSSC absent / too weakened). The verdict never
appears in the blind file, so you annotate blind. When every line is filled,
`analyze_blind_test.py` joins the two files and reports agreement.

Sampling is deterministic: rows are identified by a stable content hash, sorted,
and picked with a seeded RNG, so (sources, seed, n) always yields the same
sample. Multiple sources (--pkl / --run_id / --manifest, all repeatable) are
POOLED and the sample is drawn from the joint set.

Usage:
  python judge_robustness_eval/make_blind_test.py \
    --manifest config/experiments/rq1/main.yaml \
    --manifest config/experiments/rq1/gpt-5.4-mini.yaml \
    --results_root /data/compaction_integrity \
    --tag balanced --n 50 --seed 0 --balance
"""

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

# Files land next to this script (judge_robustness_eval/) unless --out_dir overrides.
DEFAULT_OUT_DIR = Path(__file__).resolve().parent

Message = dict[str, str]


def _render_messages_for_judge(messages: list[Message]) -> str:
    """Render a compacted context the way the LLM judge saw it, so the human
    reads the identical artifact."""
    return "\n\n".join(f"{m['role']}:\n{m['content']}" for m in messages)


def _stable_row_key(row: dict[str, Any]) -> str:
    """Position-independent content identity of a row, used as its id."""
    payload = "|".join(
        [
            str(row.get("dataset")),
            str(row.get("source_row_index")),
            str(row.get("sssc_id")),
            json.dumps(row.get("sssc_attrs"), ensure_ascii=True, sort_keys=True, default=str),
            json.dumps(row.get("probe"), ensure_ascii=True, sort_keys=True, default=str),
            json.dumps(row.get("compactor"), ensure_ascii=True, sort_keys=True, default=str),
        ]
    )
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]


def _judge_name(row: dict[str, Any]) -> str:
    ev = row.get("evaluator")
    if isinstance(ev, dict):
        return str(ev.get("name") or ev.get("model") or "unknown_judge")
    return "unknown_judge"


def _compactor_name(row: dict[str, Any]) -> str:
    comp = row.get("compactor")
    if isinstance(comp, dict):
        return str(comp.get("name") or comp.get("model") or "unknown_compactor")
    return "unknown_compactor"


def _eligible(row: dict[str, Any]) -> bool:
    ret = row.get("retention")
    if ret is None or (isinstance(ret, float) and pd.isna(ret)):
        return False
    cc = row.get("compacted_context")
    return isinstance(cc, list) and len(cc) > 0


def _load_manifest_run_ids(manifest_path: Path) -> list[str]:
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    return [str(entry["run_id"]) for entry in manifest.get("runs", [])]


def _select(records: list[dict[str, Any]], keys: list[str], n: int, seed: int,
            balance: bool) -> list[int]:
    """Deterministic index selection given (keys, n, seed)."""
    rng = random.Random(seed)
    order = sorted(range(len(records)), key=lambda i: keys[i])
    if not balance:
        return sorted(rng.sample(order, min(n, len(order))))
    yes = [i for i in order if bool(records[i]["retention"])]
    no = [i for i in order if not bool(records[i]["retention"])]
    half = n // 2
    n_yes = min(half, len(yes))
    n_no = min(n - n_yes, len(no))
    n_yes = min(n - n_no, len(yes))  # backfill if a class is short
    return sorted(rng.sample(yes, n_yes) + rng.sample(no, n_no))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pkl", type=Path, action="append", default=[],
                        help="Path to an evaluation_results.pkl (repeatable).")
    parser.add_argument("--run_id", type=str, action="append", default=[],
                        help="Run id under <results_root>/runs (repeatable).")
    parser.add_argument("--manifest", type=Path, action="append", default=[],
                        help="Experiment manifest yaml whose runs are all pooled (repeatable).")
    parser.add_argument("--results_root", type=Path, default=Path("/data/compaction_integrity"))
    parser.add_argument("--out_dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--tag", type=str, default="blind_test",
                        help="Basename for the output files.")
    parser.add_argument("--n", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--balance", action="store_true",
                        help="Sample ~n/2 retained + ~n/2 not-retained (per the judge). NOTE: raw "
                            "agreement on a balanced sample is not population agreement -- reweight "
                            "by class prevalence.")
    parser.add_argument("--overwrite", action="store_true",
                        help="Allow overwriting an existing _blind.jsonl (default: refuse, so "
                            "in-progress answers are never clobbered).")
    args = parser.parse_args()

    # Resolve sources to (run_id, pkl_path), order-preserving + de-duped.
    sources: list[tuple[str, Path]] = []
    seen: set[str] = set()

    def _add(run_id: str, pkl_path: Path) -> None:
        if run_id not in seen:
            seen.add(run_id)
            sources.append((run_id, pkl_path))

    for p in args.pkl:
        _add(p.parent.name, p)
    for rid in args.run_id:
        _add(rid, args.results_root / "runs" / rid / "evaluation_results.pkl")
    for man in args.manifest:
        for rid in _load_manifest_run_ids(man):
            _add(rid, args.results_root / "runs" / rid / "evaluation_results.pkl")
    if not sources:
        raise SystemExit("Provide at least one of --pkl / --run_id / --manifest.")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    blind_path = args.out_dir / f"{args.tag}_blind.jsonl"
    key_path = args.out_dir / f"{args.tag}_key.jsonl"
    if blind_path.exists() and not args.overwrite:
        raise SystemExit(f"{blind_path} already exists. Pass --overwrite to replace it "
                         "(this discards any answers already filled in).")

    # Pool eligible rows across all sources.
    eligible: list[dict[str, Any]] = []
    keys: list[str] = []
    n_missing = 0
    for run_id, pkl_path in sources:
        if not pkl_path.exists():
            print(f"[skip] missing pickle: {pkl_path}")
            n_missing += 1
            continue
        df = pd.read_pickle(pkl_path)
        for r in df.to_dict(orient="records"):
            if _eligible(r):
                r["_run_id"] = run_id
                eligible.append(r)
                keys.append(_stable_row_key(r))
    if not eligible:
        raise SystemExit("No rows with a judge verdict + compacted context in any source.")
    if len(set(keys)) != len(keys):
        print("WARNING: duplicate row keys detected; sample determinism may be affected.")

    chosen = _select(eligible, keys, args.n, args.seed, args.balance)

    n_yes = 0
    with blind_path.open("w", encoding="utf-8") as fb, key_path.open("w", encoding="utf-8") as fk:
        for i in chosen:
            r = eligible[i]
            rid = keys[i]
            n_yes += int(bool(r["retention"]))
            # "label" first + empty: this is the ONLY field you edit. Set it to
            # "yes" (SSSC present in the summary) or "no" (absent/too weakened).
            fb.write(json.dumps({
                "label": "",
                "id": rid,
                "sssc_message": str(r.get("sssc_message", "")),
                "compacted_summary": _render_messages_for_judge(r["compacted_context"]),
            }, ensure_ascii=False) + "\n")
            fk.write(json.dumps({
                "id": rid,
                "judge_retention": bool(r["retention"]),
                "judge_name": _judge_name(r),
                "compactor": _compactor_name(r),
                "dataset": str(r.get("dataset")),
                "source_row_index": r.get("source_row_index"),
                "sssc_id": r.get("sssc_id"),
                "run_id": r.get("_run_id"),
            }, ensure_ascii=False, default=str) + "\n")

    n_runs_hit = len({eligible[i]["_run_id"] for i in chosen})
    print(f"Sources        : {len(sources)} run(s)" + (f", {n_missing} missing" if n_missing else ""))
    print(f"Eligible rows  : {len(eligible)} pooled  (judge YES rate "
          f"{sum(bool(r['retention']) for r in eligible)/len(eligible):.1%})")
    print(f"Sampled        : {len(chosen)} from {n_runs_hit} run(s)  "
          f"(judge YES={n_yes}, NO={len(chosen)-n_yes})")
    print(f"\nFill in (edit the \"label\" field to yes/no on every line):\n  {blind_path}")
    print(f"Held-aside key (do not open until done):\n  {key_path}")
    print(f"\nWhen every line is filled:\n  python judge_robustness_eval/analyze_blind_test.py "
          f"--blind {blind_path} --key {key_path}")


if __name__ == "__main__":
    main()
