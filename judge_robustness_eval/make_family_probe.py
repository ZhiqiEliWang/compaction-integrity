"""Draw the family-bias retention probe: a large sample that CONTAINS the human-50.

Purpose
-------
The reviewer's concern is model-family bias: the retention judge is GPT-5.4, and
"evaluator-specific interpretation and potential model-family bias cannot be
ruled out, particularly when comparing GPT-5.4-mini against other compactors."

To answer that we build ONE sample used for two claims:
  * CORRECTNESS (n=50): the already-labeled human blind set is nested inside this
    sample, so both judges (GPT-5.4 and gemini) can be scored against the human
    ground truth on the identical 50 rows.
  * ROBUSTNESS + FAMILY BIAS (n=N, default 2000): the remaining rows are drawn at
    population rate so gemini-vs-GPT-5.4 agreement is a clean, unweighted number
    with tight CIs, sliced by the compactor's model family.

At population rate (~11% judged-YES) N=2000 still yields ~220 retained rows, so
we do NOT need the balancing trick the human-50 used -- and thus no reweighting
caveat. The human-50 ids are force-included; the rest are sampled deterministically
from the eligible pool with the human ids removed, so (sources, seed, N) always
yields the same sample and the 50 are always a subset.

Family tagging (--> `compactor_family`)
---------------------------------------
Judges:  gpt-5.4 = OpenAI family ; gemini = Google family.
Compactors are mapped from their name/model (see FAMILY_MAP). Per the chosen
taxonomy gpt-oss-120b counts as OpenAI (lineage), so the OpenAI (== gpt-5.4
judge) family covers {gpt_5_4_mini, gpt_oss_120b}; gemma_4 is Google (== gemini
judge); qwen30b is Alibaba; recent_5 / llmlingua2 are non-LLM baselines.

Output
------
  <out>/<tag>_sample.jsonl   one row/line with everything the judge + analysis
                             need: id, provenance, compactor + family, the frozen
                             GPT-5.4 verdict, in_human_50 flag, sssc_message, and
                             the raw compacted_context (list[message]).

Usage
-----
  python judge_robustness_eval/make_family_probe.py \
    --manifest config/experiments/rq1/main.yaml \
    --manifest config/experiments/rq1/gpt-5.4-mini.yaml \
    --results_root /data/compaction_integrity \
    --human_key judge_robustness_eval/balanced_key.jsonl \
    --n 2000 --seed 0
"""

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

DEFAULT_OUT_DIR = Path(__file__).resolve().parent

Message = dict[str, str]


# ---------------------------------------------------------------------------
# Row identity + eligibility -- IDENTICAL to make_blind_test.py, so the human-50
# ids computed there line up with the ids we compute here (the nesting depends
# on this being byte-for-byte the same key).
# ---------------------------------------------------------------------------

def _stable_row_key(row: dict[str, Any]) -> str:
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


def _eligible(row: dict[str, Any]) -> bool:
    ret = row.get("retention")
    if ret is None or (isinstance(ret, float) and pd.isna(ret)):
        return False
    cc = row.get("compacted_context")
    return isinstance(cc, list) and len(cc) > 0


# ---------------------------------------------------------------------------
# Family taxonomy.  Judge families: gpt-5.4 -> openai, gemini -> google.
# gpt-oss-120b is OpenAI (lineage), so it shares the gpt-5.4 judge's family.
# ---------------------------------------------------------------------------

# (substring, family) checked in order against "name model" (lowercased).
FAMILY_MAP: list[tuple[str, str]] = [
    ("gpt_5_4_mini", "openai"),
    ("gpt-5.4-mini", "openai"),
    ("gpt_oss", "openai"),
    ("gpt-oss", "openai"),
    ("gemma", "google"),
    ("qwen", "alibaba"),
    ("recent", "baseline"),
    ("llmlingua", "baseline"),
]

# Which judge (by family) each compactor family is "same-family" with.
JUDGE_FAMILY = {"gpt_5_4": "openai", "gpt-5.4": "openai", "gemini": "google"}


def _compactor_field(row: dict[str, Any], key: str) -> str:
    comp = row.get("compactor")
    if isinstance(comp, dict):
        return str(comp.get(key) or "")
    return ""


def _compactor_name(row: dict[str, Any]) -> str:
    return _compactor_field(row, "name") or "unknown_compactor"


def _compactor_family(row: dict[str, Any]) -> str:
    hay = f"{_compactor_field(row, 'name')} {_compactor_field(row, 'model')}".lower()
    for needle, fam in FAMILY_MAP:
        if needle in hay:
            return fam
    return "other"


def _load_manifest_run_ids(manifest_path: Path) -> list[str]:
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    return [str(entry["run_id"]) for entry in manifest.get("runs", [])]


def _load_human_ids(key_path: Path) -> set[str]:
    if not key_path.exists():
        raise SystemExit(f"--human_key not found: {key_path}")
    ids: set[str] = set()
    for line in key_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            ids.add(str(json.loads(line)["id"]))
    return ids


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", type=Path, action="append", default=[],
                        help="Experiment manifest yaml whose runs are pooled (repeatable).")
    parser.add_argument("--results_root", type=Path, default=Path("/data/compaction_integrity"))
    parser.add_argument("--human_key", type=Path,
                        default=DEFAULT_OUT_DIR / "balanced_key.jsonl",
                        help="Held-aside key of the human-labeled blind set; its ids are "
                            "force-included so the human subset nests inside this sample.")
    parser.add_argument("--out_dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--tag", type=str, default="family_probe")
    parser.add_argument("--n", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if not args.manifest:
        raise SystemExit("Provide at least one --manifest.")

    out_path = args.out_dir / f"{args.tag}_sample.jsonl"
    if out_path.exists() and not args.overwrite:
        raise SystemExit(f"{out_path} already exists. Pass --overwrite to replace it.")

    # Resolve run ids (order-preserving, de-duped) across manifests.
    run_ids: list[str] = []
    seen_runs: set[str] = set()
    for man in args.manifest:
        for rid in _load_manifest_run_ids(man):
            if rid not in seen_runs:
                seen_runs.add(rid)
                run_ids.append(rid)

    # Pool eligible rows.
    eligible: list[dict[str, Any]] = []
    keys: list[str] = []
    n_missing = 0
    for rid in run_ids:
        pkl_path = args.results_root / "runs" / rid / "evaluation_results.pkl"
        if not pkl_path.exists():
            print(f"[skip] missing pickle: {pkl_path}")
            n_missing += 1
            continue
        df = pd.read_pickle(pkl_path)
        for r in df.to_dict(orient="records"):
            if _eligible(r):
                r["_run_id"] = rid
                eligible.append(r)
                keys.append(_stable_row_key(r))
    if not eligible:
        raise SystemExit("No eligible rows (judge verdict + compacted context) in any run.")
    if len(set(keys)) != len(keys):
        print("WARNING: duplicate row keys detected; sample determinism may be affected.")

    key_to_idx = {k: i for i, k in enumerate(keys)}  # last-wins on dupes

    # Force-include the human-50, then population-sample the rest deterministically.
    human_ids = _load_human_ids(args.human_key)
    forced = [key_to_idx[k] for k in human_ids if k in key_to_idx]
    missing_human = sorted(human_ids - set(keys))
    if missing_human:
        print(f"WARNING: {len(missing_human)} human id(s) not found in the pool "
              f"(they will be absent from the nested subset): {missing_human[:5]}...")

    forced_set = set(forced)
    remaining = sorted((i for i in range(len(eligible)) if i not in forced_set),
                       key=lambda i: keys[i])
    n_more = max(0, args.n - len(forced))
    rng = random.Random(args.seed)
    sampled = rng.sample(remaining, min(n_more, len(remaining)))

    chosen = sorted(set(forced) | set(sampled), key=lambda i: keys[i])

    n_yes = 0
    with out_path.open("w", encoding="utf-8") as f:
        for i in chosen:
            r = eligible[i]
            rid = keys[i]
            judge_ret = bool(r["retention"])
            n_yes += int(judge_ret)
            f.write(json.dumps({
                "id": rid,
                "run_id": r.get("_run_id"),
                "dataset": str(r.get("dataset")),
                "source_row_index": r.get("source_row_index"),
                "sssc_id": r.get("sssc_id"),
                "compactor": _compactor_name(r),
                "compactor_model": _compactor_field(r, "model"),
                "compactor_family": _compactor_family(r),
                "judge_gpt54_retention": judge_ret,
                "in_human_50": rid in human_ids,
                "sssc_message": str(r.get("sssc_message", "")),
                "compacted_context": r["compacted_context"],
            }, ensure_ascii=False, default=str) + "\n")

    n_human = sum(1 for i in chosen if keys[i] in human_ids)
    print(f"Runs pooled    : {len(run_ids)}" + (f", {n_missing} missing" if n_missing else ""))
    print(f"Eligible rows  : {len(eligible)}  (GPT-5.4 YES rate "
          f"{sum(bool(r['retention']) for r in eligible)/len(eligible):.1%})")
    print(f"Sampled        : {len(chosen)}  (forced human={n_human}, "
          f"GPT-5.4 YES={n_yes}, NO={len(chosen)-n_yes})")
    print("family breakdown (compactor):")
    from collections import Counter
    fam = Counter(_compactor_family(eligible[i]) for i in chosen)
    for k in sorted(fam):
        print(f"  {fam[k]:>5}  {k}")
    print(f"\nWrote sample -> {out_path}")
    print("Next: python judge_robustness_eval/judge_family_probe.py "
          f"--sample {out_path} --provider gemini --model gemini-flash-latest")


if __name__ == "__main__":
    main()
