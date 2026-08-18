"""Score a filled-in blind retention test against the LLM judge.

Reads the blind file you filled (`<tag>_blind.jsonl`, each line's "label" set to
yes/no) and the held-aside `<tag>_key.jsonl`, joins them on id, and reports
human-vs-judge agreement.

Completeness gate: if ANY line is still unfilled or has an unrecognized label,
this prints what's missing and STOPS -- no analysis is produced until every
line is labeled. Only when the file is complete does it report:

  - raw agreement + Cohen's kappa
  - the 2x2 confusion (judge x human)
  - per-compactor agreement
  - the list of disagreements (with the SSSC)

Accepted labels (case-insensitive): yes/y/true/1  and  no/n/false/0.

Usage:
  python judge_robustness_eval/analyze_blind_test.py \
    --blind judge_robustness_eval/balanced_blind.jsonl \
    --key   judge_robustness_eval/balanced_key.jsonl \
    [--csv  judge_robustness_eval/balanced_join.csv]
"""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

_YES = {"yes", "y", "true", "1", "present"}
_NO = {"no", "n", "false", "0", "absent"}


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise SystemExit(f"File not found: {path}")
    out = []
    for ln, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError as e:
            raise SystemExit(f"{path} line {ln}: invalid JSON ({e}). "
                             "Edit only the label value; don't break the quoting.")
    return out


def _parse_label(raw: Any) -> bool | None:
    if raw is None:
        return None
    s = str(raw).strip().lower()
    if s in _YES:
        return True
    if s in _NO:
        return False
    return None  # empty or unrecognized -> not yet a valid answer


def _cohen_kappa(a: list[bool], b: list[bool]) -> float:
    n = len(a)
    if n == 0:
        return float("nan")
    po = sum(1 for x, y in zip(a, b) if x == y) / n
    pa, pb = sum(a) / n, sum(b) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return 1.0 if pe == 1.0 else (po - pe) / (1 - pe)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--blind", type=Path, required=True)
    parser.add_argument("--key", type=Path, required=True)
    parser.add_argument("--csv", type=Path, default=None)
    args = parser.parse_args()

    blind = _load_jsonl(args.blind)
    key = {r["id"]: r for r in _load_jsonl(args.key)}

    # --- Completeness gate -------------------------------------------------
    missing: list[str] = []
    bad: list[tuple[str, Any]] = []
    for i, item in enumerate(blind, 1):
        lab = _parse_label(item.get("label"))
        if lab is None:
            raw = item.get("label")
            if raw is None or str(raw).strip() == "":
                missing.append(f"line {i} (id={item.get('id')})")
            else:
                bad.append((f"line {i} (id={item.get('id')})", raw))

    if missing or bad:
        done = len(blind) - len(missing) - len(bad)
        print(f"NOT COMPLETE -- {done}/{len(blind)} labeled. No analysis produced.\n")
        if missing:
            print(f"Unfilled labels ({len(missing)}):")
            for m in missing[:20]:
                print(f"  {m}")
            if len(missing) > 20:
                print(f"  ... and {len(missing) - 20} more")
        if bad:
            print(f"\nUnrecognized labels ({len(bad)}) -- use yes/no:")
            for loc, raw in bad[:20]:
                print(f"  {loc}: {raw!r}")
        raise SystemExit(1)

    # --- Analysis (only reached when every line is a valid yes/no) ---------
    rows: list[dict[str, Any]] = []
    hj: list[bool] = []
    jj: list[bool] = []
    n_missing_key = 0
    for item in blind:
        rid = item["id"]
        krec = key.get(rid)
        if krec is None:
            n_missing_key += 1
            continue
        h = _parse_label(item["label"])
        j = bool(krec["judge_retention"])
        hj.append(h)
        jj.append(j)
        rows.append({
            "id": rid,
            "compactor": krec.get("compactor"),
            "judge_name": krec.get("judge_name"),
            "run_id": krec.get("run_id"),
            "judge_retention": j,
            "human_retention": h,
            "agree": h == j,
            "sssc_message": item.get("sssc_message", ""),
        })

    n = len(rows)
    if n == 0:
        raise SystemExit(f"No overlap between blind and key on id (missing_key={n_missing_key}).")

    agree = sum(1 for r in rows if r["agree"])
    jy_hy = sum(1 for j, h in zip(jj, hj) if j and h)
    jy_hn = sum(1 for j, h in zip(jj, hj) if j and not h)
    jn_hy = sum(1 for j, h in zip(jj, hj) if not j and h)
    jn_hn = sum(1 for j, h in zip(jj, hj) if not j and not h)

    print("=" * 70)
    print("Retention: human vs LLM judge (blind subset)")
    print("=" * 70)
    print(f"compared items : {n}" + (f"   (missing key: {n_missing_key})" if n_missing_key else ""))
    print(f"raw agreement  : {agree}/{n} = {agree / n:.1%}")
    print(f"Cohen's kappa  : {_cohen_kappa(jj, hj):.3f}")
    print(f"judge YES rate : {sum(jj) / n:.1%}    human YES rate : {sum(hj) / n:.1%}")
    print("\nconfusion (judge \\ human)     human=YES   human=NO")
    print(f"  judge=YES                      {jy_hy:>6}    {jy_hn:>6}")
    print(f"  judge=NO                       {jn_hy:>6}    {jn_hn:>6}")

    by_comp: dict[str, list[bool]] = defaultdict(list)
    for r in rows:
        by_comp[r["compactor"]].append(r["agree"])
    print("\nper-compactor agreement:")
    for comp in sorted(by_comp, key=lambda c: (sum(by_comp[c]) / len(by_comp[c]))):
        a = by_comp[comp]
        print(f"  {sum(a)}/{len(a)} = {sum(a)/len(a):5.0%}   {comp}")

    disagree = [r for r in rows if not r["agree"]]
    print(f"\ndisagreements: {len(disagree)}")
    for r in disagree:
        print(f"  - id={r['id']}  judge={'YES' if r['judge_retention'] else 'NO':<3} "
              f"human={'YES' if r['human_retention'] else 'NO':<3}  compactor={r['compactor']}")
        if r["sssc_message"]:
            print(f"      SSSC: {r['sssc_message'][:120]}")

    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\nWrote per-item join -> {args.csv}")


if __name__ == "__main__":
    main()
