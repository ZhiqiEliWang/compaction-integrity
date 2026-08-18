"""Retention judge validity: 3-way human/GPT-5.4/gemini agreement + family-bias.

Addresses the single-judge / model-family-bias critique of the retention metric
using the family probe (judge_robustness_eval/make_family_probe.py -> judge_family_probe.py),
a population-rate sample that NESTS the 50-row human blind set. Two reports:

  PART A -- 3-WAY on the human-50 (ground truth = human)
    All three pairwise agreements (human<->GPT-5.4, human<->gemini,
    GPT-5.4<->gemini) with Cohen's kappa + confusion, the full joint 8-cell
    (human, gpt5.4, gemini) distribution, and the unanimous-agreement rate. If
    both judges track the human, retention measures what a human means by
    "SC preserved" -- not a GPT-5.4 idiosyncrasy.

  PART B -- 2-WAY on the full sample (GPT-5.4 vs gemini) + MODEL-FAMILY BIAS
    Population-rate agreement + kappa overall, then per-compactor signed gap
        delta = (GPT-5.4 YES-rate) - (gemini YES-rate)
    and family buckets. Judges: GPT-5.4 = OpenAI family, gemini = Google family;
    gpt-oss-120b counts as OpenAI (lineage). Self-favoring would show as
        OpenAI compactors (== GPT-5.4 family) -> delta > 0,
        Google compactor  (== gemini family)  -> delta < 0,
        neutral compactors                    -> delta ~ 0.
    A flat delta is the evidence no model-family bias drives the retention
    comparison (incl. GPT-5.4-mini vs other compactors).

Inputs are the two jsonl files produced by the family probe; nothing here
recomputes a judge verdict.

Usage:
  python -m compaction_integrity.analyze.judge_family_bias \
    --judged judge_robustness_eval/family_probe_judged.jsonl \
    --human  judge_robustness_eval/balanced_blind_labeled.jsonl \
    [--out_dir judge_robustness_eval]
"""

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

_YES = {"yes", "y", "true", "1", "present"}
_NO = {"no", "n", "false", "0", "absent"}

# Compactor families that coincide with a judge's family (see FAMILY_MAP in
# judge_robustness_eval/make_family_probe.py). GPT-5.4 -> openai, gemini -> google.
GPT54_JUDGE_FAMILY = "openai"
GEMINI_JUDGE_FAMILY = "google"


# ---------------------------------------------------------------------------
# IO + label parsing
# ---------------------------------------------------------------------------

def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise SystemExit(f"File not found: {path}")
    out: list[dict[str, Any]] = []
    for ln, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError as e:
            raise SystemExit(f"{path} line {ln}: invalid JSON ({e}).")
    return out


def _parse_label(raw: Any) -> bool | None:
    if isinstance(raw, bool):
        return raw
    if raw is None:
        return None
    s = str(raw).strip().lower()
    if s in _YES:
        return True
    if s in _NO:
        return False
    return None


# ---------------------------------------------------------------------------
# Pairwise agreement
# ---------------------------------------------------------------------------

def _cohen_kappa(a: list[bool], b: list[bool]) -> float:
    n = len(a)
    if n == 0:
        return float("nan")
    po = sum(1 for x, y in zip(a, b) if x == y) / n
    pa, pb = sum(a) / n, sum(b) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return 1.0 if pe == 1.0 else (po - pe) / (1 - pe)


def _pair_stats(x: list[bool], y: list[bool]) -> dict[str, Any]:
    """Agreement of verdict list x (reference) vs y over the paired rows."""
    n = len(x)
    return {
        "n": n,
        "agree": sum(1 for a, b in zip(x, y) if a == b),
        "agree_rate": (sum(1 for a, b in zip(x, y) if a == b) / n) if n else float("nan"),
        "kappa": _cohen_kappa(x, y),
        "x_yes_rate": (sum(x) / n) if n else float("nan"),
        "y_yes_rate": (sum(y) / n) if n else float("nan"),
        "xt_yt": sum(1 for a, b in zip(x, y) if a and b),
        "xt_yn": sum(1 for a, b in zip(x, y) if a and not b),
        "xn_yt": sum(1 for a, b in zip(x, y) if not a and b),
        "xn_yn": sum(1 for a, b in zip(x, y) if not a and not b),
    }


def _print_pair(title: str, ref: str, other: str, s: dict[str, Any]) -> None:
    print(f"\n{title}  (n={s['n']})")
    print(f"  agreement : {s['agree']}/{s['n']} = {s['agree_rate']:.1%}   kappa = {s['kappa']:.3f}")
    print(f"  {ref} YES rate = {s['x_yes_rate']:.1%}    {other} YES rate = {s['y_yes_rate']:.1%}")
    print(f"  confusion ({ref} \\ {other})    {other}=YES  {other}=NO")
    print(f"    {ref}=YES              {s['xt_yt']:>6}  {s['xt_yn']:>6}")
    print(f"    {ref}=NO               {s['xn_yt']:>6}  {s['xn_yn']:>6}")


# ---------------------------------------------------------------------------
# PART A -- 3-way on the human-50
# ---------------------------------------------------------------------------

def _part_a(rows: list[dict[str, Any]], human: dict[str, bool | None],
            out_dir: Path) -> None:
    print("=" * 72)
    print("PART A  -- 3-way agreement on the human-50 (ground truth = human)")
    print("=" * 72)

    h: list[bool] = []
    g54: list[bool] = []
    gem: list[bool] = []
    triples: list[dict[str, Any]] = []
    n_dropped = 0
    for r in rows:
        if not r.get("in_human_50"):
            continue
        hv = human.get(str(r["id"]))
        gemv = r.get("gemini_retention")
        if hv is None or gemv is None:
            n_dropped += 1
            continue
        hb, g4b, geb = bool(hv), bool(r["judge_gpt54_retention"]), bool(gemv)
        h.append(hb)
        g54.append(g4b)
        gem.append(geb)
        triples.append({"id": r["id"], "compactor": r.get("compactor"),
                        "compactor_family": r.get("compactor_family"),
                        "human": hb, "gpt54": g4b, "gemini": geb})

    if not h:
        print("No human-labeled rows with both judge verdicts found.\n")
        return
    print(f"rows compared: {len(h)}"
          + (f"   (dropped {n_dropped} unlabeled/failed)" if n_dropped else ""))

    s_hg = _pair_stats(h, g54)
    s_hm = _pair_stats(h, gem)
    s_gm = _pair_stats(g54, gem)
    _print_pair("human vs GPT-5.4", "human", "gpt5.4", s_hg)
    _print_pair("human vs gemini ", "human", "gemini", s_hm)
    _print_pair("GPT-5.4 vs gemini", "gpt5.4", "gemini", s_gm)

    # Joint 8-cell (human, gpt5.4, gemini) distribution.
    joint = Counter((t["human"], t["gpt54"], t["gemini"]) for t in triples)
    unanimous = sum(v for (a, b, c), v in joint.items() if a == b == c)
    print(f"\njoint (human, gpt5.4, gemini) distribution  [unanimous "
          f"{unanimous}/{len(h)} = {unanimous/len(h):.1%}]")
    print(f"  {'human':>6} {'gpt5.4':>7} {'gemini':>7}   {'count':>5}")
    for combo in sorted(joint, reverse=True):
        tag = "  (unanimous)" if combo[0] == combo[1] == combo[2] else ""
        y = lambda v: "YES" if v else "NO"
        print(f"  {y(combo[0]):>6} {y(combo[1]):>7} {y(combo[2]):>7}   {joint[combo]:>5}{tag}")

    # Both judges vs human as ground truth: who's closer.
    print(f"\n  --> vs human: GPT-5.4 {s_hg['agree_rate']:.1%} (kappa {s_hg['kappa']:.3f}), "
          f"gemini {s_hm['agree_rate']:.1%} (kappa {s_hm['kappa']:.3f}); "
          f"the two judges agree with each other {s_gm['agree_rate']:.1%}.")

    # Any row where a judge diverges from the human.
    disagree = [t for t in triples if not (t["human"] == t["gpt54"] == t["gemini"])]
    if disagree:
        print(f"\nnon-unanimous rows ({len(disagree)}):")
        for t in disagree:
            print(f"  id={t['id']}  human={'Y' if t['human'] else 'N'} "
                  f"gpt5.4={'Y' if t['gpt54'] else 'N'} gemini={'Y' if t['gemini'] else 'N'}  "
                  f"{t['compactor']}")

    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "family_probe_3way_human50.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(triples[0].keys()))
        w.writeheader()
        w.writerows(triples)
    print(f"\nWrote 3-way per-row table -> {csv_path}")


# ---------------------------------------------------------------------------
# PART B -- 2-way + family bias on the full sample
# ---------------------------------------------------------------------------

def _part_b(rows: list[dict[str, Any]], out_dir: Path) -> None:
    print("\n" + "=" * 72)
    print("PART B  -- GPT-5.4 vs gemini on the full sample + model-family bias")
    print("=" * 72)

    ref: list[bool] = []
    oth: list[bool] = []
    by_comp: dict[str, dict[str, list[bool]]] = defaultdict(lambda: {"ref": [], "oth": []})
    fam_of: dict[str, str] = {}
    n_dropped = 0
    for r in rows:
        gemv = r.get("gemini_retention")
        if gemv is None:
            n_dropped += 1
            continue
        rv, ov = bool(r["judge_gpt54_retention"]), bool(gemv)
        ref.append(rv)
        oth.append(ov)
        comp = str(r.get("compactor"))
        fam_of[comp] = str(r.get("compactor_family"))
        by_comp[comp]["ref"].append(rv)
        by_comp[comp]["oth"].append(ov)

    if not ref:
        raise SystemExit("No rows with a gemini verdict; nothing to compare.")

    overall = _pair_stats(ref, oth)
    print(f"paired rows: {overall['n']}"
          + (f"   (dropped {n_dropped} null gemini)" if n_dropped else ""))
    _print_pair("GPT-5.4 vs gemini (overall)", "gpt5.4", "gemini", overall)

    comp_rows: list[dict[str, Any]] = []
    for comp in sorted(by_comp, key=lambda c: (fam_of[c], c)):
        s = _pair_stats(by_comp[comp]["ref"], by_comp[comp]["oth"])
        comp_rows.append({
            "compactor": comp, "family": fam_of[comp], "n": s["n"],
            "gpt54_yes_rate": s["x_yes_rate"], "gemini_yes_rate": s["y_yes_rate"],
            "delta": s["x_yes_rate"] - s["y_yes_rate"],
            "agree_rate": s["agree_rate"], "kappa": s["kappa"],
        })

    print("\nper-compactor  (delta = gpt5.4 YES-rate - gemini YES-rate; + = gpt5.4 more lenient)")
    print(f"  {'compactor':<30} {'family':<9} {'n':>5} {'gpt5.4':>7} {'gemini':>7} "
          f"{'delta':>7} {'agree':>6} {'kappa':>6}")
    for c in comp_rows:
        print(f"  {c['compactor']:<30} {c['family']:<9} {c['n']:>5} "
              f"{c['gpt54_yes_rate']:>6.1%} {c['gemini_yes_rate']:>6.1%} "
              f"{c['delta']:>+6.1%} {c['agree_rate']:>5.0%} {c['kappa']:>6.3f}")

    def _bucket(pred) -> dict[str, Any]:
        x = [bool(r["judge_gpt54_retention"]) for r in rows
             if r.get("gemini_retention") is not None and pred(r)]
        y = [bool(r["gemini_retention"]) for r in rows
             if r.get("gemini_retention") is not None and pred(r)]
        if not x:
            return {"n": 0}
        s = _pair_stats(x, y)
        s["delta"] = s["x_yes_rate"] - s["y_yes_rate"]
        return s

    buckets = [
        ("OpenAI compactors  (== GPT-5.4 judge family)",
         lambda r: r.get("compactor_family") == GPT54_JUDGE_FAMILY),
        ("Google compactors  (== gemini judge family)",
         lambda r: r.get("compactor_family") == GEMINI_JUDGE_FAMILY),
        ("neutral compactors (neither judge's family)",
         lambda r: r.get("compactor_family") not in (GPT54_JUDGE_FAMILY, GEMINI_JUDGE_FAMILY)),
    ]
    print("\nfamily buckets  (bias shows as delta swinging + for OpenAI, - for Google)")
    print(f"  {'bucket':<46} {'n':>6} {'gpt5.4':>7} {'gemini':>7} {'delta':>7} {'agree':>6} {'kappa':>6}")
    bucket_rows: list[dict[str, Any]] = []
    for name, pred in buckets:
        s = _bucket(pred)
        if not s.get("n"):
            print(f"  {name:<46} {'0':>6}   (no rows)")
            continue
        print(f"  {name:<46} {s['n']:>6} {s['x_yes_rate']:>6.1%} {s['y_yes_rate']:>6.1%} "
              f"{s['delta']:>+6.1%} {s['agree_rate']:>5.0%} {s['kappa']:>6.3f}")
        bucket_rows.append({"bucket": name, "n": s["n"],
                            "gpt54_yes_rate": s["x_yes_rate"], "gemini_yes_rate": s["y_yes_rate"],
                            "delta": s["delta"], "agree_rate": s["agree_rate"], "kappa": s["kappa"]})

    out_dir.mkdir(parents=True, exist_ok=True)
    comp_csv = out_dir / "family_probe_per_compactor.csv"
    with comp_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(comp_rows[0].keys()))
        w.writeheader()
        w.writerows(comp_rows)
    fam_csv = out_dir / "family_probe_family_buckets.csv"
    if bucket_rows:
        with fam_csv.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(bucket_rows[0].keys()))
            w.writeheader()
            w.writerows(bucket_rows)
    print(f"\nWrote per-compactor table -> {comp_csv}")
    if bucket_rows:
        print(f"Wrote family-bucket table -> {fam_csv}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--judged", type=Path, required=True,
                        help="family_probe_judged.jsonl from judge_family_probe.py.")
    parser.add_argument("--human", type=Path, required=True,
                        help="Human-labeled blind file (balanced_blind_labeled.jsonl).")
    parser.add_argument("--out_dir", type=Path, default=None,
                        help="Where to write CSVs (default: alongside --judged).")
    args = parser.parse_args()

    rows = _load_jsonl(args.judged)
    human = {str(r["id"]): _parse_label(r.get("label")) for r in _load_jsonl(args.human)}
    out_dir = args.out_dir or args.judged.parent

    _part_a(rows, human, out_dir)
    _part_b(rows, out_dir)


if __name__ == "__main__":
    main()
