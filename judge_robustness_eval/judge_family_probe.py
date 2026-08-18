"""Judge the family-bias probe sample with a second LLM judge (gemini).

Reads <tag>_sample.jsonl (from make_family_probe.py), runs the retention judge on
each row's stored compacted_context, and writes <tag>_judged.jsonl with a new
`gemini_retention` column next to the frozen `judge_gpt54_retention`.

The judge harness is reused verbatim from scripts.rejudge_retention -- same
prompt builder (build_retention_judge_prompts), same context rendering, same
YES/NO parse, same 429-aware backoff -- so gemini runs at parity with the GPT-5.4
judge (reasoning_effort='low' mirrors OpenAIRuntime's hardcoded effort). Only the
api key + base_url differ, selected by --provider.

Resumable: rows already present with a non-null verdict in an existing
<tag>_judged.jsonl are reused (keyed by id) unless --overwrite.

Usage:
  python judge_robustness_eval/judge_family_probe.py \
    --sample judge_robustness_eval/family_probe_sample.jsonl \
    --provider gemini --model gemini-flash-latest \
    --reasoning_effort low --concurrency 8
"""

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

# Make the in-repo package importable when run as a plain script.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tqdm.auto import tqdm  # noqa: E402

from compaction_integrity.scripts.rejudge_retention import (  # noqa: E402
    _build_client,
    _judge_one,
)


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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sample", type=Path, required=True,
                        help="<tag>_sample.jsonl from make_family_probe.py.")
    parser.add_argument("--out", type=Path, default=None,
                        help="Output path (default: alongside --sample as <tag>_judged.jsonl).")
    parser.add_argument("--provider", default="gemini", choices=["gemini", "openai"])
    parser.add_argument("--model", default="gemini-flash-latest")
    parser.add_argument("--reasoning_effort", default="low",
                        help="Thinking level; 'low' mirrors the GPT-5.4 judge.")
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--max_rows", type=int, default=None,
                        help="Smoke-test cap: judge only the first N sample rows.")
    parser.add_argument("--overwrite", action="store_true",
                        help="Re-judge from scratch instead of resuming an existing out file.")
    args = parser.parse_args()

    rows = _load_jsonl(args.sample)
    if args.max_rows is not None:
        rows = rows[: args.max_rows]

    out_path = args.out or args.sample.with_name(
        args.sample.name.replace("_sample.jsonl", "_judged.jsonl")
    )
    if out_path == args.sample:
        out_path = args.sample.with_suffix(".judged.jsonl")

    # Resume: reuse prior non-null verdicts keyed by id.
    prior: dict[str, bool] = {}
    if out_path.exists() and not args.overwrite:
        for r in _load_jsonl(out_path):
            v = r.get("gemini_retention")
            if isinstance(v, bool):
                prior[str(r["id"])] = v
        print(f"[resume] reusing {len(prior)} prior verdict(s) from {out_path}")

    client = _build_client(args.provider)

    pending = [i for i, r in enumerate(rows) if str(r["id"]) not in prior]
    verdicts: dict[str, bool | None] = dict(prior)

    def _work(i: int) -> tuple[str, bool | None]:
        r = rows[i]
        try:
            v = _judge_one(
                client,
                args.model,
                args.reasoning_effort,
                str(r.get("sssc_message", "")),
                r["compacted_context"],
            )
            return str(r["id"]), v
        except Exception as exc:  # judge exhausted its retries -> record as null
            print(f"[fail] id={r['id']}: {exc}")
            return str(r["id"]), None

    if pending:
        with ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as pool:
            for rid, v in tqdm(pool.map(_work, pending), total=len(pending),
                               desc=f"{args.provider}:{args.model}", unit="judge",
                               dynamic_ncols=True):
                verdicts[rid] = v

    n_ok = n_fail = 0
    with out_path.open("w", encoding="utf-8") as f:
        for r in rows:
            v = verdicts.get(str(r["id"]))
            n_ok += int(v is not None)
            n_fail += int(v is None)
            out = dict(r)
            out["gemini_retention"] = v
            out["gemini_judge_model"] = args.model
            f.write(json.dumps(out, ensure_ascii=False, default=str) + "\n")

    print(f"Judged {n_ok} row(s); {n_fail} failed/null.")
    print(f"Wrote -> {out_path}")
    print(f"Next: python judge_robustness_eval/analyze_family_probe.py --judged {out_path} "
          "--human judge_robustness_eval/balanced_blind_labeled.jsonl")


if __name__ == "__main__":
    main()
