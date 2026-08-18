"""Re-judge retention with an alternate LLM judge, reusing stored contexts.

The main evaluation (`scripts/evaluation.py`) only uses an LLM judge for one
quantity: `retention` (case 3). That verdict is a pure function of two columns
already saved in every `runs/<run_id>/evaluation_results.pkl`:

    compacted_context   (list[message])  -> rendered for the judge
    sssc_message        (str)            -> the injected SSSC to look for

This script reuses those columns and re-runs *only* the retention judge with a
different model (default: gemini-flash-latest via Google's OpenAI-compatible
endpoint), writing a new run directory per source run. The new pickle is a copy
of the source with exactly two columns overwritten -- `retention` and
`evaluator` -- so it has an identical schema/rows and drops straight into
`analyze/main_exp.py`.

Outputs (under `results_root`):
  runs/<new_run_id>/evaluation_results.pkl   new judge verdicts (same schema)
  runs/<new_run_id>/metadata.json            run_spec with the new evaluator
  <log_dir>/rejudge_run_map__<evaluator>.txt old_run_id -> new_run_id mapping
  <log_dir>/agreement__<evaluator>.{txt,csv} new-vs-old judge agreement

Usage:
  python -m compaction_integrity.scripts.rejudge_retention \
    --manifest_path config/experiments/rq1/main.yaml \
    --results_root /data/compaction_integrity \
    --new_evaluator_name llm_gemini_flash \
    --model gemini-flash-latest
"""

import argparse
import copy
import csv
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from openai import OpenAI
from tqdm.auto import tqdm

from compaction_integrity.api_keys import gemini_api_key, openai_api_key
from compaction_integrity.prompts import build_retention_judge_prompts
from compaction_integrity.scripts.eval_run_layout import build_run_id, write_run_metadata

# Google's OpenAI-compatible endpoint. Gemini models are reachable through the
# openai client by pointing base_url here and using chat.completions.
GEMINI_OPENAI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"

Message = dict[str, str]


def _build_client(provider: str) -> OpenAI:
    """All judges are called through the openai client + chat.completions;
    provider only selects the api key + base_url."""
    provider = provider.strip().lower()
    if provider == "gemini":
        api_key = str(gemini_api_key).strip()
        if not api_key:
            raise ValueError("gemini_api_key is empty in api_keys.py; set it before running.")
        return OpenAI(api_key=api_key, base_url=GEMINI_OPENAI_BASE_URL)
    if provider == "openai":
        api_key = str(openai_api_key).strip()
        if not api_key:
            raise ValueError("openai_api_key is empty in api_keys.py; set it before running.")
        return OpenAI(api_key=api_key)
    raise ValueError(f"Unsupported provider={provider!r} (expected 'gemini' or 'openai').")


def _omit_reasoning(reasoning_effort: str | None) -> bool:
    """Reasoning-effort is only meaningful for thinking models. An empty value
    or the sentinel 'omit' means: don't send the kwarg at all (e.g. for
    non-reasoning OpenAI judges that would reject it)."""
    return reasoning_effort is None or reasoning_effort.strip().lower() in ("", "omit")


# ---------------------------------------------------------------------------
# Small helpers (re-implemented locally so we don't import evaluation.py, which
# pulls in transformers/vllm).
# ---------------------------------------------------------------------------

def _flatten_dict(data: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    flattened: dict[str, Any] = {}
    for key, value in data.items():
        flattened_key = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            flattened.update(_flatten_dict(value, prefix=flattened_key))
            continue
        flattened[flattened_key] = value
    return flattened


def _render_messages_for_judge(messages: list[Message]) -> str:
    return "\n\n".join(f"{m['role']}:\n{m['content']}" for m in messages)


def _parse_retention_output(output_text: str) -> bool:
    normalized = output_text.strip().upper()
    if normalized.startswith("YES"):
        return True
    if normalized.startswith("NO"):
        return False
    raise ValueError(f"Retention judge must return YES or NO, got: {output_text!r}")


def _wide_row_key(row: dict[str, Any]) -> str:
    """Identity of a wide row *excluding* the evaluator (so the same row across
    two judges maps to the same key)."""
    return "|".join(
        [
            str(row["dataset"]),
            str(row["source_row_index"]),
            str(row["sssc_id"]),
            json.dumps(row["sssc_attrs"], ensure_ascii=True, sort_keys=True),
            json.dumps(row["probe"], ensure_ascii=True, sort_keys=True),
            json.dumps(row["compactor"], ensure_ascii=True, sort_keys=True),
        ]
    )


# ---------------------------------------------------------------------------
# Judge
# ---------------------------------------------------------------------------

def _retry_after_seconds(exc: Exception) -> float | None:
    """Best-effort extract of a server-requested retry delay from a rate-limit
    error (e.g. Gemini/OpenAI 'retryDelay': '31s' or a Retry-After header)."""
    header = getattr(getattr(exc, "response", None), "headers", None)
    if header is not None:
        val = header.get("retry-after") or header.get("Retry-After")
        if val:
            try:
                return float(val)
            except ValueError:
                pass
    m = re.search(r"retry(?:_?delay|-after)['\"]?\s*[:=]\s*['\"]?(\d+(?:\.\d+)?)", str(exc), re.I)
    if m:
        return float(m.group(1))
    return None


def _is_rate_limit(exc: Exception) -> bool:
    return getattr(exc, "status_code", None) == 429 or "429" in str(exc) or \
        "RESOURCE_EXHAUSTED" in str(exc)


def _judge_one(
    client: OpenAI,
    model: str,
    reasoning_effort: str,
    sssc_message: str,
    compacted_context: list[Message],
    max_attempts: int = 6,
) -> bool:
    system_prompt, user_prompt = build_retention_judge_prompts(
        injected_sssc=sssc_message,
        compacted_context=_render_messages_for_judge(compacted_context),
    )
    extra: dict[str, Any] = {}
    if not _omit_reasoning(reasoning_effort):
        # 'low' mirrors the GPT-5.4 judge; 'none' disables Gemini "thinking".
        extra["reasoning_effort"] = reasoning_effort
    last_exc: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                **extra,
            )
            return _parse_retention_output(resp.choices[0].message.content or "")
        except Exception as exc:  # includes parse failures and transient API errors
            last_exc = exc
            if attempt == max_attempts:
                break
            # Honor a server retry hint on rate limits; else exponential backoff.
            if _is_rate_limit(exc):
                delay = _retry_after_seconds(exc) or min(60.0, 2.0 ** attempt)
            else:
                delay = min(30.0, 1.5 ** attempt)
            print(f"Retention judge retry {attempt}/{max_attempts} in {delay:.0f}s: {exc}")
            time.sleep(delay)
    raise ValueError(f"Retention judge failed after {max_attempts} attempts: {last_exc}")


def _rejudge_dataframe(
    df: pd.DataFrame,
    client: OpenAI,
    model: str,
    reasoning_effort: str,
    concurrency: int,
    existing_retention_by_key: dict[str, bool | None],
    desc: str,
) -> list[bool | None]:
    """Return a new `retention` column aligned to df rows. Rows whose
    compacted_context is None get None; rows already judged in a prior run (via
    `existing_retention_by_key`) are reused for resume."""
    records = df.to_dict(orient="records")
    verdicts: list[bool | None] = [None] * len(records)
    pending: list[int] = []
    for i, row in enumerate(records):
        if row.get("compacted_context") is None:
            verdicts[i] = None
            continue
        prior = existing_retention_by_key.get(_wide_row_key(row))
        if prior is not None:
            verdicts[i] = bool(prior)
        else:
            pending.append(i)

    if not pending:
        return verdicts

    def _work(i: int) -> tuple[int, bool]:
        row = records[i]
        return i, _judge_one(
            client,
            model,
            reasoning_effort,
            str(row["sssc_message"]),
            row["compacted_context"],
        )

    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
        for i, verdict in tqdm(
            pool.map(_work, pending),
            total=len(pending),
            desc=desc,
            unit="judge",
            dynamic_ncols=True,
        ):
            verdicts[i] = verdict
    return verdicts


# ---------------------------------------------------------------------------
# Agreement
# ---------------------------------------------------------------------------

def _cohen_kappa(old: list[bool], new: list[bool]) -> float:
    n = len(old)
    if n == 0:
        return float("nan")
    po = sum(1 for a, b in zip(old, new) if a == b) / n
    # Marginals over {True, False}.
    p_old_true = sum(1 for a in old if a) / n
    p_new_true = sum(1 for b in new if b) / n
    pe = p_old_true * p_new_true + (1 - p_old_true) * (1 - p_new_true)
    if pe == 1.0:
        return 1.0
    return (po - pe) / (1 - pe)


def _agreement_stats(old_ret: list[bool | None], new_ret: list[bool | None]) -> dict[str, Any]:
    """Paired agreement over rows where BOTH judges returned a verdict."""
    old_p: list[bool] = []
    new_p: list[bool] = []
    for a, b in zip(old_ret, new_ret):
        if a is None or b is None:
            continue
        old_p.append(bool(a))
        new_p.append(bool(b))
    n = len(old_p)
    # Confusion cells: old \ new.
    tt = sum(1 for a, b in zip(old_p, new_p) if a and b)
    tf = sum(1 for a, b in zip(old_p, new_p) if a and not b)
    ft = sum(1 for a, b in zip(old_p, new_p) if not a and b)
    ff = sum(1 for a, b in zip(old_p, new_p) if not a and not b)
    agree = tt + ff
    return {
        "n_paired": n,
        "agree": agree,
        "agree_rate": (agree / n) if n else float("nan"),
        "old_true_new_true": tt,
        "old_true_new_false": tf,
        "old_false_new_true": ft,
        "old_false_new_false": ff,
        "old_retain_rate": (sum(old_p) / n) if n else float("nan"),
        "new_retain_rate": (sum(new_p) / n) if n else float("nan"),
        "cohen_kappa": _cohen_kappa(old_p, new_p),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _load_manifest_run_ids(manifest_path: Path) -> list[tuple[str, str]]:
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    out: list[tuple[str, str]] = []
    for entry in manifest["runs"]:
        out.append((str(entry["run_id"]), str(entry.get("label", ""))))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest_path", required=True, type=Path)
    parser.add_argument("--results_root", type=Path, default=Path("/data/compaction_integrity"))
    parser.add_argument("--provider", default="gemini", choices=["gemini", "openai"],
                        help="Selects api key + base_url; all judges use chat.completions.")
    parser.add_argument("--new_evaluator_name", default="llm_gemini_flash")
    parser.add_argument("--model", default="gemini-flash-latest")
    parser.add_argument("--reasoning_effort", default="low",
                        help="Thinking level. Default 'low' mirrors the GPT-5.4 judge "
                            "(OpenAIRuntime hardcodes reasoning effort 'low'). Use 'none' to "
                            "disable Gemini thinking, or '' / 'omit' for non-reasoning models "
                            "(e.g. gpt-4-mini) that reject the kwarg.")
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--max_rows", type=int, default=None,
                        help="Smoke-test cap: judge only the first N rows of each run.")
    parser.add_argument("--overwrite", action="store_true",
                        help="Re-judge from scratch instead of resuming existing new-judge pickles.")
    parser.add_argument("--log_dir", type=Path, default=None,
                        help="Where to write the run-id map and agreement report "
                            "(default: <manifest dir>).")
    args = parser.parse_args()

    client = _build_client(args.provider)

    runs_dir = args.results_root / "runs"
    log_dir = args.log_dir or args.manifest_path.parent
    log_dir.mkdir(parents=True, exist_ok=True)

    new_eval_config = {
        "model": args.model,
        "provider": args.provider,
        "kwargs": {"batch": False, "reasoning_effort": args.reasoning_effort},
    }
    flattened_new_eval = {"name": args.new_evaluator_name, **_flatten_dict(new_eval_config)}

    run_ids = _load_manifest_run_ids(args.manifest_path)
    print(f"Loaded {len(run_ids)} run(s) from {args.manifest_path}")

    mapping_rows: list[dict[str, Any]] = []
    per_run_agreement: list[dict[str, Any]] = []
    all_old: list[bool | None] = []
    all_new: list[bool | None] = []

    for old_run_id, label in run_ids:
        src_dir = runs_dir / old_run_id
        src_pkl = src_dir / "evaluation_results.pkl"
        src_meta = src_dir / "metadata.json"
        if not src_pkl.exists():
            print(f"[skip] missing source pickle: {src_pkl}")
            continue

        run_spec = json.loads(src_meta.read_text(encoding="utf-8"))["run_spec"]
        new_spec = copy.deepcopy(run_spec)
        new_spec["evaluator"] = {"name": args.new_evaluator_name, "config": new_eval_config}
        new_run_id = build_run_id(new_spec)
        new_dir = runs_dir / new_run_id
        new_pkl = new_dir / "evaluation_results.pkl"

        df = pd.read_pickle(src_pkl)
        if args.max_rows is not None:
            df = df.head(args.max_rows).copy()
        old_retention = [None if pd.isna(v) else bool(v) for v in df["retention"].tolist()]

        existing_by_key: dict[str, bool | None] = {}
        if new_pkl.exists() and not args.overwrite:
            prev = pd.read_pickle(new_pkl)
            for r in prev.to_dict(orient="records"):
                v = r.get("retention")
                if v is not None and not (isinstance(v, float) and pd.isna(v)):
                    existing_by_key[_wide_row_key(r)] = bool(v)
            print(f"[resume] {new_run_id}: reusing {len(existing_by_key)} prior verdict(s)")

        new_retention = _rejudge_dataframe(
            df=df,
            client=client,
            model=args.model,
            reasoning_effort=args.reasoning_effort,
            concurrency=args.concurrency,
            existing_retention_by_key=existing_by_key,
            desc=f"{args.new_evaluator_name} {label or old_run_id}",
        )

        out_df = df.copy()
        out_df["retention"] = new_retention
        out_df["evaluator"] = [dict(flattened_new_eval) for _ in range(len(out_df))]
        new_dir.mkdir(parents=True, exist_ok=True)
        out_df.to_pickle(new_pkl)
        write_run_metadata(new_dir, new_run_id, new_spec)

        stats = _agreement_stats(old_retention, new_retention)
        per_run_agreement.append({"label": label, "old_run_id": old_run_id,
                                "new_run_id": new_run_id, **stats})
        all_old.extend(old_retention)
        all_new.extend(new_retention)
        mapping_rows.append({"label": label, "old_run_id": old_run_id,
                            "new_run_id": new_run_id, "new_path": str(new_pkl)})
        print(f"[done] {label or old_run_id}\n"
            f"       old={old_run_id}\n       new={new_run_id}\n"
            f"       agree={stats['agree_rate']:.3f} kappa={stats['cohen_kappa']:.3f} "
            f"(n={stats['n_paired']})")

    # -- run-id mapping log ------------------------------------------------
    ts = datetime.now(timezone.utc).isoformat()
    map_path = log_dir / f"rejudge_run_map__{args.new_evaluator_name}.txt"
    with map_path.open("w", encoding="utf-8") as f:
        f.write(f"# retention re-judge run map\n# generated: {ts}\n")
        f.write(f"# manifest: {args.manifest_path}\n")
        f.write(f"# judge: {args.model} (evaluator={args.new_evaluator_name}, "
                f"reasoning_effort={args.reasoning_effort})\n")
        f.write(f"# runs_dir: {runs_dir}\n\n")
        for m in mapping_rows:
            f.write(f"{m['label']}\n")
            f.write(f"  old_run_id: {m['old_run_id']}\n")
            f.write(f"  new_run_id: {m['new_run_id']}\n")
            f.write(f"  new_path:   {m['new_path']}\n\n")
    print(f"\nWrote run-id map -> {map_path}")

    # -- agreement report --------------------------------------------------
    overall = _agreement_stats(all_old, all_new)
    agree_txt = log_dir / f"agreement__{args.new_evaluator_name}.txt"
    with agree_txt.open("w", encoding="utf-8") as f:
        f.write(f"# retention judge agreement: {args.new_evaluator_name} vs gpt-5.4 (old)\n")
        f.write(f"# generated: {ts}\n# manifest: {args.manifest_path}\n\n")
        f.write("== OVERALL ==\n")
        f.write(f"paired rows        : {overall['n_paired']}\n")
        f.write(f"agreement rate     : {overall['agree_rate']:.4f}\n")
        f.write(f"cohen's kappa      : {overall['cohen_kappa']:.4f}\n")
        f.write(f"old retain rate    : {overall['old_retain_rate']:.4f}\n")
        f.write(f"new retain rate    : {overall['new_retain_rate']:.4f}\n")
        f.write("confusion (old \\ new):\n")
        f.write(f"                new=RETAIN   new=DROP\n")
        f.write(f"  old=RETAIN    {overall['old_true_new_true']:>10d}  {overall['old_true_new_false']:>9d}\n")
        f.write(f"  old=DROP      {overall['old_false_new_true']:>10d}  {overall['old_false_new_false']:>9d}\n\n")
        f.write("== PER RUN ==\n")
        for r in per_run_agreement:
            f.write(f"{r['label'] or r['old_run_id']}: agree={r['agree_rate']:.3f} "
                    f"kappa={r['cohen_kappa']:.3f} n={r['n_paired']} "
                    f"(old_retain={r['old_retain_rate']:.3f} new_retain={r['new_retain_rate']:.3f})\n")

    agree_csv = log_dir / f"agreement__{args.new_evaluator_name}.csv"
    fieldnames = ["label", "old_run_id", "new_run_id", "n_paired", "agree", "agree_rate",
                "cohen_kappa", "old_retain_rate", "new_retain_rate",
                "old_true_new_true", "old_true_new_false",
                "old_false_new_true", "old_false_new_false"]
    with agree_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in per_run_agreement:
            w.writerow({k: r.get(k) for k in fieldnames})

    print(f"Wrote agreement report -> {agree_txt}")
    print(f"Wrote agreement csv    -> {agree_csv}")
    print(f"\nOVERALL agreement={overall['agree_rate']:.4f} "
        f"kappa={overall['cohen_kappa']:.4f} (n={overall['n_paired']})")


if __name__ == "__main__":
    main()
