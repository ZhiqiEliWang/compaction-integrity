"""Evaluation runner for the SC extractor prompt.

For each (dataset_row, sssc) pair:
  1. Inject the SSSC into the row's messages (reuses `_inject_sssc` from
     evaluation.py, controlled by `sssc_attrs`).
  2. Walk the conversation turn-by-turn. At every user turn, call the SC
     extractor SLM with `build_sc_extraction_prompts(...)`, passing the
     running registry of SCs detected so far in this pair and the immediately
     preceding assistant turn.
  3. Append new `text` items to the pair's registry.

Then an LLM-as-judge pass uses `build_retention_judge_prompts(...)` to decide
whether the originally injected SSSC text is "PRESENT" in the rendered
registry. Same OpenAI evaluator as evaluation.py.

Output: {results_root}/sc_extractor_runs[_test]/<run_id>/
    metadata.json, resolved_config.yaml,
    extraction_results.pkl  (per-pair registry; resumable)
    judgment_results.pkl    (adds retention column; resumable)
"""

from __future__ import annotations

import gc
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import hydra
import pandas as pd
from omegaconf import DictConfig
from tqdm.auto import tqdm

from compaction_integrity.dataset.eval_loader import EvalDatasetLoader, EvalDatasetRow
from compaction_integrity.prompts import (
    build_retention_judge_prompts,
    build_sc_extraction_prompts,
)
from compaction_integrity.runtime.env import apply_runtime_environment
from compaction_integrity.runtime.openai_runtime import OpenAIRuntime
from compaction_integrity.runtime.vllm_runtime import VLLMRuntime
from compaction_integrity.runtime.vllm_serve_runtime import VLLMServeRuntime
from compaction_integrity.scripts.eval_run_layout import (
    _content_hash,
    _load_dataset_manifest_identity,
    resolve_evaluator_cfg,
    to_container,
    write_run_metadata,
)
from compaction_integrity.scripts.evaluation import _inject_sssc
from compaction_integrity.sssc import SSSCS
from compaction_integrity.tokenization import count_tokens_text


Message = dict[str, str]

_EXTRACTION_COLUMNS = [
    "dataset", "dataset_path", "source_row_index",
    "sssc_id", "sssc_type", "sssc_message", "sssc_attrs",
    "sc_extractor", "evaluator",
    "num_user_turns", "num_extractions",
    "registry", "registry_text", "per_turn",
]
_JUDGMENT_COLUMNS = _EXTRACTION_COLUMNS + ["retention", "retention_raw"]

_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _flatten(data: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in data.items():
        nk = f"{prefix}.{k}" if prefix else str(k)
        if isinstance(v, dict):
            out.update(_flatten(v, nk))
        else:
            out[nk] = v
    return out


def _pair_key(row: dict[str, Any]) -> str:
    return "|".join([
        str(row["dataset"]), str(row["source_row_index"]), str(row["sssc_id"]),
        json.dumps(row["sssc_attrs"], sort_keys=True),
        json.dumps(row["sc_extractor"], sort_keys=True),
        json.dumps(row["evaluator"], sort_keys=True),
    ])


def _persist(rows: list[dict[str, Any]], cols: list[str], path: Path) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=cols)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_pickle(path)
    return df


def _render_registry(registry: list[dict[str, Any]]) -> str:
    if not registry:
        return "(none)"
    return "\n".join(f"{i}. {e['text']}" for i, e in enumerate(registry, start=1))


def _parse_extractor_output(raw: str) -> list[dict[str, str]]:
    text = (raw or "").strip()
    if not text:
        return []
    # Try in order: fenced JSON; the trailing `{...}` block (reasoning models
    # often emit prose then JSON); the whole text.
    candidates: list[str] = []
    match = _JSON_FENCE_RE.search(text)
    if match is not None:
        candidates.append(match.group(1).strip())
    last_open = text.rfind("{")
    last_close = text.rfind("}")
    if last_open != -1 and last_close > last_open:
        candidates.append(text[last_open : last_close + 1])
    candidates.append(text)
    data: Any = None
    for candidate in candidates:
        try:
            data = json.loads(candidate)
            break
        except json.JSONDecodeError:
            continue
    if data is None:
        return []
    items = data.get("scs", []) if isinstance(data, dict) else []
    out: list[dict[str, str]] = []
    for item in items if isinstance(items, list) else []:
        if not isinstance(item, dict):
            continue
        sc_text = item.get("text")
        if not isinstance(sc_text, str) or not sc_text.strip():
            continue
        out.append({"text": sc_text.strip(), "evidence": str(item.get("evidence", "")).strip()})
    return out


def _parse_retention_output(text: str) -> bool | None:
    n = (text or "").strip().upper()
    if n.startswith("YES"):
        return True
    if n.startswith("NO"):
        return False
    return None


def _build_extractor_runtime(model: str, provider: str, kwargs: dict[str, Any]) -> Any:
    runtime_kwargs = {k: v for k, v in kwargs.items() if k not in {"batch", "batch_size"}}
    if provider == "vllm":
        return VLLMRuntime(config={"model": model, **runtime_kwargs, "use_tqdm": False})
    if provider == "vllm_serve":
        return VLLMServeRuntime(config={"model": model, **runtime_kwargs})
    if provider == "openai":
        return OpenAIRuntime(config={"model": model, **runtime_kwargs})
    raise ValueError(f"Unsupported provider={provider} for SC extractor.")


# ---------------------------------------------------------------------------
# Per-pair scratch state
#
# Registry growth is sequential within a pair (turn N+1's prompt includes the
# registry produced through turn N). We batch across pairs at the same
# turn-step index for vLLM throughput, so each pair carries its own state
# across step iterations.
# ---------------------------------------------------------------------------

@dataclass
class _PairState:
    meta: dict[str, Any]
    messages: list[Message]
    user_turn_indices: list[int]
    registry: list[dict[str, Any]]
    per_turn: list[dict[str, Any]]


def _build_pair_states(
    *,
    dataset_name: str,
    dataset_path: str,
    rows: list[EvalDatasetRow],
    sssc_attrs: dict[str, Any],
    flattened_extractor_cfg: dict[str, Any],
    flattened_evaluator_cfg: dict[str, Any],
    global_seed: int,
) -> list[_PairState]:
    inject_rng = random.Random(global_seed)
    states: list[_PairState] = []
    for row in rows:
        for sssc in SSSCS:
            injected = _inject_sssc(
                list(row.messages),
                sssc_text=str(sssc["sssc"]),
                sssc_attrs=sssc_attrs,
                inject_rng=inject_rng,
            )
            user_indices = [i for i, m in enumerate(injected) if m["role"] == "user"]
            states.append(_PairState(
                meta={
                    "dataset": dataset_name,
                    "dataset_path": dataset_path,
                    "source_row_index": row.source_row_index,
                    "sssc_id": int(sssc["id"]),
                    "sssc_type": str(sssc["type"]),
                    "sssc_message": str(sssc["sssc"]),
                    "sssc_attrs": dict(sssc_attrs),
                    "sc_extractor": flattened_extractor_cfg,
                    "evaluator": flattened_evaluator_cfg,
                    "num_user_turns": len(user_indices),
                },
                messages=injected,
                user_turn_indices=user_indices,
                registry=[],
                per_turn=[],
            ))
    return states


def _prev_assistant(messages: list[Message], turn_idx: int) -> str | None:
    for j in range(turn_idx - 1, -1, -1):
        if messages[j]["role"] == "assistant":
            return messages[j]["content"]
    return None


# ---------------------------------------------------------------------------
# Phase 1: extraction (resumable)
# ---------------------------------------------------------------------------

def _run_extraction_phase(
    *,
    dataset_name: str,
    states: list[_PairState],
    extractor_cfg: dict[str, Any],
    save_path: Path,
    overwrite: bool,
) -> pd.DataFrame:
    if overwrite and save_path.exists():
        save_path.unlink()

    done: dict[str, dict[str, Any]] = {}
    if save_path.exists():
        for r in pd.read_pickle(save_path).to_dict(orient="records"):
            done[_pair_key(r)] = r
        print(f"Resume: {len(done)} pair(s) already in {save_path}")

    pending = [s for s in states if _pair_key(s.meta) not in done]
    if not pending:
        print(f"Extraction already complete for {dataset_name} ({len(states)} pairs).")
        return _persist(list(done.values()), _EXTRACTION_COLUMNS, save_path)

    print(f"Extraction: dataset={dataset_name} pending={len(pending)}/{len(states)}")

    raw_kwargs = dict(extractor_cfg.get("kwargs", {}))
    # batch_size only gates how often we flush the checkpoint pickle; vLLM
    # schedules internally. Default = no manual chunking (one flush per step).
    batch_size = raw_kwargs.get("batch_size")
    extractor = _build_extractor_runtime(
        str(extractor_cfg["model"]), str(extractor_cfg["provider"]), raw_kwargs,
    )

    max_steps = max((len(s.user_turn_indices) for s in pending), default=0)
    total_calls = sum(len(s.user_turn_indices) for s in pending)
    bar = tqdm(total=total_calls, desc=f"SC extraction ({dataset_name})",
               unit="turn", dynamic_ncols=True)

    try:
        for step in range(max_steps):
            active = [s for s in pending if step < len(s.user_turn_indices)]
            if not active:
                continue

            convos: list[list[Message]] = []
            for s in active:
                turn_idx = s.user_turn_indices[step]
                sys_p, user_p = build_sc_extraction_prompts(
                    user_turn=s.messages[turn_idx]["content"],
                    existing_scs=[e["text"] for e in s.registry],
                    prev_assistant_turn=_prev_assistant(s.messages, turn_idx),
                )
                convos.append([
                    {"role": "system", "content": sys_p},
                    {"role": "user", "content": user_p},
                ])

            chunk = int(batch_size) if batch_size else len(active)
            for start in range(0, len(active), chunk):
                end = min(start + chunk, len(active))
                outputs = extractor.batch_generate(convos[start:end])
                for s, response in zip(active[start:end], outputs):
                    raw = (response.text or "").strip()
                    parsed = _parse_extractor_output(raw)
                    seen = {e["text"] for e in s.registry}
                    added: list[dict[str, str]] = []
                    for item in parsed:
                        if item["text"] in seen:
                            continue
                        s.registry.append({**item, "source_turn_index": s.user_turn_indices[step]})
                        added.append(item)
                        seen.add(item["text"])
                    s.per_turn.append({
                        "turn_index": s.user_turn_indices[step],
                        "raw_output": raw,
                        "added": added,
                    })
                    bar.update(1)

            # Persist completed pairs incrementally so a crash mid-run resumes.
            for s in active:
                if len(s.per_turn) == len(s.user_turn_indices):
                    done[_pair_key(s.meta)] = {
                        **s.meta,
                        "num_extractions": sum(len(t["added"]) for t in s.per_turn),
                        "registry": list(s.registry),
                        "registry_text": _render_registry(s.registry),
                        "per_turn": list(s.per_turn),
                    }
            _persist(list(done.values()), _EXTRACTION_COLUMNS, save_path)
    finally:
        bar.close()
        close = getattr(extractor, "close", None)
        if close is not None:
            close()
        gc.collect()

    df = _persist(list(done.values()), _EXTRACTION_COLUMNS, save_path)
    print(f"Extraction saved: {save_path} ({len(df)} pair(s)).")
    return df


# ---------------------------------------------------------------------------
# Phase 2: retention judgment (resumable)
# ---------------------------------------------------------------------------

def _run_judgment_phase(
    *,
    dataset_name: str,
    extraction_df: pd.DataFrame,
    evaluator_cfg: dict[str, Any],
    save_path: Path,
    overwrite: bool,
) -> pd.DataFrame:
    if overwrite and save_path.exists():
        save_path.unlink()

    done: dict[str, dict[str, Any]] = {}
    if save_path.exists():
        for r in pd.read_pickle(save_path).to_dict(orient="records"):
            done[_pair_key(r)] = r
        print(f"Resume: {len(done)} judgment row(s) already in {save_path}")

    pending = [
        r for r in extraction_df.to_dict(orient="records")
        if _pair_key(r) not in done or done[_pair_key(r)].get("retention") is None
    ]
    if not pending:
        print(f"Judgment already complete for {dataset_name}.")
        return _persist(list(done.values()), _JUDGMENT_COLUMNS, save_path)

    runtime = OpenAIRuntime(config={})
    flex = bool(evaluator_cfg.get("kwargs", {}).get("flex", False))
    model = str(evaluator_cfg["model"])

    bar = tqdm(total=len(pending), desc=f"Retention judge ({dataset_name})",
               unit="pair", dynamic_ncols=True)
    try:
        for r in pending:
            sys_p, user_p = build_retention_judge_prompts(
                injected_sssc=str(r["sssc_message"]),
                compacted_context=str(r["registry_text"]),
            )
            parsed: bool | None = None
            raw = ""
            for attempt in range(1, 4):
                response = runtime.generate(
                    messages=[
                        {"role": "system", "content": sys_p},
                        {"role": "user", "content": user_p},
                    ],
                    model=model,
                    flex=flex,
                )
                raw = response.text or ""
                parsed = _parse_retention_output(raw)
                if parsed is not None:
                    break
                print(f"Retention judge non-YES/NO (attempt {attempt}/3): {raw!r}")
            done[_pair_key(r)] = {**r, "retention": parsed, "retention_raw": raw}
            bar.update(1)
            _persist(list(done.values()), _JUDGMENT_COLUMNS, save_path)
    finally:
        bar.close()

    df = _persist(list(done.values()), _JUDGMENT_COLUMNS, save_path)
    print(f"Judgment saved: {save_path}")
    return df


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

def _print_stats(dataset_name: str, judgment_df: pd.DataFrame) -> None:
    n = len(judgment_df)
    if n == 0:
        print(f"[{dataset_name}] no pairs.")
        return
    item_counts = judgment_df["registry"].apply(lambda r: len(r) if isinstance(r, list) else 0)
    reg_tokens = judgment_df.apply(
        lambda row: count_tokens_text(str(row["registry_text"])) if row["registry"] else 0,
        axis=1,
    )
    judged = judgment_df[judgment_df["retention"].notna()]
    retention_rate = (
        float(judged["retention"].astype(bool).mean()) if len(judged) > 0 else float("nan")
    )
    print(
        f"\n[{dataset_name}] stats over {n} pair(s):\n"
        f"  avg registry tokens:  {reg_tokens.mean():.2f}  "
        f"(min={int(reg_tokens.min())}, max={int(reg_tokens.max())})\n"
        f"  avg #SC items:        {item_counts.mean():.2f}  "
        f"(min={int(item_counts.min())}, max={int(item_counts.max())})\n"
        f"  retention rate:       {retention_rate:.3f}  ({len(judged)} judged)\n"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

@hydra.main(version_base=None, config_path="../../../config/experiments/rq4", config_name=None)
def main(cfg: DictConfig) -> None:
    apply_runtime_environment()

    test = bool(cfg.test)
    overwrite = bool(cfg.overwrite)
    global_seed = int(cfg.global_seed)
    results_root = Path(str(cfg.results_root))

    sssc_attrs = to_container(cfg.sssc_attrs)
    sssc_attrs["position"] = str(sssc_attrs["position"])
    sssc_attrs["repeat"] = int(sssc_attrs["repeat"])
    sssc_attrs["explicitness"] = bool(sssc_attrs["explicitness"])
    sssc_attrs["hard"] = bool(sssc_attrs["hard"])

    extractors = to_container(cfg.sc_extractors)
    if not isinstance(extractors, dict):
        raise TypeError(f"sc_extractors must be a mapping, got {type(extractors)!r}")
    evaluator_name, evaluator_cfg = resolve_evaluator_cfg(cfg.evaluators)
    flattened_evaluator_cfg = {"name": evaluator_name, **_flatten(evaluator_cfg)}

    runs_dir = results_root / ("sc_extractor_runs_test" if test else "sc_extractor_runs")
    runs_dir.mkdir(parents=True, exist_ok=True)

    for dataset_name in cfg.datasets:
        dataset_cfg = cfg.datasets[dataset_name]
        dataset_dir = Path(str(dataset_cfg.dir))
        num_rows = None if dataset_cfg.num_rows is None else int(dataset_cfg.num_rows)
        dataset_path = dataset_dir / "stitched_dataset"

        rows = EvalDatasetLoader.load(
            dataset_path=dataset_path, test_mode=test, num_rows=num_rows,
        ).rows()
        print(f"[{dataset_name}] loaded {len(rows)} row(s) from {dataset_path}")

        manifest_path, manifest_identity = _load_dataset_manifest_identity(dataset_dir)

        for extractor_name, extractor_cfg in extractors.items():
            extractor_cfg = to_container(extractor_cfg)
            flattened_extractor_cfg = {"name": extractor_name, **_flatten(extractor_cfg)}

            run_spec = {
                "task": "eval_sc_extractor",
                "test": test,
                "global_seed": global_seed,
                "dataset": {
                    "name": dataset_name, "dir": str(dataset_dir), "num_rows": num_rows,
                    "generation_manifest_path": manifest_path,
                    "generation_manifest_hash": _content_hash(manifest_identity),
                    "generation_manifest_identity": manifest_identity,
                },
                "sssc_attrs": dict(sssc_attrs),
                "sc_extractor": {"name": extractor_name, "config": extractor_cfg or {}},
                "evaluator": {"name": evaluator_name, "config": evaluator_cfg or {}},
            }
            run_id = f"{dataset_name}__{extractor_name}__{evaluator_name}"
            run_dir = runs_dir / run_id
            write_run_metadata(run_dir, run_id, run_spec)
            extraction_path = run_dir / "extraction_results.pkl"
            judgment_path = run_dir / "judgment_results.pkl"

            print(f"[STEP 1] extraction: dataset={dataset_name} extractor={extractor_name} run_id={run_id}")
            states = _build_pair_states(
                dataset_name=dataset_name,
                dataset_path=str(dataset_path),
                rows=rows,
                sssc_attrs=sssc_attrs,
                flattened_extractor_cfg=flattened_extractor_cfg,
                flattened_evaluator_cfg=flattened_evaluator_cfg,
                global_seed=global_seed,
            )
            extraction_df = _run_extraction_phase(
                dataset_name=dataset_name, states=states,
                extractor_cfg=extractor_cfg,
                save_path=extraction_path, overwrite=overwrite,
            )

            print(f"[STEP 2] judgment: dataset={dataset_name} extractor={extractor_name} run_id={run_id}")
            judgment_df = _run_judgment_phase(
                dataset_name=dataset_name, extraction_df=extraction_df,
                evaluator_cfg=evaluator_cfg,
                save_path=judgment_path, overwrite=overwrite,
            )

            _print_stats(f"{dataset_name}/{extractor_name}", judgment_df)


if __name__ == "__main__":
    main()
