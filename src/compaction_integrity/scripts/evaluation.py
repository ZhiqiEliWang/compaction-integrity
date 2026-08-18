"""Four-experiment SSSC evaluation runner.

For each (dataset_row, sssc) pair, computes:
  1. full_with_sssc        - probe on full context with SSSC injected (no compaction)
  2. full_without_sssc     - probe on full context without SSSC (no compaction)
  3. compacted_with_sssc   - probe on compacted context (SSSC injected pre-compaction)
  4. compacted_post_sssc   - probe on the same compacted context as case 3, with the
                             wrapped SSSC re-presented as a user turn merged into the
                             probe prompt (i.e. SSSC appears post-compaction)

Compliance is graded by exact A/B letter match against a forced-choice probe
(no LLM judge). Retention is LLM-judged and only computed for case 3.

Cases 1 & 2 are compactor-independent and cached at:
    {results_root}/_full_probe_cache[/_test]/<shared_id>.pkl

Cases 3 & 4 are compactor-dependent. Wide per-compactor results live at:
    {results_root}/runs[_test]/<run_id>/evaluation_results.pkl

================================================================================
Schema of the wide per-compactor result file (`evaluation_results.pkl`)
================================================================================
The pickle contains a pandas DataFrame with one row per
(dataset_row, sssc, compactor, evaluator, probe) tuple. Columns:

Identity / configuration (constant within a run):
  dataset                       Dataset name (e.g. "wildchat", "hermes").
  dataset_path                  Filesystem path to the stitched dataset used.
  source_row_index              Row index into the source dataset.
  sssc_id                       Integer id of the SSSC (from `SSSCS`).
  sssc_type                     SSSC category label.
  sssc_message                  The raw SSSC text injected into the context.
  sssc_probe                    The probe question text (forced-choice).
  sssc_correct                  The compliant answer (string).
  sssc_incorrect                The non-compliant answer (string).
  sssc_attrs                    Dict of injection knobs:
                                  position ("top"/"middle"/"bottom"),
                                  repeat (int >= 1),
                                  explicitness (bool), hard (bool).
  probe                         Flattened probe-runtime config dict
                                (model, provider, kwargs, name).
  compactor                     Flattened compactor config dict
                                (name, model, provider, kwargs, prompt, ...).
  evaluator                     Flattened retention-judge config dict.
  swap_seed                     Per-pair RNG seed used to randomize A/B letter
                                assignment in the probe prompt. Identical across
                                cases 1-4 so grades are directly comparable.
  compliant_letter              "A" or "B" — the letter that maps to the
                                compliant (correct) answer for this row, given
                                `swap_seed`.

Case 1 — full_with_sssc (probe on full context, SSSC injected):
  full_with_sssc_messages       The full message list with SSSC injected
                                (system prompt NOT included).
  full_with_sssc_probe_prompt   Final user message containing the probe.
  full_with_sssc_output         Raw model text (analysis + final, joined).
  full_with_sssc_compliant      True/False if model picked the compliant
                                letter; None if response wasn't a clean A/B.

Case 2 — full_without_sssc (probe on full context, no SSSC):
  full_without_sssc_probe_prompt   Same probe prompt as case 1.
  full_without_sssc_output         Raw model text.
  full_without_sssc_compliant      True/False/None (same grading rule).

Case 3 — compacted_with_sssc (probe on compacted context):
  compacted_context             Compacted message list (output of the
                                compactor on the SSSC-injected messages).
                                None when compaction failed after retries.
  compaction_status             "success" or "error".
  compaction_error              Exception string if compaction failed, else None.
  compacted_probe_prompt        Final user message (= the probe text).
                                None if compaction failed.
  compacted_output              Raw model text. None if compaction failed.
  compacted_compliant           True/False/None.
  retention                     LLM judge verdict (True if compacted context
                                still contains the SSSC content, else False;
                                None if compaction failed).

Case 4 — compacted_post_sssc (re-present SSSC as a user turn after compaction):
  compacted_post_sssc_user_turn       The wrapped SSSC string
                                      (sssc_to_prompt(sssc_message, ...)).
                                      None if compaction failed.
  compacted_post_sssc_probe_prompt    The user message actually sent: the
                                      wrapped SSSC followed by the probe
                                      ("{wrapped}\\n\\n{probe}"). None if
                                      compaction failed.
  compacted_post_sssc_output          Raw model text. None if compaction failed.
  compacted_post_sssc_compliant       True/False/None (same grading rule).

Notes on resume:
  * If the wide pickle already has cases 1-3 but not case 4 (older runs),
    re-running fills in the case-4 columns in place; rows are not duplicated.
  * Rows where compaction failed (`compacted_context is None`) skip both
    case-3 and case-4 probes; their probe/output/compliant fields stay None.
"""

import gc
import json
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import hydra
import pandas as pd
from omegaconf import DictConfig
from tqdm.auto import tqdm
from transformers import AutoTokenizer

from compaction_integrity.compactors.base import Compactor
from compaction_integrity.compactors.llm_summarize import LLMSummarizeCompactor
from compaction_integrity.compactors.llmlingua import build_llmlingua2_from_config
from compaction_integrity.compactors.recent_n import RecentNTurnsCompactor
from compaction_integrity.dataset.ds_system_prompts import get_dataset_system_prompt
from compaction_integrity.dataset.eval_loader import EvalDatasetLoader, EvalDatasetRow
from compaction_integrity.prompts import (
    build_retention_judge_prompts as _build_retention_judge_prompts,
    get_sssc_evaluation_tool_message,
)
from compaction_integrity.runtime.base import ModelResponse, ModelRuntime
from compaction_integrity.runtime.env import apply_runtime_environment
from compaction_integrity.runtime.openai_runtime import OpenAIRuntime
from compaction_integrity.runtime.vllm_runtime import VLLMRuntime
from compaction_integrity.runtime.vllm_serve_runtime import VLLMServeRuntime
from compaction_integrity.scripts.eval_run_layout import (
    build_run_id,
    build_run_spec,
    normalize_compactors_container,
    normalize_probe_container,
    resolve_evaluator_cfg,
    runs_dir as _runs_dir,
    shared_cache_dir as _shared_cache_dir,
    to_container,
    write_run_metadata,
)
from compaction_integrity.sssc import SSSCS, sssc_to_prompt, probe_to_user_prompt


Message = dict[str, str]

# ---------------------------------------------------------------------------
# Result schemas
# ---------------------------------------------------------------------------

# Phase-1 (compactor-independent) shared cache.
# Keyed by (dataset, source_row_index, sssc_id, sssc_attrs, probe).
_FULL_PROBE_COLUMNS = [
    "dataset",
    "dataset_path",
    "source_row_index",
    "sssc_id",
    "sssc_type",
    "sssc_message",
    "sssc_probe",
    "sssc_correct",
    "sssc_incorrect",
    "sssc_attrs",
    "probe",
    "swap_seed",
    "compliant_letter",
    "full_with_sssc_messages",
    "full_with_sssc_probe_prompt",
    "full_with_sssc_output",
    "full_with_sssc_compliant",
    "full_without_sssc_probe_prompt",
    "full_without_sssc_output",
    "full_without_sssc_compliant",
]

# Phase-2 (per-compactor) wide file. Includes phase-1 columns plus case-3 columns.
_EVAL_WIDE_COLUMNS = _FULL_PROBE_COLUMNS + [
    "compactor",
    "evaluator",
    "compacted_context",
    "compaction_status",
    "compaction_error",
    "compacted_probe_prompt",
    "compacted_output",
    "compacted_compliant",
    "retention",
    "compacted_post_sssc_user_turn",
    "compacted_post_sssc_probe_prompt",
    "compacted_post_sssc_output",
    "compacted_post_sssc_compliant",
]


# ---------------------------------------------------------------------------
# Helpers
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


def _json_key(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True)


def _persist_dataframe(
    rows: list[dict[str, Any]],
    columns: list[str],
    save_path: str | Path | None,
) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=columns)
    if save_path is None:
        return df
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_pickle(save_path)
    return df


# OpenAI per-model enqueued-prompt-token cap budget for a single batch file. We
# aim below the documented 40M cap (e.g. gpt-5.4-mini) to leave headroom for
# concurrent batches in the same org.
_OPENAI_BATCH_TOKEN_BUDGET = 30_000_000


def _dataset_context_tokens(dataset_name: str) -> int | None:
    """Parse the trailing `Nk` / `NNNk` context-length hint from a dataset name
    (e.g. 'hermes_cat_100k' -> 100_000). Returns None if no hint is present."""
    match = re.search(r"(\d+)k(?:[_-]|$)", dataset_name.lower())
    if match is None:
        return None
    return int(match.group(1)) * 1000


def _openai_batch_size_for_dataset(
    dataset_name: str,
    max_output_tokens: int,
    num_pending: int,
    token_budget: int = _OPENAI_BATCH_TOKEN_BUDGET,
) -> int:
    """Cap batch size so one batch file's input + reserved output tokens stay
    under `token_budget`. Falls back to `num_pending` if the dataset name has no
    context-length hint."""
    context_tokens = _dataset_context_tokens(dataset_name)
    if context_tokens is None:
        return num_pending
    per_request = max(1, context_tokens + max(0, max_output_tokens))
    cap = max(1, token_budget // per_request)
    return max(1, min(num_pending, cap))


def _batched_indices(total_size: int, batch_size: int) -> list[tuple[int, int]]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    return [
        (start, min(start + batch_size, total_size))
        for start in range(0, total_size, batch_size)
    ]


# ---------------------------------------------------------------------------
# Compactor helpers (inlined from run_compaction.py)
# ---------------------------------------------------------------------------

def _build_compactor(compactor_name: str, compactor_cfg: dict[str, Any] | None) -> Compactor:
    cfg = compactor_cfg or {}
    if compactor_name == "llmlingua2" or cfg.get("model") == "llmlingua2":
        return build_llmlingua2_from_config(cfg)
    if cfg.get("model") == "recent-n":
        match = re.match(r"^recent_(\d+)$", compactor_name)
        if match is None:
            raise ValueError(
                f"recent-n compactor must be named 'recent_<N>', got {compactor_name!r}"
            )
        return RecentNTurnsCompactor(recent_n_turns=int(match.group(1)))

    runtime_kwargs = {**cfg["kwargs"], "use_tqdm": False}
    return LLMSummarizeCompactor(
        model=cfg["model"],
        provider=cfg["provider"],
        prompt_template=cfg["prompt"],
        runtime_kwargs=runtime_kwargs,
    )


def _compact_batch_with_retry(
    compactor: Compactor,
    batch_messages: list[list[Message]],
    batch_start: int,
    max_attempts: int,
    retry_sleep_seconds: float,
) -> list[Any]:
    if max_attempts <= 0:
        raise ValueError("max_attempts must be positive.")
    last_exc: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            return compactor.compact_batch(batch_messages)
        except Exception as exc:
            last_exc = exc
            print(
                f"Compaction batch failed: start={batch_start} size={len(batch_messages)} "
                f"attempt={attempt}/{max_attempts} error={exc}"
            )
            if attempt == max_attempts:
                break
            if retry_sleep_seconds > 0:
                time.sleep(retry_sleep_seconds)
    raise RuntimeError(
        f"Compaction batch failed after {max_attempts} attempts: "
        f"start={batch_start} size={len(batch_messages)}"
    ) from last_exc


def _compact_single_with_retry(
    compactor: Compactor,
    messages: list[Message],
    item_index: int,
    max_attempts: int,
    retry_sleep_seconds: float,
) -> Any:
    if max_attempts <= 0:
        raise ValueError("max_attempts must be positive.")
    last_exc: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            return compactor.compact(messages)
        except Exception as exc:
            last_exc = exc
            print(
                f"Compaction entry failed: index={item_index} "
                f"attempt={attempt}/{max_attempts} error={exc}"
            )
            if attempt == max_attempts:
                break
            if retry_sleep_seconds > 0:
                time.sleep(retry_sleep_seconds)
    raise RuntimeError(
        f"Compaction entry failed after {max_attempts} attempts: index={item_index}"
    ) from last_exc


# ---------------------------------------------------------------------------
# Probing runtime helpers (inlined from run_probing.py)
# ---------------------------------------------------------------------------

def _retrieve_sys_prompt(dataset_name: str, model_name: str) -> str:
    if dataset_name.startswith("wildchat"):
        resolved_model_name = model_name.removeprefix("vllm/")
        if resolved_model_name == "openai/gpt-oss-120b":
            tokenizer = AutoTokenizer.from_pretrained(resolved_model_name)
            rendered_prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": ""}],
                tokenize=False,
                add_generation_prompt=False,
            )
            system_start = "<|start|>system<|message|>"
            start_index = rendered_prompt.index(system_start) + len(system_start)
            end_index = rendered_prompt.index("<|end|>", start_index)
            return rendered_prompt[start_index:end_index]
        raise ValueError(f"Unsupported model_name={model_name} for default system prompt.")
    if dataset_name.startswith("hermes"):
        return str(get_dataset_system_prompt("hermes"))
    if dataset_name.startswith("openresearcher"):
        return str(get_dataset_system_prompt("openresearcher"))
    raise ValueError(f"Unsupported dataset_name={dataset_name} for default system prompt.")


def _build_probe_runtime(
    if_model: str,
    provider: str,
    probe_kwargs: dict[str, Any],
) -> ModelRuntime:
    if provider == "openai":
        return OpenAIRuntime(config={"model": if_model, **probe_kwargs})
    if provider == "vllm":
        return VLLMRuntime(config={"model": if_model, **probe_kwargs, "use_tqdm": False})
    if provider == "vllm_serve":
        return VLLMServeRuntime(config={"model": if_model, **probe_kwargs})
    raise ValueError(f"Unsupported provider={provider} for inference.")


def _format_for_log(response: ModelResponse) -> str:
    """Return analysis + final text concatenated, for logging."""
    parts: list[str] = []
    thinking = response.raw.get("thinking")
    if thinking:
        parts.append(f"[analysis]\n{thinking}")
    if response.text:
        parts.append(response.text)
    return "\n\n".join(parts).strip()


def parse_output(text: str, role: str | None = None) -> dict[str, str]:
    result: dict[str, str] = {}
    if not text:
        return result
    parts = re.split(r"\[(\w+)\]\n", text.strip())
    if parts[0].strip():
        result["text"] = parts[0].strip()
    for label, content in zip(parts[1::2], parts[2::2]):
        result[label] = content.strip()
    if role is not None:
        return {role: result.get(role, "")}
    return result


def _render_messages_for_judge(messages: list[Message]) -> str:
    return "\n\n".join(f"{m['role']}:\n{m['content']}" for m in messages)


def _parse_retention_output(output_text: str) -> bool:
    normalized = output_text.strip().upper()
    if normalized.startswith("YES"):
        return True
    if normalized.startswith("NO"):
        return False
    raise ValueError(f"Retention judge must return YES or NO, got: {output_text!r}")


# ---------------------------------------------------------------------------
# SSSC injection + probe message
# ---------------------------------------------------------------------------

def _inject_sssc(
    messages: list[Message],
    sssc_text: str,
    sssc_attrs: dict[str, Any],
    inject_rng: random.Random,
) -> list[Message]:
    """Return a new messages list with the SSSC text prepended to one or more
    user turns. Wraps the SSSC via sssc_to_prompt(...) before injection.

    repeat == 1 -> single injection at the configured `position`.
    repeat > 1  -> N random user turns picked via inject_rng (position ignored).
    """
    explicitness = bool(sssc_attrs["explicitness"])
    hard = bool(sssc_attrs["hard"])
    repeat = int(sssc_attrs["repeat"])
    position = str(sssc_attrs["position"])

    wrapped = sssc_to_prompt(sssc_text, explicitness, hard)

    user_idxs = [i for i, m in enumerate(messages) if m["role"] == "user"]
    if not user_idxs:
        raise ValueError("Cannot inject SSSC: row has no user turns.")

    if repeat <= 0:
        raise ValueError(f"repeat must be >= 1, got {repeat}")

    if repeat == 1:
        if position == "top":
            targets = [user_idxs[0]]
        elif position == "middle":
            targets = [user_idxs[len(user_idxs) // 2]]
        elif position == "bottom":
            targets = [user_idxs[-1]]
        else:
            raise ValueError(f"Unsupported position={position!r}")
    else:
        k = min(repeat, len(user_idxs))
        targets = sorted(inject_rng.sample(user_idxs, k=k))

    out = [dict(m) for m in messages]
    for i in targets:
        out[i] = {**out[i], "content": f"{wrapped}\n{out[i]['content']}"}
    return out


def _grade_letter(text: str, compliant_letter: str) -> bool | None:
    cleaned = text.strip().upper()
    if cleaned not in ("A", "B"):
        return None
    return cleaned == compliant_letter


# ---------------------------------------------------------------------------
# Row keys
# ---------------------------------------------------------------------------

def _full_probe_row_key(row: dict[str, Any]) -> str:
    return "|".join(
        [
            str(row["dataset"]),
            str(row["source_row_index"]),
            str(row["sssc_id"]),
            _json_key(row["sssc_attrs"]),
            _json_key(row["probe"]),
        ]
    )


def _wide_row_key(row: dict[str, Any]) -> str:
    return "|".join(
        [
            str(row["dataset"]),
            str(row["source_row_index"]),
            str(row["sssc_id"]),
            _json_key(row["sssc_attrs"]),
            _json_key(row["probe"]),
            _json_key(row["compactor"]),
            _json_key(row["evaluator"]),
        ]
    )


# ---------------------------------------------------------------------------
# Phase 1: compactor-independent full-context probes (cases 1 & 2)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _PairKey:
    source_row_index: int
    sssc_id: int


def _compute_pair_swap_seeds(
    rows: list[EvalDatasetRow],
    ssscs: list[dict[str, Any]],
    global_seed: int,
) -> dict[_PairKey, int]:
    """Pre-compute a swap seed per (row, sssc) pair, deterministic given
    global_seed and pair iteration order. Used identically for cases 1, 2, 3
    so the A/B order matches across cases."""
    rng = random.Random(global_seed)
    seeds: dict[_PairKey, int] = {}
    for row in rows:
        for sssc in ssscs:
            key = _PairKey(row.source_row_index, int(sssc["id"]))
            seeds[key] = rng.randint(0, 2**31 - 1)
    return seeds


def _build_full_probe_metadata(
    *,
    dataset_name: str,
    dataset_path: str,
    row: EvalDatasetRow,
    sssc: dict[str, Any],
    sssc_attrs: dict[str, Any],
    flattened_probe_cfg: dict[str, Any],
    swap_seed: int,
    compliant_letter: str,
) -> dict[str, Any]:
    return {
        "dataset": dataset_name,
        "dataset_path": dataset_path,
        "source_row_index": row.source_row_index,
        "sssc_id": int(sssc["id"]),
        "sssc_type": str(sssc["type"]),
        "sssc_message": str(sssc["sssc"]),
        "sssc_probe": str(sssc["probe"]),
        "sssc_correct": str(sssc["correct_answer"]),
        "sssc_incorrect": str(sssc["incorrect_answer"]),
        "sssc_attrs": dict(sssc_attrs),
        "probe": flattened_probe_cfg,
        "swap_seed": int(swap_seed),
        "compliant_letter": str(compliant_letter),
    }


def _build_retention_only_full_probe_df(
    *,
    dataset_name: str,
    dataset_path: str,
    rows: list[EvalDatasetRow],
    ssscs: list[dict[str, Any]],
    sssc_attrs: dict[str, Any],
    swap_seeds: dict[_PairKey, int],
    flattened_probe_cfg: dict[str, Any],
) -> pd.DataFrame:
    """Phase-1 substitute for retention-only mode: emits the identity columns
    needed by phase 2, with all probe-output fields left as None."""
    out_rows: list[dict[str, Any]] = []
    for row in rows:
        for sssc in ssscs:
            pair_key = _PairKey(row.source_row_index, int(sssc["id"]))
            swap_seed = swap_seeds[pair_key]
            _, compliant_letter = probe_to_user_prompt(
                probe=str(sssc["probe"]),
                correct_answer=str(sssc["correct_answer"]),
                incorrect_answer=str(sssc["incorrect_answer"]),
                seed=swap_seed,
            )
            meta = _build_full_probe_metadata(
                dataset_name=dataset_name,
                dataset_path=dataset_path,
                row=row,
                sssc=sssc,
                sssc_attrs=sssc_attrs,
                flattened_probe_cfg=flattened_probe_cfg,
                swap_seed=swap_seed,
                compliant_letter=compliant_letter,
            )
            full = {col: None for col in _FULL_PROBE_COLUMNS}
            full.update(meta)
            out_rows.append(full)
    return pd.DataFrame(out_rows, columns=_FULL_PROBE_COLUMNS)


def _ensure_full_probes(
    *,
    dataset_name: str,
    dataset_path: str,
    rows: list[EvalDatasetRow],
    ssscs: list[dict[str, Any]],
    sssc_attrs: dict[str, Any],
    swap_seeds: dict[_PairKey, int],
    flattened_probe_cfg: dict[str, Any],
    probe_kwargs: dict[str, Any],
    if_model: str,
    provider: str,
    system_prompt: str,
    global_seed: int,
    save_path: Path,
    overwrite: bool,
) -> pd.DataFrame:
    """Compute (or resume) cases 1 + 2 for every (row, sssc) pair and persist
    incrementally to `save_path`. Returns the full dataframe at the end.

    Resume semantics: each case resumes independently. A (row, sssc) pair
    skips `full_with_sssc` or `full_without_sssc` when that case's compliance
    column has already been written."""
    if overwrite and save_path.exists():
        save_path.unlink()

    existing_by_key: dict[str, dict[str, Any]] = {}
    if save_path.exists():
        existing_df = pd.read_pickle(save_path)
        for row in existing_df.to_dict(orient="records"):
            existing_by_key[_full_probe_row_key(row)] = row
        print(
            f"Found existing full-probe cache at {save_path}, "
            f"resuming with {len(existing_by_key)} existing row(s)."
        )

    # Build the full work list and inject SSSCs (deterministic across phases
    # because inject_rng is seeded on global_seed and advances in fixed order).
    inject_rng = random.Random(global_seed)
    pending_with_metas: list[dict[str, Any]] = []
    pending_with_contexts: list[list[Message]] = []
    pending_with_prompts: list[str] = []
    pending_without_metas: list[dict[str, Any]] = []
    pending_without_contexts: list[list[Message]] = []
    pending_without_prompts: list[str] = []

    for row in rows:
        for sssc in ssscs:
            pair_key = _PairKey(row.source_row_index, int(sssc["id"]))
            swap_seed = swap_seeds[pair_key]
            probe_prompt, compliant_letter = probe_to_user_prompt(
                probe=str(sssc["probe"]),
                correct_answer=str(sssc["correct_answer"]),
                incorrect_answer=str(sssc["incorrect_answer"]),
                seed=swap_seed,
            )
            metadata = _build_full_probe_metadata(
                dataset_name=dataset_name,
                dataset_path=dataset_path,
                row=row,
                sssc=sssc,
                sssc_attrs=sssc_attrs,
                flattened_probe_cfg=flattened_probe_cfg,
                swap_seed=swap_seed,
                compliant_letter=compliant_letter,
            )
            injected_messages = _inject_sssc(
                list(row.messages),
                sssc_text=str(sssc["sssc"]),
                sssc_attrs=sssc_attrs,
                inject_rng=inject_rng,
            )
            existing = existing_by_key.get(_full_probe_row_key(metadata))
            if existing is None or existing.get("full_with_sssc_compliant") is None:
                pending_with_metas.append(metadata)
                pending_with_contexts.append(injected_messages)
                pending_with_prompts.append(probe_prompt)
            if existing is None or existing.get("full_without_sssc_compliant") is None:
                pending_without_metas.append(metadata)
                pending_without_contexts.append(list(row.messages))
                pending_without_prompts.append(probe_prompt)

    if not pending_with_metas and not pending_without_metas:
        print(f"Full-probe cache already complete: {save_path}")
        return _persist_dataframe(
            list(existing_by_key.values()), _FULL_PROBE_COLUMNS, save_path
        )

    probe_runtime = _build_probe_runtime(if_model, provider, probe_kwargs)
    batch_size = int(probe_kwargs.get("batch_size", 32))
    use_batch = bool(probe_kwargs.get("batch", True))
    if not use_batch:
        batch_size = 1

    # Existing rows are copied in so case-specific resume does not drop the
    # other half of a partially completed pair.
    working: dict[str, dict[str, Any]] = {
        k: dict(v) for k, v in existing_by_key.items()
    }

    try:
        for (
            case_name,
            bar_label,
            pending_metas,
            contexts,
            prompts,
            prompt_col,
            text_col,
            compliant_col,
        ) in (
            (
                "with_sssc",
                "full_with_sssc",
                pending_with_metas,
                pending_with_contexts,
                pending_with_prompts,
                "full_with_sssc_probe_prompt",
                "full_with_sssc_output",
                "full_with_sssc_compliant",
            ),
            (
                "without_sssc",
                "full_without_sssc",
                pending_without_metas,
                pending_without_contexts,
                pending_without_prompts,
                "full_without_sssc_probe_prompt",
                "full_without_sssc_output",
                "full_without_sssc_compliant",
            ),
        ):
            if not pending_metas:
                print(f"{bar_label} already complete for dataset={dataset_name}.")
                continue
            progress_bar = tqdm(
                total=len(pending_metas),
                desc=f"{bar_label} ({dataset_name}, batch_size={batch_size})",
                unit="probe",
                dynamic_ncols=True,
            )
            try:
                for start, end in _batched_indices(len(pending_metas), batch_size):
                    batch_metas = pending_metas[start:end]
                    batch_prompts = prompts[start:end]
                    batch_contexts = contexts[start:end]
                    batch_convos = [
                        [
                            {"role": "system", "content": system_prompt},
                            get_sssc_evaluation_tool_message(),
                            *ctx,
                            {"role": "user", "content": probe_msg},
                        ]
                        for ctx, probe_msg in zip(batch_contexts, batch_prompts)
                    ]
                    outputs = probe_runtime.batch_generate(batch_convos)
                    for meta, probe_msg, ctx_msgs, response in zip(
                        batch_metas, batch_prompts, batch_contexts, outputs
                    ):
                        key = _full_probe_row_key(meta)
                        row_dict = working.get(key)
                        if row_dict is None:
                            row_dict = {col: None for col in _FULL_PROBE_COLUMNS}
                            row_dict.update(meta)
                            working[key] = row_dict
                        if case_name == "with_sssc":
                            row_dict["full_with_sssc_messages"] = ctx_msgs
                        row_dict[prompt_col] = probe_msg
                        formatted = _format_for_log(response)
                        row_dict[text_col] = formatted
                        parsed = parse_output(formatted)
                        grade_text = parsed.get("final") or parsed.get("text") or ""
                        row_dict[compliant_col] = _grade_letter(
                            grade_text, str(meta["compliant_letter"])
                        )
                        progress_bar.update(1)

                    _persist_dataframe(
                        list(working.values()), _FULL_PROBE_COLUMNS, save_path
                    )
            finally:
                progress_bar.close()
    finally:
        close = getattr(probe_runtime, "close", None)
        if close is not None:
            close()
        gc.collect()

    final_df = _persist_dataframe(
        list(working.values()), _FULL_PROBE_COLUMNS, save_path
    )
    print(f"Full-probe cache saved to {save_path} ({len(final_df)} row(s)).")
    return final_df


# ---------------------------------------------------------------------------
# Phase 2: per-compactor compacted-context probe (case 3) + retention
# ---------------------------------------------------------------------------

def _compact_one_with_recovery(
    *,
    compactor: Compactor,
    messages: list[Message],
    item_index: int,
    max_attempts: int,
    retry_sleep_seconds: float,
) -> tuple[list[Message] | None, str, str | None]:
    """Returns (compacted_messages, status, error). On failure after retries,
    records error rather than re-raising."""
    try:
        result = _compact_single_with_retry(
            compactor=compactor,
            messages=messages,
            item_index=item_index,
            max_attempts=max_attempts,
            retry_sleep_seconds=retry_sleep_seconds,
        )
        return result.messages, "success", None
    except Exception as exc:
        return None, "error", str(exc)


def _compact_batch_with_recovery(
    *,
    compactor: Compactor,
    batch_messages: list[list[Message]],
    batch_start: int,
    max_attempts: int,
    retry_sleep_seconds: float,
) -> list[tuple[list[Message] | None, str, str | None]]:
    """Batch wrapper that, if the whole batch fails after retries, retries each
    item individually so partial successes still get recorded."""
    try:
        results = _compact_batch_with_retry(
            compactor=compactor,
            batch_messages=batch_messages,
            batch_start=batch_start,
            max_attempts=max_attempts,
            retry_sleep_seconds=retry_sleep_seconds,
        )
        return [(r.messages, "success", None) for r in results]
    except Exception as batch_exc:
        print(
            f"Batch compaction failed after {max_attempts} attempts; falling back "
            f"to per-item retry. start={batch_start} error={batch_exc}"
        )
        out: list[tuple[list[Message] | None, str, str | None]] = []
        for offset, msgs in enumerate(batch_messages):
            out.append(
                _compact_one_with_recovery(
                    compactor=compactor,
                    messages=msgs,
                    item_index=batch_start + offset,
                    max_attempts=max_attempts,
                    retry_sleep_seconds=retry_sleep_seconds,
                )
            )
        return out


def _judge_retention_batch(
    runtime: OpenAIRuntime,
    evaluator_cfg: dict[str, Any],
    sssc_messages: list[str],
    compacted_contexts: list[list[Message] | None],
) -> list[bool | None]:
    judgments: list[bool | None] = []
    flex = bool(evaluator_cfg.get("kwargs", {}).get("flex", False))
    for sssc_text, compacted in tqdm(
        zip(sssc_messages, compacted_contexts),
        total=len(sssc_messages),
        desc="Judging retention",
        unit="sample",
        dynamic_ncols=True,
        leave=False,
    ):
        if compacted is None:
            judgments.append(None)
            continue
        system_prompt, user_prompt = _build_retention_judge_prompts(
            injected_sssc=sssc_text,
            compacted_context=_render_messages_for_judge(compacted),
        )
        max_attempts = 3
        last_exc: Exception | None = None
        parsed: bool | None = None
        for attempt in range(1, max_attempts + 1):
            response = runtime.generate(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                model=str(evaluator_cfg["model"]),
                flex=flex,
            )
            try:
                parsed = _parse_retention_output(response.text)
                break
            except ValueError as exc:
                last_exc = exc
                print(
                    f"Retention judge parse failed (attempt {attempt}/{max_attempts}): {exc}"
                )
        if parsed is None:
            raise ValueError(
                f"Retention judge failed after {max_attempts} attempts: {last_exc}"
            )
        judgments.append(parsed)
    return judgments


def _run_compacted_phase(
    *,
    dataset_name: str,
    rows: list[EvalDatasetRow],
    ssscs: list[dict[str, Any]],
    sssc_attrs: dict[str, Any],
    swap_seeds: dict[_PairKey, int],
    full_probe_df: pd.DataFrame,
    compactor_name: str,
    compactor_cfg: dict[str, Any] | None,
    flattened_compactor_cfg: dict[str, Any],
    flattened_probe_cfg: dict[str, Any],
    flattened_evaluator_cfg: dict[str, Any],
    evaluator_cfg: dict[str, Any],
    probe_kwargs: dict[str, Any],
    if_model: str,
    provider: str,
    system_prompt: str,
    global_seed: int,
    save_path: Path,
    overwrite: bool,
) -> None:
    if overwrite and save_path.exists():
        save_path.unlink()

    working: dict[str, dict[str, Any]] = {}
    if save_path.exists():
        existing_df = pd.read_pickle(save_path)
        for r in existing_df.to_dict(orient="records"):
            working[_wide_row_key(r)] = r
        print(
            f"Found existing wide results at {save_path}, "
            f"resuming with {len(working)} existing row(s)."
        )

    full_by_pair: dict[_PairKey, dict[str, Any]] = {
        _PairKey(int(r["source_row_index"]), int(r["sssc_id"])): r
        for r in full_probe_df.to_dict(orient="records")
    }

    # Categorize work:
    #   pending_full: row missing entirely → run compaction + phase-3 + phase-4.
    #   pending_post: row exists, phase-3 done, phase-4 missing, compaction
    #                 succeeded (compacted_context not None) → phase-4 backfill.
    pending_full_metas: list[dict[str, Any]] = []
    pending_full_injected: list[list[Message]] = []
    pending_post_keys: list[str] = []

    # Same global_seed → same injection points as phase 1.
    inject_rng = random.Random(global_seed)
    for row in rows:
        for sssc in ssscs:
            pair_key = _PairKey(row.source_row_index, int(sssc["id"]))
            full_row = full_by_pair[pair_key]
            wide_meta: dict[str, Any] = {
                **full_row,
                "compactor": flattened_compactor_cfg,
                "evaluator": flattened_evaluator_cfg,
            }
            injected = _inject_sssc(
                list(row.messages),
                sssc_text=str(sssc["sssc"]),
                sssc_attrs=sssc_attrs,
                inject_rng=inject_rng,
            )
            key = _wide_row_key(wide_meta)
            existing = working.get(key)
            if existing is None:
                pending_full_metas.append(wide_meta)
                pending_full_injected.append(injected)
            elif (
                existing.get("compacted_post_sssc_compliant") is None
                and existing.get("compacted_context") is not None
            ):
                pending_post_keys.append(key)

    if not pending_full_metas and not pending_post_keys:
        print(
            f"Skipping completed compacted-phase workload for "
            f"compactor={compactor_name} dataset={dataset_name}."
        )
        return

    # ------------------------------------------------------------------
    # Phase A: compaction (only for pending_full)
    # ------------------------------------------------------------------
    compacted_contexts: list[list[Message] | None] = []
    compaction_statuses: list[str] = []
    compaction_errors: list[str | None] = []

    if pending_full_metas:
        cfg = compactor_cfg or {}
        compactor_kwargs = dict(cfg.get("kwargs", {}))
        use_batch = bool(compactor_kwargs.get("batch", False))
        batch_size = int(compactor_kwargs.get("batch_size", 32))
        if use_batch and str(cfg.get("provider", "")).strip().lower() == "openai":
            batch_size = _openai_batch_size_for_dataset(
                dataset_name=dataset_name,
                max_output_tokens=int(compactor_kwargs.get("max_tokens", 0)),
                num_pending=len(pending_full_metas),
            )
        retry_attempts = int(compactor_kwargs.get("retry_attempts", 5))
        retry_sleep_seconds = float(compactor_kwargs.get("retry_sleep_seconds", 0.0))

        compactor = _build_compactor(compactor_name, cfg)
        compacted_contexts = [None] * len(pending_full_metas)
        compaction_statuses = [""] * len(pending_full_metas)
        compaction_errors = [None] * len(pending_full_metas)

        progress_bar = tqdm(
            total=len(pending_full_metas),
            desc=f"Compaction {compactor_name}/{dataset_name} (batch_size={batch_size if use_batch else 1})",
            unit="sample",
            dynamic_ncols=True,
        )

        if use_batch and str(cfg.get("provider", "")).strip().lower() == "openai":
            def _on_batch_poll(batch_obj: Any) -> None:
                counts = getattr(batch_obj, "request_counts", None)
                postfix: dict[str, Any] = {"status": getattr(batch_obj, "status", "?")}
                if counts is not None:
                    postfix["done"] = getattr(counts, "completed", None)
                    postfix["failed"] = getattr(counts, "failed", None)
                    postfix["of"] = getattr(counts, "total", None)
                progress_bar.set_postfix(postfix, refresh=True)

            try:
                compactor.runtime.batch_progress_callback = _on_batch_poll
            except AttributeError:
                pass

        try:
            if use_batch:
                for start, end in _batched_indices(len(pending_full_metas), batch_size):
                    results = _compact_batch_with_recovery(
                        compactor=compactor,
                        batch_messages=pending_full_injected[start:end],
                        batch_start=start,
                        max_attempts=retry_attempts,
                        retry_sleep_seconds=retry_sleep_seconds,
                    )
                    for offset, (compacted, status, err) in enumerate(results):
                        compacted_contexts[start + offset] = compacted
                        compaction_statuses[start + offset] = status
                        compaction_errors[start + offset] = err
                        progress_bar.update(1)
            else:
                for idx, msgs in enumerate(pending_full_injected):
                    compacted, status, err = _compact_one_with_recovery(
                        compactor=compactor,
                        messages=msgs,
                        item_index=idx,
                        max_attempts=retry_attempts,
                        retry_sleep_seconds=retry_sleep_seconds,
                    )
                    compacted_contexts[idx] = compacted
                    compaction_statuses[idx] = status
                    compaction_errors[idx] = err
                    progress_bar.update(1)
        finally:
            progress_bar.close()
            close = getattr(compactor, "close", None)
            if close is not None:
                close()
            gc.collect()

    # ------------------------------------------------------------------
    # Phase B: probes (3 + 4 for pending_full; 4 only for pending_post)
    # ------------------------------------------------------------------
    inference_runtime = _build_probe_runtime(if_model, provider, probe_kwargs)
    evaluator_runtime = OpenAIRuntime(config={}) if pending_full_metas else None

    probe_batch_size = int(probe_kwargs.get("batch_size", 32))
    use_probe_batch = bool(probe_kwargs.get("batch", True))
    if not use_probe_batch:
        probe_batch_size = 1

    def _build_probe_messages(metas: list[dict[str, Any]]) -> list[str]:
        return [
            probe_to_user_prompt(
                probe=str(m["sssc_probe"]),
                correct_answer=str(m["sssc_correct"]),
                incorrect_answer=str(m["sssc_incorrect"]),
                seed=int(m["swap_seed"]),
            )[0]
            for m in metas
        ]

    def _build_phase4_user_turns(metas: list[dict[str, Any]]) -> list[str]:
        return [
            sssc_to_prompt(
                str(m["sssc_message"]),
                bool(m["sssc_attrs"]["explicitness"]),
                bool(m["sssc_attrs"]["hard"]),
            )
            for m in metas
        ]

    def _run_probe_batch(
        metas: list[dict[str, Any]],
        compacted_batch: list[list[Message] | None],
        user_contents: list[str],
    ) -> tuple[list[str | None], list[bool | None]]:
        convos: list[list[Message]] = []
        convo_indices: list[int] = []
        for off, (compacted, content) in enumerate(zip(compacted_batch, user_contents)):
            if compacted is None:
                continue
            convos.append(
                [
                    {"role": "system", "content": system_prompt},
                    get_sssc_evaluation_tool_message(),
                    *compacted,
                    {"role": "user", "content": content},
                ]
            )
            convo_indices.append(off)
        outs_per_meta: list[str | None] = [None] * len(metas)
        grades: list[bool | None] = [None] * len(metas)
        if convos:
            outputs = inference_runtime.batch_generate(convos)
            for off, out in zip(convo_indices, outputs):
                formatted = _format_for_log(out)
                outs_per_meta[off] = formatted
                parsed = parse_output(formatted)
                grade_text = parsed.get("final") or parsed.get("text") or ""
                grades[off] = _grade_letter(
                    grade_text, str(metas[off]["compliant_letter"])
                )
        return outs_per_meta, grades

    try:
        if pending_full_metas:
            progress_bar = tqdm(
                total=len(pending_full_metas),
                desc=f"Compacted probes {compactor_name}/{dataset_name} (batch_size={probe_batch_size})",
                unit="sample",
                dynamic_ncols=True,
            )
            try:
                for start, end in _batched_indices(len(pending_full_metas), probe_batch_size):
                    batch_metas = pending_full_metas[start:end]
                    batch_compacted = compacted_contexts[start:end]
                    batch_statuses = compaction_statuses[start:end]
                    batch_errors = compaction_errors[start:end]

                    probe_messages = _build_probe_messages(batch_metas)

                    p3_outputs, p3_grades = _run_probe_batch(
                        batch_metas, batch_compacted, probe_messages
                    )

                    p4_user_turns = _build_phase4_user_turns(batch_metas)
                    p4_merged = [
                        f"{u}\n\n{p}" for u, p in zip(p4_user_turns, probe_messages)
                    ]
                    p4_outputs, p4_grades = _run_probe_batch(
                        batch_metas, batch_compacted, p4_merged
                    )

                    retentions = _judge_retention_batch(
                        runtime=evaluator_runtime,
                        evaluator_cfg=evaluator_cfg,
                        sssc_messages=[str(m["sssc_message"]) for m in batch_metas],
                        compacted_contexts=batch_compacted,
                    )

                    for (
                        meta,
                        compacted,
                        status,
                        err,
                        probe_msg,
                        p3_out,
                        p3_grade,
                        p4_user,
                        p4_merge,
                        p4_out,
                        p4_grade,
                        retention,
                    ) in zip(
                        batch_metas,
                        batch_compacted,
                        batch_statuses,
                        batch_errors,
                        probe_messages,
                        p3_outputs,
                        p3_grades,
                        p4_user_turns,
                        p4_merged,
                        p4_outputs,
                        p4_grades,
                        retentions,
                    ):
                        wide_row = {
                            **meta,
                            "compacted_context": compacted,
                            "compaction_status": status,
                            "compaction_error": err,
                            "compacted_probe_prompt": probe_msg if compacted is not None else None,
                            "compacted_output": p3_out,
                            "compacted_compliant": p3_grade,
                            "retention": retention,
                            "compacted_post_sssc_user_turn": p4_user if compacted is not None else None,
                            "compacted_post_sssc_probe_prompt": p4_merge if compacted is not None else None,
                            "compacted_post_sssc_output": p4_out,
                            "compacted_post_sssc_compliant": p4_grade,
                        }
                        working[_wide_row_key(meta)] = wide_row
                        progress_bar.update(1)

                    _persist_dataframe(
                        list(working.values()), _EVAL_WIDE_COLUMNS, save_path
                    )
            finally:
                progress_bar.close()

        if pending_post_keys:
            progress_bar = tqdm(
                total=len(pending_post_keys),
                desc=f"Phase-4 backfill {compactor_name}/{dataset_name} (batch_size={probe_batch_size})",
                unit="sample",
                dynamic_ncols=True,
            )
            try:
                for start, end in _batched_indices(len(pending_post_keys), probe_batch_size):
                    batch_keys = pending_post_keys[start:end]
                    batch_metas = [working[k] for k in batch_keys]
                    batch_compacted = [m["compacted_context"] for m in batch_metas]

                    probe_messages = _build_probe_messages(batch_metas)
                    p4_user_turns = _build_phase4_user_turns(batch_metas)
                    p4_merged = [
                        f"{u}\n\n{p}" for u, p in zip(p4_user_turns, probe_messages)
                    ]
                    p4_outputs, p4_grades = _run_probe_batch(
                        batch_metas, batch_compacted, p4_merged
                    )

                    for meta, compacted, p4_user, p4_merge, p4_out, p4_grade in zip(
                        batch_metas,
                        batch_compacted,
                        p4_user_turns,
                        p4_merged,
                        p4_outputs,
                        p4_grades,
                    ):
                        # Mutates the row stored in `working` in place.
                        meta["compacted_post_sssc_user_turn"] = p4_user if compacted is not None else None
                        meta["compacted_post_sssc_probe_prompt"] = p4_merge if compacted is not None else None
                        meta["compacted_post_sssc_output"] = p4_out
                        meta["compacted_post_sssc_compliant"] = p4_grade
                        progress_bar.update(1)

                    _persist_dataframe(
                        list(working.values()), _EVAL_WIDE_COLUMNS, save_path
                    )
            finally:
                progress_bar.close()
    finally:
        close = getattr(inference_runtime, "close", None)
        if close is not None:
            close()
        gc.collect()

    _persist_dataframe(list(working.values()), _EVAL_WIDE_COLUMNS, save_path)
    print(f"Wide evaluation results saved to {save_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

@hydra.main(version_base=None, config_path="../../../config/tasks/eval", config_name=None)
def main(cfg: DictConfig) -> None:
    apply_runtime_environment()

    test = bool(cfg.test)
    overwrite = bool(cfg.overwrite)
    retention_rate_only = bool(cfg.get("retention_rate_only", False))
    global_seed = int(cfg.global_seed)
    results_root = Path(str(cfg.results_root))
    results_root.mkdir(parents=True, exist_ok=True)

    sssc_attrs = to_container(cfg.sssc_attrs)
    sssc_attrs["position"] = str(sssc_attrs["position"])
    sssc_attrs["repeat"] = int(sssc_attrs["repeat"])
    sssc_attrs["explicitness"] = bool(sssc_attrs["explicitness"])
    sssc_attrs["hard"] = bool(sssc_attrs["hard"])

    compactors_container = normalize_compactors_container(cfg.compactors)
    if "probe" in cfg:
        probe_container = normalize_probe_container(cfg.probe)
    else:
        raise KeyError("cfg.probe is required")
    evaluator_name, evaluator_cfg = resolve_evaluator_cfg(cfg.evaluators)
    flattened_evaluator_cfg = {"name": evaluator_name, **_flatten_dict(evaluator_cfg)}

    runs_dir = _runs_dir(results_root, test)
    shared_cache_dir = _shared_cache_dir(results_root, test)
    runs_dir.mkdir(parents=True, exist_ok=True)
    shared_cache_dir.mkdir(parents=True, exist_ok=True)

    for dataset_name in cfg.datasets:
        dataset_cfg = cfg.datasets[dataset_name]
        dataset_dir = Path(str(dataset_cfg.dir))
        num_rows = None if dataset_cfg.num_rows is None else int(dataset_cfg.num_rows)
        dataset_path = dataset_dir / "stitched_dataset"

        loader = EvalDatasetLoader.load(
            dataset_path=dataset_path,
            test_mode=test,
            num_rows=num_rows,
        )
        rows = loader.rows()
        print(f"[{dataset_name}] loaded {len(rows)} row(s) from {dataset_path}")

        swap_seeds = _compute_pair_swap_seeds(rows, SSSCS, global_seed)

        for probe_name, probe_cfg in probe_container.items():
            probe_kwargs = dict(probe_cfg.get("kwargs", {}))
            if_model = str(probe_cfg["model"])
            provider = str(probe_cfg["provider"])
            flattened_probe_cfg = {"name": probe_name, **_flatten_dict(probe_cfg)}

            system_prompt = _retrieve_sys_prompt(dataset_name, if_model)

            shared_spec = build_run_spec(
                test=test,
                dataset_name=dataset_name,
                dataset_dir=dataset_dir,
                num_rows=num_rows,
                sssc_attrs=sssc_attrs,
                global_seed=global_seed,
                probe_name=probe_name,
                probe_cfg=probe_cfg,
                compactor_name=None,
                compactor_cfg=None,
                evaluator_name=None,
                evaluator_cfg=None,
            )
            shared_id = build_run_id(shared_spec)
            shared_path = shared_cache_dir / f"{shared_id}.pkl"
            shared_metadata_path = shared_cache_dir / f"{shared_id}.metadata.json"
            shared_metadata_path.write_text(
                json.dumps({"shared_id": shared_id, "shared_spec": shared_spec}, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )

            if retention_rate_only:
                print(
                    f"[STEP 1] Skipped (retention_rate_only=True) "
                    f"dataset={dataset_name} probe={probe_name}"
                )
                full_probe_df = _build_retention_only_full_probe_df(
                    dataset_name=dataset_name,
                    dataset_path=str(dataset_path),
                    rows=rows,
                    ssscs=SSSCS,
                    sssc_attrs=sssc_attrs,
                    swap_seeds=swap_seeds,
                    flattened_probe_cfg=flattened_probe_cfg,
                )
            else:
                print(
                    f"[STEP 1] Full-context probes (cases 1+2) "
                    f"dataset={dataset_name} probe={probe_name} shared_id={shared_id}"
                )
                full_probe_df = _ensure_full_probes(
                    dataset_name=dataset_name,
                    dataset_path=str(dataset_path),
                    rows=rows,
                    ssscs=SSSCS,
                    sssc_attrs=sssc_attrs,
                    swap_seeds=swap_seeds,
                    flattened_probe_cfg=flattened_probe_cfg,
                    probe_kwargs=probe_kwargs,
                    if_model=if_model,
                    provider=provider,
                    system_prompt=system_prompt,
                    global_seed=global_seed,
                    save_path=shared_path,
                    overwrite=overwrite,
                )

            for compactor_name, compactor_cfg in compactors_container.items():
                flattened_compactor_cfg = {
                    "name": compactor_name,
                    **_flatten_dict(compactor_cfg or {}),
                }
                run_spec = build_run_spec(
                    test=test,
                    dataset_name=dataset_name,
                    dataset_dir=dataset_dir,
                    num_rows=num_rows,
                    sssc_attrs=sssc_attrs,
                    global_seed=global_seed,
                    probe_name=probe_name,
                    probe_cfg=probe_cfg,
                    compactor_name=compactor_name,
                    compactor_cfg=compactor_cfg,
                    evaluator_name=evaluator_name,
                    evaluator_cfg=evaluator_cfg,
                )
                run_id = build_run_id(run_spec)
                run_dir = runs_dir / run_id
                write_run_metadata(run_dir, run_id, run_spec)
                save_path = run_dir / "evaluation_results.pkl"

                print(
                    f"[STEP 2] Compacted-context probe + retention "
                    f"dataset={dataset_name} compactor={compactor_name} "
                    f"probe={probe_name} run_id={run_id}"
                )
                _run_compacted_phase(
                    dataset_name=dataset_name,
                    rows=rows,
                    ssscs=SSSCS,
                    sssc_attrs=sssc_attrs,
                    swap_seeds=swap_seeds,
                    full_probe_df=full_probe_df,
                    compactor_name=compactor_name,
                    compactor_cfg=compactor_cfg,
                    flattened_compactor_cfg=flattened_compactor_cfg,
                    flattened_probe_cfg=flattened_probe_cfg,
                    flattened_evaluator_cfg=flattened_evaluator_cfg,
                    evaluator_cfg=evaluator_cfg,
                    probe_kwargs=probe_kwargs,
                    if_model=if_model,
                    provider=provider,
                    system_prompt=system_prompt,
                    global_seed=global_seed,
                    save_path=save_path,
                    overwrite=overwrite,
                )


if __name__ == "__main__":
    main()
