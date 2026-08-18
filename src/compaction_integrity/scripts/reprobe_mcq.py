"""Re-probe completed MCQ evaluations with an alternate downstream prober.

Robustness/multi-prober validation for the compliance metric. The reviewer
concern this answers: compliance is measured with a single fixed probing model
(gpt-oss-120b), so a compliance score may reflect compactor-to-prober format
compatibility rather than semantic preservation. This entrypoint holds the
compactor output fixed -- it reuses the *identical* stored contexts and the
*identical* stored MCQ prompts from a completed run -- and swaps only the
downstream prober (e.g. Qwen3-30B, Gemma-4-E4B), recomputing compliance.

Because nothing but the probing model changes, the output is a drop-in
`evaluation_results.pkl` with the same schema as the source run (only the probe
config + the probe-output/compliance columns differ). It slots directly into a
manifest for `analyze/main_exp.py`, exactly like a native evaluation run.

What is reused vs recomputed
  reused (copied from source, per row):
    full_with_sssc_messages, full_without_sssc via dataset reload,
    compacted_context, *_probe_prompt (the exact A/B prompt text),
    swap_seed, compliant_letter, and `retention` (prober-independent).
  recomputed with the new prober (overwritten in place):
    <case>_output and <case>_compliant for each enabled case.

Cases mirror evaluation.py:
  full_with_sssc, full_without_sssc, compacted, compacted_post_sssc.

Config: config/tasks/reprobe_mcq/<name>.yaml  (one prober per file).
Run:
  python -m compaction_integrity.scripts.reprobe_mcq --config-name hermes100k_qwen30b
"""

import copy
import gc
import json
import re
from pathlib import Path
from typing import Any

import hydra
import pandas as pd
from omegaconf import DictConfig
from tqdm.auto import tqdm
from transformers import AutoTokenizer

from compaction_integrity.dataset.ds_system_prompts import get_dataset_system_prompt
from compaction_integrity.dataset.eval_loader import EvalDatasetLoader
from compaction_integrity.prompts import get_sssc_evaluation_tool_message
from compaction_integrity.runtime.base import ModelResponse, ModelRuntime
from compaction_integrity.runtime.env import apply_runtime_environment
from compaction_integrity.runtime.openai_runtime import OpenAIRuntime
from compaction_integrity.runtime.vllm_runtime import VLLMRuntime
from compaction_integrity.runtime.vllm_serve_runtime import VLLMServeRuntime
from compaction_integrity.scripts.eval_run_layout import (
    build_run_id,
    to_container,
    write_run_metadata,
)


Message = dict[str, str]

# (case_name, context_col, prompt_col, output_col, compliant_col). The context
# for full_without_sssc is not stored on the wide row (no SSSC injected), so it
# is reconstructed from the source dataset; every other case reuses a stored
# column verbatim.
_CASES: tuple[tuple[str, str | None, str, str, str], ...] = (
    (
        "full_with_sssc",
        "full_with_sssc_messages",
        "full_with_sssc_probe_prompt",
        "full_with_sssc_output",
        "full_with_sssc_compliant",
    ),
    (
        "full_without_sssc",
        None,  # reconstructed from the dataset (un-injected messages)
        "full_without_sssc_probe_prompt",
        "full_without_sssc_output",
        "full_without_sssc_compliant",
    ),
    (
        "compacted",
        "compacted_context",
        "compacted_probe_prompt",
        "compacted_output",
        "compacted_compliant",
    ),
    (
        "compacted_post_sssc",
        "compacted_context",
        "compacted_post_sssc_probe_prompt",
        "compacted_post_sssc_output",
        "compacted_post_sssc_compliant",
    ),
)


def _flatten_dict(data: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    flattened: dict[str, Any] = {}
    for key, value in data.items():
        flattened_key = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            flattened.update(_flatten_dict(value, prefix=flattened_key))
            continue
        flattened[flattened_key] = value
    return flattened


def _retrieve_sys_prompt(dataset_name: str, model_name: str) -> str:
    if dataset_name.startswith("wildchat"):
        resolved_model_name = model_name.removeprefix("vllm/")
        if resolved_model_name != "openai/gpt-oss-120b":
            raise ValueError(f"Unsupported model_name={model_name} for default system prompt.")
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
    if dataset_name.startswith("hermes"):
        return str(get_dataset_system_prompt("hermes"))
    if dataset_name.startswith("openresearcher"):
        return str(get_dataset_system_prompt("openresearcher"))
    raise ValueError(f"Unsupported dataset_name={dataset_name} for default system prompt.")


def _build_runtime(model: str, provider: str, kwargs: dict[str, Any]) -> ModelRuntime:
    if provider == "openai":
        return OpenAIRuntime(config={"model": model, **kwargs})
    if provider == "vllm":
        return VLLMRuntime(config={"model": model, **kwargs, "use_tqdm": False})
    if provider == "vllm_serve":
        return VLLMServeRuntime(config={"model": model, **kwargs})
    raise ValueError(f"Unsupported provider={provider} for inference.")


def _format_for_log(response: ModelResponse) -> str:
    parts: list[str] = []
    thinking = response.raw.get("thinking")
    if thinking:
        parts.append(f"[analysis]\n{thinking}")
    if response.text:
        parts.append(response.text)
    return "\n\n".join(parts).strip()


def _parse_output(text: str) -> dict[str, str]:
    result: dict[str, str] = {}
    if not text:
        return result
    parts = re.split(r"\[(\w+)\]\n", text.strip())
    if parts[0].strip():
        result["text"] = parts[0].strip()
    for label, content in zip(parts[1::2], parts[2::2]):
        result[label] = content.strip()
    return result


def _grade_letter(text: str, compliant_letter: str) -> bool | None:
    cleaned = text.strip().upper()
    if cleaned not in ("A", "B"):
        return None
    return cleaned == compliant_letter


def _case_context(
    row: dict[str, Any],
    context_col: str | None,
    base_messages_by_index: dict[int, list[Message]],
) -> list[Message] | None:
    if context_col is None:
        return base_messages_by_index[int(row["source_row_index"])]
    return row[context_col]


def _run_case(
    *,
    df: pd.DataFrame,
    case: str,
    context_col: str | None,
    prompt_col: str,
    output_col: str,
    compliant_col: str,
    runtime: ModelRuntime,
    system_prompt: str,
    base_messages_by_index: dict[int, list[Message]],
    batch_size: int,
    overwrite: bool,
) -> None:
    records = df.to_dict(orient="records")
    pending: list[int] = []
    for i, row in enumerate(records):
        context = _case_context(row, context_col, base_messages_by_index)
        prompt = row.get(prompt_col)
        # Skip rows the source run never probed (e.g. compaction failed -> the
        # context/prompt columns are None). Their compliance stays None.
        if context is None or prompt is None or (isinstance(prompt, float) and pd.isna(prompt)):
            continue
        if not overwrite and pd.notna(row.get(compliant_col)):
            continue
        pending.append(i)

    if not pending:
        print(f"  [{case}] nothing pending.")
        return

    progress_bar = tqdm(
        total=len(pending),
        desc=f"{case} (batch_size={batch_size})",
        unit="probe",
        dynamic_ncols=True,
    )
    try:
        for start in range(0, len(pending), batch_size):
            indices = pending[start:start + batch_size]
            conversations: list[list[Message]] = []
            for i in indices:
                row = records[i]
                context = _case_context(row, context_col, base_messages_by_index)
                conversations.append(
                    [
                        {"role": "system", "content": system_prompt},
                        get_sssc_evaluation_tool_message(mcq=True),
                        *context,
                        {"role": "user", "content": str(row[prompt_col])},
                    ]
                )
            outputs = runtime.batch_generate(conversations)
            for i, response in zip(indices, outputs):
                # Store the full log (analysis + answer) for transparency, but
                # grade from the answer channel only. gpt-oss puts "[final]\nA"
                # in response.text; Qwen/Gemma (thinking split into
                # raw["thinking"]) put a bare "A". Grading _format_for_log
                # instead would let the prepended "[analysis]" swallow the
                # answer and grade every thinking prober None.
                df.at[i, output_col] = _format_for_log(response)
                parsed = _parse_output(response.text or "")
                grade_text = parsed.get("final") or parsed.get("text") or ""
                df.at[i, compliant_col] = _grade_letter(
                    grade_text, str(records[i]["compliant_letter"])
                )
            progress_bar.update(len(indices))
    finally:
        progress_bar.close()


@hydra.main(version_base=None, config_path="../../../config/tasks/reprobe_mcq", config_name=None)
def main(cfg: DictConfig) -> None:
    apply_runtime_environment()

    results_root = Path(str(cfg.results_root))
    runs_dir = results_root / ("runs_test" if bool(cfg.test) else "runs")
    overwrite = bool(cfg.overwrite)

    probe_cfg = to_container(cfg.probe)
    probe_name = str(probe_cfg.pop("name"))
    model = str(probe_cfg["model"])
    provider = str(probe_cfg["provider"])
    probe_kwargs = dict(probe_cfg.get("kwargs", {}))
    batch_size = int(probe_kwargs.get("batch_size", 32))
    flattened_probe_cfg = {"name": probe_name, **_flatten_dict(probe_cfg)}

    enabled_cases = [c for c in _CASES if bool(cfg.cases.get(c[0], False))]
    if not enabled_cases:
        raise ValueError("No cases enabled in cfg.cases.")

    for source in to_container(cfg.source_runs):
        source_run_id = str(source["run_id"])
        source_dir = runs_dir / source_run_id
        source_df = pd.read_pickle(source_dir / "evaluation_results.pkl")
        source_spec = json.loads(
            (source_dir / "metadata.json").read_text(encoding="utf-8")
        )["run_spec"]

        # Faithful evaluation spec with only the prober swapped, so the new
        # run_id matches what a native evaluation.py run with this prober would
        # produce and drops straight into main_exp.py.
        new_spec = copy.deepcopy(source_spec)
        new_spec["probe"] = {"name": probe_name, "config": probe_cfg}
        new_run_id = build_run_id(new_spec)
        new_dir = runs_dir / new_run_id
        output_path = new_dir / "evaluation_results.pkl"
        new_dir.mkdir(parents=True, exist_ok=True)

        if output_path.exists() and not overwrite:
            # Genuine resume from a prior run of THIS prober: keep its verdicts.
            out_df = pd.read_pickle(output_path)
        else:
            # Fresh start: source_df carries the ORIGINAL prober's grades in the
            # output/compliance columns. Clear the prober-dependent columns for
            # the enabled cases so the swapped prober actually recomputes them --
            # otherwise the resume check (pd.notna(compliant_col)) would treat
            # the source's gpt-oss grades as done and skip every row.
            out_df = source_df.copy()
            for _case, _ctx, _prompt, out_col, comp_col in enabled_cases:
                out_df[out_col] = None
                out_df[comp_col] = None
        # Relabel the prober so main_exp attributes these rows to the new model.
        out_df["probe"] = [dict(flattened_probe_cfg) for _ in range(len(out_df))]

        dataset_cfg = source_spec["dataset"]
        dataset_name = str(dataset_cfg["name"])
        rows = EvalDatasetLoader.load(
            dataset_path=Path(dataset_cfg["dir"]) / "stitched_dataset",
            test_mode=bool(source_spec["test"]),
            num_rows=dataset_cfg["num_rows"],
        ).rows()
        base_messages_by_index = {r.source_row_index: r.messages for r in rows}
        system_prompt = _retrieve_sys_prompt(dataset_name, model)

        print(
            f"[reprobe-mcq] source={source_run_id}\n"
            f"             prober={probe_name} ({provider}:{model}) new_run={new_run_id}"
        )
        runtime = _build_runtime(model, provider, probe_kwargs)
        try:
            for case, ctx_col, prompt_col, out_col, comp_col in enabled_cases:
                _run_case(
                    df=out_df,
                    case=case,
                    context_col=ctx_col,
                    prompt_col=prompt_col,
                    output_col=out_col,
                    compliant_col=comp_col,
                    runtime=runtime,
                    system_prompt=system_prompt,
                    base_messages_by_index=base_messages_by_index,
                    batch_size=batch_size,
                    overwrite=overwrite,
                )
                out_df.to_pickle(output_path)
        finally:
            close = getattr(runtime, "close", None)
            if close is not None:
                close()
            gc.collect()

        write_run_metadata(new_dir, new_run_id, new_spec)
        # Provenance kept out of the hashed run_spec so the run_id stays faithful.
        (new_dir / "reprobe_source.json").write_text(
            json.dumps(
                {
                    "task": "mcq_reprobe",
                    "source_run_id": source_run_id,
                    "new_run_id": new_run_id,
                    "prober": flattened_probe_cfg,
                    "cases": [c[0] for c in enabled_cases],
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"[reprobe-mcq] saved -> {output_path}")


if __name__ == "__main__":
    main()
