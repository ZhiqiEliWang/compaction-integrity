"""Reprobe completed MCQ evaluations with free GPT-OSS tool generations.

This entrypoint reuses the full and compacted contexts from a completed
evaluation run. It reruns the probing model and the instruction-following judge
for free generations, writing those results beside the original MCQ columns in
a new run directory.
"""

import copy
import gc
import json
from pathlib import Path
from typing import Any

import hydra
import pandas as pd
from omegaconf import DictConfig
from transformers import AutoTokenizer

from compaction_integrity.dataset.ds_system_prompts import get_dataset_system_prompt
from compaction_integrity.dataset.eval_loader import EvalDatasetLoader
from compaction_integrity.prompts import (
    build_instruction_following_judge_prompts,
    get_sssc_evaluation_tool_message,
    get_sssc_evaluation_tools,
)
from compaction_integrity.runtime.base import ModelResponse, ModelRuntime
from compaction_integrity.runtime.env import apply_runtime_environment
from compaction_integrity.runtime.openai_runtime import OpenAIRuntime
from compaction_integrity.runtime.vllm_runtime import VLLMRuntime
from compaction_integrity.runtime.vllm_serve_runtime import VLLMServeRuntime
from compaction_integrity.scripts.eval_run_layout import build_run_id, to_container, write_run_metadata
from compaction_integrity.sssc import probe_to_user_prompt


Message = dict[str, str]
_GENERATIVE_CASES = (
    "full_with_sssc",
    "full_without_sssc",
    "compacted",
    "compacted_post_sssc",
)


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
    full_output = response.raw.get("full_output")
    if full_output:
        return str(full_output)
    parts: list[str] = []
    thinking = response.raw.get("thinking")
    if thinking:
        parts.append(f"[analysis]\n{thinking}")
    if response.text:
        parts.append(response.text)
    return "\n\n".join(parts).strip()


def _case_context_and_prompt(
    row: dict[str, Any],
    case: str,
    base_messages_by_index: dict[int, list[Message]],
) -> tuple[list[Message] | None, str | None]:
    probe_prompt, _ = probe_to_user_prompt(str(row["sssc_probe"]), mcq=False)
    if case == "full_with_sssc":
        return row["full_with_sssc_messages"], probe_prompt
    if case == "full_without_sssc":
        return base_messages_by_index[int(row["source_row_index"])], probe_prompt
    if case == "compacted":
        return row["compacted_context"], probe_prompt
    if case == "compacted_post_sssc":
        context = row["compacted_context"]
        post_sssc = row["compacted_post_sssc_user_turn"]
        if context is None or post_sssc is None:
            return None, None
        return context, f"{post_sssc}\n\n{probe_prompt}"
    raise ValueError(f"Unsupported generative case: {case}")


def _run_case(
    df: pd.DataFrame,
    case: str,
    runtime: ModelRuntime,
    system_prompt: str,
    base_messages_by_index: dict[int, list[Message]],
    batch_size: int,
    overwrite: bool,
) -> None:
    prompt_col = f"{case}_generative_prompt"
    output_col = f"{case}_generative_output"
    tool_calls_col = f"{case}_generative_tool_calls"
    for column in (prompt_col, output_col, tool_calls_col):
        if column not in df:
            df[column] = None

    records = df.to_dict(orient="records")
    pending: list[int] = []
    for i, row in enumerate(records):
        context, prompt = _case_context_and_prompt(row, case, base_messages_by_index)
        if context is None:
            continue
        if not overwrite and pd.notna(row.get(output_col)):
            continue
        df.at[i, prompt_col] = prompt
        pending.append(i)

    for start in range(0, len(pending), batch_size):
        indices = pending[start:start + batch_size]
        conversations: list[list[Message]] = []
        for i in indices:
            row = df.iloc[i].to_dict()
            context, prompt = _case_context_and_prompt(row, case, base_messages_by_index)
            conversations.append(
                [
                    {"role": "system", "content": system_prompt},
                    get_sssc_evaluation_tool_message(mcq=False),
                    *context,
                    {"role": "user", "content": prompt},
                ]
            )
        outputs = runtime.batch_generate(
            conversations,
            params={"tools": get_sssc_evaluation_tools()},
        )
        for i, response in zip(indices, outputs):
            df.at[i, output_col] = _format_for_log(response)
            df.at[i, tool_calls_col] = response.raw.get("tool_calls", [])


def _parse_judge_verdict(output_text: str) -> bool | str:
    verdict = json.loads(output_text)["verdict"]
    if verdict not in (True, False, "not_enough_information"):
        raise ValueError(
            "Judge verdict must be true, false, or not_enough_information, "
            f"got: {output_text!r}"
        )
    return verdict


def _judge_case(
    df: pd.DataFrame,
    case: str,
    runtime: ModelRuntime,
    evaluator_cfg: dict[str, Any],
    overwrite: bool,
) -> None:
    probe_col = f"{case}_generative_prompt"
    output_col = f"{case}_generative_output"
    judge_prompt_col = f"{case}_generative_judge_prompt"
    judge_output_col = f"{case}_generative_judge_output"
    judge_verdict_col = f"{case}_generative_judge_verdict"
    for column in (judge_prompt_col, judge_output_col, judge_verdict_col):
        if column not in df:
            df[column] = None

    flex = bool(evaluator_cfg.get("kwargs", {}).get("flex", False))
    for i, row in enumerate(df.to_dict(orient="records")):
        output = row.get(output_col)
        if output is None or (not overwrite and pd.notna(row.get(judge_output_col))):
            continue
        system_prompt, user_prompt = build_instruction_following_judge_prompts(
            SSSC=str(row["sssc_message"]),
            probing_prompt=str(row[probe_col]),
            output_message=str(output),
            mcq=False,
        )
        response = runtime.generate(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            model=str(evaluator_cfg["model"]),
            flex=flex,
        )
        df.at[i, judge_prompt_col] = {"system": system_prompt, "user": user_prompt}
        df.at[i, judge_output_col] = response.text
        df.at[i, judge_verdict_col] = _parse_judge_verdict(response.text)


@hydra.main(version_base=None, config_path=None, config_name=None)
def main(cfg: DictConfig) -> None:
    apply_runtime_environment()

    results_root = Path(str(cfg.results_root))
    runs_dir = results_root / ("runs_test" if bool(cfg.test) else "runs")
    probe_cfg = to_container(cfg.probe)
    model = str(probe_cfg["model"])
    provider = str(probe_cfg["provider"])
    probe_kwargs = dict(probe_cfg.get("kwargs", {}))
    enabled_cases = [case for case in _GENERATIVE_CASES if bool(cfg.cases.get(case, False))]

    for source in to_container(cfg.source_runs):
        source_run_id = str(source["run_id"])
        source_dir = runs_dir / source_run_id
        source_df = pd.read_pickle(source_dir / "evaluation_results.pkl")
        source_spec = json.loads((source_dir / "metadata.json").read_text(encoding="utf-8"))["run_spec"]

        new_spec = copy.deepcopy(source_spec)
        new_spec["task"] = "generative_reprobe"
        new_spec["source_run_id"] = source_run_id
        new_spec["probe"] = {
            "name": str(probe_cfg["name"]),
            "config": {**probe_cfg, "mcq": False, "tool_contract": str(cfg.tool_contract)},
        }
        new_spec["instruction_following_judge"] = {
            **source_spec["evaluator"],
            "mcq": False,
        }
        sssc_ids = [int(sssc_id) for sssc_id in cfg.get("sssc_ids", [])]
        new_spec["sssc_ids"] = sssc_ids
        new_run_id = build_run_id(new_spec)
        new_dir = runs_dir / new_run_id
        output_path = new_dir / "generative_probe_results.pkl"
        new_dir.mkdir(parents=True, exist_ok=True)

        if bool(cfg.overwrite):
            output_path.unlink(missing_ok=True)
            output_df = source_df.copy()
        elif output_path.exists():
            output_df = pd.read_pickle(output_path)
        else:
            output_df = source_df.copy()
        if sssc_ids:
            output_df = output_df[output_df["sssc_id"].isin(sssc_ids)].reset_index(drop=True)
        output_df["generative_source_run_id"] = source_run_id

        dataset_cfg = source_spec["dataset"]
        rows = EvalDatasetLoader.load(
            dataset_path=Path(dataset_cfg["dir"]) / "stitched_dataset",
            test_mode=bool(source_spec["test"]),
            num_rows=dataset_cfg["num_rows"],
        ).rows()
        base_messages_by_index = {row.source_row_index: row.messages for row in rows}
        system_prompt = _retrieve_sys_prompt(str(dataset_cfg["name"]), model)
        runtime = _build_runtime(model, provider, probe_kwargs)
        try:
            batch_size = int(probe_kwargs.get("batch_size", 32))
            for case in enabled_cases:
                _run_case(
                    df=output_df,
                    case=case,
                    runtime=runtime,
                    system_prompt=system_prompt,
                    base_messages_by_index=base_messages_by_index,
                    batch_size=batch_size,
                    overwrite=bool(cfg.overwrite),
                )
                output_df.to_pickle(output_path)
        finally:
            runtime.close()
            gc.collect()

        evaluator_cfg = dict(source_spec["evaluator"]["config"])
        evaluator_runtime = _build_runtime(
            str(evaluator_cfg["model"]),
            str(evaluator_cfg["provider"]),
            dict(evaluator_cfg.get("kwargs", {})),
        )
        try:
            for case in enabled_cases:
                _judge_case(
                    df=output_df,
                    case=case,
                    runtime=evaluator_runtime,
                    evaluator_cfg=evaluator_cfg,
                    overwrite=bool(cfg.overwrite),
                )
                output_df.to_pickle(output_path)
        finally:
            evaluator_runtime.close()
            gc.collect()

        write_run_metadata(new_dir, new_run_id, new_spec)
        print(f"Generative reprobe saved to {output_path}")


if __name__ == "__main__":
    main()
