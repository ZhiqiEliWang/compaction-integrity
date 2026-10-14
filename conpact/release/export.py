"""Export the conpact release to its Hugging Face dataset repo.

Run from the repository root (needs `compaction_integrity` installed):
  python conpact/release/export.py
  python conpact/release/export.py push=true

Writes, under `out_dir`:
  {wildchat,hermes,openresearcher,swe_natural}/test-00000-of-00001.parquet
      The paper's RQ1 instances, one row per (context, SC). The SC
      is injected at the top with direct + preferential framing; SWE-Natural
      rows carry their native SC.
  haystacks/{wildchat,hermes,openresearcher}-00000-of-00001.parquet
      The SC-free ~220k histories that the `conpact` package cuts
      to a requested length, with per-message token counts.
  README.md
      The dataset card, copied from `card`.

Static rows are built with the evaluation code itself and checked against the
paper's full-probe caches (`verify`): injected history, probe prompt, swap seed
and A/B key must match, so the released rows are the paper's inputs.

Row schema (static configs; the `conpact` package's `Instance` has the same fields):
  id                   "<haystack_id>:sc<sc_id>"
  source               wildchat | hermes | openresearcher | swe_natural
  haystack_id          "<dataset name>:<row index>"
  target_length        nominal context length in tokens (100000)
  token_length         tokens of `messages_without_sc` (gpt-oss tokenizer)
  position             top | middle | bottom | multi | native (SWE-Natural)
  repeat               requested repetition count (multi only; else 1)
  strict, direct       framing: "This is an important constraint:" / "For the rest of this session."
  sc_id, sc_type       SC id and taxonomy type
  sc_text              the SC as written (the retention judge's reference)
  sc_rendered          the SC with its framing, as injected and as re-presented in the upper-bound condition
  sc_message_indices   indices into `messages` of the user turns that carry the SC
  system_prompt        system message for the probe model
  developer_prompt     developer message (simulated tools) for the probe model
  messages             history with the SC (compactor input; condition 1)
  messages_without_sc  history without the SC (condition 2)
  probe, correct_answer, incorrect_answer
  swap_seed            seed of the A/B order
  compliant_letter     A | B
  probe_prompt         user turn for the full + SC, full and compaction conditions
  post_sc_probe_prompt user turn for the upper-bound condition: sc_rendered + "\\n\\n" + probe_prompt
  meta                 JSON string of source-specific fields
"""

import json
import shutil
from pathlib import Path
from typing import Any

import hydra
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from datasets import Dataset
from hydra.utils import to_absolute_path
from omegaconf import DictConfig

from compaction_integrity.dataset.ds_system_prompts import get_dataset_system_prompt
from compaction_integrity.dataset.eval_loader import EvalDatasetLoader
from compaction_integrity.prompts import get_sssc_evaluation_tool_message
from compaction_integrity.scripts.evaluation import _build_pairs, _compute_pair_swap_seeds
from compaction_integrity.sssc import probe_to_user_prompt, sssc_to_prompt
from compaction_integrity.tokenization import count_tokens_messages_batch


# The paper reads the WildChat probe system prompt out of gpt-oss-120b's chat
# template at run time (`evaluation._retrieve_sys_prompt`), which stamps the run
# date. The release freezes that string without the "Current date:" line.
WILDCHAT_SYSTEM_PROMPT = (
    "You are ChatGPT, a large language model trained by OpenAI.\n"
    "Knowledge cutoff: 2024-06\n\n"
    "Reasoning: medium\n\n"
    "# Valid channels: analysis, commentary, final. Channel must be included for every message."
)

SYSTEM_PROMPTS = {
    "wildchat": WILDCHAT_SYSTEM_PROMPT,
    "hermes": get_dataset_system_prompt("hermes"),
    "openresearcher": get_dataset_system_prompt("openresearcher"),
}

# Static rows are ordered haystack-major, 15 SCs per haystack. One row group per
# haystack lets parquet's dictionary encoding store each shared history once
# instead of 15 times (wildchat: ~690 MB -> ~13 MB, lossless).
ROW_GROUP_SIZE = 15

SWE_META_COLUMNS = [
    "sc_kind",
    "sc_clauses",
    "annotation_id",
    "instance_id",
    "trajectory_id",
    "repo",
    "resolved",
    "exit_status",
    "n_turns",
    "n_spans_removed",
    "n_echo_spans_removed",
    "keyword_token_frac",
]


def _static_rows(
    source: str,
    dataset_dir: Path,
    num_rows: int,
    sssc_attrs: dict[str, Any],
    global_seed: int,
    verify_path: Path,
) -> list[dict[str, Any]]:
    loader = EvalDatasetLoader.load(dataset_dir / "stitched_dataset", test_mode=False, num_rows=num_rows)
    rows = loader.rows()
    natural = rows[0].sssc is not None
    pairs = _build_pairs(rows, sssc_attrs, SYSTEM_PROMPTS.get(source), global_seed)
    swap_seeds = _compute_pair_swap_seeds(pairs, global_seed)
    paper = {
        (int(r["source_row_index"]), int(r["sssc_id"])): r
        for r in pd.read_pickle(verify_path).to_dict(orient="records")
    }
    developer_prompt = get_sssc_evaluation_tool_message()["content"]

    out: list[dict[str, Any]] = []
    for pair in pairs:
        sc = pair.sssc
        row_index = pair.row.source_row_index
        src = loader.dataset[row_index]
        swap_seed = swap_seeds[pair.key]
        probe_prompt, compliant_letter = probe_to_user_prompt(
            probe=str(sc["probe"]),
            correct_answer=str(sc["correct_answer"]),
            incorrect_answer=str(sc["incorrect_answer"]),
            seed=swap_seed,
        )
        sc_rendered = sssc_to_prompt(str(sc["sssc"]), sssc_attrs["explicitness"], sssc_attrs["hard"])

        expected = paper[(row_index, int(sc["id"]))]
        if (
            pair.with_messages != list(expected["full_with_sssc_messages"])
            or probe_prompt != expected["full_with_sssc_probe_prompt"]
            or swap_seed != int(expected["swap_seed"])
            or compliant_letter != expected["compliant_letter"]
        ):
            raise ValueError(f"{source} row {row_index} sc {sc['id']} differs from {verify_path}")

        if natural:
            system_offset = len(src["messages"]) - len(pair.row.messages)
            sc_message_indices = [int(src["sc_issue_message_index"]) - system_offset]
            token_length = int(src["token_length_without_sc"])
            meta = {k: src[k] for k in SWE_META_COLUMNS}
        else:
            sc_message_indices = [
                i for i, (a, b) in enumerate(zip(pair.with_messages, pair.without_messages)) if a != b
            ]
            token_length = int(src["token_length"])
            meta = {"source_conversation_count": int(src["source_conversation_count"])}

        haystack_id = f"{dataset_dir.name}:{row_index:03d}"
        out.append(
            {
                "id": f"{haystack_id}:sc{int(sc['id']):02d}",
                "source": source,
                "haystack_id": haystack_id,
                "target_length": 100_000,
                "token_length": token_length,
                "position": "native" if natural else sssc_attrs["position"],
                "repeat": int(sssc_attrs["repeat"]),
                "strict": bool(sssc_attrs["hard"]),
                "direct": bool(sssc_attrs["explicitness"]),
                "sc_id": int(sc["id"]),
                "sc_type": str(sc["type"]),
                "sc_text": str(sc["sssc"]),
                "sc_rendered": sc_rendered,
                "sc_message_indices": sc_message_indices,
                "system_prompt": pair.system_prompt,
                "developer_prompt": developer_prompt,
                "messages": pair.with_messages,
                "messages_without_sc": pair.without_messages,
                "probe": str(sc["probe"]),
                "correct_answer": str(sc["correct_answer"]),
                "incorrect_answer": str(sc["incorrect_answer"]),
                "swap_seed": swap_seed,
                "compliant_letter": compliant_letter,
                "probe_prompt": probe_prompt,
                "post_sc_probe_prompt": f"{sc_rendered}\n\n{probe_prompt}",
                "meta": json.dumps(meta),
            }
        )
    print(f"[static] {source}: {len(out)} rows from {dataset_dir} (verified against {verify_path.name})")
    return out


def _haystack_rows(source: str, dataset_dir: Path) -> list[dict[str, Any]]:
    ds = Dataset.load_from_disk(str(dataset_dir / "stitched_dataset"))
    histories = [[{"role": m["role"], "content": m["content"]} for m in row] for row in ds["messages"]]
    flat_tokens = count_tokens_messages_batch([[m] for messages in histories for m in messages])

    out: list[dict[str, Any]] = []
    offset = 0
    for i, messages in enumerate(histories):
        message_tokens = flat_tokens[offset : offset + len(messages)]
        offset += len(messages)
        out.append(
            {
                "haystack_id": f"{dataset_dir.name}:{i:03d}",
                "source": source,
                "system_prompt": SYSTEM_PROMPTS[source],
                "messages": messages,
                "message_tokens": message_tokens,
                "token_length": sum(message_tokens),
                "source_conversation_count": int(ds[i]["source_conversation_count"]),
            }
        )
    print(f"[haystacks] {source}: {len(out)} haystacks from {dataset_dir}")
    return out


def _write(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.Table.from_pylist(rows),
        path,
        row_group_size=ROW_GROUP_SIZE,
        compression="zstd",
        dictionary_pagesize_limit=64 << 20,
    )


@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg: DictConfig) -> None:
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    global_seed = int(cfg.global_seed)

    static_attrs = dict(cfg.static.sssc_attrs)
    for source, dataset_dir in cfg.static.datasets.items():
        rows = _static_rows(
            source,
            Path(to_absolute_path(dataset_dir)),
            int(cfg.static.num_rows),
            static_attrs,
            global_seed,
            Path(cfg.verify[source]),
        )
        _write(rows, out_dir / source / "test-00000-of-00001.parquet")

    swe = cfg.static.swe_natural
    rows = _static_rows(
        "swe_natural",
        Path(to_absolute_path(swe.dir)),
        int(swe.num_rows),
        dict(swe.sssc_attrs),
        global_seed,
        Path(cfg.verify.swe_natural),
    )
    _write(rows, out_dir / "swe_natural" / "test-00000-of-00001.parquet")

    for source, dataset_dir in cfg.haystacks.items():
        rows = _haystack_rows(source, Path(dataset_dir))
        _write(rows, out_dir / "haystacks" / f"{source}-00000-of-00001.parquet")

    shutil.copyfile(to_absolute_path(cfg.card), out_dir / "README.md")
    print(f"Release written to {out_dir}")

    if cfg.push:
        from huggingface_hub import HfApi

        HfApi().upload_folder(
            folder_path=str(out_dir),
            repo_id=str(cfg.hf_repo),
            repo_type="dataset",
            commit_message="Export conpact release",
        )
        print(f"Pushed {out_dir} to https://huggingface.co/datasets/{cfg.hf_repo}")


if __name__ == "__main__":
    main()
