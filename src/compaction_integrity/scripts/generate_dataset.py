import json
from dataclasses import asdict, dataclass
from pathlib import Path

import hydra
from datasets import Dataset
from omegaconf import DictConfig

from compaction_integrity.dataset.embedding_cache import (
    build_stitched_cache_path,
    resolve_knn_k,
)
from compaction_integrity.dataset.injection import require_openai_messages
from compaction_integrity.dataset.loader import Convo, load_conversations
from compaction_integrity.dataset.stitching import (
    concat_dataset,
    parse_target_length,
    topic_cohesive_dataset,
)
from compaction_integrity.runtime.env import apply_runtime_environment
from compaction_integrity.seed import set_seed
from compaction_integrity.tokenization import count_tokens_messages_batch


_OPENRESEARCHER_DATASET = "openresearcher"


@dataclass(frozen=True, slots=True)
class DatasetGenerationManifest:
    dataset: str
    test: bool
    num_convo: int
    target_size: str | int
    stitching_method: str


def write_generation_manifest(
    manifest: DatasetGenerationManifest,
    output_path: Path,
) -> None:
    output_path.write_text(json.dumps(asdict(manifest), indent=2) + "\n")


def _print_basic_stats(ds: Convo) -> None:
    dataset = ds.unwrap()
    token_lengths = ds.token_lengths()
    row_count = len(dataset)

    if row_count == 0:
        print(f"dataset={ds.source_name} rows=0")
        return

    stats_parts = [
        f"dataset={ds.source_name}",
        f"rows={row_count}",
        f"tokens_min={min(token_lengths)}",
        f"tokens_max={max(token_lengths)}",
        f"tokens_avg={sum(token_lengths) / row_count:.2f}",
    ]
    if "source_conversation_count" in dataset.column_names:
        source_counts = list(dataset["source_conversation_count"])
        stats_parts.append(
            f"source_rows_avg={sum(source_counts) / len(source_counts):.2f}"
        )

    print(" ".join(stats_parts))


def _trim_messages_to_at_least_target(
    messages: list[dict[str, str]],
    target_tokens: int,
) -> tuple[list[dict[str, str]], int]:
    trimmed_messages = list(messages)
    token_lengths = count_tokens_messages_batch([[message] for message in trimmed_messages])
    current_token_length = sum(token_lengths)

    while len(trimmed_messages) > 1 and current_token_length - token_lengths[-1] >= target_tokens:
        current_token_length -= token_lengths.pop()
        trimmed_messages.pop()

    return trimmed_messages, current_token_length


_OPENRESEARCHER_PAIR_TARGET_TOKENS = 220_000


def _prepare_openresearcher_dataset(
    ds: Convo,
    target_length: int | str,
    num_convo: int,
) -> Convo:
    target_tokens = parse_target_length(target_length)

    if target_tokens == _OPENRESEARCHER_PAIR_TARGET_TOKENS:
        return _prepare_openresearcher_paired_dataset(ds, target_tokens, num_convo)

    selected_rows: list[dict[str, object]] = []

    for row in ds.unwrap():
        if num_convo > 0 and len(selected_rows) >= num_convo:
            break
        if int(row["token_length"]) < target_tokens:
            continue

        messages = require_openai_messages(row["messages"])
        trimmed_messages, token_length = _trim_messages_to_at_least_target(
            messages,
            target_tokens,
        )
        selected_rows.append(
            {
                "messages": trimmed_messages,
                "token_length": token_length,
                "source_conversation_count": 1,
            }
        )

    if not selected_rows:
        raise ValueError(
            "OpenResearcher preparation produced no full rows. Reduce target_length or use a larger source dataset."
        )
    if num_convo > 0 and len(selected_rows) < num_convo:
        raise ValueError(
            f"OpenResearcher preparation produced only {len(selected_rows)} eligible rows, "
            f"but num_convo={num_convo}. Increase source data or lower target_size."
        )

    return Convo(
        dataset=Dataset.from_list(selected_rows),
        source_name=ds.source_name,
    )


def _prepare_openresearcher_paired_dataset(
    ds: Convo,
    target_tokens: int,
    num_convo: int,
) -> Convo:
    half_target = target_tokens // 2
    trimmed_singles: list[tuple[list[dict[str, str]], int]] = []
    needed = num_convo * 2 if num_convo > 0 else 0

    for row in ds.unwrap():
        if needed > 0 and len(trimmed_singles) >= needed:
            break
        if int(row["token_length"]) < half_target:
            continue

        messages = require_openai_messages(row["messages"])
        trimmed_messages, token_length = _trim_messages_to_at_least_target(
            messages,
            half_target,
        )
        trimmed_singles.append((trimmed_messages, token_length))

    pair_count = len(trimmed_singles) // 2
    if num_convo > 0 and pair_count < num_convo:
        raise ValueError(
            f"OpenResearcher paired preparation produced only {pair_count} pairs "
            f"(from {len(trimmed_singles)} ~{half_target}-token rows), but num_convo={num_convo}. "
            f"Increase source data or lower target_size."
        )
    if pair_count == 0:
        raise ValueError(
            "OpenResearcher paired preparation produced no pairs. Reduce target_length or use a larger source dataset."
        )

    if num_convo > 0:
        trimmed_singles = trimmed_singles[: num_convo * 2]

    selected_rows: list[dict[str, object]] = []
    for i in range(0, len(trimmed_singles), 2):
        first_messages, first_tokens = trimmed_singles[i]
        second_messages, second_tokens = trimmed_singles[i + 1]
        second_no_system = [m for m in second_messages if m.get("role") != "system"]
        # Recount tokens for the second conversation after dropping its system prompt.
        if len(second_no_system) != len(second_messages):
            second_token_lengths = count_tokens_messages_batch(
                [[message] for message in second_no_system]
            )
            second_tokens = sum(second_token_lengths)
        combined_messages = first_messages + second_no_system
        selected_rows.append(
            {
                "messages": combined_messages,
                "token_length": first_tokens + second_tokens,
                "source_conversation_count": 2,
            }
        )

    return Convo(
        dataset=Dataset.from_list(selected_rows),
        source_name=ds.source_name,
    )


_KEEP_COLUMNS = {"messages", "token_length", "source_conversation_count"}


def _project_to_keep_columns(ds: Convo) -> Convo:
    """Drop any columns outside _KEEP_COLUMNS so the saved dataset is lean."""
    dataset = ds.unwrap()
    extra_columns = [c for c in dataset.column_names if c not in _KEEP_COLUMNS]
    if extra_columns:
        dataset = dataset.remove_columns(extra_columns)
    return Convo(dataset=dataset, source_name=ds.source_name)


@hydra.main(version_base=None, config_path="../../../config/tasks/generate_dataset", config_name=None)
def main(cfg: DictConfig) -> None:
    output_dir = Path(str(cfg.output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)
    env_config = apply_runtime_environment(cfg)
    embedding_col_raw = cfg.get("embedding_col")
    embedding_col = None if embedding_col_raw is None else str(embedding_col_raw).strip() or None

    stitched_save_path = output_dir / "stitched_dataset"

    embedding_cache_root_dir: Path | None = None
    embedding_cache_cfg = cfg.get("embedding_cache")
    if embedding_cache_cfg is not None and embedding_cache_cfg.get("path") is not None:
        embedding_cache_root_dir = Path(str(embedding_cache_cfg.path))
        stitched_save_path = build_stitched_cache_path(embedding_cache_root_dir, cfg)

    # Manifest lives next to the dataset so eval consumers only need one path.
    manifest_path = stitched_save_path.parent / "generation_manifest.json"

    print(f"STEP 1: Loading dataset from {cfg.dataset}...")
    ds = load_conversations(
        str(cfg.dataset),
        preserve_columns=[embedding_col] if embedding_col is not None else None,
    )

    print("STEP 2: Preparing target-length message rows...")
    if str(cfg.dataset) == _OPENRESEARCHER_DATASET:
        if stitched_save_path.exists():
            print("Found existing prepared dataset, loading from disk...")
            ds = Convo(
                dataset=Dataset.load_from_disk(str(stitched_save_path)),
                source_name=str(cfg.dataset),
            )
        else:
            num_convo = 5 if bool(cfg.test) else int(cfg.num_convo)
            ds = _prepare_openresearcher_dataset(
                ds,
                cfg.target_size,
                num_convo,
            )
            ds = _project_to_keep_columns(ds)
            ds.dataset.save_to_disk(str(stitched_save_path))
    elif cfg.concat:
        if stitched_save_path.exists():
            print("Found existing stitched dataset, loading from disk...")
            ds = Convo(
                dataset=Dataset.load_from_disk(str(stitched_save_path)),
                source_name=str(cfg.dataset),
            )
        else:
            num_convo = 5 if bool(cfg.test) else int(cfg.num_convo)
            if cfg.stitching_method == "topic_cohesive":
                ds = topic_cohesive_dataset(
                    ds,
                    cfg.target_size,
                    num_convo,
                    embedding_source_column=embedding_col or "messages",
                    embedding_cache_root_dir=embedding_cache_root_dir,
                    knn_k=resolve_knn_k(cfg.get("knn_k")),
                    env_config=env_config,
                )
            elif cfg.stitching_method == "vanilla":
                ds = concat_dataset(
                    ds,
                    cfg.target_size,
                    num_convo,
                )
            else:
                raise ValueError(f"Unsupported stitching_method: {cfg.stitching_method}")
            ds = _project_to_keep_columns(ds)
            ds.dataset.save_to_disk(str(stitched_save_path))
    _print_basic_stats(ds)

    write_generation_manifest(
        DatasetGenerationManifest(
            dataset=str(cfg.dataset),
            test=bool(cfg.test),
            num_convo=int(cfg.num_convo),
            target_size=str(cfg.target_size),
            stitching_method=str(cfg.stitching_method),
        ),
        manifest_path,
    )
    print(f"saved_rows={len(ds.unwrap())} output_path={stitched_save_path}")


if __name__ == "__main__":
    set_seed(42)
    main()
