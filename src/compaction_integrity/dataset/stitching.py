from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from datasets import Dataset
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm

from compaction_integrity.dataset.embedding_cache import (
    DEFAULT_EMBEDDING_MODEL_NAME,
    DEFAULT_TOPIC_COHESIVE_KNN_K,
    batch_embedding,
    build_or_load_knn_index,
    build_vllm_embedding_model,
    load_or_compute_normalized_embeddings,
    normalize_embeddings,
    resolve_knn_k,
)
from compaction_integrity.dataset.injection import (
    Message,
    require_openai_messages,
)
from compaction_integrity.dataset.loader import Convo
from compaction_integrity.tokenization import count_tokens_messages_batch

# Datasets whose conversations include a system prompt that must be deduplicated on stitching.
_SYSTEM_PROMPT_DEDUP_KEYWORDS: frozenset[str] = frozenset({"hermes"})


def _needs_system_prompt_dedup(source_name: str) -> bool:
    lower = source_name.lower()
    return any(kw in lower for kw in _SYSTEM_PROMPT_DEDUP_KEYWORDS)


def _validate_system_prompt_position(messages: list[Message]) -> None:
    """Raise if any system message appears after the first position."""
    for i, msg in enumerate(messages):
        if msg.get("role") == "system" and i != 0:
            raise ValueError(
                f"System message found at index {i} but system prompts must be the first message. "
                f"Offending message: {msg}"
            )


def _strip_leading_system_prompt(messages: list[Message]) -> list[Message]:
    if messages and messages[0].get("role") == "system":
        return messages[1:]
    return messages


def _stitching_row_token_length(
    row: "StitchingRow",
    *,
    dedup_system: bool,
    first_source_row: bool,
) -> int:
    if (
        dedup_system
        and not first_source_row
        and row.messages
        and row.messages[0].get("role") == "system"
    ):
        return row.token_length - count_tokens_messages_batch([[row.messages[0]]])[0]
    return row.token_length


def _build_stitched_messages(source_rows: "list[StitchingRow]", dedup_system: bool) -> list[Message]:
    """Concatenate messages from source rows, optionally deduplicating system prompts."""
    result: list[Message] = []
    for i, row in enumerate(source_rows):
        msgs = row.messages
        _validate_system_prompt_position(msgs)
        if dedup_system and i > 0:
            msgs = _strip_leading_system_prompt(msgs)
        result.extend(msgs)
    return result


def _stitched_token_length(source_rows: "list[StitchingRow]", dedup_system: bool) -> int:
    return sum(
        _stitching_row_token_length(
            row,
            dedup_system=dedup_system,
            first_source_row=i == 0,
        )
        for i, row in enumerate(source_rows)
    )


@dataclass(frozen=True, slots=True)
class StitchingRow:
    messages: list[Message]
    token_length: int


def parse_target_length(target_length: int | str) -> int:
    if isinstance(target_length, int):
        return target_length

    normalized = target_length.strip().lower().replace("_", "")
    suffix_multipliers = {"k": 1000, "m": 1000000}
    suffix = normalized[-1]
    if suffix in suffix_multipliers:
        return int(float(normalized[:-1]) * suffix_multipliers[suffix])
    return int(normalized)


def parse_stitching_row(row: dict[str, Any]) -> StitchingRow:
    messages = require_openai_messages(row.get("messages"))
    return StitchingRow(
        messages=messages,
        token_length=int(row["token_length"]),
    )


def build_stitching_rows(source_dataset: Dataset) -> list[StitchingRow]:
    return [parse_stitching_row(row) for row in source_dataset]


def normalize_vector(vector: np.ndarray) -> np.ndarray:
    return vector / np.linalg.norm(vector)


def find_next_knn_candidate(
    knn_index: NearestNeighbors,
    query_vector: np.ndarray,
    candidate_mask: np.ndarray,
    initial_neighbor_count: int,
) -> int | None:
    total_count = len(candidate_mask)
    neighbor_count = min(total_count, initial_neighbor_count)

    while True:
        neighbor_indices = knn_index.kneighbors(
            query_vector.reshape(1, -1),
            n_neighbors=neighbor_count,
            return_distance=False,
        )[0]
        for candidate_idx in neighbor_indices:
            candidate_idx_int = int(candidate_idx)
            if candidate_mask[candidate_idx_int]:
                return candidate_idx_int
        if neighbor_count == total_count:
            return None
        neighbor_count = min(total_count, neighbor_count * 2)


def _crop_source_rows_to_max_tokens(
    source_rows: list[StitchingRow],
    max_tokens: int,
    dedup_system: bool,
) -> list[StitchingRow]:
    """Remove rows from the tail until total token count <= max_tokens."""
    total = _stitched_token_length(source_rows, dedup_system)
    if total <= max_tokens:
        return source_rows
    kept = list(source_rows)
    while kept and total > max_tokens:
        kept.pop()
        total = _stitched_token_length(kept, dedup_system)
    return kept


def concat_dataset(
    ds: Convo,
    target_length: int | str,
    num_convo: int,
) -> Convo:
    target_tokens = parse_target_length(target_length)
    max_tokens = int(target_tokens / 0.8)
    dedup_system = _needs_system_prompt_dedup(ds.source_name)
    source_dataset = ds.unwrap()
    required_columns = {"messages", "token_length"}
    missing_columns = required_columns.difference(source_dataset.column_names)
    if missing_columns:
        raise ValueError(
            f"Dataset is missing required columns: {sorted(missing_columns)}."
        )

    concatenated_rows: list[dict[str, Any]] = []
    current_source_rows: list[StitchingRow] = []
    current_token_length = 0

    for row in source_dataset:
        if num_convo > 0 and len(concatenated_rows) >= num_convo:
            print(f"Stopping after {num_convo} concatenated rows.")
            break

        parsed_row = parse_stitching_row(row)
        current_source_rows.append(parsed_row)
        current_token_length = _stitched_token_length(current_source_rows, dedup_system)

        if current_token_length >= target_tokens:
            if current_token_length > max_tokens:
                current_source_rows = _crop_source_rows_to_max_tokens(
                    current_source_rows,
                    max_tokens,
                    dedup_system,
                )
                current_token_length = _stitched_token_length(current_source_rows, dedup_system)
            concatenated_rows.append(
                {
                    "messages": _build_stitched_messages(current_source_rows, dedup_system),
                    "token_length": current_token_length,
                    "source_conversation_count": len(current_source_rows),
                }
            )
            current_source_rows = []
            current_token_length = 0

    if not concatenated_rows:
        raise ValueError(
            "Concatenation produced no full rows. Reduce target_length or use a larger source dataset."
        )
    if num_convo > 0 and len(concatenated_rows) < num_convo:
        raise ValueError(
            f"Concatenation produced only {len(concatenated_rows)} eligible rows, "
            f"but num_convo={num_convo}. Increase source data or lower target_size."
        )

    return Convo(
        dataset=Dataset.from_list(concatenated_rows),
        source_name=ds.source_name,
    )


def topic_cohesive_dataset(
    ds: Convo,
    target_length: int | str,
    num_convo: int,
    *,
    embedding_source_column: str = "messages",
    embedding_cache_root_dir: Path | None = None,
    knn_k: int = DEFAULT_TOPIC_COHESIVE_KNN_K,
    embedding_model_name: str = DEFAULT_EMBEDDING_MODEL_NAME,
    env_config: Any = None,
) -> Convo:
    target_tokens = parse_target_length(target_length)
    dedup_system = _needs_system_prompt_dedup(ds.source_name)
    source_dataset = ds.unwrap()
    resolved_knn_k = resolve_knn_k(knn_k)

    required_columns = {"messages", "token_length"}
    missing_columns = required_columns.difference(source_dataset.column_names)
    if missing_columns:
        raise ValueError(
            f"Dataset is missing required columns: {sorted(missing_columns)}."
        )
    if len(source_dataset) == 0:
        raise ValueError("Dataset is empty.")

    if embedding_cache_root_dir is None:
        embedding_model = build_vllm_embedding_model(
            embedding_model_name=embedding_model_name,
            env_config=env_config,
        )
        embeddings = batch_embedding(
            ds,
            embedding_model,
            embedding_source_column=embedding_source_column,
        )
        normalized_embeddings = normalize_embeddings(embeddings).astype(np.float32, copy=False)
    else:
        normalized_embeddings = load_or_compute_normalized_embeddings(
            ds,
            embedding_cache_root_dir,
            embedding_model_name=embedding_model_name,
            embedding_source_column=embedding_source_column,
            env_config=env_config,
        )

    rows = build_stitching_rows(source_dataset)
    remaining_mask = np.ones(len(rows), dtype=bool)
    stitched_rows: list[dict[str, Any]] = []
    max_stitch_attempts = num_convo if num_convo > 0 else len(rows)
    max_tokens = int(target_tokens / 0.8)
    knn_index = build_or_load_knn_index(
        ds,
        normalized_embeddings,
        resolved_knn_k,
        cache_root_dir=embedding_cache_root_dir,
        embedding_model_name=embedding_model_name,
        embedding_source_column=embedding_source_column,
    )

    print(f"Using k-NN topic-cohesive stitching with knn_k={resolved_knn_k}.")

    for _ in tqdm(range(max_stitch_attempts), desc="Stitching conversations"):
        if not remaining_mask.any():
            break

        seed_idx = int(np.flatnonzero(remaining_mask)[0])
        used_mask = np.zeros(len(rows), dtype=bool)
        used_mask[seed_idx] = True

        seed_row = rows[seed_idx]
        current_source_indices: list[int] = [seed_idx]
        current_token_length = seed_row.token_length
        centroid_sum = normalized_embeddings[seed_idx].copy()

        while current_token_length < target_tokens:
            candidate_mask = remaining_mask & ~used_mask
            if not candidate_mask.any():
                break

            current_source_count = len(current_source_indices)
            centroid = normalize_vector(centroid_sum / current_source_count)
            next_idx = find_next_knn_candidate(
                knn_index=knn_index,
                query_vector=centroid,
                candidate_mask=candidate_mask,
                initial_neighbor_count=max(1, resolved_knn_k + current_source_count),
            )
            if next_idx is None:
                raise ValueError(
                    "k-NN topic-cohesive stitching exhausted the neighbor set before completing a row. "
                    f"knn_k={resolved_knn_k} current_source_count={current_source_count} "
                    f"remaining_candidates={int(candidate_mask.sum())}."
                )

            used_mask[next_idx] = True
            next_row = rows[next_idx]
            current_source_indices.append(next_idx)
            current_token_length += _stitching_row_token_length(
                next_row,
                dedup_system=dedup_system,
                first_source_row=False,
            )
            centroid_sum += normalized_embeddings[next_idx]

        if current_token_length < target_tokens:
            raise ValueError(
                "Topic-cohesive stitching could not complete a valid row from the remaining pool. "
                f"current_token_length={current_token_length} target_tokens={target_tokens}."
            )

        if current_token_length > max_tokens:
            # Crop from the most recently added source rows, restoring them to the available pool.
            while len(current_source_indices) > 1 and current_token_length > max_tokens:
                dropped_idx = current_source_indices.pop()
                current_token_length -= _stitching_row_token_length(
                    rows[dropped_idx],
                    dedup_system=dedup_system,
                    first_source_row=False,
                )
                used_mask[dropped_idx] = False

        source_rows_for_row = [rows[idx] for idx in current_source_indices]
        stitched_rows.append(
            {
                "messages": _build_stitched_messages(source_rows_for_row, dedup_system),
                "token_length": current_token_length,
                "source_conversation_count": len(current_source_indices),
            }
        )
        remaining_mask[used_mask] = False

    if not stitched_rows:
        raise ValueError(
            "Topic-cohesive stitching produced no full rows. Reduce target_length or use a larger source dataset."
        )
    if num_convo > 0 and len(stitched_rows) < num_convo:
        raise ValueError(
            f"Topic-cohesive stitching produced only {len(stitched_rows)} eligible rows, "
            f"but num_convo={num_convo}. Increase source data or lower target_size."
        )

    return Convo(
        dataset=Dataset.from_list(stitched_rows),
        source_name=ds.source_name,
    )
