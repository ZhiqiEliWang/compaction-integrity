import json
import pickle
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.neighbors import NearestNeighbors

from compaction_integrity.dataset.injection import Message, require_openai_messages
from compaction_integrity.dataset.loader import Convo
from compaction_integrity.runtime.env import apply_runtime_environment


DEFAULT_TOPIC_COHESIVE_KNN_K = 32
DEFAULT_EMBEDDING_MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"


@dataclass(frozen=True, slots=True)
class EmbeddingCacheManifest:
    dataset_name: str
    row_count: int
    embedding_model_name: str
    embedding_source_column: str
    max_model_len: int
    embedding_dim: int
    dataset_fingerprint: str | None = None


@dataclass(frozen=True, slots=True)
class EmbeddingCachePaths:
    cache_dir: Path
    manifest_path: Path
    embeddings_path: Path


@dataclass(frozen=True, slots=True)
class KNNIndexCacheManifest:
    dataset_name: str
    row_count: int
    embedding_model_name: str
    embedding_source_column: str
    knn_k: int
    dataset_fingerprint: str | None = None


@dataclass(frozen=True, slots=True)
class KNNIndexCachePaths:
    cache_dir: Path
    manifest_path: Path
    index_path: Path


def resolve_knn_k(knn_k: Any) -> int:
    if knn_k is None:
        return DEFAULT_TOPIC_COHESIVE_KNN_K
    return int(knn_k)


def sanitize_cache_key_part(value: Any) -> str:
    return str(value).strip().replace("/", "--")


def build_stitched_cache_path(
    cache_root_dir: Path,
    cfg: Any,
) -> Path:
    ds_name = str(cfg.get("ds_name", "")).strip()
    if ds_name:
        cache_key = sanitize_cache_key_part(ds_name)
    else:
        cache_key = "__".join(
            [
                sanitize_cache_key_part(cfg.dataset),
                sanitize_cache_key_part(cfg.stitching_method),
                sanitize_cache_key_part(cfg.target_size),
                f"num_convo={int(cfg.num_convo)}",
                f"test={str(bool(cfg.test)).lower()}",
                f"knn_k={resolve_knn_k(cfg.get('knn_k'))}",
                f"embedding_col={sanitize_cache_key_part(cfg.get('embedding_col') or 'messages')}",
            ]
        )
    return cache_root_dir / "stitched_datasets" / cache_key / "stitched_dataset"


def build_vllm_embedding_model(
    *,
    embedding_model_name: str = DEFAULT_EMBEDDING_MODEL_NAME,
    env_config: Any = None,
) -> Any:
    apply_runtime_environment(
        env_config,
        worker_multiproc_method="spawn",
    )
    from vllm import LLM

    return LLM(model=embedding_model_name, runner="pooling")


def embedding_model_max_len(embedding_model: Any) -> int:
    return int(embedding_model.model_config.max_model_len)


def dataset_fingerprint(ds: Convo) -> str:
    return ds.unwrap()._fingerprint


def build_embedding_cache_paths(
    cache_root_dir: Path,
    ds: Convo,
    *,
    embedding_model_name: str = DEFAULT_EMBEDDING_MODEL_NAME,
    embedding_source_column: str = "messages",
) -> EmbeddingCachePaths:
    cache_dir = (
        cache_root_dir
        / ds.source_name
        / embedding_model_name.replace("/", "--")
        / f"embedding_col={sanitize_cache_key_part(embedding_source_column)}"
    )
    return EmbeddingCachePaths(
        cache_dir=cache_dir,
        manifest_path=cache_dir / "manifest.json",
        embeddings_path=cache_dir / "normalized_embeddings.npy",
    )


def build_knn_index_cache_paths(
    cache_root_dir: Path,
    ds: Convo,
    knn_k: int,
    *,
    embedding_model_name: str = DEFAULT_EMBEDDING_MODEL_NAME,
    embedding_source_column: str = "messages",
) -> KNNIndexCachePaths:
    embedding_cache_dir = build_embedding_cache_paths(
        cache_root_dir,
        ds,
        embedding_model_name=embedding_model_name,
        embedding_source_column=embedding_source_column,
    ).cache_dir
    cache_dir = embedding_cache_dir / f"knn_k={knn_k}"
    return KNNIndexCachePaths(
        cache_dir=cache_dir,
        manifest_path=cache_dir / "manifest.json",
        index_path=cache_dir / "nearest_neighbors.pkl",
    )


def normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings / norms


def embedding_cache_manifest_matches(
    manifest: EmbeddingCacheManifest,
    ds: Convo,
    row_count: int,
    *,
    embedding_model_name: str = DEFAULT_EMBEDDING_MODEL_NAME,
    embedding_source_column: str = "messages",
) -> bool:
    return (
        manifest.dataset_name == ds.source_name
        and manifest.row_count == row_count
        and manifest.embedding_model_name == embedding_model_name
        and manifest.embedding_source_column == embedding_source_column
    )


def knn_index_cache_manifest_matches(
    manifest: KNNIndexCacheManifest,
    ds: Convo,
    row_count: int,
    knn_k: int,
    *,
    embedding_model_name: str = DEFAULT_EMBEDDING_MODEL_NAME,
    embedding_source_column: str = "messages",
) -> bool:
    return (
        manifest.dataset_name == ds.source_name
        and manifest.row_count == row_count
        and manifest.embedding_model_name == embedding_model_name
        and manifest.embedding_source_column == embedding_source_column
        and manifest.knn_k == knn_k
    )


def batch_embedding(
    ds: Convo,
    embedding_model: Any,
    embedding_source_column: str = "messages",
) -> np.ndarray:
    print("Computing embeddings for dataset...")
    source_dataset = ds.unwrap()
    if embedding_source_column == "messages":
        embedding_inputs = [
            "\n\n".join(
                message["content"]
                for message in require_openai_messages(messages_raw)
            )
            for messages_raw in source_dataset["messages"]
        ]
    else:
        embedding_inputs = [str(value) for value in source_dataset[embedding_source_column]]
    max_model_len = embedding_model_max_len(embedding_model)
    outputs = embedding_model.embed(
        embedding_inputs,
        use_tqdm=True,
        tokenization_kwargs={"truncate_prompt_tokens": max_model_len - 20},
    )
    print("Finished computing embeddings.")
    return np.asarray(
        [output.outputs.embedding for output in outputs],
        dtype=np.float32,
    )


def load_or_compute_normalized_embeddings(
    ds: Convo,
    cache_root_dir: Path,
    *,
    embedding_model_name: str = DEFAULT_EMBEDDING_MODEL_NAME,
    embedding_source_column: str = "messages",
    env_config: Any = None,
) -> np.ndarray:
    source_dataset = ds.unwrap()
    cache_paths = build_embedding_cache_paths(
        cache_root_dir,
        ds,
        embedding_model_name=embedding_model_name,
        embedding_source_column=embedding_source_column,
    )

    if cache_paths.manifest_path.exists() and cache_paths.embeddings_path.exists():
        manifest = EmbeddingCacheManifest(**json.loads(cache_paths.manifest_path.read_text()))
        if embedding_cache_manifest_matches(
            manifest,
            ds,
            len(source_dataset),
            embedding_model_name=embedding_model_name,
            embedding_source_column=embedding_source_column,
        ):
            print(f"Loading cached embeddings from {cache_paths.embeddings_path}...")
            normalized_embeddings = np.load(cache_paths.embeddings_path)
            return normalized_embeddings.astype(np.float32, copy=False)
        print(f"Embedding cache mismatch at {cache_paths.cache_dir}; recomputing embeddings...")

    embedding_model = build_vllm_embedding_model(
        embedding_model_name=embedding_model_name,
        env_config=env_config,
    )
    max_model_len = embedding_model_max_len(embedding_model)
    embeddings = batch_embedding(
        ds,
        embedding_model,
        embedding_source_column=embedding_source_column,
    )
    normalized_embeddings = normalize_embeddings(embeddings).astype(np.float32, copy=False)

    cache_paths.cache_dir.mkdir(parents=True, exist_ok=True)
    np.save(cache_paths.embeddings_path, normalized_embeddings)
    cache_paths.manifest_path.write_text(
        json.dumps(
            asdict(
                EmbeddingCacheManifest(
                    dataset_name=ds.source_name,
                    dataset_fingerprint=dataset_fingerprint(ds),
                    row_count=len(source_dataset),
                    embedding_model_name=embedding_model_name,
                    embedding_source_column=embedding_source_column,
                    max_model_len=max_model_len,
                    embedding_dim=int(normalized_embeddings.shape[1]),
                )
            ),
            indent=2,
        )
        + "\n"
    )
    print(f"Saved embedding cache to {cache_paths.cache_dir}.")
    return normalized_embeddings


def build_or_load_knn_index(
    ds: Convo,
    normalized_embeddings: np.ndarray,
    knn_k: int,
    *,
    cache_root_dir: Path | None = None,
    embedding_model_name: str = DEFAULT_EMBEDDING_MODEL_NAME,
    embedding_source_column: str = "messages",
) -> NearestNeighbors:
    if len(normalized_embeddings) == 0:
        raise ValueError("Cannot build a k-NN index for an empty embedding set.")

    if cache_root_dir is None:
        knn_index = NearestNeighbors(metric="euclidean", algorithm="brute")
        knn_index.fit(normalized_embeddings)
        return knn_index

    row_count = len(ds.unwrap())
    cache_paths = build_knn_index_cache_paths(
        cache_root_dir,
        ds,
        knn_k,
        embedding_model_name=embedding_model_name,
        embedding_source_column=embedding_source_column,
    )
    if cache_paths.manifest_path.exists() and cache_paths.index_path.exists():
        manifest = KNNIndexCacheManifest(**json.loads(cache_paths.manifest_path.read_text()))
        if knn_index_cache_manifest_matches(
            manifest,
            ds,
            row_count,
            knn_k,
            embedding_model_name=embedding_model_name,
            embedding_source_column=embedding_source_column,
        ):
            print(f"Loading cached k-NN index from {cache_paths.index_path}...")
            with cache_paths.index_path.open("rb") as handle:
                return pickle.load(handle)
        print(f"k-NN index cache mismatch at {cache_paths.cache_dir}; rebuilding index...")

    knn_index = NearestNeighbors(metric="euclidean", algorithm="brute")
    knn_index.fit(normalized_embeddings)
    print(f"k-NN requested algorithm=auto, chosen backend={knn_index._fit_method}")
    cache_paths.cache_dir.mkdir(parents=True, exist_ok=True)
    with cache_paths.index_path.open("wb") as handle:
        pickle.dump(knn_index, handle)
    cache_paths.manifest_path.write_text(
        json.dumps(
            asdict(
                KNNIndexCacheManifest(
                    dataset_name=ds.source_name,
                    row_count=row_count,
                    embedding_model_name=embedding_model_name,
                    embedding_source_column=embedding_source_column,
                    knn_k=knn_k,
                    dataset_fingerprint=dataset_fingerprint(ds),
                )
            ),
            indent=2,
        )
        + "\n"
    )
    print(f"Saved k-NN index cache to {cache_paths.cache_dir}.")
    return knn_index
