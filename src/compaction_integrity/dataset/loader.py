import json
import zipfile
from dataclasses import dataclass
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from datasets import Dataset, load_dataset
from huggingface_hub import hf_hub_download, list_repo_files

from compaction_integrity.dataset.normalizer import normalize_to_openai
from compaction_integrity.tokenization import count_tokens_messages_batch

_OPENRESEARCHER_CONVERTED_PARQUET_REVISION = "refs/convert/parquet"


@dataclass(frozen=True)
class DatasetSource:
    name: str
    hf_path: str
    cache_dir: Path
    split: str
    config_name: str | None = None
    subset_names: tuple[str, ...] | None = None

@dataclass(frozen=True)
class Convo:
    """
    Wrapper for a messages dataset with associated metadata.
    dataset: Hugging Face Dataset with
        - a 'messages' column containing normalized messages in OpenAI chat format.
        - a 'token_length' column containing the token count of each row.
    """
    dataset: Dataset
    source_name: str

    @classmethod
    def load(cls, ds_name: str, preserve_columns: list[str] | None = None) -> "Convo":
        source_name = ds_name
        if source_name not in DATASETS:
            supported = ", ".join(sorted(DATASETS))
            raise ValueError(
                f"Unsupported dataset source: {source_name}. Supported sources: {supported}."
            )
        ds = _load_source_dataset(source_name)

        if source_name == "wildchat" or source_name == "lmsyschat":
            ds = ds.filter(lambda x: x["language"] == "English")

        ds = ds.map(
            lambda x: normalize_to_openai(
                x,
                source_name=source_name,
                preserve_columns=preserve_columns,
            ),
            remove_columns=ds.column_names,
        )
        ds = _append_token_lengths(ds)
        if not _has_nonempty_messages(ds):
            raise ValueError(
                f"Normalized {source_name} has only empty messages. "
                "Check the source schema and restart the kernel to reload local normalizer code."
            )
        return cls(dataset=ds, source_name=source_name)

    def token_lengths(self) -> list[int]:
        return list(self.dataset["token_length"])

    def unwrap(self) -> Dataset:
        return self.dataset


DATASETS: dict[str, DatasetSource] = {
    "sharegpt": DatasetSource(
        name="sharegpt",
        hf_path="liyucheng/ShareGPT90K",
        cache_dir=Path("/data/hf_dataset_cache/sharegpt"),
        split="train",
    ),
    "wildchat": DatasetSource(
        name="wildchat",
        hf_path="allenai/WildChat",
        cache_dir=Path("/data/hf_dataset_cache/WildChat"),
        split="train",
    ),
    "ultrachat": DatasetSource(
        name="ultrachat",
        hf_path="HuggingFaceH4/ultrachat_200k",
        cache_dir=Path("/data/hf_dataset_cache/ultrachat"),
        split="train_sft",
    ),
    "mrcr": DatasetSource(
        name="mrcr",
        hf_path="openai/mrcr",
        cache_dir=Path("/data/hf_dataset_cache/mrcr"),
        split="train",
    ),
    "lmsyschat": DatasetSource(
        name="lmsys-chat-1m",
        hf_path="lmsys/lmsys-chat-1m",
        cache_dir=Path("/data/hf_dataset_cache/lmsys-chat-1m"),
        split="train",
    ),
    "hermes": DatasetSource(
        name="hermes",
        hf_path="lambda/hermes-agent-reasoning-traces",
        cache_dir=Path("/data/hf_dataset_cache/hermes-agent-reasoning-traces"),
        config_name="glm-5.1",
        split="train",
    ),
    "longbench": DatasetSource(
        name="longbench",
        hf_path="THUDM/LongBench",
        cache_dir=Path("/data/hf_dataset_cache/longbench"),
        split="test",
        subset_names=("gov_report", "qmsum", "multi_news"),
    ),
    "openresearcher": DatasetSource(
        name="openresearcher",
        hf_path="OpenResearcher/OpenResearcher-Dataset",
        cache_dir=Path("/data/hf_dataset_cache/openresearcher"),
        config_name="seed_42",
        split="train",
    ),
}


def _load_source_dataset(ds_name: str) -> Dataset:
    ds_cfg = DATASETS[ds_name]
    if ds_name == "longbench":
        return _load_longbench_dataset(ds_cfg)
    if ds_name == "openresearcher":
        return _load_openresearcher_dataset(ds_cfg)
    try:
        return load_dataset(
            ds_cfg.hf_path,
            ds_cfg.config_name,
            split=ds_cfg.split,
            cache_dir=str(ds_cfg.cache_dir),
        )
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load source dataset for {ds_name} from {ds_cfg.hf_path} "
            f"using cache_dir={ds_cfg.cache_dir}."
        ) from exc


def _load_longbench_dataset(ds_cfg: DatasetSource) -> Dataset:
    archive_path = hf_hub_download(
        ds_cfg.hf_path,
        filename="data.zip",
        repo_type="dataset",
        cache_dir=str(ds_cfg.cache_dir),
    )
    rows: list[dict[str, object]] = []
    with zipfile.ZipFile(archive_path) as archive:
        for subset_name in ds_cfg.subset_names or ():
            member_name = f"data/{subset_name}.jsonl"
            with archive.open(member_name) as source_file:
                for raw_line in source_file:
                    rows.append(json.loads(raw_line))
    return Dataset.from_list(rows)


def _load_openresearcher_dataset(ds_cfg: DatasetSource) -> Dataset:
    shard_prefix = f"{ds_cfg.config_name}/{ds_cfg.split}/"
    shard_names = sorted(
        filename
        for filename in list_repo_files(
            ds_cfg.hf_path,
            repo_type="dataset",
            revision=_OPENRESEARCHER_CONVERTED_PARQUET_REVISION,
        )
        if filename.startswith(shard_prefix) and filename.endswith(".parquet")
    )
    tables = []
    for shard_name in shard_names:
        shard_path = hf_hub_download(
            ds_cfg.hf_path,
            filename=shard_name,
            repo_type="dataset",
            revision=_OPENRESEARCHER_CONVERTED_PARQUET_REVISION,
            cache_dir=str(ds_cfg.cache_dir),
        )
        tables.append(pq.read_table(shard_path, columns=["messages"]))
    return Dataset(pa.concat_tables(tables))


def _append_token_lengths(ds: Dataset) -> Dataset:
    if "token_length" in ds.column_names:
        return ds

    def _token_length_batch(batch: dict[str, list[object]]) -> dict[str, list[int]]:
        messages_batch = batch.get("messages")
        if not isinstance(messages_batch, list):
            raise TypeError(
                "Each normalized dataset batch must contain a list-valued 'messages' field."
            )
        for messages in messages_batch:
            if not isinstance(messages, list):
                raise TypeError(
                    "Each normalized dataset row must contain a list-valued 'messages' field."
                )
        return {"token_length": count_tokens_messages_batch(messages_batch)}

    return ds.map(_token_length_batch, batched=True)


def _has_nonempty_messages(ds: Dataset, probe_size: int = 64) -> bool:
    sample_size = min(int(probe_size), len(ds))
    for i in range(sample_size):
        messages = ds[i].get("messages", [])
        if isinstance(messages, list) and len(messages) > 0:
            return True
    return False



def load_conversations(
    ds_name: str,
    preserve_columns: list[str] | None = None,
) -> Convo:
    return Convo.load(ds_name, preserve_columns=preserve_columns)
