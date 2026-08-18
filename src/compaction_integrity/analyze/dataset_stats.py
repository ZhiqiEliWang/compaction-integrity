import argparse
from dataclasses import dataclass
from pathlib import Path

from datasets import Dataset


DEFAULT_DATASET_ROOT = Path("/data/compaction_integrity/default_ds")
DEFAULT_EMBEDDING_CACHE_ROOT = Path(
    "/data/compaction_integrity/topic_cohesive_embedding_cache"
)
DEFAULT_DATASETS = ("hermes", "wildchat", "openresearcher")
DEFAULT_CONTEXT_LENGTHS = ("10k", "50k", "100k")
DEFAULT_CONTEXT_LENGTH_ORDER = {"10k": 10_000, "50k": 50_000, "100k": 100_000}


@dataclass(frozen=True, slots=True)
class DatasetStats:
    dataset: str
    context_length: str
    avg_tokens: float
    avg_turns: float
    avg_user_turns: float
    avg_source_rows: float


def _load_dataset(
    dataset_root: Path,
    embedding_cache_root: Path,
    dataset: str,
    context_length: str,
    num_rows: int | None,
) -> Dataset:
    dataset_name = f"{dataset}_cat_{context_length}"
    dataset_path = dataset_root / dataset_name / "stitched_dataset"
    cache_dataset_path = (
        embedding_cache_root / "stitched_datasets" / dataset_name / "stitched_dataset"
    )
    if _is_saved_dataset(dataset_path):
        loaded_dataset = Dataset.load_from_disk(str(dataset_path))
    else:
        loaded_dataset = Dataset.load_from_disk(str(cache_dataset_path))
    if num_rows is not None:
        loaded_dataset = loaded_dataset.select(list(range(num_rows)))
    return loaded_dataset


def _is_saved_dataset(path: Path) -> bool:
    return (path / "dataset_info.json").exists() and (path / "state.json").exists()


def _default_dataset_specs() -> list[tuple[str, str]]:
    return [
        (dataset, context_length)
        for dataset in DEFAULT_DATASETS
        for context_length in DEFAULT_CONTEXT_LENGTHS
    ]


def _discover_dataset_specs(
    dataset_root: Path,
    embedding_cache_root: Path,
) -> list[tuple[str, str]]:
    specs: list[tuple[str, str]] = []
    dataset_dirs = list(dataset_root.glob("*_cat_*")) + list(
        (embedding_cache_root / "stitched_datasets").glob("*_cat_*")
    )
    for dataset_dir in dataset_dirs:
        dataset_path = dataset_dir / "stitched_dataset"
        if not _is_saved_dataset(dataset_path):
            continue
        dataset, context_length = dataset_dir.name.rsplit("_cat_", 1)
        specs.append((dataset, context_length))
    return sorted(
        set(specs),
        key=lambda item: (item[0], DEFAULT_CONTEXT_LENGTH_ORDER.get(item[1], item[1])),
    )


def _summarize_dataset(
    dataset_root: Path,
    embedding_cache_root: Path,
    dataset: str,
    context_length: str,
    num_rows: int | None,
) -> DatasetStats:
    loaded_dataset = _load_dataset(
        dataset_root,
        embedding_cache_root,
        dataset,
        context_length,
        num_rows,
    )
    rows = len(loaded_dataset)
    token_lengths = loaded_dataset["token_length"]
    messages_rows = loaded_dataset["messages"]
    turn_counts = [len(messages) for messages in messages_rows]
    user_turn_counts = [
        sum(1 for message in messages if message["role"] == "user")
        for messages in messages_rows
    ]
    source_row_counts = loaded_dataset["source_conversation_count"]

    return DatasetStats(
        dataset=dataset,
        context_length=context_length,
        avg_tokens=sum(token_lengths) / rows,
        avg_turns=sum(turn_counts) / rows,
        avg_user_turns=sum(user_turn_counts) / rows,
        avg_source_rows=sum(source_row_counts) / rows,
    )


def _format_table(stats: list[DatasetStats]) -> str:
    headers = (
        "dataset",
        "context_length",
        "avg_tokens",
        "avg_turns",
        "avg_user_turns",
        "avg_source_rows",
    )
    rows = [
        (
            item.dataset,
            item.context_length,
            f"{item.avg_tokens:.2f}",
            f"{item.avg_turns:.2f}",
            f"{item.avg_user_turns:.2f}",
            f"{item.avg_source_rows:.2f}",
        )
        for item in stats
    ]
    widths = [
        max(len(str(value)) for value in column)
        for column in zip(headers, *rows, strict=True)
    ]
    lines = [
        "  ".join(value.ljust(width) for value, width in zip(headers, widths, strict=True))
    ]
    lines.extend(
        "  ".join(value.ljust(width) for value, width in zip(row, widths, strict=True))
        for row in rows
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="List basic statistics for default stitched datasets."
    )
    parser.add_argument(
        "--dataset_root",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
        help="Root directory containing <dataset>_cat_<length>/stitched_dataset artifacts.",
    )
    parser.add_argument(
        "--embedding_cache_root",
        type=Path,
        default=DEFAULT_EMBEDDING_CACHE_ROOT,
        help="Root for topic-cohesive stitched dataset cache artifacts.",
    )
    parser.add_argument(
        "--dataset_specs",
        nargs="+",
        help="Optional explicit specs like openresearcher:10k hermes:50k. Defaults to hermes/wildchat/openresearcher x 10k/50k/100k.",
    )
    parser.add_argument(
        "--discover",
        action="store_true",
        help="Discover valid saved datasets instead of using the default 3 x 3 grid.",
    )
    parser.add_argument(
        "--num_rows",
        type=int,
        default=50,
        help="Number of leading rows to include per dataset. Use -1 for all rows.",
    )
    args = parser.parse_args()
    num_rows = None if args.num_rows == -1 else args.num_rows

    if args.dataset_specs is not None:
        dataset_specs = [tuple(spec.split(":", 1)) for spec in args.dataset_specs]
    elif args.discover:
        dataset_specs = _discover_dataset_specs(
            args.dataset_root,
            args.embedding_cache_root,
        )
    else:
        dataset_specs = _default_dataset_specs()

    stats = (
        _summarize_dataset(
            args.dataset_root,
            args.embedding_cache_root,
            dataset,
            context_length,
            num_rows,
        )
        for dataset, context_length in dataset_specs
    )
    print(_format_table(list(stats)))


if __name__ == "__main__":
    main()
