import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path

from datasets import Dataset


DEFAULT_DATASET_ROOT = Path("/data/compaction_integrity/default_ds")
DEFAULT_EMBEDDING_CACHE_ROOT = Path(
    "/data/compaction_integrity/topic_cohesive_embedding_cache"
)
DEFAULT_DATASETS = ("hermes", "wildchat", "openresearcher")
DEFAULT_CONTEXT_LENGTHS = ("10k", "50k", "100k")
# COMPINT-SWE-Natural exists at one length only (n126 build), so it is listed as an
# explicit spec rather than joining the DEFAULT_DATASETS x DEFAULT_CONTEXT_LENGTHS grid.
DEFAULT_EXTRA_SPECS = (("swe_natural_sc", "100k_n126"),)
DEFAULT_CONTEXT_LENGTH_ORDER = {"10k": 10_000, "50k": 50_000, "100k": 100_000}
# Stitched datasets are saved as <dataset>_cat_<length>; the natural-SC datasets
# are not stitched and are saved as plain <dataset>_<length>.
DIRECTORY_TEMPLATES = ("{dataset}_cat_{context_length}", "{dataset}_{context_length}")
CONTEXT_LENGTH_PATTERN = re.compile(r"(\d+(?:\.\d+)?)k")


@dataclass(frozen=True, slots=True)
class DatasetStats:
    dataset: str
    context_length: str
    avg_tokens: float
    avg_turns: float
    avg_user_turns: float
    avg_tool_turns: float
    avg_source_rows: float


def _candidate_paths(
    dataset_root: Path,
    embedding_cache_root: Path,
    dataset: str,
    context_length: str,
) -> list[Path]:
    roots = (dataset_root, embedding_cache_root / "stitched_datasets")
    return [
        root / template.format(dataset=dataset, context_length=context_length)
        / "stitched_dataset"
        for template in DIRECTORY_TEMPLATES
        for root in roots
    ]


def _resolve_dataset_path(
    dataset_root: Path,
    embedding_cache_root: Path,
    dataset: str,
    context_length: str,
) -> Path:
    candidates = _candidate_paths(
        dataset_root,
        embedding_cache_root,
        dataset,
        context_length,
    )
    dataset_path = next(
        (path for path in candidates if _is_saved_dataset(path)),
        None,
    )
    if dataset_path is None:
        raise FileNotFoundError(
            f"No saved dataset for {dataset}:{context_length}; tried "
            + ", ".join(str(path) for path in candidates)
        )
    return dataset_path


def _load_dataset(dataset_path: Path, num_rows: int | None) -> Dataset:
    loaded_dataset = Dataset.load_from_disk(str(dataset_path))
    if num_rows is not None:
        loaded_dataset = loaded_dataset.select(
            list(range(min(num_rows, len(loaded_dataset))))
        )
    return loaded_dataset


def _tool_turns_folded_into_user(dataset_path: Path) -> bool:
    """Did the builder rewrite `tool` observations as `user` messages?

    dataset/swe_natural_curation/dataset.py build does this so the agent trajectories
    render under a plain-chat template, and records it in the manifest. Without
    undoing it here, avg_user_turns would report ~90 user turns for a SWE
    trajectory that has exactly one: the GitHub issue.
    """
    manifest_path = dataset_path.parent / "generation_manifest.json"
    if not manifest_path.exists():
        return False
    return _manifest_folds_tool(json.loads(manifest_path.read_text()))


def _manifest_folds_tool(manifest: dict) -> bool:
    # Merged builds (n126) record the normalization per part, not at the top.
    if manifest.get("role_normalization", {}).get("tool") == "user":
        return True
    return any(_manifest_folds_tool(part) for part in manifest.get("parts", []))


def _is_saved_dataset(path: Path) -> bool:
    return (path / "dataset_info.json").exists() and (path / "state.json").exists()


def _default_dataset_specs() -> list[tuple[str, str]]:
    return [
        (dataset, context_length)
        for dataset in DEFAULT_DATASETS
        for context_length in DEFAULT_CONTEXT_LENGTHS
    ] + list(DEFAULT_EXTRA_SPECS)


def _split_dataset_dir(name: str) -> tuple[str, str] | None:
    """Split a saved directory name into its (dataset, context_length) spec."""
    if "_cat_" in name:
        dataset, context_length = name.rsplit("_cat_", 1)
        return dataset, context_length
    dataset, _, context_length = name.rpartition("_")
    if dataset and CONTEXT_LENGTH_PATTERN.fullmatch(context_length):
        return dataset, context_length
    return None


def _context_length_sort_key(context_length: str) -> tuple[float, str]:
    if context_length in DEFAULT_CONTEXT_LENGTH_ORDER:
        return float(DEFAULT_CONTEXT_LENGTH_ORDER[context_length]), context_length
    match = CONTEXT_LENGTH_PATTERN.fullmatch(context_length)
    if match is None:
        return float("inf"), context_length
    return float(match.group(1)) * 1_000, context_length


def _discover_dataset_specs(
    dataset_root: Path,
    embedding_cache_root: Path,
) -> list[tuple[str, str]]:
    specs: list[tuple[str, str]] = []
    dataset_dirs = list(dataset_root.glob("*")) + list(
        (embedding_cache_root / "stitched_datasets").glob("*")
    )
    for dataset_dir in dataset_dirs:
        if not _is_saved_dataset(dataset_dir / "stitched_dataset"):
            continue
        spec = _split_dataset_dir(dataset_dir.name)
        if spec is not None:
            specs.append(spec)
    return sorted(
        set(specs),
        key=lambda item: (item[0], _context_length_sort_key(item[1])),
    )


def _summarize_dataset(
    dataset_root: Path,
    embedding_cache_root: Path,
    dataset: str,
    context_length: str,
    num_rows: int | None,
) -> DatasetStats:
    dataset_path = _resolve_dataset_path(
        dataset_root,
        embedding_cache_root,
        dataset,
        context_length,
    )
    loaded_dataset = _load_dataset(dataset_path, num_rows)
    folded = _tool_turns_folded_into_user(dataset_path)
    rows = len(loaded_dataset)
    token_lengths = loaded_dataset["token_length"]
    messages_rows = loaded_dataset["messages"]
    turn_counts = [len(messages) for messages in messages_rows]
    user_turn_counts: list[int] = []
    tool_turn_counts: list[int] = []
    for messages in messages_rows:
        users = sum(1 for message in messages if message["role"] == "user")
        tools = sum(1 for message in messages if message["role"] == "tool")
        if folded:
            # One row is one trajectory, and its only human turn is the issue at
            # the head; every later `user` message is a folded tool observation.
            tools += max(users - 1, 0)
            users = min(users, 1)
        user_turn_counts.append(users)
        tool_turn_counts.append(tools)
    source_row_counts = loaded_dataset["source_conversation_count"]

    return DatasetStats(
        dataset=dataset,
        context_length=context_length,
        avg_tokens=sum(token_lengths) / rows,
        avg_turns=sum(turn_counts) / rows,
        avg_user_turns=sum(user_turn_counts) / rows,
        avg_tool_turns=sum(tool_turn_counts) / rows,
        avg_source_rows=sum(source_row_counts) / rows,
    )


def _format_table(stats: list[DatasetStats]) -> str:
    headers = (
        "dataset",
        "context_length",
        "avg_tokens",
        "avg_turns",
        "avg_user_turns",
        "avg_tool_turns",
        "avg_source_rows",
    )
    rows = [
        (
            item.dataset,
            item.context_length,
            f"{item.avg_tokens:.2f}",
            f"{item.avg_turns:.2f}",
            f"{item.avg_user_turns:.2f}",
            f"{item.avg_tool_turns:.2f}",
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
        description="List basic statistics for the default eval datasets."
    )
    parser.add_argument(
        "--dataset_root",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
        help="Root directory containing <dataset>[_cat]_<length>/stitched_dataset artifacts.",
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
        help="Optional explicit specs like openresearcher:10k swe_natural_sc:100k_n126. Defaults to hermes/wildchat/openresearcher x 10k/50k/100k plus swe_natural_sc:100k_n126.",
    )
    parser.add_argument(
        "--discover",
        action="store_true",
        help="Discover valid saved datasets instead of using the default grid.",
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
