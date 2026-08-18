"""Shared helpers for run layout plus dataset/compactor label formatting."""

import ast
from contextlib import contextmanager
import hashlib
import json
import re
from pathlib import Path
import sys
from typing import Any

from omegaconf import DictConfig, OmegaConf
import pandas as pd


_DATASET_MANIFEST_IDENTITY_FIELDS = (
    "dataset",
    "test",
    "concat",
    "num_convo",
    "target_size",
    "stitching_method",
    "embedding_col",
    "knn_k",
)


def _sanitize_slug(value: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_")
    return sanitized or "run"


def _to_container(value: Any) -> Any:
    if OmegaConf.is_config(value):
        return OmegaConf.to_container(value, resolve=True)
    return value


def _content_hash(value: Any) -> str:
    serialized = json.dumps(value, ensure_ascii=True, sort_keys=True)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _load_dataset_manifest_identity(
    dataset_dir: str | Path,
) -> tuple[str, dict[str, Any]]:
    manifest_path = Path(dataset_dir) / "generation_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    identity = {
        field: manifest[field]
        for field in _DATASET_MANIFEST_IDENTITY_FIELDS
    }
    return str(manifest_path), identity


def build_run_spec(
    *,
    test: bool,
    dataset_name: str,
    dataset_dir: str | Path,
    num_rows: int | None,
    compactor_name: str,
    compactor_cfg: dict[str, Any] | DictConfig | None,
    evaluators_cfg: dict[str, Any] | DictConfig,
) -> dict[str, Any]:
    generation_manifest_path, generation_manifest_identity = _load_dataset_manifest_identity(
        dataset_dir
    )
    return {
        "task": "run_compaction",
        "test": test,
        "dataset": {
            "name": dataset_name,
            "dir": str(dataset_dir),
            "num_rows": num_rows,
            "generation_manifest_path": generation_manifest_path,
            "generation_manifest_hash": _content_hash(generation_manifest_identity),
            "generation_manifest_identity": generation_manifest_identity,
        },
        "compactor": {
            "name": compactor_name,
            "config": _to_container(compactor_cfg) or {},
        },
        "evaluators": _to_container(evaluators_cfg),
    }


def build_run_id(run_spec: dict[str, Any]) -> str:
    serialized = json.dumps(run_spec, ensure_ascii=True, sort_keys=True)
    short_hash = hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:10]
    dataset_name = _sanitize_slug(str(run_spec["dataset"]["name"]))
    compactor_name = _sanitize_slug(str(run_spec["compactor"]["name"]))
    evaluator_names = "_".join(
        _sanitize_slug(str(name)) for name in run_spec["evaluators"].keys()
    )
    num_rows = run_spec["dataset"]["num_rows"]
    row_tag = f"n{num_rows}" if num_rows is not None else "all"
    test_tag = "test" if bool(run_spec["test"]) else "full"
    return "__".join(
        [
            dataset_name,
            compactor_name,
            evaluator_names,
            row_tag,
            test_tag,
            short_hash,
        ]
    )


def get_run_dir(results_root: str | Path, run_id: str) -> Path:
    return Path(results_root) / "runs" / run_id


class _Tee:
    def __init__(self, *files: Any) -> None:
        self.files = files

    def write(self, data: str) -> int:
        for file in self.files:
            file.write(data)
        return len(data)

    def flush(self) -> None:
        for file in self.files:
            file.flush()


@contextmanager
def tee_stdout(log_path: str | Path) -> Any:
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    original_stdout = sys.stdout
    with log_path.open("w", encoding="utf-8") as log_file:
        sys.stdout = _Tee(original_stdout, log_file)
        try:
            yield
        finally:
            sys.stdout = original_stdout


def load_manifest_results(
    *,
    manifest_path: str | Path,
    results_root: str | Path,
    result_file_name: str = "evaluation_results.pkl",
) -> pd.DataFrame:
    manifest = OmegaConf.load(manifest_path)
    results_root = Path(results_root)
    experiment = str(manifest.get("experiment", ""))

    frames: list[pd.DataFrame] = []
    for run_entry in manifest.runs:
        run_id = str(run_entry.run_id)
        runs_dir = "runs_test" if experiment == "test" or "__test__" in run_id else "runs"
        run_dir = results_root / runs_dir / run_id
        df = pd.read_pickle(run_dir / result_file_name)
        df["_run_dir"] = str(run_dir)
        df["run_id"] = run_id
        df["experiment"] = experiment
        if "label" in run_entry:
            df["run_label"] = str(run_entry.label)
        frames.append(df)

    return pd.concat(frames, ignore_index=True)


def write_run_metadata(
    *,
    run_dir: str | Path,
    run_id: str,
    run_spec: dict[str, Any],
) -> None:
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "run_id": run_id,
        "run_spec": run_spec,
    }
    (run_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    OmegaConf.save(
        config=OmegaConf.create(run_spec),
        f=run_dir / "resolved_config.yaml",
    )


def extract_context_length(dataset_name: str) -> str:
    """Return the trailing context-length token, e.g. '10k' from 'hermes_cat_10k'."""
    return dataset_name.rsplit("_", 1)[-1]


def extract_dataset_config(dataset_name: str) -> str:
    """Strip the trailing context-length suffix, e.g. 'hermes_cat' from 'hermes_cat_10k'."""
    return dataset_name.rsplit("_", 1)[0]


def extract_compactor_name(compactor: object) -> str:
    """Return the compactor name string from a dict, serialised dict, or plain string."""
    if isinstance(compactor, dict):
        return str(compactor["name"])
    if isinstance(compactor, str) and compactor.startswith("{"):
        parsed = ast.literal_eval(compactor)
        return str(parsed["name"])
    return str(compactor)


_COMPACTOR_MODEL_LABELS: dict[str, str] = {
    "gpt_oss_120b": "gpt-oss",
    "qwen30b": "Qwen3",
    "gemma_4": "Gemma-4",
}

_COMPACTOR_PROMPT_LABELS: dict[str, str] = {
    "anthropic_sc_targeted": "Anthropic + SC target",
    "anthropic": "Anthropic",
    "pi_mono": "pi-mono",
}


COMPACTOR_NAME_ORDER: list[str] = [
    "recent_5",
    "llmlingua2_t500",
    "gpt_oss_120b_anthropic_prompt",
    "gpt_oss_120b_anthropic_sc_targeted_prompt",
    "gpt_oss_120b_pi_mono_prompt",
    "qwen30b_anthropic_prompt",
    "qwen30b_anthropic_sc_targeted_prompt",
]


def ordered_values(values: pd.Series | list[str], preferred_order: list[str]) -> list[str]:
    """Return values intersected with preferred_order first, then any extras sorted."""
    present = set(values)
    ordered = [v for v in preferred_order if v in present]
    return ordered + sorted(present - set(ordered))


def ordered_compactor_names(values: pd.Series | list[str]) -> list[str]:
    """Canonical order for compactor *raw names* (the keys used in configs)."""
    return ordered_values(values, COMPACTOR_NAME_ORDER)


def ordered_compactor_labels(df: pd.DataFrame) -> list[str]:
    """Canonical order for compactor *display labels*, derived from a df with
    both 'compactor_name' and 'compactor_name_label' columns."""
    label_by_name = (
        df[["compactor_name", "compactor_name_label"]]
        .drop_duplicates()
        .set_index("compactor_name")["compactor_name_label"]
        .to_dict()
    )
    ordered = [
        label_by_name[name] for name in COMPACTOR_NAME_ORDER if name in label_by_name
    ]
    extras = sorted(
        label for name, label in label_by_name.items() if name not in COMPACTOR_NAME_ORDER
    )
    return ordered + extras


def fmt_compactor_label(name: str) -> str:
    """Human-readable compactor label, combining model and prompt mappings.

    Strips a trailing '_prompt' suffix, then splits on a known prompt key to
    look up each part separately. Falls back to raw segment when label is unset.
    """
    if name.endswith("_prompt"):
        name = name[: -len("_prompt")]
    for prompt_key in _COMPACTOR_PROMPT_LABELS:
        if name.endswith(f"_{prompt_key}"):
            model_key = name[: -len(f"_{prompt_key}")]
            model_label = _COMPACTOR_MODEL_LABELS.get(model_key, "") or model_key
            prompt_label = _COMPACTOR_PROMPT_LABELS.get(prompt_key, "") or prompt_key
            return f"{model_label} ({prompt_label})"
    return name.replace("_", " ").title()


_DATASET_LABELS: dict[str, str] = {
    "hermes_cat": "HermesAgent",
    "wildchat_cat": "WildChat",
    "openresearcher_cat": "OpenResearcher",
}


def fmt_dataset_label(config: str) -> str:
    """Human-readable dataset label, looked up from _DATASET_LABELS. Falls back to raw config when label is unset."""
    label = _DATASET_LABELS.get(config, "")
    return label if label else config
