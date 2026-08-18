"""Run-id construction and on-disk layout for the evaluation pipeline.

Shared between `evaluation.py` (writes results) and `print_eval_run_ids.py`
(enumerates expected run_ids/paths from a config without running anything).

Schemas here are eval-specific and intentionally distinct from
`compaction_integrity.analyze.utils` (which serves the run_compaction task).
"""

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf


_DATASET_MANIFEST_IDENTITY_FIELDS = (
    "dataset",
    "test",
    "num_convo",
    "target_size",
    "stitching_method",
)


def to_container(value: Any) -> Any:
    if OmegaConf.is_config(value):
        return OmegaConf.to_container(value, resolve=True)
    return value


def _sanitize_slug(value: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_")
    return sanitized or "run"


def _content_hash(value: Any) -> str:
    serialized = json.dumps(value, ensure_ascii=True, sort_keys=True)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _load_dataset_manifest_identity(dataset_dir: str | Path) -> tuple[str, dict[str, Any]]:
    manifest_path = Path(dataset_dir) / "generation_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    identity = {field: manifest[field] for field in _DATASET_MANIFEST_IDENTITY_FIELDS}
    return str(manifest_path), identity


def normalize_compactors_container(compactors_cfg: Any) -> dict[str, Any]:
    container = to_container(compactors_cfg)
    if isinstance(container, str):
        return {container: None}
    if isinstance(container, list):
        return {str(name): None for name in container}
    if isinstance(container, dict):
        return container
    raise TypeError(f"Unsupported compactors config: {type(container)!r}")


def normalize_probe_container(probe_cfg: Any) -> dict[str, Any]:
    container = to_container(probe_cfg)
    if isinstance(container, str):
        return {container: {"model": container}}
    if isinstance(container, dict):
        if "model" in container:
            return {str(container["model"]): container}
        return container
    raise TypeError(f"Unsupported probe config: {type(container)!r}")


def resolve_evaluator_cfg(evaluators_cfg: Any) -> tuple[str, dict[str, Any]]:
    container = to_container(evaluators_cfg)
    if len(container) > 1:
        raise NotImplementedError("Currently don't expect more than 1 evaluator.")
    name, cfg = next(iter(container.items()))
    if cfg["provider"] != "openai":
        raise AssertionError(f"Expected openai evaluator, got provider={cfg['provider']}")
    return name, cfg


def build_run_spec(
    *,
    test: bool,
    dataset_name: str,
    dataset_dir: str | Path,
    num_rows: int | None,
    sssc_attrs: dict[str, Any],
    global_seed: int,
    probe_name: str,
    probe_cfg: dict[str, Any],
    compactor_name: str | None,
    compactor_cfg: dict[str, Any] | None,
    evaluator_name: str | None,
    evaluator_cfg: dict[str, Any] | None,
) -> dict[str, Any]:
    manifest_path, manifest_identity = _load_dataset_manifest_identity(dataset_dir)
    spec: dict[str, Any] = {
        "task": "evaluation",
        "test": test,
        "global_seed": global_seed,
        "dataset": {
            "name": dataset_name,
            "dir": str(dataset_dir),
            "num_rows": num_rows,
            "generation_manifest_path": manifest_path,
            "generation_manifest_hash": _content_hash(manifest_identity),
            "generation_manifest_identity": manifest_identity,
        },
        "sssc_attrs": dict(sssc_attrs),
        "probe": {"name": probe_name, "config": to_container(probe_cfg) or {}},
    }
    if compactor_name is not None:
        spec["compactor"] = {
            "name": compactor_name,
            "config": to_container(compactor_cfg) or {},
        }
    if evaluator_name is not None:
        spec["evaluator"] = {
            "name": evaluator_name,
            "config": to_container(evaluator_cfg) or {},
        }
    return spec


def build_run_id(spec: dict[str, Any]) -> str:
    serialized = json.dumps(spec, ensure_ascii=True, sort_keys=True)
    short_hash = hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:10]
    parts = [_sanitize_slug(str(spec["dataset"]["name"]))]
    if "compactor" in spec:
        parts.append(_sanitize_slug(str(spec["compactor"]["name"])))
    parts.append(_sanitize_slug(str(spec["probe"]["name"])))
    if "evaluator" in spec:
        parts.append(_sanitize_slug(str(spec["evaluator"]["name"])))
    num_rows = spec["dataset"]["num_rows"]
    parts.append(f"n{num_rows}" if num_rows is not None else "all")
    parts.append("test" if bool(spec["test"]) else "full")
    parts.append(short_hash)
    return "__".join(parts)


def runs_dir(results_root: Path, test: bool) -> Path:
    return results_root / ("runs_test" if test else "runs")


def shared_cache_dir(results_root: Path, test: bool) -> Path:
    return results_root / ("_full_probe_cache_test" if test else "_full_probe_cache")


def write_run_metadata(run_dir: Path, run_id: str, spec: dict[str, Any]) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "metadata.json").write_text(
        json.dumps({"run_id": run_id, "run_spec": spec}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    OmegaConf.save(config=OmegaConf.create(spec), f=run_dir / "resolved_config.yaml")
