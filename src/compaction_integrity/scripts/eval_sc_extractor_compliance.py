"""Compliance of the SC-aware extractor condition (RQ4, Eq. 9) on cached RQ1 runs.

Adds a fifth condition to the four in evaluation.py:

  5. compacted_extracted - probe on C(H) ⊕ S_t: the run's cached compacted context,
                           followed by the extractor's registry S_t
                           (eval_sc_extractor.py) and the probe.

S_t sits where case 4 puts the SC (K_ub = C(H) ⊕ s): merged into the user turn
ahead of the probe, so cases 4 and 5 differ only in what is appended, the
verbatim SC or the extracted registry. An empty registry leaves the bare probe,
i.e. case 3's input, which is probed again rather than copied.

Cases 1-4 and the compactor's retention verdict are copied from the RQ1 run, the
registry's retention verdict from the extractor run, so one file per RQ1 run holds
every number ER needs.

Output: {results_root}/sc_extractor_compliance_runs/<rq1_run_id>/<extractor_run_id>/
    metadata.json, resolved_config.yaml, results.pkl
"""

from __future__ import annotations

import gc
from pathlib import Path
from typing import Any

import hydra
import pandas as pd
from omegaconf import DictConfig, OmegaConf
from tqdm.auto import tqdm

from compaction_integrity.dataset.eval_loader import EvalDatasetLoader
from compaction_integrity.prompts import get_sssc_evaluation_tool_message
from compaction_integrity.runtime.env import apply_runtime_environment
from compaction_integrity.sc_extractor import REGISTRY_HEADER, render_registry_turn
from compaction_integrity.eval_run_layout import to_container, write_run_metadata
from compaction_integrity.scripts.evaluation import (
    _build_probe_runtime,
    _format_for_log,
    _grade_letter,
    _retrieve_sys_prompt,
    parse_output,
)


REPO_ROOT = Path(__file__).resolve().parents[3]

_COPIED_COLUMNS = [
    "dataset",
    "source_row_index",
    "sssc_id",
    "sssc_type",
    "sssc_message",
    "compliant_letter",
    "compaction_status",
    "retention",
    "full_with_sssc_compliant",
    "full_without_sssc_compliant",
    "compacted_compliant",
    "compacted_post_sssc_compliant",
]
_COLUMNS = _COPIED_COLUMNS + [
    "compactor_name",
    "extractor_retention",
    "registry",
    "registry_text",
    "compacted_extracted_user_turn",
    "compacted_extracted_probe_prompt",
    "compacted_extracted_output",
    "compacted_extracted_compliant",
]

_CHUNK = 256


def _system_prompts(dataset: str, dataset_path: str, probe_model: str) -> dict[int, str]:
    """Per-row system prompt exactly as evaluation._build_pairs resolves it."""
    rows = EvalDatasetLoader.load(dataset_path=dataset_path, test_mode=False, num_rows=None).rows()
    natural_sc = rows[0].sssc is not None
    dataset_system = None if natural_sc else _retrieve_sys_prompt(dataset, probe_model)
    return {r.source_row_index: str(r.system_prompt or dataset_system) for r in rows}


def _build_rows(
    wide: pd.DataFrame,
    registry_df: pd.DataFrame,
) -> list[dict[str, Any]]:
    # One extractor run dir holds every framing it was run with (its run id omits
    # sssc_attrs), so match the RQ1 run's framing.
    attrs = wide["sssc_attrs"].iloc[0]
    by_pair = {
        (int(r["source_row_index"]), int(r["sssc_id"])): r
        for r in registry_df.to_dict(orient="records")
        if r["sssc_attrs"] == attrs
    }
    out: list[dict[str, Any]] = []
    for w in wide.to_dict(orient="records"):
        ext = by_pair[(int(w["source_row_index"]), int(w["sssc_id"]))]
        block = render_registry_turn(ext["registry"])
        probe = w["compacted_probe_prompt"]
        out.append({
            **{c: w[c] for c in _COPIED_COLUMNS},
            "compactor_name": w["compactor"]["name"],
            "extractor_retention": ext["retention"],
            "registry": ext["registry"],
            "registry_text": ext["registry_text"],
            "compacted_extracted_user_turn": block,
            "compacted_extracted_probe_prompt": (
                None if probe is None else (f"{block}\n\n{probe}" if block else probe)
            ),
            "compacted_extracted_output": None,
            "compacted_extracted_compliant": None,
            "_context": w["compacted_context"],
        })
    return out


def _persist(rows: list[dict[str, Any]], path: Path) -> None:
    pd.DataFrame(rows, columns=_COLUMNS).to_pickle(path)


@hydra.main(version_base=None, config_path="../../../config/tasks/sc_extractor", config_name=None)
def main(cfg: DictConfig) -> None:
    apply_runtime_environment()

    overwrite = bool(cfg.overwrite)
    results_root = Path(str(cfg.results_root))
    extractor_runs = to_container(cfg.extractor_runs)
    probe_name, probe_cfg = next(iter(to_container(cfg.probe).items()))
    probe_kwargs = dict(probe_cfg.get("kwargs", {}))
    probe_model = str(probe_cfg["model"])

    run_ids = [
        str(entry.run_id)
        for manifest in cfg.manifests
        for entry in OmegaConf.load(REPO_ROOT / str(manifest)).runs
    ]

    runtime = None
    system_cache: dict[str, dict[int, str]] = {}
    try:
        for run_id in run_ids:
            wide = pd.read_pickle(results_root / "runs" / run_id / "evaluation_results.pkl")
            dataset = str(wide["dataset"].iloc[0])
            extractor_run_id = str(extractor_runs[dataset])
            registry_df = pd.read_pickle(
                results_root / "sc_extractor_runs" / extractor_run_id / "judgment_results.pkl"
            )

            out_dir = results_root / "sc_extractor_compliance_runs" / run_id / extractor_run_id
            write_run_metadata(out_dir, f"{run_id}/{extractor_run_id}", {
                "task": "eval_sc_extractor_compliance",
                "rq1_run_id": run_id,
                "extractor_run_id": extractor_run_id,
                "probe": {"name": probe_name, "config": probe_cfg},
                "registry_header": REGISTRY_HEADER,
                "placement": "registry merged into the probe's user turn (case-4 position)",
            })
            save_path = out_dir / "results.pkl"
            if overwrite and save_path.exists():
                save_path.unlink()

            rows = _build_rows(wide, registry_df)
            if save_path.exists():
                done = {
                    (int(r["source_row_index"]), int(r["sssc_id"])): r
                    for r in pd.read_pickle(save_path).to_dict(orient="records")
                }
                for r in rows:
                    prev = done.get((int(r["source_row_index"]), int(r["sssc_id"])))
                    if prev is not None and prev["compacted_extracted_output"] is not None:
                        r["compacted_extracted_output"] = prev["compacted_extracted_output"]
                        r["compacted_extracted_compliant"] = prev["compacted_extracted_compliant"]

            pending = [
                r for r in rows
                if r["_context"] is not None and r["compacted_extracted_output"] is None
            ]
            print(f"[{run_id}] pending={len(pending)}/{len(rows)}")
            if not pending:
                _persist(rows, save_path)
                continue

            if dataset not in system_cache:
                system_cache[dataset] = _system_prompts(
                    dataset, str(wide["dataset_path"].iloc[0]), probe_model
                )
            systems = system_cache[dataset]
            if runtime is None:
                runtime = _build_probe_runtime(probe_model, str(probe_cfg["provider"]), probe_kwargs)

            bar = tqdm(total=len(pending), desc=f"Extractor probes {run_id}", unit="probe",
                       dynamic_ncols=True)
            for start in range(0, len(pending), _CHUNK):
                chunk = pending[start : start + _CHUNK]
                convos = [
                    [
                        {"role": "system", "content": systems[int(r["source_row_index"])]},
                        get_sssc_evaluation_tool_message(),
                        *r["_context"],
                        {"role": "user", "content": r["compacted_extracted_probe_prompt"]},
                    ]
                    for r in chunk
                ]
                for r, response in zip(chunk, runtime.batch_generate(convos)):
                    formatted = _format_for_log(response)
                    parsed = parse_output(formatted)
                    grade_text = parsed.get("final") or parsed.get("text") or ""
                    r["compacted_extracted_output"] = formatted
                    r["compacted_extracted_compliant"] = _grade_letter(
                        grade_text, str(r["compliant_letter"])
                    )
                bar.update(len(chunk))
                _persist(rows, save_path)
            bar.close()

            graded = pd.DataFrame(rows, columns=_COLUMNS)
            rate = lambda col: graded[col].dropna().astype(float).mean()  # noqa: E731
            print(
                f"[{run_id}] compliance: compacted={rate('compacted_compliant'):.3f} "
                f"extracted={rate('compacted_extracted_compliant'):.3f} "
                f"upper_bound={rate('compacted_post_sssc_compliant'):.3f}"
            )
    finally:
        if runtime is not None:
            runtime.close()
        gc.collect()


if __name__ == "__main__":
    main()
