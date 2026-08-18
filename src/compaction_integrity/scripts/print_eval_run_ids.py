"""Print the run_ids and result paths an evaluation config would generate.

Usage:
    python -m compaction_integrity.scripts.print_eval_run_ids \
        config/tasks/eval/wildchat100k-repeat3.yaml
"""

import argparse
from pathlib import Path

from omegaconf import OmegaConf

from compaction_integrity.scripts.eval_run_layout import (
    build_run_id,
    build_run_spec,
    normalize_compactors_container,
    normalize_probe_container,
    resolve_evaluator_cfg,
    runs_dir,
    shared_cache_dir,
    to_container,
)


def enumerate_outputs(cfg_path: Path) -> list[dict[str, str]]:
    cfg = OmegaConf.load(cfg_path)

    test = bool(cfg.test)
    global_seed = int(cfg.global_seed)
    results_root = Path(str(cfg.results_root))

    sssc_attrs = to_container(cfg.sssc_attrs)
    sssc_attrs["position"] = str(sssc_attrs["position"])
    sssc_attrs["repeat"] = int(sssc_attrs["repeat"])
    sssc_attrs["explicitness"] = bool(sssc_attrs["explicitness"])
    sssc_attrs["hard"] = bool(sssc_attrs["hard"])

    retention_rate_only = bool(cfg.get("retention_rate_only", False))

    compactors_container = normalize_compactors_container(cfg.compactors)
    if "probe" in cfg:
        probe_container = normalize_probe_container(cfg.probe)
    elif retention_rate_only:
        # Match evaluation.py: in retention-only mode the probe is unused but
        # we still need one iteration to drive the per-compactor loop and
        # produce stable run_ids.
        probe_container = {
            "no_probe": {"model": "n/a", "provider": "n/a", "kwargs": {}}
        }
    else:
        raise KeyError("cfg.probe is required unless retention_rate_only=True")
    evaluator_name, evaluator_cfg = resolve_evaluator_cfg(cfg.evaluators)

    runs_root = runs_dir(results_root, test)
    shared_root = shared_cache_dir(results_root, test)

    outputs: list[dict[str, str]] = []
    for dataset_name in cfg.datasets:
        dataset_cfg = cfg.datasets[dataset_name]
        dataset_dir = Path(str(dataset_cfg.dir))
        num_rows = None if dataset_cfg.num_rows is None else int(dataset_cfg.num_rows)

        for probe_name, probe_cfg in probe_container.items():
            shared_spec = build_run_spec(
                test=test,
                dataset_name=dataset_name,
                dataset_dir=dataset_dir,
                num_rows=num_rows,
                sssc_attrs=sssc_attrs,
                global_seed=global_seed,
                probe_name=probe_name,
                probe_cfg=probe_cfg,
                compactor_name=None,
                compactor_cfg=None,
                evaluator_name=None,
                evaluator_cfg=None,
            )
            shared_id = build_run_id(shared_spec)
            shared_path = shared_root / f"{shared_id}.pkl"

            for compactor_name, compactor_cfg in compactors_container.items():
                run_spec = build_run_spec(
                    test=test,
                    dataset_name=dataset_name,
                    dataset_dir=dataset_dir,
                    num_rows=num_rows,
                    sssc_attrs=sssc_attrs,
                    global_seed=global_seed,
                    probe_name=probe_name,
                    probe_cfg=probe_cfg,
                    compactor_name=compactor_name,
                    compactor_cfg=compactor_cfg,
                    evaluator_name=evaluator_name,
                    evaluator_cfg=evaluator_cfg,
                )
                run_id = build_run_id(run_spec)
                save_path = runs_root / run_id / "evaluation_results.pkl"
                outputs.append(
                    {
                        "dataset": dataset_name,
                        "probe": probe_name,
                        "compactor": compactor_name,
                        "shared_id": shared_id,
                        "shared_path": str(shared_path),
                        "run_id": run_id,
                        "evaluation_results_path": str(save_path),
                    }
                )
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path, help="Path to evaluation YAML config.")
    args = parser.parse_args()

    rows = enumerate_outputs(args.config)
    for row in rows:
        print(
            f"[{row['dataset']} | probe={row['probe']} | compactor={row['compactor']}]"
        )
        print(f"  run_id    : {row['run_id']}")
        print(f"  results   : {row['evaluation_results_path']}")


if __name__ == "__main__":
    main()
