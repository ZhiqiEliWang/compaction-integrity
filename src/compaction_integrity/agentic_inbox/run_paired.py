"""Resume both continuations from shared inbox checkpoints (Hydra)."""
from copy import deepcopy
import csv
import hashlib
import json
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from compaction_integrity.agentic_inbox.runtime import build_runtime


CONDITIONS = ("compacted", "compacted_post_sc")


def run_pairs(cfg: DictConfig, runtime=None) -> dict:
    import pandas as pd

    from compaction_integrity.agentic_inbox import loop
    from compaction_integrity.agentic_inbox.analyze import judge_retention
    from compaction_integrity.agentic_inbox.session import SC_REPORT_TEXT, load_compaction_checkpoint, run_session
    from compaction_integrity.runtime.env import apply_runtime_environment

    apply_runtime_environment(cfg)
    loop.EMAIL_BODY_FORMAT = cfg.email_format
    root = Path(cfg.out_dir)
    root.mkdir(parents=True, exist_ok=True)
    source_root = Path(cfg.checkpoint_dir)
    sources = []
    for seed in range(cfg.seeds):
        path = source_root / f"seed{seed}.json"
        checkpoint, inbox = load_compaction_checkpoint(path)
        assert checkpoint.trace.seed == seed
        assert checkpoint.trace.compactor == cfg.compactor.name
        assert checkpoint.trace.sc_text == SC_REPORT_TEXT
        sources.append((path, checkpoint, inbox))

    # Judge each shared summary once, before either continuation restates the SC.
    judge_retention(pd.DataFrame({
        "summaries": [[checkpoint.trace.compaction_events[0]["summary"]] for _, checkpoint, _ in sources],
    }), Path(cfg.retention_cache), SC_REPORT_TEXT)
    cache = json.loads(Path(cfg.retention_cache).read_text())
    settings = cfg.session
    retained = 0
    deletion_sessions = {"before": 0, "after": 0, "reminder": 0}
    pairs = []
    for path, checkpoint, inbox in sources:
        source = json.loads(checkpoint.trace.to_json())
        traces = []
        for condition in CONDITIONS:
            output = root / f"{condition}__{source['compactor']}__seed{source['seed']}.jsonl"
            if output.exists() and not cfg.overwrite:
                result = json.loads(output.read_text())
                print(f"skip (exists) {output.name}", flush=True)
            else:
                if runtime is None:
                    runtime = build_runtime(cfg.agent)
                print(f"running {condition}: seed={source['seed']} next_turn={checkpoint.next_turn}", flush=True)
                trace = run_session(
                    runtime=runtime, inbox=inbox.model_copy(deep=True), condition=condition,
                    compactor=None, compactor_name=source["compactor"], seed=source["seed"],
                    sc_text=source["sc_text"], post_compaction_sc=condition == "compacted_post_sc",
                    threshold_tokens=settings.threshold_tokens,
                    max_context_tokens=settings.max_context_tokens, max_turns=settings.max_turns,
                    max_reminders=settings.max_reminders,
                    max_tool_result_tokens=settings.max_tool_result_tokens,
                    max_gen_tokens=settings.max_gen_tokens, resume_from=deepcopy(checkpoint),
                )
                result = json.loads(trace.to_json())
                result.update(paired_source_checkpoint=str(path), paired_shared_summary=True,
                              post_compaction_max_reminders=settings.max_reminders)
            # The same contract applies to generated and resumed files.
            assert result["post_compaction_max_reminders"] == settings.max_reminders
            assert result["compaction_events"] == source["compaction_events"]
            assert [c for c in result["tool_calls"] if c["n_compactions_so_far"] == 0] == source["tool_calls"]
            if not output.exists() or cfg.overwrite:
                output.write_text(json.dumps(result, ensure_ascii=False))
                print(f"finished {output.name}: stop={result['stop_reason']}", flush=True)
            traces.append(result)

        key = hashlib.sha256((source["sc_text"] + "\x00" + source["compaction_events"][0]["summary"]).encode()).hexdigest()
        retained += cache[key]
        row = {"seed": source["seed"], "retained": cache[key]}
        for name, trace, phase in zip(("before", "after", "reminder"), (source, *traces), (0, 1, 1)):
            row[name] = sum(c["name"] == "delete_email" and c["error"] is None
                            and c["n_compactions_so_far"] == phase for c in trace["tool_calls"])
            deletion_sessions[name] += row[name] > 0
        pairs.append(row)
    with (root / "paired_sessions.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(pairs[0]))
        writer.writeheader()
        writer.writerows(pairs)
    summary = {"sessions": len(sources), "retained_summaries": retained,
               "deletion_sessions": deletion_sessions, "shared_summaries": True,
               "source_dir": str(source_root), "post_compaction_max_reminders": settings.max_reminders}
    (root / "paired_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)
    return summary


@hydra.main(version_base=None, config_path="../../../config/tasks/agentic_inbox", config_name="config")
def main(cfg: DictConfig) -> None:
    root = Path(cfg.out_dir)
    root.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, root / "experiment_config.yaml", resolve=True)
    run_pairs(cfg)


if __name__ == "__main__":
    main()
