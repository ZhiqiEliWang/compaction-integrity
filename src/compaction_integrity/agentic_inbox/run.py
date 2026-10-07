"""Collect shared inbox checkpoints, then run both paired continuations (Hydra)."""
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from compaction_integrity.agentic_inbox.run_paired import run_pairs
from compaction_integrity.agentic_inbox.runtime import build_compactor, build_runtime
from compaction_integrity.runtime.env import apply_runtime_environment


@hydra.main(version_base=None, config_path="../../../config/tasks/agentic_inbox", config_name="config")
def main(cfg: DictConfig) -> None:
    from compaction_integrity.agentic_inbox import loop
    from compaction_integrity.agentic_inbox.loop import TriageInbox
    from compaction_integrity.agentic_inbox.session import SC_REPORT_TEXT, run_session

    apply_runtime_environment(cfg)
    loop.EMAIL_BODY_FORMAT = cfg.email_format
    root = Path(cfg.out_dir)
    root.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, root / "experiment_config.yaml", resolve=True)
    inbox_raw = Path(cfg.inbox_path).read_text()
    settings = cfg.session
    runtime = compactor = None
    print(f"SC: {SC_REPORT_TEXT}; continue_reply={settings.continue_reply!r}; threshold={settings.threshold_tokens}; max_reminders={settings.max_reminders}", flush=True)

    for seed in range(cfg.seeds):
        checkpoint_path = Path(cfg.checkpoint_dir) / f"seed{seed}.json"
        if checkpoint_path.exists() and not cfg.overwrite:
            print(f"skip (exists) {checkpoint_path.name}", flush=True)
            continue
        if runtime is None:
            runtime = build_runtime(cfg.agent)
            compactor = build_compactor(cfg.compactor, runtime, cfg.agent)
        while True:
            trace = run_session(
                runtime=runtime, inbox=TriageInbox.model_validate_json(inbox_raw),
                condition="before_compaction", compactor=compactor,
                compactor_name=cfg.compactor.name, seed=seed, sc_text=SC_REPORT_TEXT,
                post_compaction_sc=False, continue_reply=settings.continue_reply,
                threshold_tokens=settings.threshold_tokens,
                max_context_tokens=settings.max_context_tokens, max_turns=settings.max_turns,
                max_reminders=settings.max_reminders,
                max_tool_result_tokens=settings.max_tool_result_tokens,
                max_gen_tokens=settings.max_gen_tokens, checkpoint_path=checkpoint_path,
            )
            if trace.compaction_events:
                break
            attempts = root / "uncompacted_attempts" / f"seed{seed}"
            attempts.mkdir(parents=True, exist_ok=True)
            attempt = len(list(attempts.glob("*.jsonl")))
            (attempts / f"attempt{attempt}.jsonl").write_text(trace.to_json())
            print(f"rerun (no compaction) seed={seed}: stop={trace.stop_reason}", flush=True)
        print(f"saved {checkpoint_path.name}: turn={trace.scoring_origin_turn}", flush=True)

    run_pairs(cfg, runtime)
    print(f"Traces written to {root}", flush=True)


if __name__ == "__main__":
    main()
