"""Runtime and compaction transforms for the inbox experiment."""
from typing import Any

from omegaconf import DictConfig, OmegaConf


def build_runtime(cfg: DictConfig) -> Any:
    from compaction_integrity.agentic_inbox.loop import ToolAwareVLLMRuntime
    from compaction_integrity.runtime.openai_runtime import OpenAIRuntime
    from compaction_integrity.runtime.vllm_serve_runtime import VLLMServeRuntime

    provider = {"vllm": ToolAwareVLLMRuntime, "openai": OpenAIRuntime, "vllm_serve": VLLMServeRuntime}
    return provider[cfg.provider](config={
        "model": cfg.model, **OmegaConf.to_container(cfg.kwargs), "use_tqdm": False,
    })


def build_compactor(cfg: DictConfig, runtime: Any, agent: DictConfig) -> Any:
    from compaction_integrity.compactors.llm_summarize import LLMSummarizeCompactor

    if cfg.model == "recent-n":
        from compaction_integrity.compactors.recent_n import RecentNTurnsCompactor
        return RecentNTurnsCompactor(recent_n_turns=cfg.recent_n_turns)
    if cfg.model == "llmlingua2":
        from compaction_integrity.compactors.llmlingua import build_llmlingua2_from_config
        return build_llmlingua2_from_config(OmegaConf.to_container(cfg))
    if cfg.model == agent.model and cfg.provider == agent.provider:
        # Reuse the resident model for summarization.
        compactor = LLMSummarizeCompactor.__new__(LLMSummarizeCompactor)
        compactor.model = cfg.model
        compactor.provider = cfg.provider
        compactor.prompt_template = cfg.prompt
        compactor.runtime = runtime
        return compactor
    return LLMSummarizeCompactor(
        model=cfg.model, provider=cfg.provider, prompt_template=cfg.prompt,
        runtime_kwargs={**OmegaConf.to_container(cfg.kwargs), "use_tqdm": False},
    )
