from typing import Any

from compaction_integrity.compactors.base import CompactionResult, Compactor
from compaction_integrity.prompts import get_summarization_prompt
from compaction_integrity.runtime.base import ModelRuntime
from compaction_integrity.runtime.openai_runtime import OpenAIRuntime
from compaction_integrity.runtime.vllm_runtime import VLLMRuntime
from compaction_integrity.runtime.vllm_serve_runtime import VLLMServeRuntime
from compaction_integrity.tokenization import count_tokens_messages


_SUPPORTED_PROVIDERS = {"openai", "vllm", "vllm_serve"}


class LLMSummarizeCompactor(Compactor):
    def __init__(
        self,
        model: str,
        provider: str,
        prompt_template: str,
        runtime_kwargs: dict[str, Any] | None = None,
    ):
        normalized_provider = str(provider).strip().lower()
        if normalized_provider not in _SUPPORTED_PROVIDERS:
            raise ValueError(
                f"Invalid provider: {provider}. Only {sorted(_SUPPORTED_PROVIDERS)} are supported."
            )

        self.model = model
        self.provider = normalized_provider
        self.prompt_template = prompt_template
        self.runtime = self._build_runtime(runtime_kwargs or {})

    def _build_runtime(self, runtime_kwargs: dict[str, Any]) -> ModelRuntime:
        runtime_config = {"model": self.model, **runtime_kwargs}
        if self.provider == "vllm":
            return VLLMRuntime(config=runtime_config)
        if self.provider == "vllm_serve":
            return VLLMServeRuntime(config=runtime_config)
        if self.provider == "openai":
            return OpenAIRuntime(config=runtime_config)
        return OpenAIRuntime(config=runtime_config)

    def name(self) -> str:
        return f"llm_summarize_{self.provider}_{self.model}_{self.prompt_template}"

    def _build_compaction_result(
        self,
        source_messages: list[dict[str, Any]],
        summary_text: str,
        target_tokens: int | None,
    ) -> CompactionResult:
        tokens_before = count_tokens_messages(source_messages)
        compacted_messages = [{"role": "assistant", "content": summary_text}]
        tokens_after = count_tokens_messages(compacted_messages)
        compression_ratio = (tokens_after / tokens_before) if tokens_before else 1.0
        return CompactionResult(
            messages=compacted_messages,
            tokens_before=tokens_before,
            tokens_after=tokens_after,
            compression_ratio=compression_ratio,
            notes={
                "provider": self.provider,
                "model": self.model,
                "prompt_template": self.prompt_template,
                "target_tokens": target_tokens,
            },
        )

    def compact(
        self,
        messages: list[dict[str, Any]],
        target_tokens: int = None,
    ) -> CompactionResult:
        response = self.runtime.generate(
            messages=get_summarization_prompt(self.prompt_template, messages),
            model=self.model,
        )
        return self._build_compaction_result(
            source_messages=messages,
            summary_text=response.text,
            target_tokens=target_tokens,
        )

    def compact_batch(
        self,
        batch_messages: list[list[dict[str, Any]]],
        target_tokens: int = None,
    ) -> list[CompactionResult]:
        if not batch_messages:
            return []

        responses = self.runtime.batch_generate(
            batch_messages=[
                get_summarization_prompt(self.prompt_template, messages)
                for messages in batch_messages
            ],
            model=self.model,
        )
        if len(responses) != len(batch_messages):
            raise RuntimeError(
                "Runtime batch_generate returned a mismatched number of responses."
            )

        return [
            self._build_compaction_result(
                source_messages=messages,
                summary_text=response.text,
                target_tokens=target_tokens,
            )
            for messages, response in zip(batch_messages, responses)
        ]

    def close(self) -> None:
        self.runtime.close()
