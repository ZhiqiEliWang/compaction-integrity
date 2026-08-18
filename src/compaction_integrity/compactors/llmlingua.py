from typing import Any

from llmlingua import PromptCompressor

from compaction_integrity.compactors.base import CompactionResult, Compactor
from compaction_integrity.tokenization import count_tokens_messages


_DEFAULT_MODEL_NAME = "microsoft/llmlingua-2-xlm-roberta-large-meetingbank"
_RUNNER_ONLY_KWARGS = {"batch", "batch_size", "retry_attempts", "retry_sleep_seconds"}


class LLMLingua2(Compactor):
    def __init__(
        self,
        model_name: str = _DEFAULT_MODEL_NAME,
        rate: float = 0.33,
        force_tokens: list[str] | None = None,
        compress_kwargs: dict[str, Any] | None = None,
        target_token: int | None = None,
    ) -> None:
        self.model_name = model_name
        self.rate = rate
        self.force_tokens = force_tokens or ["\n", "?"]
        self.compress_kwargs = compress_kwargs or {}
        self.target_token = target_token
        self.compressor = PromptCompressor(
            model_name=self.model_name,
            use_llmlingua2=True,
        )

    def name(self) -> str:
        return "llmlingua2"

    @staticmethod
    def _pre_process_messages(messages: list[dict[str, Any]]) -> list[str]:
        return [
            f"[{message['role']}]: {message['content']}"
            for message in messages
        ]

    def compact(
        self,
        messages: list[dict[str, Any]],
        target_tokens: int = None,
    ) -> CompactionResult:
        if target_tokens is not None:
            resolved_target = int(target_tokens)
        elif self.target_token is not None:
            resolved_target = int(self.target_token)
        else:
            resolved_target = -1
        compressed = self.compressor.compress_prompt(
            context=self._pre_process_messages(messages),
            rate=self.rate,
            target_token=resolved_target,
            force_tokens=self.force_tokens,
            **self.compress_kwargs,
        )
        compacted_messages = [
            {"role": "assistant", "content": compressed["compressed_prompt"]}
        ]
        tokens_before = count_tokens_messages(messages)
        tokens_after = count_tokens_messages(compacted_messages)
        compression_ratio = (tokens_after / tokens_before) if tokens_before else 1.0
        return CompactionResult(
            messages=compacted_messages,
            tokens_before=tokens_before,
            tokens_after=tokens_after,
            compression_ratio=compression_ratio,
            notes={
                "model_name": self.model_name,
                "target_tokens": target_tokens,
                "llmlingua_rate": self.rate,
                "force_tokens": self.force_tokens,
                "origin_tokens": compressed["origin_tokens"],
                "compressed_tokens": compressed["compressed_tokens"],
                "ratio": compressed["ratio"],
                "rate": compressed["rate"],
            },
        )

    def compact_batch(
        self,
        batch_messages: list[list[dict[str, Any]]],
        target_tokens: int = None,
    ) -> list[CompactionResult]:
        return [
            self.compact(messages=messages, target_tokens=target_tokens)
            for messages in batch_messages
        ]

    def close(self) -> None:
        return None


def build_llmlingua2_from_config(compactor_cfg: dict[str, Any] | None) -> LLMLingua2:
    cfg = compactor_cfg or {}
    kwargs = dict(cfg.get("kwargs", {}))
    compress_kwargs = {
        key: value
        for key, value in kwargs.items()
        if key not in _RUNNER_ONLY_KWARGS | {"rate", "force_tokens", "target_token"}
    }
    raw_model = str(cfg.get("model", _DEFAULT_MODEL_NAME))
    model_name = _DEFAULT_MODEL_NAME if raw_model == "llmlingua2" else raw_model
    target_token = cfg.get("target_token", kwargs.get("target_token"))
    return LLMLingua2(
        model_name=model_name,
        rate=float(kwargs.get("rate", 0.33)),
        force_tokens=list(kwargs.get("force_tokens", ["\n", "?"])),
        compress_kwargs=compress_kwargs,
        target_token=None if target_token is None else int(target_token),
    )
