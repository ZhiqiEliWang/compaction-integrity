from typing import Any

from compaction_integrity.compactors.base import CompactionResult, Compactor
from compaction_integrity.tokenization import count_tokens_messages


class RecentNTurnsCompactor(Compactor):
    """Keep the last N messages. Each message counts as one turn."""

    def __init__(self, recent_n_turns: int):
        if recent_n_turns < 1:
            raise ValueError(f"recent_n_turns must be >= 1, got {recent_n_turns}")
        self.recent_n_turns = int(recent_n_turns)

    def name(self) -> str:
        return "recent_n"

    def compact(
        self,
        messages: list[dict[str, Any]],
        target_tokens: int = None,
    ) -> CompactionResult:
        tokens_before = count_tokens_messages(messages)
        compacted = messages[-self.recent_n_turns:]
        tokens_after = count_tokens_messages(compacted)
        ratio = (tokens_after / tokens_before) if tokens_before else 1.0
        return CompactionResult(
            messages=compacted,
            tokens_before=tokens_before,
            tokens_after=tokens_after,
            compression_ratio=ratio,
            notes={"target_tokens": target_tokens, "recent_n_turns": self.recent_n_turns},
        )

    def compact_batch(
        self,
        batch_messages: list[list[dict[str, Any]]],
        target_tokens: int = None,
    ) -> list[CompactionResult]:
        return [self.compact(messages, target_tokens) for messages in batch_messages]
