from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class CompactionResult:
    messages: list[dict[str, Any]]
    tokens_before: int
    tokens_after: int
    compression_ratio: float
    notes: dict[str, Any] = field(default_factory=dict)


class Compactor(ABC):
    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def compact(
        self,
        messages: list[dict[str, Any]],
        target_tokens: int = None,
    ) -> CompactionResult:
        raise NotImplementedError

    @abstractmethod
    def compact_batch(
        self,
        batch_messages: list[list[dict[str, Any]]],
        target_tokens: int = None,
    ) -> list[CompactionResult]:
        raise NotImplementedError
