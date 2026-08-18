from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class ModelResponse:
    text: str
    model: str
    usage: dict[str, int] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict)


class ModelRuntime(ABC):
    def __init__(self, config: dict[str, Any]):
        self.config = config

    @staticmethod
    def _normalize_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """
        normalize messages to a standard format with "role" and "content" keys.
        """
        role_map = {
            "human": "user",
            "user": "user",
            "system": "system",
            "assistant": "assistant",
            "ai": "assistant",
        }
        normalized: list[dict[str, Any]] = []
        for idx, message in enumerate(messages):
            role = str(message.get("role", message.get("from", "user"))).lower()
            if "content" in message:
                content = message["content"]
            elif "value" in message:
                content = message["value"]
            else:
                raise ValueError(f"Message at index {idx} is missing content/value.")
            normalized.append({"role": role_map.get(role, role), "content": content})
        return normalized

    @abstractmethod
    def generate(
        self,
        messages: list[dict[str, Any]],
        model: str | None = None,
        params: dict[str, Any] | None = None,
    ) -> ModelResponse:
        raise NotImplementedError

    def batch_generate(
        self,
        batch_messages: list[list[dict[str, Any]]],
        model: str | None = None,
        params: dict[str, Any] | None = None,
    ) -> list[ModelResponse]:
        return [self.generate(messages, model=model, params=params) for messages in batch_messages]

    def close(self) -> None:
        return None
