from dataclasses import dataclass
from typing import Any

from compaction_integrity.dataset.loader import Convo


Message = dict[str, str]
DEFAULT_CONTEXT_RADIUS = 3


@dataclass(frozen=True, slots=True)
class InjectionRow:
    messages: list[Message]
    token_length: int
    middle_user_index: int
    injection_position: int
    target_user_message: str
    local_context: list[Message]

    def to_dict(self) -> dict[str, Any]:
        return {
            "messages": self.messages,
            "token_length": self.token_length,
            "middle_user_index": self.middle_user_index,
            "injection_position": self.injection_position,
            "target_user_message": self.target_user_message,
            "local_context": self.local_context,
        }


def require_openai_messages(messages_raw: Any) -> list[Message]:
    return [
        {"role": message["role"], "content": message["content"]}
        for message in messages_raw
    ]


def get_user_message_indices(messages: list[Message]) -> list[int]:
    user_indices: list[int] = []
    for index, message in enumerate(messages):
        role = str(message.get("role", "")).strip().lower()
        if role == "user":
            user_indices.append(index)
    return user_indices


def required_user_turn_count(context_radius: int = DEFAULT_CONTEXT_RADIUS) -> int:
    return 2 * context_radius + 1


def has_centered_local_context(
    user_indices: list[int],
    context_radius: int = DEFAULT_CONTEXT_RADIUS,
) -> bool:
    return len(user_indices) >= required_user_turn_count(context_radius)


def build_local_context(
    messages: list[Message],
    user_indices: list[int],
    middle_user_offset: int,
    context_radius: int = DEFAULT_CONTEXT_RADIUS,
) -> list[Message]:
    start_index = user_indices[middle_user_offset - context_radius]
    end_user_index = user_indices[middle_user_offset + context_radius]

    end_index = end_user_index + 1
    if end_index < len(messages):
        next_role = str(messages[end_index].get("role", "")).strip().lower()
        if next_role == "assistant":
            end_index += 1

    return messages[start_index:end_index]


def build_injection_rows(
    ds: Convo,
    *,
    context_radius: int = DEFAULT_CONTEXT_RADIUS,
) -> list[InjectionRow]:
    injection_rows: list[InjectionRow] = []
    for row_index, row in enumerate(ds.unwrap()):
        messages = require_openai_messages(row.get("messages"))
        user_message_indices = get_user_message_indices(messages)
        if not user_message_indices:
            print(f"Skipping row {row_index}: no user messages found.")
            continue

        middle_user_offset = len(user_message_indices) // 2
        if not has_centered_local_context(
            user_message_indices,
            context_radius=context_radius,
        ):
            print(
                f"Skipping row {row_index}: need at least "
                f"{required_user_turn_count(context_radius)} user turns "
                "for centered local context."
            )
            continue

        middle_user_index = user_message_indices[middle_user_offset]
        target_user_message = str(messages[middle_user_index].get("content", "")).strip()
        if not target_user_message:
            raise ValueError(f"Row {row_index} has an empty middle user message.")

        local_context = build_local_context(
            messages,
            user_message_indices,
            middle_user_offset,
            context_radius=context_radius,
        )
        injection_rows.append(
            InjectionRow(
                messages=messages,
                token_length=int(row["token_length"]),
                middle_user_index=middle_user_index,
                injection_position=middle_user_index + 1,
                target_user_message=target_user_message,
                local_context=local_context,
            )
        )
    return injection_rows
