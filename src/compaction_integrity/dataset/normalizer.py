import json
from collections.abc import Iterable, Iterator
from typing import Any


_USER_ROLES = {"human", "user"}
_SYSTEM_ROLES = {"system", "developer"}
_ASSISTANT_ROLES = {"assistant", "gpt", "bot", "model", "tool"}
_DEFAULT_SUMMARIZATION_PROMPT = "Summarize the following document."


def _map_message_role(role_key: Any) -> str:
    role = str(role_key).strip().lower()
    if role in _USER_ROLES:
        return "user"
    if role in _SYSTEM_ROLES:
        return "system"
    if role in _ASSISTANT_ROLES:
        return "assistant"
    return "assistant"


def _iter_columnar_turns(container: dict[str, Any]) -> Iterator[tuple[Any, Any]]:
    role_col = container.get("from")
    value_col = container.get("value")
    if isinstance(role_col, list) and isinstance(value_col, list):
        for role_key, content in zip(role_col, value_col):
            yield role_key, content
        return
    if role_col is not None or value_col is not None:
        yield role_col or "", value_col or ""


def _iter_list_turns(container: list[Any]) -> Iterator[tuple[Any, Any]]:
    for turn in container:
        if not isinstance(turn, dict):
            continue
        role_key = turn.get("role", turn.get("from", ""))
        content = turn.get("content", turn.get("value", ""))
        yield role_key, content


def _iter_sharegpt_turns(record: dict[str, Any]) -> Iterator[tuple[Any, Any]]:
    conv = record.get("conversations")
    if isinstance(conv, list):
        yield from _iter_list_turns(conv)
        return
    if isinstance(conv, dict):
        yield from _iter_columnar_turns(conv)
        return
    yield from _iter_columnar_turns(record)


def _iter_messages_turns(record: dict[str, Any]) -> Iterator[tuple[Any, Any]]:
    for key in ("messages", "conversation"):
        turns = record.get(key)
        if isinstance(turns, list):
            yield from _iter_list_turns(turns)
            return

    role_col = record.get("role")
    content_col = record.get("content")
    if isinstance(role_col, list) and isinstance(content_col, list):
        for role_key, content in zip(role_col, content_col):
            yield role_key, content
        return
    if role_col is not None or content_col is not None:
        yield role_col or "", content_col or ""


def _normalize_from_pairs(turn_pairs: Iterable[tuple[Any, Any]]) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    for role_key, content_raw in turn_pairs:
        content = str(content_raw).strip()
        if not content:
            continue
        messages.append({"role": _map_message_role(role_key), "content": content})
    return messages


def _normalize_sharegpt(record: dict[str, Any]) -> list[dict[str, str]]:
    return _normalize_from_pairs(_iter_sharegpt_turns(record))


def _normalize_wildchat(record: dict[str, Any]) -> list[dict[str, str]]:
    return _normalize_from_pairs(_iter_messages_turns(record))


def _normalize_ultrachat(record: dict[str, Any]) -> list[dict[str, str]]:
    return _normalize_from_pairs(_iter_messages_turns(record))


def _normalize_mrcr(record: dict[str, Any]) -> list[dict[str, str]]:
    prompt = record.get("prompt", "")
    if isinstance(prompt, str):
        try:
            turns = json.loads(prompt)
        except json.JSONDecodeError as exc:
            raise ValueError("MRCR prompt is not valid JSON.") from exc
    else:
        turns = prompt

    if isinstance(turns, list):
        messages = _normalize_from_pairs(_iter_list_turns(turns))
    elif isinstance(turns, dict):
        messages = _normalize_from_pairs(_iter_columnar_turns(turns))
    else:
        raise TypeError(f"MRCR prompt must decode to list/dict, got {type(turns).__name__}.")

    if not messages and str(prompt).strip():
        raise ValueError("MRCR prompt decoded but produced zero messages.")
    return messages


def _normalize_generic(record: dict[str, Any]) -> list[dict[str, str]]:
    sharegpt_like = _normalize_from_pairs(_iter_sharegpt_turns(record))
    if sharegpt_like:
        return sharegpt_like
    return _normalize_from_pairs(_iter_messages_turns(record))


def _normalize_memory_agent_bench_answer(answer_raw: Any) -> str:
    candidate = answer_raw
    if isinstance(answer_raw, list):
        if not answer_raw:
            raise ValueError("memory_agent_bench answer list is empty.")
        candidate = answer_raw[0]

    answer = str(candidate).strip()
    if not answer:
        raise ValueError("memory_agent_bench answer is empty.")
    return answer


def _normalize_memory_agent_bench(record: dict[str, Any]) -> list[dict[str, str]]:
    context_raw = record.get("context")
    questions = record.get("questions")
    answers = record.get("answers")

    context = str(context_raw).strip()
    if not context:
        raise ValueError("memory_agent_bench record is missing a non-empty context.")
    if not isinstance(questions, list):
        raise TypeError(
            f"memory_agent_bench questions must be a list, got {type(questions).__name__}."
        )
    if not isinstance(answers, list):
        raise TypeError(
            f"memory_agent_bench answers must be a list, got {type(answers).__name__}."
        )
    if len(questions) != len(answers):
        raise ValueError("memory_agent_bench questions and answers must have the same length.")

    messages = [{"role": "user", "content": f"Memorize the following content:\n{context}\n."}]
    for question_raw, answer_raw in zip(questions, answers):
        question = str(question_raw).strip()
        if not question:
            raise ValueError("memory_agent_bench question is empty.")
        messages.append({"role": "user", "content": question})
        messages.append(
            {"role": "assistant", "content": _normalize_memory_agent_bench_answer(answer_raw)}
        )

    return messages


def _normalize_longbench(record: dict[str, Any]) -> list[dict[str, str]]:
    input_text = str(record.get("input", "")).strip() or _DEFAULT_SUMMARIZATION_PROMPT
    context_text = str(record.get("context", "")).strip()
    answer_text = str(record["answers"][0]).strip()

    return [
        {"role": "user", "content": input_text},
        {"role": "user", "content": context_text},
        {"role": "assistant", "content": answer_text},
    ]


def _openresearcher_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    text_parts: list[str] = []
    for part in content:
        if not isinstance(part, dict):
            continue
        text = str(part.get("text") or "").strip()
        if text:
            text_parts.append(text)
    return "\n".join(text_parts)


def _iter_openresearcher_turns(record: dict[str, Any]) -> Iterator[tuple[Any, Any]]:
    for message in record["messages"]:
        yield message.get("role", ""), _openresearcher_text(message.get("content"))


def _normalize_openresearcher(record: dict[str, Any]) -> list[dict[str, str]]:
    return _normalize_from_pairs(_iter_openresearcher_turns(record))


def normalize_record(record: dict[str, Any], source_name: str = "") -> list[dict[str, str]]:
    source = source_name.lower()
    if source == "sharegpt":
        return _normalize_sharegpt(record)
    if source == "wildchat":
        return _normalize_wildchat(record)
    if source == "ultrachat":
        return _normalize_ultrachat(record)
    if source == "mrcr":
        return _normalize_mrcr(record)
    if source == "longbench":
        return _normalize_longbench(record)
    if source == "openresearcher":
        return _normalize_openresearcher(record)
    if source == "memory_agent_bench":
        return _normalize_memory_agent_bench(record)
    return _normalize_generic(record)


def normalize_to_openai(
    record: dict[str, Any],
    source_name: str = "",
    preserve_columns: list[str] | None = None,
) -> dict[str, Any]:
    """
    normalize to openai chat format: {"messages": [{"role": "user/assistant/system", "content": "..."}]}
    """
    normalized = {"messages": normalize_record(record, source_name=source_name)}
    if preserve_columns is None:
        return normalized
    for column_name in preserve_columns:
        normalized[column_name] = record[column_name]
    return normalized
