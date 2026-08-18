# Tokenizer for evaluating context length.
# Note that in evaluation, each model/agent use its own tokenizer.

import os
from functools import lru_cache
from typing import Any


_TOKENIZER_MODEL = os.getenv("COMPACTION_TOKENIZER_MODEL", "openai/gpt-oss-20b")

@lru_cache(maxsize=1)
def _get_hf_tokenizer() -> Any:
    try:
        from transformers import AutoTokenizer  # type: ignore

        return AutoTokenizer.from_pretrained(
            _TOKENIZER_MODEL,
            use_fast=True,
        )
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load tokenizer '{_TOKENIZER_MODEL}'. "
            "Install transformers and ensure tokenizer files are available."
        )


def _count_tokens_texts(texts: list[str]) -> list[int]:
    if not texts:
        return []
    tokenizer = _get_hf_tokenizer()
    encoded = tokenizer(
        texts,
        add_special_tokens=False,
        return_length=True,
        return_attention_mask=False,
        return_token_type_ids=False,
    )
    return [int(length) for length in encoded["length"]]


def count_tokens_text(text: str) -> int:
    return _count_tokens_texts([text])[0]


def count_tokens_messages_batch(messages_batch: list[list[dict[str, Any]]]) -> list[int]:
    """
    Count tokens for a list (batch) of messages. 
    """
    texts: list[str] = []
    text_counts_per_message_list: list[int] = []
    for messages in messages_batch:
        count = 0
        for message in messages:
            texts.append(str(message.get("role", message.get("from", ""))))
            texts.append(str(message.get("content", message.get("value", ""))))
            count += 2
        text_counts_per_message_list.append(count)

    text_lengths = _count_tokens_texts(texts)
    totals: list[int] = []
    offset = 0
    for messages, text_count in zip(messages_batch, text_counts_per_message_list):
        totals.append(sum(text_lengths[offset : offset + text_count]) + len(messages))
        offset += text_count
    return totals


def count_tokens_messages(messages: list[dict[str, Any]]) -> int:
    return count_tokens_messages_batch([messages])[0]
