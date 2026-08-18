import re
from functools import lru_cache
from typing import Any
from openai_harmony import HarmonyEncodingName, Role, TextContent, load_harmony_encoding
from vllm import LLM, SamplingParams

from compaction_integrity.runtime.env import apply_runtime_environment
from compaction_integrity.runtime.base import ModelResponse, ModelRuntime


@lru_cache(maxsize=1)
def _get_gpt_oss_harmony_encoding() -> Any:
    return load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)


class VLLMRuntime(ModelRuntime):
    def __init__(self, config: dict[str, Any]):
        super().__init__(config)
        if "model" not in config:
            raise ValueError("VLLMRuntime requires a 'model' in the config.")

        apply_runtime_environment(config)
        self.model = self._resolve_model_name(str(config["model"]))
        llm_kwargs = {
            key: value
            for key, value in config.items()
            if key not in {
                "model",
                "batch",
                "batch_size",
                "max_tokens",
                "retry_attempts",
                "retry_sleep_seconds",
                "use_tqdm",
                "env",
                "enable_thinking",
                "truncate_prompt_tokens",
            }
        }
        llm_kwargs.setdefault("enable_prefix_caching", True)
        self.llm = LLM(model=self.model, **llm_kwargs)

    @staticmethod
    def _resolve_model_name(model: str) -> str:
        if model.startswith("vllm/"):
            resolved = model.split("/", 1)[1]
            if not resolved:
                raise ValueError("Invalid model: missing vLLM model name after provider prefix.")
            return resolved
        return model

    def _resolve_request_model(self, model: str | None) -> str:
        if model is None:
            return self.model
        resolved_model = self._resolve_model_name(model)
        if resolved_model != self.model:
            raise ValueError(
                f"VLLMRuntime is bound to model {self.model}, got request for {resolved_model}."
            )
        return resolved_model

    def _default_chat_template_kwargs(self, model: str) -> dict[str, Any] | None:
        """Build per-request chat_template_kwargs.

        - gpt-oss models default to `reasoning_effort=low`.
        - Any model can disable thinking via `enable_thinking: false` in the
          runtime config. For Qwen3 family this short-circuits the
          `<think>...</think>` prelude. See
          https://docs.vllm.ai/en/latest/features/reasoning_outputs/#online-serving
        """
        kwargs: dict[str, Any] = {}
        if "gpt-oss" in model.lower():
            kwargs["reasoning_effort"] = "low"
        if "enable_thinking" in self.config:
            kwargs["enable_thinking"] = bool(self.config["enable_thinking"])
        return kwargs or None

    @staticmethod
    def _is_gpt_oss_model(model: str) -> bool:
        return "gpt-oss" in model.lower()

    @staticmethod
    def _trim_trailing_stop_tokens(token_ids: list[int]) -> list[int]:
        if not token_ids:
            return token_ids
        stop_tokens = set(_get_gpt_oss_harmony_encoding().stop_tokens_for_assistant_actions())
        trimmed = list(token_ids)
        while trimmed and trimmed[-1] in stop_tokens:
            trimmed.pop()
        return trimmed

    @staticmethod
    def _message_text(message: Any) -> str:
        return "".join(
            content.text
            for content in message.content
            if isinstance(content, TextContent)
        ).strip()

    @staticmethod
    def _truncate_text(text: str, limit: int = 400) -> str:
        if len(text) <= limit:
            return text
        return f"{text[:limit]}..."

    def _parse_gpt_oss_completion(
        self,
        token_ids: list[int] | None,
    ) -> list[Any]:
        if token_ids is None:
            raise RuntimeError("gpt-oss completion is missing token_ids for Harmony parsing.")
        trimmed_token_ids = self._trim_trailing_stop_tokens(token_ids)
        try:
            return _get_gpt_oss_harmony_encoding().parse_messages_from_completion_tokens(
                trimmed_token_ids,
                Role.ASSISTANT,
            )
        except Exception as exc:
            decoded_completion = _get_gpt_oss_harmony_encoding().decode(trimmed_token_ids)
            raise RuntimeError(
                "Harmony parsing failed for gpt-oss completion. "
                f"decoded_completion={self._truncate_text(decoded_completion)!r}"
            ) from exc

    @staticmethod
    def _regex_extract_harmony_channels(decoded: str) -> dict[str, str]:
        """Best-effort recovery for malformed harmony streams.

        Why: gpt-oss occasionally emits `<|start|>final<|message|>...` instead
        of `<|start|>assistant<|channel|>final<|message|>...`, which the
        harmony parser rejects. Falling back to regex lets evaluation
        continue instead of failing the whole batch.
        """
        channels: dict[str, list[str]] = {}
        pattern = re.compile(
            r"(?:<\|channel\|>|<\|start\|>)(\w+)<\|message\|>(.*?)"
            r"(?=<\|end\|>|<\|start\|>|<\|channel\|>|\Z)",
            re.DOTALL,
        )
        for label, text in pattern.findall(decoded):
            channels.setdefault(label, []).append(text.strip())
        return {label: "\n\n".join(t for t in texts if t) for label, texts in channels.items()}

    @staticmethod
    def _regex_extract_harmony_tool_calls(decoded: str) -> list[dict[str, str | None]]:
        tool_calls: list[dict[str, str | None]] = []
        pattern = re.compile(
            r"<\|channel\|>(\w+)[^<]*?\bto=([\w.-]+)[^<]*"
            r"(?:<\|constrain\|>[^<]*)?<\|message\|>(.*?)"
            r"(?=<\|end\|>|<\|start\|>|<\|channel\|>|\Z)",
            re.DOTALL,
        )
        for channel, recipient, arguments in pattern.findall(decoded):
            tool_calls.append(
                {
                    "channel": channel,
                    "recipient": recipient,
                    "name": (
                        recipient.removeprefix("functions.")
                        if recipient.startswith("functions.")
                        else recipient
                    ),
                    "arguments": arguments.strip(),
                    "content_type": None,
                }
            )
        return tool_calls

    def _extract_harmony_channel_text(
        self,
        messages: list[Any],
        channel: str,
    ) -> str:
        texts = [
            self._message_text(message)
            for message in messages
            if message.author.role == Role.ASSISTANT and message.channel == channel
        ]
        return "\n\n".join(text for text in texts if text).strip()

    def _extract_harmony_response_text(self, messages: list[Any]) -> str:
        """All assistant output the user/judge should see: every non-`analysis`
        channel (typically `final` and/or `commentary`), preserving emission order.
        gpt-oss may emit only a `commentary` tool-call when no `final` is produced;
        we surface that as the response so the IF judge can score it."""
        parts: list[str] = []
        for message in messages:
            if message.author.role != Role.ASSISTANT:
                continue
            if message.channel == "analysis":
                continue
            text = self._message_text(message)
            if not text:
                continue
            label = message.channel or "unknown"
            parts.append(f"[{label}]\n{text}")
        return "\n\n".join(parts).strip()

    def _format_harmony_messages(self, messages: list[Any]) -> str:
        parts: list[str] = []
        for message in messages:
            if message.author.role != Role.ASSISTANT:
                continue
            text = self._message_text(message)
            header = message.channel or "unknown"
            if message.recipient is not None:
                header += f" to={message.recipient}"
            parts.append(f"[{header}]\n{text}".rstrip())
        return "\n\n".join(parts).strip()

    def _extract_harmony_tool_calls(self, messages: list[Any]) -> list[dict[str, str | None]]:
        tool_calls: list[dict[str, str | None]] = []
        for message in messages:
            if message.author.role != Role.ASSISTANT:
                continue
            recipient = message.recipient
            if recipient is None:
                continue
            tool_calls.append(
                {
                    "channel": message.channel,
                    "recipient": recipient,
                    "name": (
                        recipient.removeprefix("functions.")
                        if recipient.startswith("functions.")
                        else recipient
                    ),
                    "arguments": self._message_text(message),
                    "content_type": message.content_type,
                }
            )
        return tool_calls

    @staticmethod
    def _extract_tagged_thinking(raw_output: str) -> tuple[str | None, str]:
        think_match = re.search(r"<think>(.*?)</think>", raw_output, re.DOTALL)
        if think_match is not None:
            thinking = think_match.group(1).strip() or None
            content = re.sub(r"<think>.*?</think>", "", raw_output, flags=re.DOTALL).strip()
            return thinking, content

        if "</think>" in raw_output: # qwen style? 
            thinking_text, content = raw_output.split("</think>", 1)
            thinking = thinking_text.strip() or None
            return thinking, content.strip()

        return None, raw_output

    def _to_model_response(
        self,
        request_output: Any,
        model: str | None = None,
    ) -> ModelResponse:
        if not getattr(request_output, "outputs", None):
            raise RuntimeError("vLLM returned no outputs.")

        completion = request_output.outputs[0]
        resolved_model = model or str(self.config["model"])
        raw_output = str(getattr(completion, "text", "") or "")
        usage: dict[str, int] = {}
        prompt_token_ids = getattr(request_output, "prompt_token_ids", None)
        if prompt_token_ids is not None:
            usage["prompt_tokens"] = len(prompt_token_ids)

        completion_token_ids = getattr(completion, "token_ids", None)
        if completion_token_ids is not None:
            usage["completion_tokens"] = len(completion_token_ids)

        if usage:
            usage["total_tokens"] = usage.get("prompt_tokens", 0) + usage.get("completion_tokens", 0)

        thinking: str | None = None
        content = raw_output
        parsed_messages: list[Any] | None = None
        tool_calls: list[dict[str, str | None]] = []
        if self._is_gpt_oss_model(resolved_model):
            try:
                parsed_messages = self._parse_gpt_oss_completion(completion_token_ids)
            except RuntimeError as parse_exc:
                decoded_completion = _get_gpt_oss_harmony_encoding().decode(
                    self._trim_trailing_stop_tokens(completion_token_ids or [])
                )
                channels = self._regex_extract_harmony_channels(decoded_completion)
                final_text = channels.get("final") or channels.get("commentary") or ""
                tool_calls = self._regex_extract_harmony_tool_calls(decoded_completion)
                print(
                    "[vllm_runtime] Harmony parse failed; recovered via regex fallback. "
                    f"resolved_model={resolved_model} error={parse_exc}",
                    flush=True,
                )
                thinking = channels.get("analysis") or None
                if final_text:
                    content = (
                        f"[final]\n{final_text}"
                        if "final" in channels
                        else f"[commentary]\n{final_text}"
                    )
                else:
                    raise RuntimeError(
                        "Harmony-recovered gpt-oss completion produced no non-analysis output."
                    ) from parse_exc
                return ModelResponse(
                    text=content,
                    model=resolved_model,
                    usage=usage,
                    raw={
                        "response": request_output,
                        "raw_output": raw_output,
                        "thinking": thinking,
                        "parsed_messages": None,
                        "tool_calls": tool_calls,
                        "full_output": decoded_completion,
                        "harmony_recovered": True,
                    },
                )
            thinking = self._extract_harmony_channel_text(parsed_messages, "analysis") or None
            content = self._extract_harmony_response_text(parsed_messages)
            tool_calls = self._extract_harmony_tool_calls(parsed_messages)
            full_output = self._format_harmony_messages(parsed_messages)
            if not content and tool_calls:
                content = "\n\n".join(
                    f"[{tool_call['channel'] or 'commentary'} to={tool_call['recipient']}]\n"
                    f"{tool_call['arguments'] or ''}".rstrip()
                    for tool_call in tool_calls
                )
            if not content:
                decoded_completion = _get_gpt_oss_harmony_encoding().decode(
                    self._trim_trailing_stop_tokens(completion_token_ids or [])
                )
                parsed_dump = [message.to_dict() for message in parsed_messages]
                print(
                    "[vllm_runtime] gpt-oss completion produced no non-analysis output.\n"
                    f"  resolved_model={resolved_model}\n"
                    f"  completion_token_count={len(completion_token_ids or [])}\n"
                    f"  finish_reason={getattr(completion, 'finish_reason', None)!r}\n"
                    f"  stop_reason={getattr(completion, 'stop_reason', None)!r}\n"
                    f"  parsed_messages={parsed_dump}\n"
                    f"  decoded_completion={decoded_completion!r}",
                    flush=True,
                )
                raise RuntimeError("Harmony-parsed gpt-oss completion produced no non-analysis output.")
        else:
            thinking, content = self._extract_tagged_thinking(raw_output)

        return ModelResponse(
            text=content,
            model=resolved_model,
            usage=usage,
            raw={
                "response": request_output,
                "raw_output": raw_output,
                "thinking": thinking,
                "tool_calls": tool_calls,
                "full_output": (
                    full_output
                    if parsed_messages is not None
                    else "\n\n".join(
                        part for part in (thinking, content) if part
                    )
                ),
                "parsed_messages": (
                    [message.to_dict() for message in parsed_messages]
                    if parsed_messages is not None
                    else None
                ),
            },
        )

    def _chat(
        self,
        batch_messages: list[list[dict[str, Any]]],
        sampling_params: SamplingParams | None,
        chat_template_kwargs: dict[str, Any] | None,
        tools: list[dict[str, Any]] | None,
        use_tqdm: bool,
    ) -> Any:
        tokenization_kwargs = None
        if "truncate_prompt_tokens" in self.config:
            tokenization_kwargs = {
                "truncate_prompt_tokens": int(self.config["truncate_prompt_tokens"])
            }
        return self.llm.chat(
            batch_messages,
            sampling_params=sampling_params,
            chat_template_kwargs=chat_template_kwargs,
            tools=tools,
            tokenization_kwargs=tokenization_kwargs,
            use_tqdm=use_tqdm,
        )

    def _generate_single_response(
        self,
        messages: list[dict[str, Any]],
        model: str,
        sampling_params: SamplingParams | None,
        chat_template_kwargs: dict[str, Any] | None,
        tools: list[dict[str, Any]] | None,
    ) -> ModelResponse:
        outputs = self._chat(
            [messages],
            sampling_params=sampling_params,
            chat_template_kwargs=chat_template_kwargs,
            tools=tools,
            use_tqdm=False,
        )
        if not outputs:
            raise RuntimeError("vLLM returned no outputs for single-request retry.")
        if len(outputs) != 1:
            raise RuntimeError(
                f"vLLM returned {len(outputs)} outputs for a single-request retry."
            )
        return self._to_model_response(outputs[0], model=model)

    def _build_sampling_params(
        self,
        model: str,
        params: dict[str, Any] | None = None,
    ) -> SamplingParams | None:
        sampling_config: dict[str, Any] = {}

        if self._is_gpt_oss_model(model):
            sampling_config["stop_token_ids"] = (
                _get_gpt_oss_harmony_encoding().stop_tokens_for_assistant_actions()
            )

        if "max_tokens" in self.config:
            sampling_config["max_tokens"] = int(self.config["max_tokens"])

        if params is not None:
            unsupported = set(params) - {"max_tokens", "tools"}
            if unsupported:
                raise ValueError(
                    f"VLLMRuntime received unsupported sampling params: {sorted(unsupported)}."
                )
            if "max_tokens" in params:
                sampling_config["max_tokens"] = int(params["max_tokens"])

        if not sampling_config:
            return None

        return SamplingParams(**sampling_config)

    def generate(
        self,
        messages: list[dict[str, Any]],
        model: str | None = None,
        params: dict[str, Any] | None = None,
    ) -> ModelResponse:
        responses = self.batch_generate([messages], model=model, params=params)
        if not responses:
            raise RuntimeError("vLLM returned no outputs.")
        return responses[0]

    def batch_generate(
        self,
        batch_messages: list[list[dict[str, Any]]],
        model: str | None = None,
        params: dict[str, Any] | None = None,
    ) -> list[ModelResponse]:
        if not batch_messages:
            return []

        resolved_model = self._resolve_request_model(model)
        normalized_batch_messages = [
            self._normalize_messages(messages) for messages in batch_messages
        ]
        chat_template_kwargs = self._default_chat_template_kwargs(resolved_model)
        sampling_params = self._build_sampling_params(resolved_model, params)
        tools = params.get("tools") if params is not None else None
        use_tqdm = bool(self.config.get("use_tqdm", False))

        outputs = self._chat(
            normalized_batch_messages,
            sampling_params=sampling_params,
            chat_template_kwargs=chat_template_kwargs,
            tools=tools,
            use_tqdm=use_tqdm,
        )
        if not outputs:
            raise RuntimeError("vLLM returned no outputs.")
        if len(outputs) != len(batch_messages):
            raise RuntimeError(
                f"vLLM returned {len(outputs)} outputs for {len(batch_messages)} requests."
            )

        responses: list[ModelResponse] = []
        for idx, request_output in enumerate(outputs):
            try:
                responses.append(self._to_model_response(request_output, model=resolved_model))
            except Exception as exc:
                if self._is_gpt_oss_model(resolved_model) and len(batch_messages) > 1:
                    try:
                        responses.append(
                            self._generate_single_response(
                                messages=normalized_batch_messages[idx],
                                model=resolved_model,
                                sampling_params=sampling_params,
                                chat_template_kwargs=chat_template_kwargs,
                                tools=tools,
                            )
                        )
                        continue
                    except Exception as retry_exc:
                        raise RuntimeError(
                            "vLLM response handling failed for batch index "
                            f"{idx}; single-request retry also failed. "
                            f"batch_error={exc} retry_error={retry_exc}"
                        ) from retry_exc
                raise RuntimeError(
                    f"vLLM response handling failed for batch index {idx}: {exc}"
                ) from exc
        return responses

    def close(self) -> None:
        self.llm.llm_engine.engine_core.shutdown()


def parse_output(text: str, role: str | None = None) -> dict[str, str]:
    """Parse formatted model output into a {channel: content} dict.

    Handles the [channel_name]\\ncontent format produced by _format_for_log.
    Unlabeled content (e.g. from non-gpt-oss models) is stored under 'text'.

    Args:
        text: Formatted output string, e.g. '[analysis]\\n...\\n\\n[final]\\nB'.
        role: If given, return only that channel's entry (empty string if absent).
    """
    result: dict[str, str] = {}
    if not text:
        return result
    parts = re.split(r"\[(\w+)\]\n", text.strip())
    # parts[0]: content before first label (may be empty for gpt-oss outputs)
    # parts[1::2]: labels; parts[2::2]: content after each label
    if parts[0].strip():
        result["text"] = parts[0].strip()
    for label, content in zip(parts[1::2], parts[2::2]):
        result[label] = content.strip()
    if role is not None:
        return {role: result.get(role, "")}
    return result
