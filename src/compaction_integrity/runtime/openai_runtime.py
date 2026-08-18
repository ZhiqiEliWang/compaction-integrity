import io
import json
import time
from typing import Any
from openai import OpenAI

from compaction_integrity.runtime.base import ModelResponse, ModelRuntime
from compaction_integrity.api_keys import openai_api_key


class OpenAIRuntime(ModelRuntime):
    def __init__(self, config: dict[str, Any]):
        super().__init__(config)
        api_key = str(config.get("api_key") or openai_api_key).strip()
        if not api_key:
            raise ValueError("OpenAI API key is required.")

        client_kwargs: dict[str, Any] = {"api_key": api_key}

        timeout_s = config.get("timeout_s")
        if timeout_s is not None:
            client_kwargs["timeout"] = float(timeout_s)

        self.client = OpenAI(**client_kwargs)

    @staticmethod
    def _serialize_usage(usage_obj: Any) -> dict[str, int]:
        if usage_obj is None:
            return {}
        return {
            "prompt_tokens": int(getattr(usage_obj, "input_tokens", 0)),
            "completion_tokens": int(getattr(usage_obj, "output_tokens", 0)),
            "total_tokens": int(getattr(usage_obj, "total_tokens", 0)),
        }

    @staticmethod
    def _serialize_response_metadata(response: Any) -> dict[str, Any]:
        if isinstance(response, dict):
            return response

        metadata: dict[str, Any] = {}
        for field_name in ("id", "model", "status", "created_at"):
            field_value = getattr(response, field_name, None)
            if field_value is not None:
                metadata[field_name] = field_value

        error = getattr(response, "error", None)
        if error is not None:
            if hasattr(error, "model_dump"):
                metadata["error"] = error.model_dump()
            else:
                metadata["error"] = str(error)

        incomplete_details = getattr(response, "incomplete_details", None)
        if incomplete_details is not None:
            if hasattr(incomplete_details, "model_dump"):
                metadata["incomplete_details"] = incomplete_details.model_dump()
            else:
                metadata["incomplete_details"] = str(incomplete_details)

        return metadata

    @staticmethod
    def _json_ready(value: Any) -> Any:
        if hasattr(value, "model_dump"):
            return value.model_dump()
        if isinstance(value, dict):
            return {str(k): OpenAIRuntime._json_ready(v) for k, v in value.items()}
        if isinstance(value, list):
            return [OpenAIRuntime._json_ready(v) for v in value]
        return value

    @staticmethod
    def _extract_reasoning_text(output_items: list[Any]) -> str | None:
        parts: list[str] = []
        for item in output_items:
            item_data = OpenAIRuntime._json_ready(item)
            if not isinstance(item_data, dict) or item_data.get("type") != "reasoning":
                continue
            for field_name in ("content", "summary"):
                for part in item_data.get(field_name) or []:
                    if isinstance(part, dict):
                        text = str(part.get("text") or "").strip()
                        if text:
                            parts.append(text)
        return "\n\n".join(parts) or None

    @staticmethod
    def _responses_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "name": tool["function"]["name"],
                "description": tool["function"]["description"],
                "parameters": tool["function"]["parameters"],
            }
            for tool in tools
        ]

    @staticmethod
    def _extract_tool_calls(output_items: list[Any]) -> list[dict[str, Any]]:
        tool_calls: list[dict[str, Any]] = []
        for item in output_items or []:
            item_data = OpenAIRuntime._json_ready(item)
            if not isinstance(item_data, dict) or item_data.get("type") != "function_call":
                continue
            name = str(item_data["name"])
            tool_calls.append(
                {
                    "channel": "commentary",
                    "recipient": f"functions.{name}",
                    "name": name,
                    "arguments": item_data.get("arguments"),
                    "call_id": item_data.get("call_id"),
                }
            )
        return tool_calls

    @staticmethod
    def _format_full_output(output_items: list[Any]) -> str:
        parts: list[str] = []
        reasoning = OpenAIRuntime._extract_reasoning_text(output_items)
        if reasoning:
            parts.append(f"[analysis]\n{reasoning}")
        for tool_call in OpenAIRuntime._extract_tool_calls(output_items):
            parts.append(
                f"[{tool_call['channel']} to={tool_call['recipient']}]\n"
                f"{tool_call['arguments'] or ''}".rstrip()
            )
        text = OpenAIRuntime._text_from_output_items(output_items)
        if text:
            parts.append(f"[final]\n{text}")
        return "\n\n".join(parts).strip()

    def _build_request_kwargs(
        self,
        messages: list[dict[str, Any]],
        resolved_model: str,
        params: dict[str, Any] | None,
        flex: bool,
    ) -> tuple[dict[str, Any], Any]:
        request_kwargs: dict[str, Any] = {
            "model": resolved_model,
            "input": self._normalize_messages(messages),
            "reasoning": {"effort": "low"},
        }

        if "max_tokens" in self.config:
            request_kwargs["max_output_tokens"] = int(self.config["max_tokens"])
        if "temperature" in self.config:
            request_kwargs["temperature"] = float(self.config["temperature"])
        text_format = None
        if params:
            response_params = dict(params)
            text_format = response_params.pop("text_format", None)
            tools = response_params.pop("tools", None)
            if tools is not None:
                request_kwargs["tools"] = self._responses_tools(tools)
            if "max_tokens" in response_params and "max_output_tokens" not in response_params:
                response_params["max_output_tokens"] = int(response_params.pop("max_tokens"))
            request_kwargs.update(response_params)

            if flex:
                request_kwargs.update({"service_tier": "flex"})

        return request_kwargs, text_format

    def generate(
        self,
        messages: list[dict[str, Any]],
        model: str | None = None,
        params: dict[str, Any] | None = None,
        flex = False,
    ) -> ModelResponse:
        resolved_model = model or str(self.config.get("model", "")).strip()
        if not resolved_model:
            raise ValueError("OpenAI model is required.")

        request_kwargs, text_format = self._build_request_kwargs(
            messages, resolved_model, params, flex
        )

        if text_format is not None:
            response = self.client.responses.parse(**request_kwargs, text_format=text_format)
        else:
            response = self.client.responses.create(**request_kwargs)

        parsed_output = getattr(response, "output_parsed", None)
        text = str(getattr(response, "output_text", "") or "")
        if not text and parsed_output is not None:
            if hasattr(parsed_output, "model_dump_json"):
                text = parsed_output.model_dump_json()
            else:
                text = json.dumps(parsed_output)
        raw_response: dict[str, Any] = {
            "response": self._serialize_response_metadata(response)
        }
        output_items = getattr(response, "output", None)
        if output_items is not None:
            raw_response["output"] = self._json_ready(output_items)
            thinking = self._extract_reasoning_text(output_items)
            if thinking:
                raw_response["thinking"] = thinking
            tool_calls = self._extract_tool_calls(output_items)
            raw_response["tool_calls"] = tool_calls
            raw_response["full_output"] = self._format_full_output(output_items)
            if not text and tool_calls:
                text = raw_response["full_output"]
        if not text and parsed_output is None:
            raise RuntimeError("OpenAI returned no text or tool-call output.")
        if parsed_output is not None:
            if hasattr(parsed_output, "model_dump"):
                raw_response["parsed_output"] = parsed_output.model_dump()
            else:
                raw_response["parsed_output"] = parsed_output

        return ModelResponse(
            text=text,
            model=str(getattr(response, "model", None) or resolved_model),
            usage=self._serialize_usage(getattr(response, "usage", None)),
            raw=raw_response,
        )

    @staticmethod
    def _text_from_output_items(output_items: list[Any]) -> str:
        parts: list[str] = []
        for item in output_items or []:
            item_data = OpenAIRuntime._json_ready(item)
            if not isinstance(item_data, dict) or item_data.get("type") != "message":
                continue
            for content_part in item_data.get("content") or []:
                if isinstance(content_part, dict):
                    text = content_part.get("text")
                    if text:
                        parts.append(str(text))
        return "".join(parts)

    def _parse_batch_response_body(
        self, body: dict[str, Any], fallback_model: str
    ) -> ModelResponse:
        output_items = body.get("output") or []
        text = str(body.get("output_text") or "") or self._text_from_output_items(output_items)
        tool_calls = self._extract_tool_calls(output_items)
        full_output = self._format_full_output(output_items)
        if not text and tool_calls:
            text = full_output
        if not text:
            raise RuntimeError("OpenAI batch response contained no text output.")

        raw_response: dict[str, Any] = {
            "response": {
                k: body.get(k)
                for k in ("id", "model", "status", "created_at")
                if body.get(k) is not None
            },
            "output": self._json_ready(output_items),
            "tool_calls": tool_calls,
            "full_output": full_output,
        }
        thinking = self._extract_reasoning_text(output_items)
        if thinking:
            raw_response["thinking"] = thinking

        usage = body.get("usage") or {}
        return ModelResponse(
            text=text,
            model=str(body.get("model") or fallback_model),
            usage={
                "prompt_tokens": int(usage.get("input_tokens", 0)),
                "completion_tokens": int(usage.get("output_tokens", 0)),
                "total_tokens": int(usage.get("total_tokens", 0)),
            },
            raw=raw_response,
        )

    # OpenAI Batch API per-file caps. Keep file-size headroom under the 200 MB
    # documented limit to allow for multipart upload overhead. The per-model
    # enqueued-token cap is enforced by the caller (e.g. evaluation.py picks a
    # batch_size that fits under the cap, since dataset names encode the
    # per-request context length).
    _BATCH_MAX_REQUESTS = 50_000
    _BATCH_MAX_FILE_BYTES = 190 * 1024 * 1024

    def _partition_jsonl_lines(
        self, line_bytes: list[bytes]
    ) -> list[tuple[int, int]]:
        """Return [(start_idx, end_idx), ...] partitions that each fit under the
        Batch API per-file caps (request count, file bytes). end_idx is exclusive."""
        max_requests = int(
            self.config.get("batch_max_requests_per_file", self._BATCH_MAX_REQUESTS)
        )
        max_bytes = int(
            self.config.get("batch_max_bytes_per_file", self._BATCH_MAX_FILE_BYTES)
        )

        partitions: list[tuple[int, int]] = []
        chunk_start = 0
        chunk_bytes = 0
        for idx, line in enumerate(line_bytes):
            if len(line) > max_bytes:
                raise RuntimeError(
                    f"Single batch request at index {idx} is {len(line)} bytes, "
                    f"exceeding the per-file cap of {max_bytes} bytes."
                )
            chunk_size = idx - chunk_start
            if chunk_size >= max_requests or (
                chunk_size > 0 and chunk_bytes + len(line) > max_bytes
            ):
                partitions.append((chunk_start, idx))
                chunk_start = idx
                chunk_bytes = 0
            chunk_bytes += len(line)
        if chunk_start < len(line_bytes):
            partitions.append((chunk_start, len(line_bytes)))
        return partitions

    def _submit_batch_chunk(
        self,
        chunk_lines: list[bytes],
        endpoint: str,
        chunk_label: str,
    ) -> dict[str, dict[str, Any]]:
        buf = io.BytesIO()
        for line in chunk_lines:
            buf.write(line)
        buf.seek(0)

        input_file = self.client.files.create(
            file=(f"batch_input_{chunk_label}.jsonl", buf, "application/jsonl"),
            purpose="batch",
        )

        batch = self.client.batches.create(
            input_file_id=input_file.id,
            endpoint=endpoint,
            completion_window=str(self.config.get("batch_completion_window", "24h")),
        )
        print(
            f"[openai-batch] submitted chunk={chunk_label} batch_id={batch.id} "
            f"requests={len(chunk_lines)} status={batch.status}",
            flush=True,
        )

        poll_interval = float(self.config.get("batch_poll_interval_s", 30))
        terminal_statuses = {"completed", "failed", "expired", "cancelled"}
        progress_callback = getattr(self, "batch_progress_callback", None)
        while batch.status not in terminal_statuses:
            if progress_callback is not None:
                try:
                    progress_callback(batch)
                except Exception:
                    pass
            counts = getattr(batch, "request_counts", None)
            counts_str = ""
            if counts is not None:
                total = getattr(counts, "total", None)
                completed = getattr(counts, "completed", None)
                failed = getattr(counts, "failed", None)
                counts_str = f" counts=(completed={completed}, failed={failed}, total={total})"
            print(
                f"[openai-batch] chunk={chunk_label} batch_id={batch.id} "
                f"status={batch.status}{counts_str}",
                flush=True,
            )
            time.sleep(poll_interval)
            batch = self.client.batches.retrieve(batch.id)
        if progress_callback is not None:
            try:
                progress_callback(batch)
            except Exception:
                pass

        if batch.status != "completed":
            raise RuntimeError(
                f"OpenAI batch {batch.id} (chunk {chunk_label}) ended in status "
                f"{batch.status!r}."
            )
        if not batch.output_file_id:
            raise RuntimeError(
                f"OpenAI batch {batch.id} (chunk {chunk_label}) completed without "
                "an output file."
            )

        output_text = self.client.files.content(batch.output_file_id).text
        entries: dict[str, dict[str, Any]] = {}
        for line in output_text.splitlines():
            if not line.strip():
                continue
            entry = json.loads(line)
            entries[str(entry.get("custom_id"))] = entry
        return entries

    def batch_generate(
        self,
        batch_messages: list[list[dict[str, Any]]],
        model: str | None = None,
        params: dict[str, Any] | None = None,
    ) -> list[ModelResponse]:
        if not batch_messages:
            return []

        resolved_model = model or str(self.config.get("model", "")).strip()
        if not resolved_model:
            raise ValueError("OpenAI model is required.")

        if params and params.get("text_format") is not None:
            raise ValueError(
                "OpenAI Batch API does not support structured `text_format` outputs."
            )

        endpoint = "/v1/responses"
        line_bytes: list[bytes] = []
        for idx, messages in enumerate(batch_messages):
            request_kwargs, _ = self._build_request_kwargs(
                messages, resolved_model, params, flex=False
            )
            line = {
                "custom_id": f"request-{idx}",
                "method": "POST",
                "url": endpoint,
                "body": request_kwargs,
            }
            line_bytes.append((json.dumps(line) + "\n").encode("utf-8"))

        partitions = self._partition_jsonl_lines(line_bytes)

        entries_by_custom_id: dict[str, dict[str, Any]] = {}
        for chunk_idx, (start, end) in enumerate(partitions):
            chunk_label = f"{chunk_idx + 1}of{len(partitions)}"
            chunk_entries = self._submit_batch_chunk(
                chunk_lines=line_bytes[start:end],
                endpoint=endpoint,
                chunk_label=chunk_label,
            )
            entries_by_custom_id.update(chunk_entries)

        responses: list[ModelResponse] = []
        for idx in range(len(batch_messages)):
            entry = entries_by_custom_id.get(f"request-{idx}")
            if entry is None:
                raise RuntimeError(f"OpenAI batch is missing result for request-{idx}.")
            if entry.get("error"):
                raise RuntimeError(
                    f"OpenAI batch request-{idx} failed: {entry['error']}"
                )
            response_obj = entry.get("response") or {}
            body = response_obj.get("body") or {}
            responses.append(self._parse_batch_response_body(body, resolved_model))

        return responses
