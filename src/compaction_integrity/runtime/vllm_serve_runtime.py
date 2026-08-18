import atexit
import os
import socket
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import requests
from openai import OpenAI

from compaction_integrity.runtime.base import ModelResponse, ModelRuntime
from compaction_integrity.runtime.env import apply_runtime_environment


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


class VLLMServeRuntime(ModelRuntime):
    def __init__(self, config: dict[str, Any]):
        super().__init__(config)
        if "model" not in config:
            raise ValueError("VLLMServeRuntime requires a 'model' in the config.")

        apply_runtime_environment(config)
        self.model = str(config["model"])

        port = _find_free_port()
        self._base_url = f"http://localhost:{port}/v1"
        self.client = OpenAI(base_url=self._base_url, api_key="EMPTY")

        cmd = ["vllm", "serve", self.model, "--port", str(port)]

        # Pass through extra vllm serve flags from config
        _reserved = {
            "model", "base_url", "api_key", "max_tokens", "cuda_visible_devices",
            "startup_timeout", "env", "batch", "batch_size", "retry_attempts",
            "retry_sleep_seconds",
        }
        for key, value in config.items():
            if key not in _reserved:
                flag = f"--{key.replace('_', '-')}"
                cmd.extend([flag, str(value)])

        serve_env = os.environ.copy()
        if "cuda_visible_devices" in config:
            serve_env["CUDA_VISIBLE_DEVICES"] = str(config["cuda_visible_devices"])
        elif "CUDA_VISIBLE_DEVICES" in os.environ:
            # Explicit re-injection (defensive — Popen would inherit anyway,
            # but make it visible/loggable that we're pinning the GPU).
            serve_env["CUDA_VISIBLE_DEVICES"] = os.environ["CUDA_VISIBLE_DEVICES"]

        print(
            f"[vllm_serve_runtime] launching with "
            f"CUDA_VISIBLE_DEVICES={serve_env.get('CUDA_VISIBLE_DEVICES', '<unset>')}"
        )

        self._process = subprocess.Popen(
            cmd,
            env=serve_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        atexit.register(self.close)
        print(f"Started vllm serve with command: {' '.join(cmd)}")
        timeout = int(config.get("startup_timeout", 300))
        self._wait_for_server(port, timeout)
        print(f"vllm serve is healthy and ready to accept requests at {self._base_url}.")

    def _wait_for_server(self, port: int, timeout: int) -> None:
        health_url = f"http://localhost:{port}/health"
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self._process.poll() is not None:
                raise RuntimeError(
                    f"vllm serve process exited early with code {self._process.returncode}."
                )
            try:
                resp = requests.get(health_url, timeout=2)
                if resp.status_code == 200:
                    return
            except requests.exceptions.ConnectionError:
                pass
            time.sleep(2)
        self._process.terminate()
        raise TimeoutError(
            f"vllm serve did not become healthy within {timeout}s on port {port}."
        )

    def generate(
        self,
        messages: list[dict[str, Any]],
        model: str | None = None,
        params: dict[str, Any] | None = None,
    ) -> ModelResponse:
        resolved_model = model or self.model
        normalized = self._normalize_messages(messages)

        kwargs: dict[str, Any] = {}
        if "max_tokens" in self.config:
            kwargs["max_tokens"] = int(self.config["max_tokens"])
        if params and "max_tokens" in params:
            kwargs["max_tokens"] = int(params["max_tokens"])
        if params and "extra_body" in params:
            kwargs["extra_body"] = params["extra_body"]
        if params and "tools" in params:
            kwargs["tools"] = params["tools"]

        completion = self.client.chat.completions.create(
            model=resolved_model,
            messages=normalized,  # type: ignore[arg-type]
            **kwargs,
        )

        message = completion.choices[0].message
        content = message.content or ""
        reasoning = str(getattr(message, "reasoning_content", "") or "").strip()
        tool_calls: list[dict[str, Any]] = []
        for tool_call in message.tool_calls or []:
            name = tool_call.function.name
            tool_calls.append(
                {
                    "channel": "commentary",
                    "recipient": f"functions.{name}",
                    "name": name,
                    "arguments": tool_call.function.arguments,
                    "call_id": tool_call.id,
                }
            )
        full_parts: list[str] = []
        if reasoning:
            full_parts.append(f"[analysis]\n{reasoning}")
        full_parts.extend(
            f"[commentary to={tool_call['recipient']}]\n{tool_call['arguments']}"
            for tool_call in tool_calls
        )
        if content:
            full_parts.append(f"[final]\n{content}")
        full_output = "\n\n".join(full_parts)
        if not content and tool_calls:
            content = full_output

        usage: dict[str, int] = {}
        if completion.usage is not None:
            usage = {
                "prompt_tokens": completion.usage.prompt_tokens,
                "completion_tokens": completion.usage.completion_tokens,
                "total_tokens": completion.usage.total_tokens,
            }

        return ModelResponse(
            text=content,
            model=resolved_model,
            usage=usage,
            raw={
                "response": completion,
                "thinking": reasoning or None,
                "tool_calls": tool_calls,
                "full_output": full_output,
            },
        )

    def close(self) -> None:
        if self._process.poll() is None:
            self._process.terminate()
            try:
                self._process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait()

    def batch_generate(
        self,
        batch_messages: list[list[dict[str, Any]]],
        model: str | None = None,
        params: dict[str, Any] | None = None,
    ) -> list[ModelResponse]:
        if not batch_messages:
            return []

        max_workers = min(
            len(batch_messages),
            int(self.config.get("batch_size", len(batch_messages))),
        )
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            return list(
                executor.map(
                    lambda messages: self.generate(messages, model=model, params=params),
                    batch_messages,
                )
            )
