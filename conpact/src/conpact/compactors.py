"""Reference compactors from the paper's baselines. A compactor is any callable
mapping a message list to the message list that replaces it."""

from dataclasses import dataclass

from conpact.evaluate import Endpoint
from conpact.instance import Message
from conpact.prompts import summarization_prompt


@dataclass(frozen=True)
class RecentN:
    """Keep the last n messages (paper: Recent-5)."""

    n: int = 5

    def __call__(self, messages: list[Message]) -> list[Message]:
        return messages[-self.n :]


@dataclass(frozen=True)
class LLMSummarize:
    """Replace the history with an LLM summary, using the Anthropic compaction
    prompt (`anthropic`) or the Pi / OpenClaw prompt (`pi_mono`)."""

    endpoint: Endpoint
    prompt: str = "anthropic"

    def __call__(self, messages: list[Message]) -> list[Message]:
        response = self.endpoint.client().chat.completions.create(
            model=self.endpoint.model,
            messages=summarization_prompt(self.prompt, messages),
            **self.endpoint.params,
        )
        return [{"role": "assistant", "content": response.choices[0].message.content or ""}]
