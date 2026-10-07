"""Variable-length instances: cut an SC-free haystack to the requested length and
inject each of the 15 SCs (paper §4.3).

Determinism follows the paper's evaluation code: (haystack, SC) pairs are
enumerated haystack-major; one `random.Random(seed)` draws every pair's A/B swap
seed in that order, and a second one draws the multi-injection turns. Run on the
paper's haystacks, this reproduces its injected inputs exactly.
"""

import json
import random
from dataclasses import dataclass

from datasets import load_dataset

from conpact.constraints import SCS, render_sc
from conpact.instance import Instance, Message
from conpact.prompts import DEVELOPER_PROMPT, probe_prompt

HF_REPO = "ZhiqiEliWang/conpact"
SOURCES = ("wildchat", "hermes", "openresearcher")
POSITIONS = ("top", "middle", "bottom", "multi")


def load_haystacks(source: str, repo_id: str = HF_REPO, revision: str | None = None) -> list[dict]:
    """The SC-free ~220k-token histories of one source (`haystacks` config)."""
    return list(load_dataset(repo_id, "haystacks", split=source, revision=revision))


def cut(messages: list[Message], message_tokens: list[int], length: int) -> tuple[list[Message], int]:
    """Longest message prefix of at most `length` tokens, and its token count."""
    total = 0
    for i, tokens in enumerate(message_tokens):
        if total + tokens > length:
            return messages[:i], total
        total += tokens
    return messages, total


def inject(
    messages: list[Message],
    sc_rendered: str,
    position: str,
    repeat: int,
    rng: random.Random,
) -> tuple[list[Message], list[int]]:
    """The paper's Inj operator: put the SC before the content of the selected
    user turns. top / middle / bottom pick the first / median / last user turn;
    multi picks min(repeat, #user turns) of them uniformly without replacement."""
    user_idxs = [i for i, m in enumerate(messages) if m["role"] == "user"]
    if position == "multi":
        targets = sorted(rng.sample(user_idxs, k=min(repeat, len(user_idxs))))
    else:
        targets = [{"top": user_idxs[0], "middle": user_idxs[len(user_idxs) // 2], "bottom": user_idxs[-1]}[position]]
    out = [dict(m) for m in messages]
    for i in targets:
        out[i] = {**out[i], "content": f"{sc_rendered}\n{out[i]['content']}"}
    return out, targets


def build_instances(
    haystacks: list[dict],
    *,
    length: int,
    position: str = "top",
    repeat: int = 1,
    strict: bool = False,
    direct: bool = True,
    seed: int = 42,
) -> list[Instance]:
    """Cross each haystack (cut to `length`) with the 15 SCs."""
    inject_rng = random.Random(seed)
    swap_rng = random.Random(seed)
    instances: list[Instance] = []
    for haystack in haystacks:
        without_sc, token_length = cut(haystack["messages"], haystack["message_tokens"], length)
        for sc in SCS:
            sc_rendered = render_sc(sc["sc"], strict, direct)
            messages, sc_message_indices = inject(without_sc, sc_rendered, position, repeat, inject_rng)
            swap_seed = swap_rng.randint(0, 2**31 - 1)
            prompt, compliant_letter = probe_prompt(sc["probe"], sc["correct_answer"], sc["incorrect_answer"], swap_seed)
            instances.append(
                Instance(
                    id=f"{haystack['haystack_id']}:sc{sc['id']:02d}",
                    source=haystack["source"],
                    haystack_id=haystack["haystack_id"],
                    target_length=length,
                    token_length=token_length,
                    position=position,
                    repeat=repeat if position == "multi" else 1,
                    strict=strict,
                    direct=direct,
                    sc_id=sc["id"],
                    sc_type=sc["type"],
                    sc_text=sc["sc"],
                    sc_rendered=sc_rendered,
                    sc_message_indices=sc_message_indices,
                    system_prompt=haystack["system_prompt"],
                    developer_prompt=DEVELOPER_PROMPT,
                    messages=messages,
                    messages_without_sc=list(without_sc),
                    probe=sc["probe"],
                    correct_answer=sc["correct_answer"],
                    incorrect_answer=sc["incorrect_answer"],
                    swap_seed=swap_seed,
                    compliant_letter=compliant_letter,
                    probe_prompt=prompt,
                    post_sc_probe_prompt=f"{sc_rendered}\n\n{prompt}",
                    meta=json.dumps({"source_conversation_count": haystack["source_conversation_count"]}),
                )
            )
    return instances


@dataclass(frozen=True)
class ConPact:
    """A variable-length COMPINT instance set.

    source:   wildchat | hermes | openresearcher
    length:   context length in tokens (gpt-oss tokenizer), up to ~220k. Haystacks
              shorter than `length` are skipped.
    position: top | middle | bottom | multi. OpenResearcher histories have a
              single task query, so only `top` is defined for them.
    repeat:   number of SC statements for `multi` (r >= 2).
    strict:   prepend "This is an important constraint:" (paper: Strict).
    direct:   prepend "For the rest of this session." (paper: Direct).
    n:        use the first n eligible haystacks (default: all).
    """

    source: str
    length: int
    position: str = "top"
    repeat: int = 1
    strict: bool = False
    direct: bool = True
    seed: int = 42
    n: int | None = None
    repo_id: str = HF_REPO
    revision: str | None = None

    def __post_init__(self) -> None:
        if self.position not in POSITIONS:
            raise ValueError(f"position must be one of {POSITIONS}, got {self.position!r}")
        if self.source == "openresearcher" and self.position != "top":
            raise ValueError("openresearcher histories have one task query; only position='top' is defined")
        if self.position == "multi" and self.repeat < 2:
            raise ValueError("position='multi' needs repeat >= 2")

    def generate(self) -> list[Instance]:
        haystacks = [h for h in load_haystacks(self.source, self.repo_id, self.revision) if h["token_length"] >= self.length]
        return build_instances(
            haystacks[: self.n],
            length=self.length,
            position=self.position,
            repeat=self.repeat,
            strict=self.strict,
            direct=self.direct,
            seed=self.seed,
        )


def load_static(source: str, repo_id: str = HF_REPO, revision: str | None = None) -> list[Instance]:
    """The fixed ~100k instance set of the paper: wildchat | hermes | openresearcher | swe_natural."""
    return [Instance(**row) for row in load_dataset(repo_id, source, split="test", revision=revision)]
