"""Online SC-aware extractor (RQ4, §5.4 / App. F.6) for live agent loops.

A strategy-layer hook, independent of the compactor: after each user turn an SLM
updates a session-scoped registry S_t, held outside the conversation history so
compaction cannot overwrite it. At compaction the harness appends the rendered
registry to the compacted context (Eq. 9).

scripts/eval_sc_extractor.py runs the same prompt offline, batched across pairs.
"""

import json
import re
from dataclasses import dataclass, field
from typing import Any

from compaction_integrity.prompts import SC_EXTRACTION_PROMPTS, sentence_windows
from compaction_integrity.runtime.base import ModelRuntime

REGISTRY_HEADER = "Constraints from earlier in this session:"

_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


def parse_extractor_output(raw: str) -> list[dict[str, str]]:
    text = (raw or "").strip()
    if not text:
        return []
    # Try in order: fenced JSON; the trailing `{...}` block (reasoning models
    # often emit prose then JSON); the whole text.
    candidates: list[str] = []
    match = _JSON_FENCE_RE.search(text)
    if match is not None:
        candidates.append(match.group(1).strip())
    last_open = text.rfind("{")
    last_close = text.rfind("}")
    if last_open != -1 and last_close > last_open:
        candidates.append(text[last_open : last_close + 1])
    candidates.append(text)
    data: Any = None
    for candidate in candidates:
        try:
            data = json.loads(candidate)
            break
        except json.JSONDecodeError:
            continue
    if data is None:
        return []
    items = data.get("scs", []) if isinstance(data, dict) else []
    out: list[dict[str, str]] = []
    for item in items if isinstance(items, list) else []:
        if not isinstance(item, dict):
            continue
        sc_text = item.get("text")
        if not isinstance(sc_text, str) or not sc_text.strip():
            continue
        out.append({"text": sc_text.strip(), "evidence": str(item.get("evidence", "")).strip()})
    return out


_QUOTES_RE = re.compile("[\"'`\u2018\u2019\u201c\u201d]")


def _normalize_span(text: str) -> str:
    # Quote characters are dropped: the model re-quotes spans ("..." -> '...').
    return " ".join(_QUOTES_RE.sub("", text.lower()).split())


_EVIDENCE_PIECES_RE = re.compile(r"(?<=[.!?])\s+|\.\.\.|\u2026")


def evidence_in_turn(evidence: str, user_turn: str) -> bool:
    """The prompt's own output contract: the evidence appears in the user turn
    (enforced with `verify_evidence: true`). Checked sentence by sentence: the model
    may quote two sentences in another order or join pieces with '...'."""
    turn = _normalize_span(user_turn)
    pieces = [_normalize_span(p).strip(" .,;:") for p in _EVIDENCE_PIECES_RE.split(evidence)]
    pieces = [p for p in pieces if p]
    return bool(pieces) and all(p in turn for p in pieces)




def render_registry_turn(registry: list[dict[str, Any]]) -> str | None:
    """The registry as appended after the compacted context; None when it is empty."""
    if not registry:
        return None
    lines = "\n".join(f"{i}. {e['text']}" for i, e in enumerate(registry, start=1))
    return f"{REGISTRY_HEADER}\n{lines}"


@dataclass
class SCExtractor:
    runtime: ModelRuntime
    prompt: str
    verify_evidence: bool
    # Read each user turn as consecutive windows of this many sentences (None: whole turn).
    window_sentences: int | None = None
    registry: list[dict[str, Any]] = field(default_factory=list)
    calls: list[dict[str, Any]] = field(default_factory=list)

    def observe(self, user_turn: str, prev_assistant_turn: str | None) -> None:
        # Windows of one turn see the registry from earlier turns only.
        existing = [e["text"] for e in self.registry]
        texts = sentence_windows(user_turn, self.window_sentences) if self.window_sentences else [user_turn]
        for text in texts:
            self._observe(text, existing, prev_assistant_turn)

    def _observe(self, user_turn: str, existing: list[str], prev_assistant_turn: str | None) -> None:
        system_prompt, user_prompt = SC_EXTRACTION_PROMPTS[self.prompt](
            user_turn=user_turn,
            existing_scs=existing,
            prev_assistant_turn=prev_assistant_turn,
        )
        response = self.runtime.generate(
            [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
        )
        raw = (response.text or "").strip()
        seen = {e["text"] for e in self.registry}
        added = []
        for item in parse_extractor_output(raw):
            if self.verify_evidence and not evidence_in_turn(item["evidence"], user_turn):
                continue
            if item["text"] in seen:
                continue
            added.append(item)
            seen.add(item["text"])
        self.registry.extend(added)
        self.calls.append({"user_turn": user_turn, "raw_output": raw, "added": added})
