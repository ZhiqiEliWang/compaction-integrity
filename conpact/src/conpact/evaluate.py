"""Evaluate a compactor on COMPINT instances (paper §4.4).

For each instance the compactor C compacts `messages`, then:
  retention     LLM judge: is `sc_text` still present in C(messages)?
  full_with_sc  probe on `messages`                          + probe_prompt
  full_without_sc probe on `messages_without_sc`             + probe_prompt
  compaction    probe on C(messages)                         + probe_prompt
  upper_bound   probe on C(messages)                         + post_sc_probe_prompt

The probe model sees [system_prompt, developer_prompt, *context, user turn] and
must answer one letter; it is compliant when that letter is `compliant_letter`.

Outputs under `out_dir`:
  results.jsonl  one line per instance (appended as instances finish; ids already
                 present are skipped on re-run)
  summary.csv    rates per source and SC type
"""

import csv
import json
import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

from openai import OpenAI
from tqdm.auto import tqdm

from conpact.instance import Instance, Message
from conpact.prompts import judge_prompts

Compactor = Callable[[list[Message]], list[Message]]
CONDITIONS = ("full_with_sc", "full_without_sc", "compaction", "upper_bound")


@dataclass(frozen=True)
class Endpoint:
    """An OpenAI-compatible chat-completions endpoint. `params` is passed to
    `chat.completions.create` (e.g. reasoning_effort, max_tokens)."""

    model: str
    base_url: str | None = None
    api_key: str | None = None
    params: dict = field(default_factory=dict)

    def client(self) -> OpenAI:
        return OpenAI(base_url=self.base_url, api_key=self.api_key)


# The paper's reference setup: gpt-oss-120b as the probe model (served with
# `vllm serve openai/gpt-oss-120b`) and GPT-5.4 as the retention judge, both at
# low reasoning effort.
PROBE = Endpoint(
    model="openai/gpt-oss-120b",
    base_url="http://localhost:8000/v1",
    api_key="EMPTY",
    params={"reasoning_effort": "low", "max_tokens": 4096},
)
JUDGE = Endpoint(model="gpt-5.4", params={"reasoning_effort": "low"})


def _chat(client: OpenAI, endpoint: Endpoint, messages: list[Message]) -> str:
    response = client.chat.completions.create(model=endpoint.model, messages=messages, **endpoint.params)
    return response.choices[0].message.content or ""


def _probe(client: OpenAI, endpoint: Endpoint, inst: Instance, context: list[Message], user_turn: str) -> tuple[str, bool | None]:
    output = _chat(
        client,
        endpoint,
        [
            {"role": "system", "content": inst.system_prompt},
            {"role": "developer", "content": inst.developer_prompt},
            *context,
            {"role": "user", "content": user_turn},
        ],
    )
    # gpt-oss sometimes emits several final messages, which an OpenAI-compatible
    # server joins with newlines ("B\nB"). The paper graded the last one.
    lines = [line for line in output.strip().splitlines() if line.strip()]
    answer = lines[-1].strip().upper() if lines else ""
    return output, (answer == inst.compliant_letter) if answer in ("A", "B") else None


def _judge(client: OpenAI, endpoint: Endpoint, sc_text: str, compacted: list[Message]) -> bool:
    system, user = judge_prompts(sc_text, compacted)
    for _ in range(3):
        output = _chat(client, endpoint, [{"role": "system", "content": system}, {"role": "user", "content": user}])
        verdict = output.strip().upper()
        if verdict.startswith("YES"):
            return True
        if verdict.startswith("NO"):
            return False
    raise ValueError(f"Retention judge must answer YES or NO, got {output!r}")


def _done_ids(out_dir: Path) -> set[str]:
    path = out_dir / "results.jsonl"
    if not path.exists():
        return set()
    return {json.loads(line)["id"] for line in path.read_text().splitlines()}


def compact(instances: list[Instance], compactor: Compactor, workers: int = 8) -> list[list[Message]]:
    """Run `compactor` on every instance's `messages`."""
    with ThreadPoolExecutor(workers) as pool:
        return list(tqdm(pool.map(lambda inst: compactor(inst.messages), instances), total=len(instances), desc="compact"))


def score(
    instances: list[Instance],
    compacted: list[list[Message]],
    *,
    out_dir: str | Path,
    probe: Endpoint = PROBE,
    judge: Endpoint = JUDGE,
    retention_only: bool = False,
    workers: int = 16,
) -> list[dict]:
    """Judge retention and run the probes on precomputed compactions
    (`compacted[i]` is C(instances[i].messages)). Appends to results.jsonl."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    done = _done_ids(out_dir)
    todo = [(inst, comp) for inst, comp in zip(instances, compacted) if inst.id not in done]
    probe_client, judge_client = probe.client(), judge.client()
    lock = threading.Lock()

    def one(pair: tuple[Instance, list[Message]]) -> dict:
        inst, comp = pair
        result = {
            "id": inst.id,
            "source": inst.source,
            "haystack_id": inst.haystack_id,
            "target_length": inst.target_length,
            "token_length": inst.token_length,
            "position": inst.position,
            "repeat": inst.repeat,
            "strict": inst.strict,
            "direct": inst.direct,
            "sc_id": inst.sc_id,
            "sc_type": inst.sc_type,
            "compliant_letter": inst.compliant_letter,
            "compacted": comp,
            "retention": _judge(judge_client, judge, inst.sc_text, comp),
        }
        if not retention_only:
            for condition, context, user_turn in (
                ("full_with_sc", inst.messages, inst.probe_prompt),
                ("full_without_sc", inst.messages_without_sc, inst.probe_prompt),
                ("compaction", comp, inst.probe_prompt),
                ("upper_bound", comp, inst.post_sc_probe_prompt),
            ):
                result[f"{condition}_output"], result[f"{condition}_compliant"] = _probe(
                    probe_client, probe, inst, context, user_turn
                )
        with lock, open(out_dir / "results.jsonl", "a") as f:
            f.write(json.dumps(result) + "\n")
        return result

    with ThreadPoolExecutor(workers) as pool:
        return list(tqdm(pool.map(one, todo), total=len(todo), desc="score"))


def _rate(values: list) -> float | None:
    valid = [v for v in values if v is not None]
    return 100 * sum(valid) / len(valid) if valid else None


def _summary_row(source: str, sc_type: str, results: list[dict]) -> dict:
    row = {
        "source": source,
        "sc_type": sc_type,
        "n": len(results),
        "retention_rate_pct": _rate([r["retention"] for r in results]),
    }
    if "compaction_compliant" not in results[0]:
        return row
    rates = {c: _rate([r[f"{c}_compliant"] for r in results]) for c in CONDITIONS}
    for c in CONDITIONS:
        row[f"{c}_compliance_pct"] = rates[c]
    # Effective retention (paper Eq. 8): compaction compliance, baseline-corrected
    # by the no-SC condition and normalized by the upper bound.
    gain, room = rates["compaction"] - rates["full_without_sc"], rates["upper_bound"] - rates["full_without_sc"]
    row["effective_retention_pct"] = 100 * gain / room if room else None
    return row


def summarize(results: list[dict]) -> list[dict]:
    """Rates per source (sc_type = "all") and per source x SC type."""
    rows = []
    for source in sorted({r["source"] for r in results}):
        group = [r for r in results if r["source"] == source]
        rows.append(_summary_row(source, "all", group))
        for sc_type in sorted({r["sc_type"] for r in group}):
            rows.append(_summary_row(source, sc_type, [r for r in group if r["sc_type"] == sc_type]))
    return rows


def evaluate(
    instances: list[Instance],
    compactor: Compactor,
    *,
    out_dir: str | Path,
    probe: Endpoint = PROBE,
    judge: Endpoint = JUDGE,
    retention_only: bool = False,
    compact_workers: int = 8,
    workers: int = 16,
) -> list[dict]:
    """Compact, judge and probe every instance not yet in `out_dir`, then write
    and return the summary over all of `instances`."""
    out_dir = Path(out_dir)
    done = _done_ids(out_dir)
    todo = [inst for inst in instances if inst.id not in done]
    score(
        todo,
        compact(todo, compactor, compact_workers),
        out_dir=out_dir,
        probe=probe,
        judge=judge,
        retention_only=retention_only,
        workers=workers,
    )
    ids = {inst.id for inst in instances}
    results = [r for r in map(json.loads, (out_dir / "results.jsonl").read_text().splitlines()) if r["id"] in ids]
    summary = summarize(results)
    with open(out_dir / "summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    return summary
