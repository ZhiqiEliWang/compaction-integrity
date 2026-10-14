"""End-to-end broad-task-constraint annotation of an SC candidate sheet.

Runs the ``annotation_guidelines/`` package next to this file (rubric
``1.0-broad-default``) as one pipeline. This script plays the package's
coordinator (ORCHESTRATION_PROMPT.md), API model calls play its workers, and
every deterministic step is the package's own frozen helper
(``scripts/pipeline.py``), imported from the run's frozen copy.

  input   candidate CSV with annotation_id, instance_id, candidate_paragraph,
          issue_text (e.g. studies/swe_natural/generated/sc_candidates.*.csv)
  output  <export_dir>/
            sc_annotations.broad_keeps.typed.csv        every broad keep, row-level
            sc_annotations.broad_keeps.probe_ready.csv  ready probes only
                                                        -> dataset.py build --annotations
            sc_annotations.broad_keeps.probes.csv       one row per probe (clause sheet)
                                                        -> build --clause_sheet <input_csv> --annotations
          <run_dir>/ every package deliverable (all_candidates.annotated.csv,
          annotations_detailed.jsonl, typed_records.jsonl, probes_final.jsonl,
          review_queue.jsonl, final_validation.json, run_report.md, ...)

Stages (LLM = model worker, H = frozen helper):

  H    prepare                              freeze source + package, issue-grouped partitions
  LLM  calibration_A, calibration_B, controls
  H    review --stage calibration, score-controls; calibration_report.md
  LLM  pass_A, pass_B                       two independent annotation passes
  H    review --stage production            positives, disagreements, flags, 20 agreed negatives
  LLM  adjudication
  H    merge, prepare-types
  LLM  type_A, type_B
  H    review-types
  LLM  type_adjudication
  H    merge-types, prepare-probes
  LLM  probe_drafts
  H    collect-probes -> probes_v1
       repeat: LLM quality review -> H prepare-revision -> LLM revision -> H apply-revision
               (after round 1 only revised drafts are re-reviewed; unchanged ones keep their verdict)
  H    finalize; export

Every call is a fresh context holding the stage's frozen prompt, rubric(s) and
schema in full plus its assigned partition as JSON lines. Each returned record
is checked with the helper's validator; errors go back into the same
conversation for correction, up to ``llm.max_attempts``. The rubric requires
clause offsets to be derived programmatically, so the model returns clause text
only and offsets are located here (tolerating whitespace encoding such as CRLF),
storing the exact source slice.

Prompts and scheduling are laid out for prompt-cache hits: every call opens with
its stage's fixed system prompt (shared files first, so related stages share a
prefix), then the partition. Passes A and B of a partition send identical prompts
(no pass id) and run back to back, so B reads A's whole prompt from cache; retries
are sent at once. Cached input tokens are logged per call and summarized per
stage in run_report.md.

Rerunning with the same run_dir resumes: finished partitions and helper steps
are skipped. Raw responses are kept under <run_dir>/attempts/ and every call is
logged to <run_dir>/dispatch_log.jsonl.

Usage::

    python -m compaction_integrity.dataset.swe_natural_curation.annotate_sc --config-name annotate \\
      input_csv=studies/swe_natural/generated/sc_candidates.open_swe_251_500.csv \\
      run_dir=/data/compaction_integrity/annotation_runs/open_swe_251_500
"""

import importlib.util
import json
import re
import shutil
import sys
import tempfile
import threading
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import hydra
from omegaconf import DictConfig, OmegaConf

from compaction_integrity.runtime.base import ModelResponse, ModelRuntime
from compaction_integrity.runtime.env import apply_runtime_environment
from compaction_integrity.runtime.openai_runtime import OpenAIRuntime

# The annotation package (prompts, rubrics, schemas, helper) a new run freezes.
PACKAGE_DIR = Path(__file__).resolve().with_name("annotation_guidelines")


# Worker prompts

# Frozen package files each worker role receives in full (ORCHESTRATION_PROMPT.md
# and DISPATCH_TEMPLATES.md). Shared rubrics and schemas come first and role
# overrides last, so roles that build on one another share a cacheable prompt
# prefix: annotate < adjudicate, type < type_adjudicate, probe_draft < probe_review,
# and probe_review / probe_revise share all but their last file.
ANNOTATE = ("SC_RUBRIC.md", "schemas/ANNOTATION_SCHEMA.json", "EVALUATION_PROMPT.md")
TYPE = ("TYPE_RUBRIC.md", "schemas/TYPE_SCHEMA.json", "TYPE_EVALUATION_PROMPT.md")
PROBE = ("SC_RUBRIC.md", "TYPE_RUBRIC.md", "schemas/PROBE_SCHEMA.json", "PROBE_GENERATION_PROMPT.md")
PROMPT_FILES = {
    "annotate": ANNOTATE,
    "adjudicate": (*ANNOTATE, "ADJUDICATION_PROMPT.md"),
    "type": TYPE,
    "type_adjudicate": (*TYPE, "TYPE_ADJUDICATION_PROMPT.md"),
    "probe_draft": PROBE,
    "probe_review": (*PROBE, "PROBE_QUALITY_PROMPT.md", "schemas/PROBE_REVIEW_SCHEMA.json"),
    "probe_revise": (*PROBE, "PROBE_QUALITY_PROMPT.md", "PROBE_REVISION_PROMPT.md"),
}
# The schema each validator kind checks returned records against.
OUTPUT_SCHEMA = {
    "annotation": "schemas/ANNOTATION_SCHEMA.json",
    "types": "schemas/TYPE_SCHEMA.json",
    "probes": "schemas/PROBE_SCHEMA.json",
    "probe-review": "schemas/PROBE_REVIEW_SCHEMA.json",
}

# The five definitions DISPATCH_TEMPLATES.md restates in every type dispatch.
TYPE_DEFINITIONS = (
    "Action: whether an action is performed or permission is required.\n"
    "Information: access, disclosure, storage, retention, or preservation of specified information.\n"
    "Process: preparation, method, procedure, sequencing, testing, implementation location, or coordination.\n"
    "Preference: choice among otherwise acceptable methods, responses, options, or sources.\n"
    "Output: surface response form, or by analogy form, fidelity, compatibility, or properties of the "
    "delivered artifact/result.\n"
)

# DISPATCH_TEMPLATES.md coordinator messages, minus the path and validation lines
# that EXECUTION replaces.
DISPATCH = {
    "annotate": (
        "The default selection label is broad_task_constraint. Keep the secondary strict comparison "
        "separately. Read only the assigned input. Do not inspect other passes, keys, proposals, or the "
        "original CSV."
    ),
    "adjudicate": (
        "Role override: inspect both proposals supplied in your assigned input. Decide from the complete "
        "frozen source under the broader-default rubric. Preserve unresolved ambiguity and return exactly one "
        "final annotation-schema record per assigned ID."
    ),
    "type": TYPE_DEFINITIONS
    + (
        "Read all clauses and full issue context. Assign one primary row type; flag material ambiguity. "
        "Do not read the other type pass."
    ),
    "type_adjudicate": TYPE_DEFINITIONS
    + (
        "Role override: you may read both type proposals in your assigned input. Re-read the complete issue, "
        "candidate, and broad clauses, and assign exactly one primary type per entry; flag material ambiguity."
    ),
    "probe_draft": (
        "Produce one draft or explicit hold per entry. Use one kept target clause and its exact source text. "
        "Write a neutral continuation and two task-equivalent alternatives differing on that clause only."
    ),
    "probe_review": (
        "Review every assigned draft against full source context. Check leakage, single controlled difference, "
        "modality, allowed alternatives, scope, lifecycle, feasibility, and gold defensibility. Use "
        "ready/targeted_revision/substantial_revision/hold and explain concrete findings."
    ),
    "probe_revise": (
        "Revise only the assigned targeted/substantial entries; address every finding. Return an explicit hold "
        "if no supported contrast is possible. Preserve exact source text."
    ),
}

EXECUTION = """## Execution in this pipeline

The coordinator is an automated pipeline, and you are dispatched as a single model call. Every file you are told to read is inlined above, and your assigned input follows in the user message as JSON lines. You cannot read or write files, run commands, call the helper, or spawn agents. The coordinator saves your result, validates it with the frozen helper (`validate-{kind}`), and returns any errors to you for correction; that replaces every instruction to save, validate, or report paths and counts.

Respond with exactly one JSON object and nothing else: {{"records": [...]}}, holding one object valid against `{schema}` per assigned annotation_id, in input order.
"""

OFFSETS = """
Clause offsets: omit `start` and `end` from clause objects. The coordinator derives them programmatically by locating each clause `text` in `candidate_paragraph` (at or after the previous clause) and stores the exact source slice as `text`; evidence quotes are located the same way in their `source`. Copy text and quotes verbatim from the decoded source. Only whitespace encoding (CRLF line endings, non-breaking spaces) and typographic look-alikes (non-breaking or en/em dashes for "-", curly quotes for straight ones) are tolerated; any other difference, including dropped Markdown such as `**`, is rejected.
"""

SC_TEXT = """
`sc_text` is checked against `final.clauses[target_clause_index].text`; copy it verbatim.
"""

# The rubric rule behind each terse helper error, appended in feedback.
RULES = {
    "clauses out of source order": "clause objects follow source order in candidate_paragraph",
    "broad gate aggregation": "clause broad_task_constraint is keep if is_directive and not_main_task are both "
    "yes, drop if either is no, else uncertain",
    "strict gate aggregation": "clause strict_sc is keep if all five gates are yes, drop if any is no, else uncertain",
    "strict keep missing types": "a strict-kept clause needs at least one sc_types entry",
    "strict drop has clause types": "sc_types must be [] for a clause whose strict_sc is drop",
    "broad keep missing kind": "a broad-kept clause needs a non-empty broad_kind",
    "aggregation": "a row label is keep if any clause has that label keep, else uncertain if any is uncertain, else drop",
    "primary_clause_index": "the index is the earliest clause kept under that label, else the earliest uncertain, "
    "else 0 when clauses exist, else -1",
    "keep exclusion reason": "a keep row has exclusion reason none",
    "uncertain exclusion reason": "an uncertain row has exclusion reason uncertain",
    "drop exclusion reason": "a drop row needs a decisive exclusion reason, not none or uncertain",
    "empty clause exclusion reason": "with clauses=[], both broad_exclusion_reason and strict_exclusion_reason "
    "are not_a_constraint",
    "review flag missing reason": "needs_review=true requires a non-empty review_reason",
    "must be flagged": "set needs_review=true with a review_reason when either row label is uncertain or any "
    "broad-kept clause has observable other than yes",
    "target not broad kept": "target_clause_index indexes a broad-kept clause in final.clauses",
    "ready/hold field policy": "a ready probe fills user, compliant, and violating; a hold leaves all three empty",
    "identical probe options": "compliant and violating must differ",
    "macros belong only in display export": "do not write \\compl{} or \\viol{} in any field",
    "held draft marked ready": "a held draft cannot be marked ready",
}


def explain(error: str) -> str:
    rule = next((rule for key, rule in RULES.items() if key in error), None)
    return f"{error} (rule: {rule})" if rule else error


FEEDBACK = """The coordinator validated your response with the frozen helper and rejected it:
{errors}

Return the complete corrected JSON object {{"records": [...]}} for all {count} assigned IDs, not only the corrected ones."""


# Contracts


@dataclass(frozen=True, slots=True)
class Stage:
    """One worker stage: a partition directory in, a result directory out."""

    name: str  # result directory under run_dir, partition file names mirrored
    inputs: str  # partition directory under run_dir/partitions
    kind: str  # helper validator: annotation | types | probes | probe-review
    prompt: str  # key of PROMPT_FILES / DISPATCH
    role: str  # key of cfg.stage_llm overrides
    pass_id: str


CALIBRATION_A = Stage("calibration_A", "calibration", "annotation", "annotate", "annotate_A", "calibration_A")
CALIBRATION_B = Stage("calibration_B", "calibration", "annotation", "annotate", "annotate_B", "calibration_B")
CONTROLS = Stage("controls", "controls", "annotation", "annotate", "controls", "controls")
PASS_A = Stage("pass_A", "production", "annotation", "annotate", "annotate_A", "A")
PASS_B = Stage("pass_B", "production", "annotation", "annotate", "annotate_B", "B")
ADJUDICATION = Stage("adjudication", "adjudication", "annotation", "adjudicate", "adjudicate", "adjudication")
TYPE_A = Stage("type_A", "types", "types", "type", "type_A", "type_A")
TYPE_B = Stage("type_B", "types", "types", "type", "type_B", "type_B")
TYPE_ADJUDICATION = Stage(
    "type_adjudication", "type_adjudication", "types", "type_adjudicate", "type_adjudicate", "type_adjudication"
)
PROBE_DRAFTS = Stage("probe_drafts", "probes", "probes", "probe_draft", "probe_draft", "probe_drafts")


def probe_review_stage(version: int) -> Stage:
    name = f"probe_review_v{version}"
    return Stage(name, name, "probe-review", "probe_review", "probe_review", name)


def probe_revision_stage(version: int) -> Stage:
    name = f"probe_revisions_v{version}"
    return Stage(name, name, "probes", "probe_revise", "probe_revise", name)


@dataclass(frozen=True, slots=True)
class WorkerConfig:
    provider: str
    model: str
    batch: bool
    max_workers: int
    max_attempts: int
    kwargs: dict[str, Any]
    params: dict[str, Any]

    @property
    def key(self) -> str:
        return json.dumps(asdict(self), sort_keys=True)


@dataclass(slots=True)
class Task:
    """One partition's conversation with its worker, across attempts."""

    stage: Stage
    worker: WorkerConfig
    input_path: Path
    output_path: Path
    expected: dict[str, dict[str, Any]]
    messages: list[dict[str, str]]
    attempt: int = 0
    errors: list[str] = field(default_factory=list)


# Helpers


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_helper(package_dir: Path) -> ModuleType:
    """Import a package's scripts/pipeline.py without writing bytecode into it."""
    spec = importlib.util.spec_from_file_location(
        f"annotation_helper_{abs(hash(package_dir))}", package_dir / "scripts" / "pipeline.py"
    )
    helper = importlib.util.module_from_spec(spec)
    previous, sys.dont_write_bytecode = sys.dont_write_bytecode, True
    spec.loader.exec_module(helper)
    sys.dont_write_bytecode = previous
    return helper


def build_runtime(worker: WorkerConfig) -> ModelRuntime:
    config = {"model": worker.model, **worker.kwargs}
    if worker.provider == "openai":
        return OpenAIRuntime(config=config)
    if worker.provider == "vllm":
        from compaction_integrity.runtime.vllm_runtime import VLLMRuntime

        return VLLMRuntime(config={**config, "use_tqdm": False})
    if worker.provider == "vllm_serve":
        from compaction_integrity.runtime.vllm_serve_runtime import VLLMServeRuntime

        return VLLMServeRuntime(config=config)
    raise ValueError(f"Unsupported provider={worker.provider}")


# Typographic characters models copy as their ASCII look-alikes (GitHub issues are
# full of U+2011 non-breaking hyphens and curly quotes). One character to one, so
# an offset in the folded text is the same offset in the source.
FOLD = str.maketrans(
    {
        **dict.fromkeys("‐‑‒–—―−", "-"),
        **dict.fromkeys("‘’‚‛′´", "'"),
        **dict.fromkeys("“”„″", '"'),
    }
)


def locate(source: str, text: str, position: int) -> tuple[int, int] | None:
    """Span of `text` in `source`, exact or up to whitespace encoding and FOLD
    look-alikes, preferring matches at or after `position`."""
    start = source.find(text, position)
    if start < 0:
        start = source.find(text)
    if start >= 0 and text:
        return start, start + len(text)
    words = text.translate(FOLD).split()
    if not words:
        return None
    folded = source.translate(FOLD)
    pattern = re.compile(r"\s+".join(map(re.escape, words)))
    match = pattern.search(folded, position) or pattern.search(folded)
    return (match.start(), match.end()) if match else None


def derive_annotation_spans(record: dict[str, Any], source: dict[str, Any]) -> list[str]:
    """Set clause offsets and exact-slice text/quotes; return location failures."""
    errors: list[str] = []
    paragraph = source["candidate_paragraph"]
    position = 0
    clauses = []
    for n, clause in enumerate(record["clauses"]):
        span = locate(paragraph, clause["text"], position)
        if span is None:
            errors.append(
                f"clauses[{n}].text is not a verbatim substring of candidate_paragraph: {clause['text']!r}"
            )
            clauses.append(clause)
            continue
        start, end = span
        position = start
        rest = {k: v for k, v in clause.items() if k not in ("text", "start", "end")}
        clauses.append({"text": paragraph[start:end], "start": start, "end": end, **rest})
        for m, evidence in enumerate(clause["evidence"]):
            if evidence["source"] not in ("candidate_paragraph", "issue_text"):
                continue
            haystack = source[evidence["source"]]
            span = locate(haystack, evidence["quote"], 0)
            if span is None:
                errors.append(
                    f"clauses[{n}].evidence[{m}].quote is not a verbatim substring of "
                    f"{evidence['source']}: {evidence['quote']!r}"
                )
                continue
            evidence["quote"] = haystack[span[0] : span[1]]
    record["clauses"] = clauses
    return errors


def derive_probe_text(record: dict[str, Any], source: dict[str, Any]) -> list[str]:
    """Store the target clause's exact text when sc_text matches it up to whitespace
    and FOLD look-alikes."""
    clauses = source["final"]["clauses"]
    index = record["target_clause_index"]
    if (
        type(index) is int
        and 0 <= index < len(clauses)
        and record["sc_text"].translate(FOLD).split() == clauses[index]["text"].translate(FOLD).split()
    ):
        record["sc_text"] = clauses[index]["text"]
    return []


DERIVE = {"annotation": derive_annotation_spans, "probes": derive_probe_text}


# Coordinator


class Coordinator:
    def __init__(self, cfg: DictConfig, run: Path, helper: ModuleType):
        self.cfg = cfg
        self.run = run
        self.H = helper
        self.package = run / "frozen" / "package"
        self._runtimes: dict[tuple[str, str, str], ModelRuntime] = {}
        self._log_lock = threading.Lock()

    # Deterministic steps

    def once(self, marker: str, step, **kwargs) -> None:
        """Run a helper step unless its last-written output already exists."""
        if (self.run / marker).exists():
            return
        result = step(SimpleNamespace(run=self.run, **kwargs))
        print(f"[{step.__name__}] {json.dumps(result, ensure_ascii=False, default=str)[:600]}", flush=True)

    # Model workers

    def worker(self, role: str) -> WorkerConfig:
        merged = OmegaConf.merge(self.cfg.llm, self.cfg.stage_llm.get(role, {}))
        return WorkerConfig(**OmegaConf.to_container(merged, resolve=True))

    def runtime(self, worker: WorkerConfig) -> ModelRuntime:
        key = (worker.provider, worker.model, json.dumps(worker.kwargs, sort_keys=True))
        if key not in self._runtimes:
            self._runtimes[key] = build_runtime(worker)
        return self._runtimes[key]

    def system_prompt(self, stage: Stage) -> str:
        sections = [
            f"=== {name} ===\n{(self.package / name).read_text(encoding='utf-8')}"
            for name in PROMPT_FILES[stage.prompt]
        ]
        note = EXECUTION.format(kind=stage.kind, schema=OUTPUT_SCHEMA[stage.kind])
        if stage.kind == "annotation":
            note += OFFSETS
        if stage.kind == "probes":
            note += SC_TEXT
        return "\n\n".join([*sections, note])

    def task(self, stage: Stage, part: Path) -> Task:
        expected = self.H.input_map(part)
        # No pass id in the prompt: passes over one partition send identical
        # requests and sample independently, and the later one reads the whole
        # prompt from cache. Some models (gpt-5.6-terra) cache only at the end of
        # the system prompt and of the full input, so any pass-specific token would
        # cut the shared prefix back to the system prompt. pass_id is logged.
        user = (
            f"{DISPATCH[stage.prompt]}\n\n"
            f"ASSIGNED INPUT ({stage.inputs}/{part.name}, one JSON object per line):\n"
            f"{part.read_text(encoding='utf-8')}\n"
            f"Assigned annotation_ids ({len(expected)}): {', '.join(expected)}"
        )
        return Task(
            stage=stage,
            worker=self.worker(stage.role),
            input_path=part,
            output_path=self.run / stage.name / part.name,
            expected=expected,
            messages=[
                {"role": "system", "content": self.system_prompt(stage)},
                {"role": "user", "content": user},
            ],
        )

    def check(self, task: Task, text: str) -> tuple[list[dict[str, Any]], list[str]]:
        """Parse a response and validate each record with the frozen helper."""
        try:
            records = json.loads(text[text.index("{") : text.rindex("}") + 1])["records"]
            ids = [r["annotation_id"] for r in records]
        except (ValueError, KeyError, TypeError) as e:
            return [], [f'response is not one JSON object {{"records": [...]}} of ID-bearing records: {e!r}']
        errors = []
        missing = [i for i in task.expected if i not in ids]
        extra = [i for i in ids if i not in task.expected]
        duplicated = [i for i, n in Counter(ids).items() if n > 1]
        if missing or extra or duplicated:
            errors.append(f"ID coverage: missing={missing}, extra={extra}, duplicated={duplicated}")
        derive = DERIVE.get(task.stage.kind)
        with tempfile.TemporaryDirectory() as tmp:
            source_path, record_path = Path(tmp) / "input.jsonl", Path(tmp) / "output.jsonl"
            for record in records:
                aid = record["annotation_id"]
                if aid not in task.expected:
                    continue
                try:
                    located = derive(record, task.expected[aid]) if derive else []
                    if located:
                        errors.extend(f"{aid}: {e}" for e in located)
                        continue
                    source_path.write_text(json.dumps(task.expected[aid], ensure_ascii=False) + "\n", encoding="utf-8")
                    record_path.write_text(json.dumps(record, ensure_ascii=False) + "\n", encoding="utf-8")
                    self.H.validate(source_path, record_path, task.stage.kind)
                except (ValueError, KeyError, TypeError, IndexError, AttributeError) as e:
                    errors.append(f"{e}" if str(e).startswith(aid) else f"{aid}: {e!r}")
        by_id = {r["annotation_id"]: r for r in records}
        return [by_id[i] for i in task.expected if i in by_id], errors

    def finish(self, task: Task, response: ModelResponse, dispatched_at: str) -> None:
        task.attempt += 1
        attempts = self.run / "attempts" / task.stage.name
        attempts.mkdir(parents=True, exist_ok=True)
        # Numbered across resumed runs, so earlier attempts are never overwritten.
        stem = f"{task.output_path.stem}.attempt{len(list(attempts.glob(f'{task.output_path.stem}.attempt*.txt'))) + 1}"
        raw_path = attempts / f"{stem}.txt"
        raw_path.write_text(response.text, encoding="utf-8")
        records, task.errors = self.check(task, response.text)
        if not task.errors:
            # Validate the saved file once more as a whole, then publish it.
            checked = attempts / f"{stem}.jsonl"
            self.H.write_jsonl(checked, records)
            self.H.validate(task.input_path, checked, task.stage.kind)
            task.output_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(checked, task.output_path)
        else:
            task.messages = task.messages + [
                {"role": "assistant", "content": response.text},
                {
                    "role": "user",
                    "content": FEEDBACK.format(
                        errors="\n".join(f"- {explain(e)}" for e in task.errors), count=len(task.expected)
                    ),
                },
            ]
        self.log(
            {
                "stage": task.stage.name,
                "pass_id": task.stage.pass_id,
                "input": str(task.input_path.relative_to(self.run)),
                "output": str(task.output_path.relative_to(self.run)),
                "expected_ids": len(task.expected),
                "attempt": task.attempt,
                "provider": task.worker.provider,
                "model_requested": task.worker.model,
                "model_reported": response.model,
                "params": task.worker.params,
                "usage": response.usage,
                "dispatched_at": dispatched_at,
                "completed_at": _now(),
                "status": "invalid" if task.errors else "valid",
                "errors": task.errors[:50],
                "raw_response": str(raw_path.relative_to(self.run)),
            }
        )
        print(
            f"[{task.stage.name}] {task.input_path.name} attempt {task.attempt}: "
            f"{'invalid, ' + str(len(task.errors)) + ' error(s)' if task.errors else 'valid'}",
            flush=True,
        )

    def log(self, entry: dict[str, Any]) -> None:
        with self._log_lock, (self.run / "dispatch_log.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def run_llm(self, stages: list[Stage]) -> None:
        """Dispatch every unfinished partition of `stages`.

        Scheduled for prompt-cache hits: calls on one partition (passes A and B)
        send the same prompt and run back to back in one thread, so the later
        call reads the earlier call's prefix from cache, which concurrent calls
        cannot. A retry extends its conversation and is sent immediately, while
        that prefix is warm. Partitions run in parallel.
        """
        tasks = [
            self.task(stage, part)
            for stage in stages
            for part in sorted((self.run / "partitions" / stage.inputs).glob("part_*.jsonl"))
            if not (self.run / stage.name / part.name).exists()
        ]
        pending = [t for t in tasks if t.worker.batch]
        while pending:
            # Batch workers: one batch per worker config per attempt round.
            by_worker: dict[str, list[Task]] = defaultdict(list)
            for task in pending:
                by_worker[task.worker.key].append(task)
            for group in by_worker.values():
                worker = group[0].worker
                dispatched_at = _now()
                responses = self.runtime(worker).batch_generate(
                    [t.messages for t in group], model=worker.model, params=worker.params
                )
                for task, response in zip(group, responses):
                    self.finish(task, response, dispatched_at)
            pending = [t for t in pending if not t.output_path.exists() and t.attempt < t.worker.max_attempts]

        by_partition: dict[Path, list[Task]] = defaultdict(list)
        for task in tasks:
            if not task.worker.batch:
                by_partition[task.input_path].append(task)

        def run_partition(group: list[Task]) -> None:
            for task in group:
                runtime = self.runtime(task.worker)
                while not task.output_path.exists() and task.attempt < task.worker.max_attempts:
                    dispatched_at = _now()
                    response = runtime.generate(task.messages, model=task.worker.model, params=task.worker.params)
                    self.finish(task, response, dispatched_at)

        with ThreadPoolExecutor(max_workers=int(self.cfg.llm.max_workers)) as pool:
            list(pool.map(run_partition, by_partition.values()))
        exhausted = [t for t in tasks if not t.output_path.exists()]
        if exhausted:
            raise RuntimeError(
                "worker output still invalid after max_attempts: "
                + "; ".join(f"{t.stage.name}/{t.input_path.name}: {t.errors[:3]}" for t in exhausted)
            )

    def run_rows(self, source: Path, output: Path, stage: Stage) -> None:
        """Review/revise a row file in bounded calls, then join with exact coverage."""
        if output.exists():
            return
        parts = self.run / "partitions" / stage.inputs
        if not parts.exists():
            rows = self.H.read_jsonl(source)
            size = int(self.cfg.probe_rows_per_call)
            for n, start in enumerate(range(0, len(rows), size), 1):
                self.H.write_jsonl(parts / f"part_{n:03d}.jsonl", rows[start : start + size])
        self.run_llm([stage])
        records = [
            r
            for part in sorted(parts.glob("part_*.jsonl"))
            for r in self.H.read_jsonl(self.run / stage.name / part.name)
        ]
        self.H.write_jsonl(output, records)
        self.H.validate(source, output, stage.kind)

    # Pipeline

    def calibrate(self) -> None:
        stages = [CALIBRATION_A, CALIBRATION_B]
        controls = any((self.run / "partitions" / "controls").glob("part_*.jsonl"))
        if controls:
            stages.append(CONTROLS)
        self.run_llm(stages)
        self.once("calibration_comparison.json", self.H.review, stage="calibration")
        if controls:
            self.once("control_scores.json", self.H.score_controls)
        if not (self.run / "calibration_report.md").exists():
            self.H.write_text(self.run / "calibration_report.md", self.calibration_report())

    def annotate(self) -> None:
        self.run_llm([PASS_A, PASS_B])
        self.once("production_comparison.json", self.H.review, stage="production")
        self.run_llm([ADJUDICATION])
        self.once("adjudication_log.jsonl", self.H.merge)

    def type(self) -> None:
        self.once("types_manifest.json", self.H.prepare_types)
        self.run_llm([TYPE_A, TYPE_B])
        self.once("type_adjudication_manifest.json", self.H.review_types)
        self.run_llm([TYPE_ADJUDICATION])
        self.once("typed_records.jsonl", self.H.merge_types)

    def probe(self) -> int:
        """Draft, then review and revise until every disposition is ready or hold."""
        self.once("probes_manifest.json", self.H.prepare_probes)
        self.run_llm([PROBE_DRAFTS])
        self.once("probes_v1.jsonl", self.H.collect_probes, output=self.run / "probes_v1.jsonl")
        version = 1
        while True:
            probes = self.run / f"probes_v{version}.jsonl"
            review_input = self.run / f"probe_review_input_v{version}.jsonl"
            review = self.run / f"probe_review_v{version}.jsonl"
            revision_input = self.run / f"probe_revision_input_v{version}.jsonl"
            revisions = self.run / f"probe_revisions_v{version}.jsonl"
            self.once(review_input.name, self.H.prepare_probe_review, probes=probes, output=review_input)
            self.review_probes(version)
            self.once(
                revision_input.name,
                self.H.prepare_revision,
                input=review_input,
                review=review,
                output=revision_input,
            )
            pending = len(self.H.read_jsonl(revision_input))
            if not pending:
                return version
            if version > int(self.cfg.max_revision_rounds):
                raise RuntimeError(
                    f"{pending} probe(s) still need revision after {self.cfg.max_revision_rounds} round(s) "
                    f"(see {review.name}); raise max_revision_rounds and rerun to resume"
                )
            self.run_rows(revision_input, revisions, probe_revision_stage(version))
            self.once(
                f"probes_v{version + 1}.jsonl",
                self.H.apply_revision,
                probes=probes,
                input=revision_input,
                revisions=revisions,
                output=self.run / f"probes_v{version + 1}.jsonl",
            )
            version += 1

    def review_probes(self, version: int) -> None:
        """Consolidated review of every draft in this version.

        A draft identical to the previous version's keeps that version's ready or
        hold verdict; only new and revised drafts go to the reviewer. Re-reviewing
        settled drafts would re-sample their verdicts, so every round would flag a
        fresh few and the loop would not converge.
        """
        review = self.run / f"probe_review_v{version}.jsonl"
        if review.exists():
            return
        review_input = self.run / f"probe_review_input_v{version}.jsonl"
        rows = self.H.read_jsonl(review_input)
        carried = {}
        if version > 1:
            previous_input = self.H.indexed(self.H.read_jsonl(self.run / f"probe_review_input_v{version - 1}.jsonl"))
            previous = self.H.indexed(self.H.read_jsonl(self.run / f"probe_review_v{version - 1}.jsonl"))
            carried = {
                r["annotation_id"]: previous[r["annotation_id"]]
                for r in rows
                if r["draft"] == previous_input[r["annotation_id"]]["draft"]
                and previous[r["annotation_id"]]["disposition"] in ("ready", "hold")
            }
        pending = self.run / f"probe_review_pending_v{version}.jsonl"
        if not pending.exists():
            self.H.write_jsonl(pending, [r for r in rows if r["annotation_id"] not in carried])
        fresh = self.run / f"probe_review_fresh_v{version}.jsonl"
        self.run_rows(pending, fresh, probe_review_stage(version))
        reviewed = self.H.indexed(self.H.read_jsonl(fresh))
        self.H.write_jsonl(review, [carried.get(r["annotation_id"]) or reviewed[r["annotation_id"]] for r in rows])
        self.H.validate(review_input, review, "probe-review")
        print(f"[probe_review_v{version}] {len(reviewed)} reviewed, {len(carried)} verdicts carried over", flush=True)

    def finalize(self, version: int) -> None:
        """Helper export and final validation, then the coordinator's provenance section."""
        if (self.run / "run_report.md").exists():
            return
        self.once(
            "run_report.md",
            self.H.finalize,
            probes=self.run / f"probes_v{version}.jsonl",
            review_input=self.run / f"probe_review_input_v{version}.jsonl",
            review=self.run / f"probe_review_v{version}.jsonl",
        )
        # Appended in the same step, so a resumed run never appends twice.
        with (self.run / "run_report.md").open("a", encoding="utf-8") as f:
            f.write(self.coordinator_report(version))

    # Reports

    def calibration_report(self) -> str:
        comparison = json.loads((self.run / "calibration_comparison.json").read_text())
        selection = json.loads((self.run / "calibration_selection.json").read_text())
        lines = [
            "# Calibration report",
            "",
            "Written by the automated coordinator (`compaction_integrity.dataset.swe_natural_curation.annotate_sc`). No researcher "
            "inspected calibration before production, and the frozen rubric was not changed between calibration "
            "and production. Calibration rows are annotated again in production and counted once there.",
            "",
            f"Selection: `{selection['mode']}`, seed {selection['seed']}, "
            f"{len(selection['annotation_ids'])} candidates.",
            "",
            "| Label | A/B agreement (candidates) | Both keep / either keep (candidates) | Pass A counts | Pass B counts |",
            "|---|---|---|---|---|",
        ]
        for label in ("broad", "strict"):
            a = comparison[label]
            lines.append(
                f"| {label} | {a['agreed']}/{a['total']} | {a['both_keep']}/{a['either_keep']} | "
                f"{a['pass_A_counts']} | {a['pass_B_counts']} |"
            )
        lines += [
            "",
            f"Calibration review queue: {comparison['review_count']} candidates in `calibration_review_queue.jsonl` "
            "(left for researcher inspection; not adjudicated).",
            "",
        ]
        if (self.run / "control_scores.json").exists():
            scores = json.loads((self.run / "control_scores.json").read_text())
            lines += [
                "Constructed controls (outside production counts), label matches the control key:",
                "",
                "| Label | Correct / controls |",
                "|---|---|",
                *(f"| {k} | {v['correct']}/{v['total']} |" for k, v in scores.items()),
                "",
            ]
        return "\n".join(lines)

    def coordinator_report(self, version: int) -> str:
        stats: dict[str, dict[str, Any]] = defaultdict(
            lambda: {"calls": 0, "valid": 0, "models": set(), "input": 0, "cached": 0, "output": 0}
        )
        for entry in self.H.read_jsonl(self.run / "dispatch_log.jsonl"):
            s = stats[entry["stage"]]
            s["calls"] += 1
            s["valid"] += entry["status"] == "valid"
            s["models"].add(f"{entry['provider']}:{entry['model_requested']} -> {entry['model_reported']}")
            s["input"] += entry["usage"].get("prompt_tokens", 0)
            s["cached"] += entry["usage"].get("cached_tokens", 0)
            s["output"] += entry["usage"].get("completion_tokens", 0)
        lines = [
            "",
            "## Coordinator (automated)",
            "",
            "Coordinator: `compaction_integrity.dataset.swe_natural_curation.annotate_sc`; no researcher review occurred during the run. "
            "Each partition of each pass is a separate API call with no shared context (passes A and B of a "
            "partition send identical prompts and share only the provider prompt cache; each call samples independently); A/B passes use the same "
            "worker configuration unless `stage_llm` overrides one. Clause offsets were derived programmatically from "
            "the returned clause text, storing exact source slices. Every worker output was validated record by "
            "record with the frozen helper; rejected attempts went back to the same worker with the errors and are "
            f"kept under `attempts/`. Probe review rounds after the first re-reviewed only revised drafts; unchanged "
            "drafts kept their earlier ready/hold verdict (`probe_review_pending_v*.jsonl` lists what each round sent). "
            f"Final probe version: v{version}.",
            "",
            "| Stage | Worker (requested -> reported model) | Valid / total calls | Input tokens "
            "| Cached input tokens (% of input) | Output tokens |",
            "|---|---|---|---|---|---|",
        ]
        for stage, s in stats.items():
            lines.append(
                f"| {stage} | {'; '.join(sorted(s['models']))} | {s['valid']}/{s['calls']} | {s['input']:,} "
                f"| {s['cached']:,} ({s['cached'] / max(s['input'], 1):.0%}) | {s['output']:,} |"
            )
        return "\n".join(lines) + "\n"

    # Export

    # The column layout of the first batches' sc_annotations.broad_keeps.typed*.csv,
    # then the probe fields those sheets left implicit.
    ROW_COLUMNS = (
        "broad_task_constraint",
        "sc_type",
        "type_rationale",
        "type_needs_review",
        "type_provenance",
        "type_pass_A",
        "type_pass_B",
        "broad_kind",
        "broad_constraint_text",
        "broad_clauses_json",
        "strict_sc",
        "rationale",
        "rubric_version",
        "annotation_provenance",
    )
    SOURCE_COLUMNS = (
        "annotation_id",
        "source_row_index",
        "trajectory_id",
        "instance_id",
        "keyword_hits",
        "n_keyword_hits",
        "candidate_paragraph",
        "issue_text",
    )
    PROBE_EXTRAS = ("sc_text", "probe_status", "probe_target_clause_index", "probe_sc_text", "probe_note")
    # sc_annotations.broad_keeps-nvidia-swe.csv layout: one row per probe.
    CLAUSE_COLUMNS = ("annotation_id", "clause_index", "sc_text", "sc_type", "status", "probe", "qa_note")

    def export(self, version: int) -> None:
        out = Path(str(self.cfg.export_dir))
        mapping = json.loads((self.run / "final_validation.json").read_text())["new_column_mapping"]
        header, broad = self.H.read_csv(self.run / "sc_annotations.broad_keeps.typed.csv")
        typed = self.H.indexed(self.H.read_jsonl(self.run / "typed_records.jsonl"))
        reviews = self.H.indexed(self.H.read_jsonl(self.run / f"probe_review_v{version}.jsonl"))
        source_columns = [c for c in self.SOURCE_COLUMNS if c in header]

        rows, clause_rows = [], []
        for row in broad:
            aid = row["annotation_id"]
            derived = {k: row[v] for k, v in mapping.items()}
            t = typed[aid]
            rows.append(
                {
                    **{c: derived[c] for c in self.ROW_COLUMNS if c in derived},
                    "type_pass_A": t["type_pass_A"]["sc_type"],
                    "type_pass_B": t["type_pass_B"]["sc_type"],
                    **{c: row[c] for c in source_columns},
                    "probe": derived["probe"],
                    **{c: derived[c] for c in self.PROBE_EXTRAS},
                }
            )
            target = t["final"]["clauses"][int(derived["probe_target_clause_index"])]
            status = derived["probe_status"]
            note = f"HOLD. {reviews[aid]['reason']}" if status == "hold" else derived["probe_note"]
            clause_rows.append(
                {
                    "annotation_id": aid,
                    "clause_index": derived["probe_target_clause_index"],
                    "sc_text": derived["probe_sc_text"],
                    "sc_type": derived["sc_type"],
                    "status": status,
                    "probe": derived["probe"],
                    "qa_note": f"[{target['broad_kind']}] {note} Offset check passed: "
                    f"paragraph[{target['start']}:{target['end']}] == clause text.",
                }
            )
        row_header = [*self.ROW_COLUMNS, *source_columns, "probe", *self.PROBE_EXTRAS]
        ready = [r for r in rows if r["probe_status"] == "ready"]
        for name, header_, data in (
            ("sc_annotations.broad_keeps.typed.csv", row_header, rows),
            ("sc_annotations.broad_keeps.probe_ready.csv", row_header, ready),
            ("sc_annotations.broad_keeps.probes.csv", list(self.CLAUSE_COLUMNS), clause_rows),
        ):
            if not (out / name).exists():
                self.H.write_csv(out / name, header_, data)
        print(
            f"[export] {len(rows)} broad keeps, {len(ready)} ready probes, "
            f"{len(rows) - len(ready)} held -> {out}",
            flush=True,
        )


# Entry point


@hydra.main(version_base=None, config_path="../../../../config/tasks/swe_natural", config_name=None)
def main(cfg: DictConfig) -> None:
    apply_runtime_environment()
    run = Path(str(cfg.run_dir)).resolve()
    if not (run / "run_manifest.json").exists():
        package = PACKAGE_DIR if cfg.package_dir is None else Path(str(cfg.package_dir)).resolve()
        prepare = load_helper(package).prepare
        result = prepare(
            SimpleNamespace(
                input=Path(str(cfg.input_csv)),
                run=run,
                seed=int(cfg.seed),
                mode=str(cfg.mode),
                calibration_ids=None if cfg.calibration_ids is None else Path(str(cfg.calibration_ids)),
            )
        )
        print(f"[prepare] {json.dumps(result, ensure_ascii=False)}", flush=True)

    coordinator = Coordinator(cfg, run, load_helper(run / "frozen" / "package"))
    coordinator.calibrate()
    if str(cfg.mode) == "calibration":
        print(f"[calibration] report: {run / 'calibration_report.md'}")
        return
    coordinator.annotate()
    coordinator.type()
    version = coordinator.probe()
    coordinator.finalize(version)
    coordinator.export(version)


if __name__ == "__main__":
    main()
