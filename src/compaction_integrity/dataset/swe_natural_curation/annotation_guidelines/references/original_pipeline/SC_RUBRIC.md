# Side-constraint annotation rubric

Version: `0.1-draft` · Prepared September 5, 2026

This is a proposed operationalization for calibration, not an approved gold standard. Preserve the written SC definition as the primary label and record a separate broader label. The broader label must never silently replace the strict label. A run may proceed under this disclosed draft; its results remain provisional until researcher review.

## 1. Definition and annotation unit

The researcher defines a side constraint (SC) as a clause in a user's prompt such that **(1)** it is not part of the user's task, **(2)** it is meant to constrain the LLM's decoding within the same session, and **(3)** it has no intended use outside that session. Linguistically, it is **generic** (targets a kind of action or output), rather than **episodic** (targets one specific action in the current turn).

Annotate whether the supplied `candidate_paragraph` **contains at least one** qualifying clause. Read the complete `issue_text` to establish the task, author, scope, and surrounding quotation/template boundaries. An SC elsewhere in the issue does not make this candidate positive. Do not reconstruct missing conversation turns from `trajectory_id`, infer a session from software terms such as pytest's `session`, or rewrite a candidate into an SC.

Treat the stored issue as the task specification supplied to a hypothetical coding agent. This makes substantive issue requirements relevant to that agent's task; it does **not** turn every embedded instruction, software specification, or error message into a session directive. These files contain no trajectories with which to establish post-compaction retention or actual compliance.

## 2. Two separate judgments

**`strict_sc`** applies the definition above. All five gates in section 3 must pass for at least one clause.

**`broad_task_constraint`** asks whether the candidate contains an author-endorsed, ancillary restriction on how the task is carried out, what collateral changes are allowed, which method is used, or how work is coordinated. It may govern a lasting artifact or one specific action. It must still be separate from the main objective or its defining acceptance requirement. Descriptions, quoted diagnostics, unadopted templates, and a restatement of the requested feature are not broad positives. Thus this field is broader in scope and duration, but is not a label for every software requirement.

Use `keep`, `drop`, or `uncertain` for each judgment. These are semantic labels, not sampling or deduplication decisions. For example, a generic instruction to the agent to ask before each send can pass strict SC; a requirement that the finished application ask before each send normally fails the session/external-use gates. A secondary compatibility guardrail can be broad-positive but strict-negative. If compatibility is the entire requested fix, it can be negative on both labels.

**Unresolved research boundary:** the eight natural examples supplied by the researcher include lasting software requirements. Their summaries do not establish session-only scope. Preserve them as boundary motivations, not unconditional strict-positive gold examples. Similar text in different issues must be judged in its own context. The current CSV has autopep8 **#466** and JWST **#7978**, not the supplied #456 and #7973; do not conflate these issue numbers.

## 3. Clause-level gates

Use `yes`, `no`, or `uncertain` independently for every gate. A failed gate does not entitle an annotator to invent answers to other gates.

| Field | Question and operational rule |
|---|---|
| `is_directive` | Does the author use or endorse this clause to specify, request, prohibit, or prefer behavior relevant to the task? Soft preferences can qualify. An expected-software-behavior requirement can pass this gate. A negated factual statement, runtime error, or unadopted third-party instruction cannot. |
| `not_main_task` | Is this an ancillary restriction, rather than the main objective or a defining acceptance requirement? First summarize the task independently. The fact that a requirement is in a later paragraph, optional implementation note, or negative sentence does not settle this gate. A restriction on an otherwise task-equivalent choice can be ancillary. |
| `session_scoped` | Does this govern the LLM/agent's outputs or actions during the interaction? Explicit session wording is sufficient when consistent with context, but is not mandatory: a clearly standing instruction about the agent's replies can imply it. Software runtime behavior, contributor policies, and persistent repository rules do not establish this scope. If the intended recipient or scope is genuinely unresolved, use `uncertain`. |
| `no_external_use` | Is the directive intended only to regulate this interaction, rather than to define reusable software behavior, a policy, documentation, or another enduring specification? Assess the **directive's intended scope**, not whether an action has lasting effects. A session instruction about how to edit files can qualify even though the edited files persist. A public API compatibility requirement normally cannot. |
| `generic` | Does this target a kind of action/output across applicable occasions, rather than one identifiable action or result? A conditional rule can be generic even if its trigger happens once or never. Mere repetition of steps in a reproduction recipe does not make them generic. Software invariants can be generic while failing other gates. |

For a clause: any `no` means `strict_sc=drop`; all `yes` means `keep`; otherwise `uncertain`. For a paragraph: any kept clause means `keep`; otherwise any uncertain clause means `uncertain`; otherwise `drop`. With no plausible operative or quoted directive to assess, return an empty clause list and `drop` with `not_a_constraint`. Do not force uncertainty when an independently decisive failed gate establishes a drop.

## 4. Observability is a separate screening field

`observable` asks whether compliance and violation can be distinguished in accessible outputs, tool calls, files, or an explicitly specified behavioral probe. Use `yes`, `no`, or `uncertain`. Record a short `observability_note`. This is an additional suitability screen in the CSV, **not a fourth defining qualifier**. Do not drop a semantically valid SC solely because a particular trajectory or probe is unavailable. Retain semantic status and flag missing evaluability for review. An output-format rule normally has an observable violation; an instruction only about unobservable private thinking may not.

## 5. Common boundaries

| Situation | Rule |
|---|---|
| “I don't know,” “cannot parse,” “never populated” | These often describe knowledge or failures. Keywords alone supply no SC evidence. Read the entire candidate for any separate directive. |
| “X should not happen” in Expected behavior | Frequently states the bug fix itself. Identify the task and target program before assigning a process or output type. |
| “Do not edit manually” in a generated file | Identify it as quoted generated-file guidance. Do not adopt it as the issue author's instruction to the agent without contextual evidence. |
| “Do not ask in both places” in an HTML issue template | It is a contributor instruction. Retain provenance; do not elevate it to a user session rule. HTML comments and code fences may start before the candidate. |
| “Use Git/PyPI sources” inside an exception | Advice emitted by software remains diagnostic content unless the author adopts it. A natural-language imperative inside a traceback is not enough. |
| “Do not modify the persistency folder” in To replicate | It applies to one reproducer and preserves its setup. Usually episodic and part of the reproduction task. |
| “Before merging, test for WIP” in a bot issue | Determine whether “merging” is the target bot's future behavior or the annotating agent's workflow. Do not decide from syntax alone. |
| “Whenever bases changes, update CI” in source comments | Generic, but normally a persistent repository maintenance rule with external use. |
| “Use linear interpolation” in an implementation note | May be an ancillary method constraint; normally also defines the delivered algorithm. Evaluate the two labels separately. |
| A soft or hedged preference | Do not require “must,” “always,” or “from now on.” Distinguish a preferred restriction from speculation or a list of equally acceptable options. |
| A one-time coordination request | May be a broad ancillary constraint. A request to ping someone when starting one identified fix is usually episodic. |
| Same paragraph in multiple records | Annotate each record in context. Exact text identity alone is not grounds to delete a record or set semantic status to `duplicate`. |

## 6. Provisional taxonomy

Use these working definitions based on the researcher's paragraph and 15 probes. The supplied LaTeX references `tables/taxonomy` but does not contain that table; these descriptions are **not** a recovered copy of the missing table. Preserve the supplied example assignments until the researcher revises them.

| Type | What it binds | Anchors from the supplied probes |
|---|---|---|
| Action | Whether the agent performs an action and whether permission is required. | No confirmations; wait for approval; draft but do not send. |
| Information | Access, disclosure, or storage of specified information. | Do not reveal a name, save a phone number, or read a confidential folder. |
| Process | Preparatory, sequencing, or completion steps in the interaction. | Search before answering; restate the question; summarize when a task finishes. |
| Preference | Choice among otherwise task-equivalent responses or sources. | Prefer arXiv papers, metric units, or primary sources. |
| Output | Surface textual properties of the agent's response. | Numbers as words; a fixed closing sentence; bullets only. |

Assign types only to strict-kept or strict-uncertain clauses, not to rejected software requirements. Use the constrained object and these anchors to resolve overlaps: information-specific access restrictions are Information; metric units and source choices are Preference; the task-completion summary is Process under the supplied taxonomy. If overlap remains material, put the best primary type first in `sc_types`, include the alternative, and flag review. Broad-only constraints use a short free-text `broad_kind`, such as `method`, `scope`, or `coordination`, rather than forcing them into this taxonomy.

## 7. Evidence and output contract

The machine-readable contract is `ANNOTATION_SCHEMA.json`, applied to each line of a JSONL result. Record one result per `annotation_id`. Keep `primary_task` to one sentence and `rationale` to one to three sentences identifying the decisive evidence, not a long deliberation.

Extract the shortest **self-contained contiguous** clause that preserves its negation, trigger, exceptions, and scope. `text` must be an exact substring of the original `candidate_paragraph`. Do not normalize line endings, non-breaking spaces, smart quotes, Markdown, or LaTeX in the machine-readable evidence. Offsets are zero-based Unicode character indices, as in Python string slicing: `candidate_paragraph[start:end] == text`. Derive offsets programmatically after choosing the span. Separate distinct constraints into separate clause objects, including dropped plausible candidates when useful for explaining a boundary. A quoted directive may be retained as a rejected clause with `is_directive=no`.

Every clause includes one or more exact supporting excerpts from either `candidate_paragraph` or `issue_text`. Context excerpts must be copied exactly. Absence of a session marker alone is not positive evidence for `no`; explain the actual target or provenance. List clauses in source order. Select the earliest kept clause as `primary_clause_index`; if none, the earliest uncertain clause; if none, the first rejected clause; use `-1` for an empty list. This primary selection controls the legacy CSV's single-span fields only; it never discards additional clauses from JSONL.

Use `strict_exclusion_reason=none` for keep and `uncertain` for unresolved row status. For drop, choose the decisive reason for the primary rejected clause, preferring `not_a_constraint`, then `main_task`, then `not_session_scoped`, then `intended_external_use`, then `episodic` when several independently apply. Explain important additional failures in the rationale.

## 8. Mapping to the existing CSV

The original first eight columns remain the only editable source columns. Preserve all other values and row order. Write a reviewed output copy first; the orchestrator handles any final replacement only as explicitly requested in its launch instruction.

| Original CSV field | Mapping |
|---|---|
| `decision (keep/drop)` | The resolved **strict** semantic label. Leave blank for unresolved strict status; identify it in a separate review queue. |
| `not_main_task (yes/no)` | Primary clause's gate. Leave blank for `uncertain` or no assessed clause; do not fabricate a Boolean. |
| `session_scoped (yes/no)` | Primary clause's gate, with the same blank policy. |
| `observable (yes/no)` | Primary clause's observability, with the same blank policy. |
| `sc_text (exact substring from candidate_paragraph)` | Primary kept clause's exact text. Blank for drop or unresolved rows. |
| `sc_type (Action/Information/Process/Preference/Output)` | Primary kept clause's first type. Flag unresolved type overlap and preserve all types in JSONL. |
| `exclusion_reason (...)` | `not_a_constraint`, `main_task`, or `not_session_scoped` directly; map `intended_external_use` to `not_session_scoped` and preserve the precise reason in the note; map `episodic` to `other` with explanation. Blank for keep or unresolved rows. `not_observable` and `duplicate` are not automatic semantic exclusions in this draft. |
| `note (optional free text)` | Concise rationale plus broad status, draft version, and any review flag. |

Preserve the separate `no_external_use`, `generic`, broad judgment, evidence, and uncertainty in the detailed JSONL. Do not overwrite a nonempty manual annotation automatically. Resolve such conflicts in a review queue.

## 9. Calibration and review

`calibration/calibration_set.csv` contains 25 original records with blank annotation cells. `calibration/calibration_input.jsonl` presents the same cases without keyword scores or labels. Ten rows were selected using Python `random.Random(20260905).sample(range(208), 10)`; fifteen additional rows were deliberately selected for boundary coverage. This mixed sample is for calibrating decisions, not estimating prevalence.

`calibration/CALIBRATION_REVIEW.md` and `calibration/proposed_annotations.jsonl` contain single-assistant proposals, not human gold or independent agreement results. Keep them and their label distributions hidden from first-pass annotators. There is no required class balance and no target number of SCs.

`calibration/controls_input.jsonl` contains benchmark-derived examples and synthetic contrasts. Their added session/task wrappers and expected judgments are marked in `calibration/controls_key.jsonl`. They are not dataset discoveries and must never be merged into the source CSV. Benchmark type assignments follow the researcher. Keep the key and its label distribution hidden during control evaluation; control IDs carry no semantic label.

Review disagreements on individual gates, not only keep/drop. Human review should cover all proposed positives, strict uncertainty, taxonomy ambiguity, and a reproducible sample of agreed negatives. If a material definition change is made, increment the rubric version and re-annotate affected records consistently across both passes. Report raw disagreement counts and class-specific agreement; an all-drop agreement rate alone does not validate the annotator. Record any remaining uncertainty rather than claiming the file is fully resolved.
