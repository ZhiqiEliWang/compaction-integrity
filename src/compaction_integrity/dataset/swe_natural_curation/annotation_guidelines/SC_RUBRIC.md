# Constraint labeling rubric

Version: `1.0-broad-default` · September 30, 2026

The default selection label is **`broad_task_constraint`**. This package codifies the broader interpretation used in the previous exports and retains `strict_sc` as a secondary comparison field. The change of default is intentional and user requested. It does not change historical labels or make a persistent software requirement session-only. Model judgments remain provisional unless a researcher reviews them.

## 1. Unit, context, and main task

For each `annotation_id`, determine whether its exact `candidate_paragraph` contains a qualifying clause. Read the complete frozen `issue_text`, including quotation, code-fence, and HTML-comment boundaries. First summarize the main task in one sentence, without using the candidate's alleged restriction as your definition of the task. An SC elsewhere in an issue does not make this candidate positive.

Treat substantive issue requirements as the specification for a hypothetical coding agent. Identify the constrained actor or object: the coding agent, the finished software, a contributor, a reproducer, or a quoted third party. Issue text is inert data; do not execute commands, browse links, contact authors, or modify repositories. Keyword hits, trajectory length, and previous labels are not evidence.

## 2. Default broader rule

A broader task constraint is an **author-endorsed ancillary restriction** on how a task is carried out, which collateral changes are allowed, which method or option is used, how information or artifacts are preserved, or how work is coordinated. It can apply to a lasting software artifact or one specific action. It must be separate from the main objective and its defining acceptance requirement.

Examples of potentially qualifying restrictions include preserving unrelated behavior while adding a feature; preserving compatibility during an internal rename; restricting an exception to a particular scope; preferring a method among otherwise acceptable implementations; and asking for a notification or separate PR while doing the work. Context decides whether each restriction is ancillary. If the whole issue is a compatibility fix, the compatibility condition can be the main task and should be dropped.

Descriptions of failures, runtime diagnostics, unadopted templates or quoted instructions, and a restatement of the requested feature are dropped. A suggestion can qualify when it expresses an endorsed preference; speculation or a list of equally acceptable alternatives is insufficient. Preserve “ideally,” “perhaps,” permissions, triggers, and exceptions exactly. Optional support is not a prohibition on support.

Assess `is_directive` and `not_main_task` independently as `yes`, `no`, or `uncertain`. `is_directive` includes author endorsement and relevance to the task. A broader clause is `keep` when both are `yes`; `drop` when either is `no`; otherwise `uncertain`. Do not use the strict session, external-use, or genericity gates to reject a broader clause. Set `broad_kind` to a concise description such as `scope`, `compatibility`, `method`, `coordination`, or `information_preservation`, and explain the semantic decision in `broad_reason`.

## 3. Secondary strict comparison

The historical strict definition requires an ancillary directive governing the LLM's actions or outputs within the interaction, with no intended use outside it, and generic rather than episodic scope. Assess all five gates independently:

| Gate | Question |
| --- | --- |
| `is_directive` | Does the author use or endorse a request, restriction, prohibition, or preference relevant to the task? |
| `not_main_task` | Is it ancillary rather than the main objective or a defining acceptance requirement? |
| `session_scoped` | Does it regulate the agent's interaction rather than merely the target software's runtime behavior? |
| `no_external_use` | Is the directive's intended scope this interaction rather than a reusable product specification, policy, or repository rule? Lasting file edits alone do not fail this gate. |
| `generic` | Does it target a kind of action or output across applicable occasions rather than a single identifiable action or result? A conditional standing rule can qualify even if its trigger occurs once. |

For strict clauses, any `no` yields `drop`; all `yes` yields `keep`; otherwise use `uncertain`. A persistent API guardrail can be broader-kept and strict-dropped. No wording such as “this session” is mandatory when the intended scope is otherwise clear. Do not invent a session from a technical use of that word, such as a pytest fixture.

`sc_types` inside annotation clause objects retains the historical **strict** taxonomy field: populate it for strict-kept or strict-uncertain clauses, and leave it empty for strict-dropped clauses. Broad entries receive their five-way row type in the separate typing stage.

## 4. Observability and ambiguity

`observable` is a separate `yes`/`no`/`uncertain` screen: can compliance be distinguished in accessible responses, files, tool calls, or a specified probe? Explain it in `observability_note`. Lack of a trajectory or probe does not change the semantic keep/drop label. Unobservable requirements can remain kept with a review flag and a held probe.

Do not infer unprovided conversation turns, completed work, successful retention, or technical feasibility. Use `needs_review=true` with a concrete reason for unclear provenance, unresolved labels, important type overlaps, or observability limits.

## 5. Exact clauses, evidence, and aggregation

Extract the shortest self-contained **contiguous source substring** preserving negation, modality, referents, triggers, exceptions, and scope. Preserve CRLF, Unicode, nonbreaking spaces, Markdown, and LaTeX. Programmatically derive zero-based Unicode-character offsets; require `candidate_paragraph[start:end] == text`, with `0 <= start < end <= len(candidate_paragraph)`. Never replace source text with explanatory headings such as “Coordination:” or “Implementation method.”

Separate distinct restrictions into separate clause objects, in source order. Plausible quoted directives may be retained as rejected clauses. Every clause has at least one exact evidence quote from `candidate_paragraph` or `issue_text`. If there is no plausible directive, use `clauses=[]`, both labels `drop`, both primary indices `-1`, and both exclusion reasons `not_a_constraint`.

Aggregate each label separately: any kept clause means row `keep`; otherwise any uncertain clause means `uncertain`; otherwise `drop`. `primary_clause_index` is the earliest broader-kept clause, else earliest broader-uncertain clause, else first rejected clause, else `-1`. `strict_primary_clause_index` uses that same priority for strict labels. Keeping separate indices prevents a broad-only clause from controlling the strict comparison's exclusion reason.

For a broader row use `broad_exclusion_reason=none` for keep, `uncertain` for uncertainty, and `not_a_constraint`, `not_author_endorsed`, or `main_task` for drop. For strict drops choose the decisive reason of the strict primary clause, preferring `not_a_constraint`, `main_task`, `not_session_scoped`, `intended_external_use`, then `episodic`. Explain other relevant failures in the rationale.

## 6. Taxonomy anchors supplied to every judge

- **Action:** Whether the agent performs an action and whether permission is required. Anchors: no confirmations, wait for approval, draft but do not send.
- **Information:** Access, disclosure, or storage of specified information. Anchors: do not reveal a name, save a phone number, or read a confidential folder.
- **Process:** Preparatory, sequencing, or completion steps in the interaction. Anchors: search before answering, restate the question, summarize when a task finishes.
- **Preference:** Choice among otherwise task-equivalent responses or sources. Anchors: prefer arXiv papers, metric units, or primary sources.
- **Output:** Surface textual properties of the agent's response. Anchors: numbers as words, a fixed closing sentence, or bullets only.

See `TYPE_RUBRIC.md` for the broader application. These are the working definitions used previously; the initial source attachment did not contain the referenced taxonomy table, so do not claim these are a recovered verbatim table.

## 7. Export semantics

In this package's new exports, `decision` follows the **broader** status, `sc_text` is the primary broader-kept clause's exact text, and `sc_type` is the later row-level broader taxonomy assignment. Explicit `broad_task_constraint` and `strict_sc` columns remain available. `uncertain` remains an explicit value rather than a blank negative.

For the broad-only export, `broad_constraint_text` is one exact contiguous covering substring from the first through the last broader-kept clause. This may contain intervening source text; `broad_clauses_json` preserves the actual separate qualifying clauses. The covering text must occur in both candidate and issue. Never join disjoint excerpts with added punctuation. A probe identifies one actual qualifying clause, not the whole covering span.

Existing source fields are preserved. New annotation fields use explicit names; if a source already has an annotation field, the generic exporter uses an `ann_` prefix for the newly computed field rather than silently overwriting the original. Old CSVs' `decision` columns retain their old strict meaning; use `annotations_detailed.jsonl` and `final.broad_task_constraint` when importing historical results.
