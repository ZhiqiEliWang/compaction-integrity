# Five-type taxonomy for broader task constraints

Version: `1.0-broad-default`. Assign exactly one primary `sc_type` per broader-kept **entry**, after semantic annotation and adjudication. This reproduces the previous row-level typing stage. Multiple clause objects remain available in the detailed record.

## Definitions and anchors

- **Action:** Whether the agent performs an action and whether permission is required. Original anchors: no confirmations, wait for approval, draft but do not send.
- **Information:** Access, disclosure, or storage of specified information. Original anchors: do not reveal a name, save a phone number, or read a confidential folder.
- **Process:** Preparatory, sequencing, or completion steps in the interaction. Original anchors: search before answering, restate the question, summarize when a task finishes.
- **Preference:** Choice among otherwise task-equivalent responses or sources. Original anchors: prefer arXiv papers, metric units, or primary sources.
- **Output:** Surface textual properties of the agent's response. Original anchors: numbers as words, a fixed closing sentence, or bullets only.

## Applying them to broader constraints

- Action: whether an operation may or must occur, including permission or prohibition boundaries.
- Information: access, disclosure, retention, storage, or preservation of specified information/data.
- Process: preparation, method, procedure, sequencing, testing, reproduction, investigation, implementation location, or coordination.
- Preference: selecting or favoring one otherwise acceptable method, source, option, or implementation alternative.
- Output: form, fidelity, compatibility, or another property of the delivered artifact or observable result.

The original definitions concern session behavior. These extension rules deliberately apply them by analogy to software work and project activity. They do not change `strict_sc`.

## Decision rules

Read the complete issue, exact candidate, and all kept clauses. Identify what the restriction principally binds. Do not infer type from `broad_kind`, a keyword, negative syntax, or the mere presence of “prefer.” For example, a hedged token-storage requirement can still be Information. Distinguish choosing an acceptable method (Preference) from prescribing workflow steps (Process), and a prohibition on performing an operation (Action) from properties of the resulting artifact (Output).

For multiple clauses, choose the type representing the central broad restriction. Explain relevant alternatives in `rationale`; set `needs_review=true` for material ambiguity. Use exactly one of the five labels, even when review is needed. Do not force class balance. The researcher can subsequently review a provisional primary choice.

Use the same rubric in both independent type passes and adjudication. The complete five definitions must be restated in every worker dispatch through `TYPE_EVALUATION_PROMPT.md`.
