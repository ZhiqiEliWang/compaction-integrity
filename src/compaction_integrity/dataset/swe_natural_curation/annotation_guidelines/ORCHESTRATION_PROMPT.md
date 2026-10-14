# Coordinator: reproduce the broader-default annotation pipeline

Read `README.md`, every stage prompt, both rubrics, and all schemas in this package completely before dispatch. Coordinate bounded workers; use `scripts/pipeline.py` for deterministic operations. Produce a complete provisional export and review queue.

## Defaults

```text
PACKAGE_DIR=<absolute path to annotation_guidelines/>
INPUT_CSV=<explicit candidate CSV>
RUN_DIR=<new absolute directory outside PACKAGE_DIR>
MODE=full
PRIMARY_LABEL=broad_task_constraint
RECORD_STRICT_LABEL=true
INDEPENDENT_ANNOTATION_PASSES=2
INDEPENDENT_TYPE_PASSES=2
MAX_CONCURRENT_AGENTS_INCLUDING_COORDINATOR=4
TARGET_CANDIDATES_PER_PARTITION=15
MAX_CANDIDATES_PER_PARTITION=20
SOFT_MAX_PARTITION_CHARACTERS=50000
SEED=20260905
PROBE_UNIT=row
REPLACE_ORIGINAL=false
```

At most three workers may run while one coordinator is active; nested coordinators consume the same cap. Do not let workers spawn more agents. Fresh isolated contexts are preferred for each partition/pass. Give each worker the full relevant prompt and required frozen rubric/schema content, or explicit instructions and paths to read all of it before acting. Do not expose opposite-pass labels through inherited context. If platform limits require reusing agents, use pass-dedicated workers on new IDs and record the independence limitation; never claim fresh workers where they were reused. If independent workers cannot be obtained, report the execution limitation rather than call one sequential context two independent passes.

## 1. Inventory, optional difference, and preparation

Check the input/header/ID counts. If the source is a new superset, apply `DEDUPLICATION_PROMPT.md` before annotation and retain its manifest. Do not remove distinct same-text IDs merely for being text duplicates.

Run `prepare` against a new run directory. It checks candidate-in-issue membership, issue-text consistency, IDs, and CSV shape; copies the source and package into `frozen/`; hashes them; preserves existing manual cells; selects calibration; and makes production/calibration/control partitions. Run all later helpers from `<RUN_DIR>/frozen/package/scripts/pipeline.py`.

Character counts are work-unit heuristics, not token counts. Account for rubrics, output, and reasoning. Whole issues remain together; oversized issues get a dedicated partition and must be read completely. If a partition is too large for the actual model, re-pack it before either pass, record the new manifest, and preserve issue grouping.

## 2. Calibration and constructed controls

Follow `CALIBRATION_PROMPT.md`. Dispatch independent A and B with `EVALUATION_PROMPT.md`, `SC_RUBRIC.md`, and the annotation schema, using `partitions/calibration` and unique `calibration_A`/`calibration_B` outputs. Dispatch fresh control workers with the same content, using `partitions/controls` and `controls` outputs. Keep reference proposals and the control key hidden.

Validate immediately, compare calibration, score controls only after completion, and write `calibration_report.md`. Retain disagreements and definition ambiguities. Correct actual rubric noncompliance with a specific correction and preserve attempts. For full mode continue to production under the frozen version; for calibration mode stop with these deliverables. The prior NVIDIA run did not repeat calibration; this package's full default performs it, matching the original full orchestration protocol rather than silently inheriting that exception.

## 3. Independent production judgments

Use `partitions/production/part_NNN.jsonl` for both passes. Dispatch fresh workers with the entire `EVALUATION_PROMPT.md`, complete rubric/schema, explicit `PASS_ID=A` or `B`, and `pass_A/part_NNN.jsonl` or `pass_B/part_NNN.jsonl`. The inputs conceal keywords, lengths, previous labels, and reference keys.

Validate every saved result. Production includes every source ID exactly once per pass, including IDs previously used for calibration. Preserve valid outputs on retry. A readable ambiguous case gets an uncertain record; a missing/invalid record is an execution failure to repair, not a semantic drop.

## 4. Comparison and source-based adjudication

Run `review --stage production`. It reviews every proposed broad positive, proposed strict positive, all label/gate/span/type/observability/provenance differences, review flags, uncertainty, existing manual annotations, same-issue duplicate inconsistencies, and 20 seeded agreed broad negatives (or all if fewer).

Dispatch `ADJUDICATION_PROMPT.md` plus the complete evaluation prompt/rubric/schema for each `partitions/adjudication` partition. Only these workers may see both proposals. Use separate `adjudication/part_NNN.jsonl` outputs. No majority vote substitutes for source reading. Validate all, then run `merge`. Retain `annotations_detailed.jsonl`, both passes, adjudication log, review reasons, and model provenance. Existing researcher annotations are surfaced, preserved, and not silently adjudicated away.

## 5. Independent five-way typing of broad keeps

Run `prepare-types`. Selection comes from `final.broad_task_constraint=keep`, even if every strict label is drop. Dispatch two independent type judges with the full `TYPE_EVALUATION_PROMPT.md` and `TYPE_RUBRIC.md` plus type schema. Restate the five definitions in each dispatch through the supplied worker prompt; do not merely list category names.

Use `type_A/part_NNN.jsonl` and `type_B/part_NNN.jsonl`. Validate, run `review-types`, and dispatch `TYPE_ADJUDICATION_PROMPT.md` plus the full type worker prompt/rubric/schema on disputed or flagged entries. Validate `type_adjudication` outputs and run `merge-types`. One row gets one primary type, with alternatives and material uncertainty preserved.

## 6. Probe drafting

Run `prepare-probes`. Dispatch `PROBE_GENERATION_PROMPT.md` with the complete SC/type rubrics and probe schema, using `partitions/probes` and `probe_drafts/part_NNN.jsonl`. Draft one probe or held record per entry. Record one kept target clause and exact target text; do not bundle multiple violations or infer coverage of untested clauses.

Validate and run `collect-probes` into `probes_v1.jsonl`. Workers return structured fields, while export creates the original three-line User/Compliant/Violation LaTeX cell. Gold markers are for authoring only.

## 7. Probe quality review and targeted revision

Run `prepare-probe-review` to create a joined input containing drafts and full source context. Dispatch a separate fresh quality reviewer with `PROBE_QUALITY_PROMPT.md`, probe-generation prompt, full rubrics, and review schema. Split into bounded row batches if necessary and consolidate only after validating each exact subset. Validate the consolidated review against the full input.

Use `prepare-revision` to select only targeted/substantial revisions. Dispatch `PROBE_REVISION_PROMPT.md` with full draft/quality/rubric/schema content. Validate the exact revision set and run `apply-revision` into a new version; ready and unassigned held rows are unchanged. Prepare a new review input for the resulting version and re-review. Preserve all versions. Repeat until no targeted/substantial revision remains; source-unsupported cases stay held with reasons. Avoid repeating unchanged reviews without new evidence.

## 8. Final validation and export

Follow `VALIDATION_EXPORT_PROMPT.md` and run `finalize` with the final probe version and a matching fresh review. The helper rejects stale review inputs or outstanding revision dispositions. Ready means ready for a pilot, not empirically validated. Held probes remain in the broad-keeps CSV with a hold status and are excluded from the probe-ready export.

Write the coordinator's calibration report and dispatch log. Augment run reporting with actual worker model metadata when available, independence exceptions, human review performed, schema repairs, calibration/control results, and the source path. Do not invent model snapshots or experiments. Re-check source and frozen digests. Deliver the README-linked outputs, all-candidates CSV, broad-only typed/probed CSV, ready subset, detailed records, validation, and pending review queue.

## Pause, resume, and error recovery

Maintain `dispatch_log.jsonl` with stage/pass, input/output, expected IDs, worker identity, exposed model/settings, dispatch/completion timestamps, status, and validation result. At a requested pause write `pause_state.json` with run/source/package digests, completed validated files and hashes, active/missing/invalid partitions, next stage, probe/review versions, and unresolved IDs. Stop active workers and preserve partial artifacts; never count them as completed.

On resume, verify all digests and validate completed files before using them. Dispatch only missing/invalid work. Preserve old attempts in an `attempts/` directory and use unique result paths. Do not reuse a valid independent pass as an adjudication. Do not mechanically fix semantic fields to satisfy a validator; offsets or primary-index repairs must be documented, source-checked, and proved not to alter other fields. Structural failures stop the affected batch; ambiguity stays in the review queue.
