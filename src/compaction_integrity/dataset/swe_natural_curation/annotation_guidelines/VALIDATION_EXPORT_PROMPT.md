# Coordinator validation, export, and handoff

Use deterministic helpers for all parsing, joining, counts, checks, and exports; use judges for semantics. Run helpers from the run's frozen package so future edits cannot change a running annotation.

Validate every worker file immediately for exact ID coverage, schema, frozen version, exact source spans and evidence, clause order, both aggregation rules, broad and strict primary indices, and allowed strict type usage. `keep` is not contingent on observability. Compare independent results before consulting keys. Review all proposed broad positives, all disagreements/uncertainty/flags, proposed strict positives, manual conflicts, duplicate inconsistencies, and a seeded sample of 20 agreed broad negatives.

Merge only after complete annotation adjudication coverage. Preserve pass A, pass B, final result, review reasons, and provenance. Broad status controls selection. Complete the separate independent typing stage, its review, and merge. Draft probes, review them, revise only the declared revision set, and review again until each row is ready for a pilot or explicitly held.

Final exports:

- `all_candidates.annotated.csv`: every source row with newly computed annotation fields; `decision` means broad by default.
- `sc_annotations.broad_keeps.typed.csv`: one row per broad-kept entry, all five-way row types, exact SC fields, and the primary display `probe` column.
- `sc_annotations.broad_keeps.probe_ready.csv`: only probes with final reviewer disposition ready.
- `annotations_detailed.jsonl`, `typed_records.jsonl`, `probes_final.jsonl`, all passes and adjudications, and pending `review_queue.jsonl`.
- `run_report.md`, `calibration_report.md`, `probe_quality_review.md`, `final_validation.json`, and a source/package digest manifest.

The helper uses a single primary exact clause for `sc_text` and a contiguous covering source span for `broad_constraint_text`; all kept clauses remain in `broad_clauses_json`. Require exact candidate and issue membership for both fields. The probe's target text must match a kept clause and its index. Never store a paraphrase as SC text.

Reparse output CSVs with `csv.DictReader`, `newline=''`, UTF-8/BOM handling. Verify source row count/order, unique IDs, every source field unchanged, correct broader keep set, allowed types, complete probe/hold coverage, and exact source substrings. New fields that collide with source names are prefixed `ann_`; document the mapping. Preserve the original input and surface manual annotation conflicts.

No structural check establishes semantic correctness or empirical discrimination. Report model versus human provenance, unresolved labels/types, held probes, observability limits, agreement denominators, and source digest changes. Keep MCQ gold outside model-visible prompts, counterbalance alternatives, and pilot no-SC/upper-bound conditions in a separate experiment. Do not claim those experiments ran as part of annotation.
