# Constraint adjudication worker

Read the frozen `SC_RUBRIC.md`, `EVALUATION_PROMPT.md`, and annotation schema completely. The coordinator supplies `INPUT_PATH`, `OUTPUT_PATH`, and frozen paths. Default selection is the broader rule.

Role override: you may inspect the original issue/candidate and both proposals included in your assigned adjudication input. This is adjudication, not an independent third annotation pass. Do not inspect any other files containing labels.

For each assigned candidate, read the whole issue and the review reasons. Evaluate the source directly; do not vote or prefer the longer rationale. Resolve differences in provenance, main-task boundary, broad/strict labels, gates, exact clauses, primary indices, and strict types. Keep uncertainty when the source does not resolve it. Review every proposed broad positive even if both initial passes agree, and the assigned sample of agreed broad negatives. Review proposed strict positives and same-issue duplicate inconsistencies too.

Write one final annotation-schema record per assigned ID. Preserve exact source offsets and supporting quotes. Explain the adjudication in its rationale and keep review flags that remain unresolved. Validate with `validate-annotation` using this input. Do not overwrite either independent pass. The coordinator will wrap this result with both pass records and adjudication provenance when merging.

Manual annotation conflicts belong in a separate report. Model adjudication does not authorize replacing a human annotation. Return output path, coverage, broad/strict counts, and unresolved IDs.
