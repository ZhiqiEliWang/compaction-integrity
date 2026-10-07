# Type adjudication judge

Read the full frozen `TYPE_RUBRIC.md`, `TYPE_EVALUATION_PROMPT.md`, and type schema. The complete Action, Information, Process, Preference, and Output definitions in that worker prompt apply to you too; the coordinator supplies that prompt in full.

Role override: you may read both type proposals in your assigned review input. Re-read the complete issue, candidate, and broad clauses. Resolve disagreement based on the central constrained object, not majority vote, keywords, or an existing `broad_kind`. Assign exactly one primary type. Explain alternatives and retain `needs_review=true` if the choice remains materially ambiguous. Do not change semantic keep/drop status or exact SC text.

Output the same four-field type schema, one record per assigned ID, to a unique adjudication file. Validate with `validate-types`. The coordinator merges it while preserving type pass A and B and records `fresh_model_adjudication` provenance. Report unresolved type IDs; a model agreement is not human gold.
