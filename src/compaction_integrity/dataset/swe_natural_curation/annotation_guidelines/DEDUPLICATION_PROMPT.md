# Optional candidate-set difference

Use this stage only when a new candidate CSV is intended to extend a previous candidate set. It is deterministic preprocessing, not semantic SC annotation.

Compare `annotation_id` values across the new and previous files. For every overlapping ID, require the same `instance_id`, exact `candidate_paragraph`, and exact `issue_text`. Conflicting content fails the operation and must be reported; do not silently treat a changed ID as already annotated. If the new file is claimed to be a superset, verify every previous ID is present. Preserve all new-file columns, source strings, row order, and distinct IDs, even when texts duplicate.

```bash
python3 scripts/pipeline.py deduplicate --input <new.csv> --previous <old.csv> --output <new-only.csv>
```

The helper writes a new CSV and a digest/count manifest. `--allow-not-superset` is an explicit alternative when a superset claim is not intended. It never overwrites either input. Do not reuse old labels automatically if context or prompt versions differ; import labels only through a separately documented researcher-approved migration.
