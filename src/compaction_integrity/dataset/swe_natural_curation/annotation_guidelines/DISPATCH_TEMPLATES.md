# Worker dispatches

Use fresh isolated contexts for independent workers, and no more than four active agents including the coordinator. Replace placeholders with absolute paths. The coordinator supplies the entire named prompt and required rubric/schema content in the message, or requires the worker to read the complete files before classification. Inputs should be bounded; no worker reads the main dataset CSV.

## Calibration, controls, and production

```text
Follow <run>/frozen/package/EVALUATION_PROMPT.md completely.
Read <run>/frozen/package/SC_RUBRIC.md and <run>/frozen/package/schemas/ANNOTATION_SCHEMA.json completely before judging.
PROJECT_DIR=<run>
RUBRIC_PATH=<run>/frozen/package/SC_RUBRIC.md
SCHEMA_PATH=<run>/frozen/package/schemas/ANNOTATION_SCHEMA.json
INPUT_PATH=<run>/partitions/<calibration|controls|production>/part_NNN.jsonl
OUTPUT_PATH=<run>/<calibration_A|calibration_B|controls|pass_A|pass_B>/part_NNN.jsonl
PASS_ID=<calibration_A|calibration_B|controls|A|B>
The default selection label is broad_task_constraint. Keep the secondary strict comparison separately. Read only assigned inputs. Do not inspect other passes, keys, proposals, or the original CSV. Do not spawn agents. Validate and report completion.
```

## Semantic adjudication

```text
Follow <run>/frozen/package/ADJUDICATION_PROMPT.md completely, with the complete EVALUATION_PROMPT.md, SC_RUBRIC.md, and annotation schema.
INPUT_PATH=<run>/partitions/adjudication/part_NNN.jsonl
OUTPUT_PATH=<run>/adjudication/part_NNN.jsonl
Role override: inspect both proposals supplied in your assigned input. Decide from the complete frozen source under the broader-default rubric. Preserve unresolved ambiguity and validate exact output coverage.
```

## Independent type judgment

```text
Follow <run>/frozen/package/TYPE_EVALUATION_PROMPT.md completely, and read TYPE_RUBRIC.md and schemas/TYPE_SCHEMA.json in that same frozen package.
Action: whether an action is performed or permission is required.
Information: access, disclosure, storage, retention, or preservation of specified information.
Process: preparation, method, procedure, sequencing, testing, implementation location, or coordination.
Preference: choice among otherwise acceptable methods, responses, options, or sources.
Output: surface response form, or by analogy form, fidelity, compatibility, or properties of the delivered artifact/result.
INPUT_PATH=<run>/partitions/types/part_NNN.jsonl
OUTPUT_PATH=<run>/<type_A|type_B>/part_NNN.jsonl
PASS_ID=<type_A|type_B>
Read all clauses and full issue context. Assign one primary row type; flag material ambiguity. Do not read the other type pass. Validate.
```

## Type adjudication

Use the full independent type dispatch above, replacing its role with `TYPE_ADJUDICATION_PROMPT.md`, input with `partitions/type_adjudication`, and output with `type_adjudication`. Include both proposals only here. Restate all five definitions for this judge too.

## Draft probes

```text
Follow <run>/frozen/package/PROBE_GENERATION_PROMPT.md completely. Read the full SC_RUBRIC.md, TYPE_RUBRIC.md, and schemas/PROBE_SCHEMA.json.
INPUT_PATH=<run>/partitions/probes/part_NNN.jsonl
OUTPUT_PATH=<run>/probe_drafts/part_NNN.jsonl
Produce one draft or explicit hold per entry. Use one kept target clause and its exact source text. Write a neutral continuation and two task-equivalent alternatives differing on that clause only. Validate.
```

## Quality review

```text
Follow <run>/frozen/package/PROBE_QUALITY_PROMPT.md completely. Read the full generation prompt, SC/type rubrics, and schemas/PROBE_REVIEW_SCHEMA.json.
INPUT_PATH=<run>/probe_review_input_vN.jsonl
OUTPUT_PATH=<run>/probe_review_vN.jsonl
Review every assigned draft against full source context. Check leakage, single controlled difference, modality, allowed alternatives, scope, lifecycle, feasibility, and gold defensibility. Use ready/targeted_revision/substantial_revision/hold and explain concrete findings. Validate.
```

For a large review input, partition it into bounded row files, give each reviewer a unique output, validate each pair, then join the files with exact unique ID coverage and validate against the full review input. This consolidation is deterministic, not another judgment.

## Revision

```text
Follow <run>/frozen/package/PROBE_REVISION_PROMPT.md completely, with the full generation and quality prompts, SC/type rubrics, and probe schema.
INPUT_PATH=<run>/probe_revision_input_vN.jsonl
OUTPUT_PATH=<run>/probe_revisions_vN.jsonl
Revise only the assigned targeted/substantial entries; address every finding. Return an explicit hold if no supported contrast is possible. Preserve exact source text and validate.
```
