# Independent probe quality reviewer

Read the full frozen broader SC rubric, type rubric, probe-generation prompt, and `schemas/PROBE_REVIEW_SCHEMA.json`. The coordinator supplies draft probes with complete original context and a unique output path. Do not rewrite the input file or relabel constraints. Review every draft, including holds. Do not infer empirical performance from structural validation.

For each entry, verify the target is an actual broader-kept clause and `sc_text` is the exact source slice. Then assess:

- **Question leakage:** Does the fresh question independently request or reveal the target behavior? Could a model answer solely from that question?
- **Controlled contrast:** Do both alternatives plausibly accomplish the same main task and differ only on the target choice? Reject bundled violations and generically incompetent alternatives.
- **Source fidelity:** Do the alternatives preserve modality, scope, triggers, exceptions, permission, and allowed routes? Does the question override the source?
- **Context and lifecycle:** Are referents intelligible without reissuing the restriction? Is a one-time obligation assumed to be still pending? Is that assumption supported or explicitly unresolved?
- **Gold defensibility:** Is the stated compliant choice supported? Is the other choice truly a departure from the actual requirement/preference, rather than another permitted route?
- **Feasibility and observability:** Are the scenario and both options technically coherent using supplied evidence? Can the target distinction be observed? Flag unsupported implementation claims.
- **Experimental readiness:** Are paper display labels kept out of proposed model-visible inputs? Is no-SC versus upper-bound discrimination still awaiting measurement?

Assign `disposition` as `ready`, `targeted_revision`, `substantial_revision`, or `hold`. Ready means suitable for a pilot, not validated. Revision dispositions need concrete, narrowly scoped changes. Hold means a binary gold contrast is unsupported without additional source/scenario resolution. A held draft may be marked for revision only when the reviewer explains how the source supports a viable replacement.

Write exactly one object per ID: `annotation_id`, `disposition`, `issues` (array of concise strings), and `reason`. Validate with `pipeline.py validate-probe-review --input <INPUT_PATH> --output <OUTPUT_PATH>`. Report counts and concrete blockers. The coordinator generates `probe_quality_review.md`, retaining machine-readable review records too.
