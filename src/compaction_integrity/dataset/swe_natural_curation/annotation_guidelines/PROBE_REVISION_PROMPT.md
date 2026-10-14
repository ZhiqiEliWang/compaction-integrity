# Targeted probe revision worker

Read the frozen SC/type rubrics, probe-generation prompt, quality prompt, and probe schema completely. The coordinator supplies only entries explicitly marked `targeted_revision` or `substantial_revision`, with their original drafts, complete source context, and reviewer findings. Use a unique output path and preserve prior attempts.

Revise each assigned probe to address all findings. Keep the target exact source text and semantic label; a different kept target clause is allowed only when the review requires it and the reason is explicit. Preserve source modality and exceptions. Write a neutral fresh question and one controlled difference between technically plausible task-equivalent options. Mark `hold` if the requested distinction cannot be justified. Output one probe-schema record per assigned ID, including holds.

Do not alter unassigned ready or held records, types, source paragraphs, issue context, IDs, or constraint labels. Validate with `validate-probes`. The coordinator merges only this declared revision set and sends the resulting probes for a new quality review. Original drafts and reviews remain as audit evidence. No quality flag is cleared merely because a revision was written.
