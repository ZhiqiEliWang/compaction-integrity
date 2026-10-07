# Coordinator calibration and controls

Prepare a calibration subset and separate controls before production. Use the same broader-default annotation rubric, schema, worker prompt, issue-preserving partitions, and independent passes A/B as production. Workers do not see proposals, review guides, keys, or each other's outputs.

For the original 208-row dataset, `prepare` reuses the 25 historical selected IDs only when every selected candidate and full issue exactly match. This reproduces the 10 seeded random plus 15 researcher-boundary selections. For other data, the helper draws up to 25 rows with `random.Random(SEED).sample`; provide `--calibration-ids <JSON array>` to specify a reviewed boundary selection. Record selection mode and actual IDs. A new random sample is not the old mixed calibration sample.

The package's historical 20 controls include 15 benchmark-derived positive anchors and five synthetic negatives. Read their input with fresh workers; keep their key hidden until the control outputs pass validation. Their old strict labels are not production labels. Score the broad and strict expected labels separately with `score-controls`. Type anchors belong to the later taxonomy stage, not a change of selection rule.

After both calibration passes finish, run `review --stage calibration` to compare labels, gates, spans, observability, and strict types. Read the historical proposals only then as provisional references, not approved gold. Write `calibration_report.md` covering coverage, A/B disagreement with denominators, controls, boundary questions, source matching, and any prompt noncompliance. Distinguish controls from sampled production records.

Correct actual failure to follow the rubric with a specific explanation and retain both attempts. Definition ambiguity should retain uncertainty rather than training workers to match a draft proposal. If the rubric changes, version and refreeze it and rerun both passes consistently. `MODE=calibration` ends after this report; `MODE=full` proceeds to fresh production passes. Calibration judgments never substitute for production judgments or add another count of those IDs.
