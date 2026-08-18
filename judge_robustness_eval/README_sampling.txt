Retention judge consistency check -- how the sample was drawn
=============================================================

Purpose
-------
The retention metric is scored by a single LLM judge (GPT-5.4) that decides, per
row, whether a compacted summary still preserves the injected SSSC. This blind
test measures how well a human agrees with that judge on a sample of its
verdicts, so evaluator-specific interpretation / model-family bias can be
assessed instead of assumed.

Source runs
-----------
Rows are pooled from every run listed in the two rq1 manifests:
  config/experiments/rq1/main.yaml          (n50 runs, various compactors)
  config/experiments/rq1/gpt-5.4-mini.yaml  (n20 gpt-5.4-mini compactor runs)
= 24 runs total, read from <results_root>/runs/<run_id>/evaluation_results.pkl
(results_root = /data/compaction_integrity).

Eligible rows
-------------
A row is eligible only if BOTH:
  - the judge returned a verdict (retention is not null), and
  - a compacted context exists (compaction did not fail).
Across the 24 runs this gave 15,225 eligible rows. The judge's population YES
(retained) rate over that pool is ~11.1%.

Sampling procedure (deterministic)
----------------------------------
1. Every eligible row is given a stable id: sha1(dataset | source_row_index |
   sssc_id | sssc_attrs | probe | compactor)[:16]. This id is position- and
   judge-independent, so the same underlying row always gets the same id.
2. Rows are sorted by that id, then sampled with a seeded RNG (random.Random).
   Given the same (sources, n, seed) the selected set is identical on any
   machine / pandas version -- rerunning reproduces the exact same sample.
3. Selection is done by judge_robustness_eval/make_blind_test.py.

The "balanced" sample (balanced_blind.jsonl)
--------------------------------------------
  n = 50, seed = 0, --balance
  => 25 rows the judge scored YES + 25 the judge scored NO.
Balancing is used so both directions of the judge's decision are stress-tested;
without it a representative sample would be ~11% YES and barely probe the
retained calls. It spans 21 of the 24 runs and all 8 compactors.

IMPORTANT: because the classes are balanced (not drawn at population rate), the
raw agreement rate on this file is NOT the population agreement. To get a
population number, either reweight by the ~11% / ~89% YES/NO prevalence, or
generate an unbalanced sample (drop --balance).

Exact command used
------------------
  python judge_robustness_eval/make_blind_test.py \
    --manifest config/experiments/rq1/main.yaml \
    --manifest config/experiments/rq1/gpt-5.4-mini.yaml \
    --results_root /data/compaction_integrity \
    --tag balanced --n 50 --seed 0 --balance

Files produced
--------------
  balanced_blind.jsonl  one item/line: {label:"", id, sssc_message,
                        compacted_summary}. The human fills "label" with
                        yes/no. Contains NO judge verdict (annotate blind).
  balanced_key.jsonl    held-aside: {id, judge_retention, judge_name,
                        compactor, dataset, source_row_index, sssc_id, run_id}.

Scoring
-------
After every line's label is filled, run:
  python judge_robustness_eval/analyze_blind_test.py \
    --blind judge_robustness_eval/balanced_blind.jsonl \
    --key   judge_robustness_eval/balanced_key.jsonl \
    --csv   judge_robustness_eval/balanced_join.csv
It refuses to produce numbers until all 50 are labeled, then reports raw
agreement, Cohen's kappa, the judge x human confusion, per-compactor agreement,
and the list of disagreements.
