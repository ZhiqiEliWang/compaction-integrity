#!/usr/bin/env bash
# Model-family-bias retention probe for the completed RQ1 runs.
#
# Answers the reviewer's single-judge / model-family-bias concern without
# re-running compaction or probing. One sample is drawn from the two rq1
# manifests that CONTAINS the existing human-50 blind set, then re-judged with a
# second-family judge (gemini) reusing each row's cached compacted_context:
#   1. make_family_probe   -- sample N rows (default 2000), nesting the human-50,
#                             population-rate; tag each row's compactor family.
#   2. judge_family_probe  -- gemini verdict per row (parity harness from
#                             scripts.rejudge_retention; reasoning_effort=low).
#   3. analyze.judge_family_bias -- PART A: 3-way human/gpt5.4/gemini on the
#                             nested n=50; PART B: gemini-vs-gpt5.4 robustness +
#                             per-family delta (same-family-as-judge bias check).
#
# Prereq: gemini_api_key set in src/compaction_integrity/api_keys.py (PAID tier;
# ~N judge calls). Override sample size / concurrency via N / CONCURRENCY.

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../_common.sh"

N="${N:-2000}"
CONCURRENCY="${CONCURRENCY:-8}"
HE_DIR="${REPO_ROOT}/judge_robustness_eval"
SAMPLE="${HE_DIR}/family_probe_sample.jsonl"
JUDGED="${HE_DIR}/family_probe_judged.jsonl"

echo "=============================================================="
echo "[family-probe] 1/3 sample N=${N} (nesting human-50)"
echo "=============================================================="
python "${HE_DIR}/make_family_probe.py" \
  --manifest "${REPO_ROOT}/config/experiments/rq1/main.yaml" \
  --manifest "${REPO_ROOT}/config/experiments/rq1/gpt-5.4-mini.yaml" \
  --results_root /data/compaction_integrity \
  --human_key "${HE_DIR}/balanced_key.jsonl" \
  --n "${N}" --seed 0

echo "=============================================================="
echo "[family-probe] 2/3 gemini judge (parity harness)"
echo "=============================================================="
python "${HE_DIR}/judge_family_probe.py" \
  --sample "${SAMPLE}" \
  --provider gemini --model gemini-flash-latest \
  --reasoning_effort low --concurrency "${CONCURRENCY}"

echo "=============================================================="
echo "[family-probe] 3/3 3-way (vs human) + family-bias report"
echo "=============================================================="
run_analyze judge_family_bias \
  --judged "${JUDGED}" \
  --human "${HE_DIR}/balanced_blind_labeled.jsonl"

echo "Family-bias retention probe complete."
