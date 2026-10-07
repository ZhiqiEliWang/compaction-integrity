#!/usr/bin/env bash
# Model-family-bias retention probe for the completed RQ1 runs.
# Tests the retention judge for single-judge / model-family bias without re-running
# compaction or probing: one sample from the two rq1 manifests that contains the human-50
# blind set is re-judged by a second-family judge (gemini) on each row's cached
# compacted_context.
#   1. make_family_probe         sample N rows (default 2000), nesting the human-50,
#                                population-rate; tag each row's compactor family.
#   2. judge_family_probe        gemini verdict per row (parity harness from
#                                scripts.rejudge_retention; reasoning_effort=low).
#   3. analyze/judge_family_bias A: 3-way human/gpt5.4/gemini on the nested n=50;
#                                B: gemini vs gpt5.4 robustness + per-family delta.
# Prereq: gemini_api_key in src/compaction_integrity/api_keys.py (paid tier; ~N judge
# calls). N and CONCURRENCY override sample size and concurrency.

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../_common.sh"

N="${N:-2000}"
CONCURRENCY="${CONCURRENCY:-8}"
STUDY_DIR="${REPO_ROOT}/studies/judge_robustness"
GEN_DIR="${STUDY_DIR}/generated"
SAMPLE="${GEN_DIR}/family_probe_sample.jsonl"
JUDGED="${GEN_DIR}/family_probe_judged.jsonl"

echo "=============================================================="
echo "[family-probe] 1/3 sample N=${N} (nesting human-50)"
echo "=============================================================="
python -m compaction_integrity.scripts.judge_robustness.make_family_probe \
  --manifest "${REPO_ROOT}/config/experiments/rq1/main.yaml" \
  --manifest "${REPO_ROOT}/config/experiments/rq1/gpt-5.4-mini.yaml" \
  --results_root /data/compaction_integrity \
  --human_key "${STUDY_DIR}/labels/balanced_key.jsonl" \
  --out_dir "${GEN_DIR}" \
  --n "${N}" --seed 0

echo "=============================================================="
echo "[family-probe] 2/3 gemini judge (parity harness)"
echo "=============================================================="
python -m compaction_integrity.scripts.judge_robustness.judge_family_probe \
  --sample "${SAMPLE}" \
  --provider gemini --model gemini-flash-latest \
  --reasoning_effort low --concurrency "${CONCURRENCY}"

echo "=============================================================="
echo "[family-probe] 3/3 3-way (vs human) + family-bias report"
echo "=============================================================="
run_analyze judge_family_bias \
  --judged "${JUDGED}" \
  --human "${STUDY_DIR}/labels/balanced_blind_labeled.jsonl"

echo "Family-bias retention probe complete."
