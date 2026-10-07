#!/usr/bin/env bash
# Different-prober MCQ re-probe for the completed HermesAgent RQ1 run.
# Reuses the source run's full and compacted contexts and its exact A/B prompts; only
# the prober changes (Qwen3-30B, Gemma-4-E4B). Each config writes a drop-in
# runs/<new_run_id>/evaluation_results.pkl with identical schema.
# Analyze:  analyze/prober_swap.py  (gpt-oss source + the two swaps: compliance/agreement table)

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../_common.sh"

CONFIGS=(
  "hermes100k_qwen30b"
  "hermes100k_gemma4"
)

for config in "${CONFIGS[@]}"; do
  echo "=============================================================="
  echo "[reprobe-mcq] config=${config}"
  echo "=============================================================="
  python -m compaction_integrity.scripts.reprobe_mcq \
    --config-path "${REPO_ROOT}/config/tasks/reprobe_mcq" \
    --config-name "${config}"
done

echo "=============================================================="
echo "[reprobe-mcq] aggregating prober-swap comparison"
echo "=============================================================="
run_analyze prober_swap \
  --manifest_path "${REPO_ROOT}/config/experiments/additional/prober_swap_hermes100k.yaml" \
  --results_root /data/compaction_integrity

echo "All different-prober MCQ re-probe passes complete."
