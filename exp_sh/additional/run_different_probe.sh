#!/usr/bin/env bash
# Different-prober MCQ re-probe for the completed HermesAgent RQ1 run.
#
# Reuses the source run's full and compacted contexts AND its exact A/B prompts.
# Only the downstream prober changes (Qwen3-30B, Gemma-4-E4B); compaction and
# retention are not recomputed. Each config writes a drop-in
# runs/<new_run_id>/evaluation_results.pkl with identical schema so it slots into
# a main_exp-style manifest next to the gpt-oss-120b baseline.
#
# After both probers finish, aggregates the three probers (gpt-oss source +
# the two swaps) into one compliance/agreement table.

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
