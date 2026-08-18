#!/usr/bin/env bash
# Multi-judge retention re-judge for the completed RQ1 runs.
#
# Reuses each source run's cached compacted contexts. Only the retention judge
# is recomputed -- with gemini-flash-latest instead of gpt-5.4 -- so compaction,
# probing, and the compliance columns are not recomputed. Each source run writes
# a drop-in runs/<new_run_id>/evaluation_results.pkl with identical schema
# (only `retention` + `evaluator` differ) so it slots into a main_exp-style
# manifest next to the gpt-5.4 baseline.
#
# For each manifest it also writes, next to that manifest:
#   - rejudge_run_map__<evaluator>.txt   (old_run_id -> new_run_id)
#   - agreement__<evaluator>.{txt,csv}   (new judge vs gpt-5.4: rate, kappa, confusion)
#
# reasoning_effort=low mirrors the gpt-5.4 judge (OpenAIRuntime hardcodes
# reasoning={'effort':'low'}). Prereq: set gemini_api_key in
# src/compaction_integrity/api_keys.py.

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../_common.sh"

CONCURRENCY="${CONCURRENCY:-8}"

MANIFESTS=(
  "${REPO_ROOT}/config/experiments/rq1/main.yaml"
  "${REPO_ROOT}/config/experiments/rq1/gpt-5.4-mini.yaml"
)

for manifest in "${MANIFESTS[@]}"; do
  echo "=============================================================="
  echo "[rejudge] manifest=$(basename "${manifest}") judge=gemini-flash-latest"
  echo "=============================================================="
  python -m compaction_integrity.scripts.rejudge_retention \
    --manifest_path "${manifest}" \
    --provider gemini \
    --model gemini-flash-latest \
    --new_evaluator_name llm_gemini_flash \
    --reasoning_effort low \
    --concurrency "${CONCURRENCY}"
done

echo "All retention re-judge passes complete."
