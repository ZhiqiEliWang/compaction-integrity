#!/usr/bin/env bash
# Sourced helpers for the rqN/*.sh experiment runners.

set -euo pipefail

EXP_SH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${EXP_SH_DIR}/.." && pwd)"
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

# Run the main evaluation pipeline (cases 1-4) for one eval config.
# Usage: run_eval <config_name>   # e.g. run_eval wildchat100k
run_eval() {
  local config_name="$1"
  python -m compaction_integrity.scripts.evaluation \
    --config-path "${REPO_ROOT}/config/tasks/eval" \
    --config-name "${config_name}"
}

# Run the SC-extractor evaluation (RQ4) for one config under config/experiments/rq4/.
# Usage: run_sc_extractor_eval <config_name>
run_sc_extractor_eval() {
  local config_name="$1"
  python -m compaction_integrity.scripts.eval_sc_extractor \
    --config-name "${config_name}"
}

# Run an analyze script under src/compaction_integrity/analyze/.
# Usage: run_analyze <script_basename> [extra args...]
run_analyze() {
  local script="$1"
  shift
  python -m "compaction_integrity.analyze.${script}" "$@"
}
