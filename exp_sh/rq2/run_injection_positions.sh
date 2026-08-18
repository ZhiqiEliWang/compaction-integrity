#!/usr/bin/env bash
# RQ2: SC injection position (top/middle/bottom) at 50k and 100k for
# wildchat and hermes. The "top" rows reuse the baseline 50k/100k runs.
# Manifest: config/experiments/rq2/injection_positions.yaml
# Analyze:
#   - analyze/injection_positions.py
#   - analyze/openresearcher_stability.py  (per-row stability sanity check)

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../_common.sh"

# Baseline (position=top) — reused from the diff_input_size runs.
run_eval wildchat100k
run_eval hermes100k
run_eval wildchat50k
run_eval hermes50k

# Middle / bottom variants.
run_eval wildchat100k-middle
run_eval wildchat100k-bottom
run_eval hermes100k-middle
run_eval hermes100k-bottom
run_eval wildchat50k-middle
run_eval wildchat50k-bottom
run_eval hermes50k-middle
run_eval hermes50k-bottom

run_analyze injection_positions \
  --manifest_path "${REPO_ROOT}/config/experiments/rq2/injection_positions.yaml"

run_analyze openresearcher_stability
