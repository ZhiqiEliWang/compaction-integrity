#!/usr/bin/env bash
# RQ3: SC injection position (top/middle/bottom) at 50k and 100k for
# wildchat and hermes. The "top" rows reuse the baseline 50k/100k runs.
# Manifest: config/experiments/rq3/injection_positions.yaml
# Analyze:
#   - analyze/injection_positions.py
#   - analyze/openresearcher_stability.py  (per-row stability sanity check)

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../_common.sh"

# Baseline (position=top).
run_eval wildchat100k
run_eval hermes100k
run_eval wildchat50k
run_eval hermes50k

run_eval wildchat100k-middle
run_eval wildchat100k-bottom
run_eval hermes100k-middle
run_eval hermes100k-bottom
run_eval wildchat50k-middle
run_eval wildchat50k-bottom
run_eval hermes50k-middle
run_eval hermes50k-bottom

run_analyze injection_positions \
  --manifest_path "${REPO_ROOT}/config/experiments/rq3/injection_positions.yaml"

run_analyze openresearcher_stability
