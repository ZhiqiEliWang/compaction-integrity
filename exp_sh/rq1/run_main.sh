#!/usr/bin/env bash
# RQ1 main experiment: full compactor/probe matrix on the 100k datasets.
# Manifest: config/experiments/rq1/main.yaml
# Analyze:
#   - analyze/main_exp.py            (retention/compliance bars, NRYC composition)
#   - analyze/compliance_decomposition.py (four-group compliance bars)

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../_common.sh"

run_eval wildchat100k
run_eval hermes100k
run_eval openresearcher100k

run_analyze main_exp \
  --manifest_path "${REPO_ROOT}/config/experiments/rq1/main.yaml"

run_analyze compliance_decomposition \
  --manifest_path "${REPO_ROOT}/config/experiments/rq1/main.yaml"
