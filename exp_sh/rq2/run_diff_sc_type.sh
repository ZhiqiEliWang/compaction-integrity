#!/usr/bin/env bash
# RQ2: cross-SC-type analysis. Uses the same 100k runs as rq1/main.
# Manifest: config/experiments/rq2/diff_sc_type.yaml
# Analyze:  analyze/diff_sc_type.py

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../_common.sh"

run_eval wildchat100k
run_eval hermes100k
run_eval openresearcher100k

run_analyze diff_sc_type \
  --manifest_path "${REPO_ROOT}/config/experiments/rq2/diff_sc_type.yaml"
