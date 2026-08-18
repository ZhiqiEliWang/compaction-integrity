#!/usr/bin/env bash
# RQ1 supplemental: gpt-5.4-mini compactor on the 220k datasets.
# Manifest: config/experiments/rq1/gpt-5.4-mini.yaml
# Analyze:  analyze/gpt_5.4_eval.py

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../_common.sh"

run_eval wildchat220k
run_eval hermes220k
run_eval openresearcher220k

run_analyze "gpt_5.4_eval" \
  --manifest "${REPO_ROOT}/config/experiments/rq1/gpt-5.4-mini.yaml"
