#!/usr/bin/env bash
# RQ2: SC repeat-count sweep (1, 2, 3, 4, 5, 6, 10, 15, 20, 25, 30) on
# wildchat_cat_100k. repeat=1 reuses the baseline wildchat100k run.
# Manifest: config/experiments/rq2/repeat.yaml
# Analyze:  analyze/repeat.py

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../_common.sh"

run_eval wildchat100k            # repeat = 1 (baseline)
run_eval wildchat100k-repeat2
run_eval wildchat100k-repeat3
run_eval wildchat100k-repeat4
run_eval wildchat100k-repeat5
run_eval wildchat100k-repeat6
run_eval wildchat100k-repeat10
run_eval wildchat100k-repeat15
run_eval wildchat100k-repeat20
run_eval wildchat100k-repeat25
run_eval wildchat100k-repeat30

run_analyze repeat \
  --manifest_path "${REPO_ROOT}/config/experiments/rq2/repeat.yaml"
