#!/usr/bin/env bash
# RQ3: SC prefix variants (explicit on/off x hard on/off) on hermes_cat_100k.
# The "explicit & non-hard" cell reuses the baseline hermes100k run.
# Manifest: config/experiments/rq3/diff_prefix.yaml
# Analyze:  analyze/diff_prefix.py

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../_common.sh"

run_eval hermes100k                     # explicit, not-hard (baseline)
run_eval hermes100k-hard                # explicit, hard
run_eval hermes100k-non-explicit        # non-explicit, not-hard
run_eval hermes100k-non-explicit-hard   # non-explicit, hard

run_analyze diff_prefix \
  --manifest_path "${REPO_ROOT}/config/experiments/rq3/diff_prefix.yaml"
