#!/usr/bin/env bash
# RQ2: how compaction quality varies with input context length (10k/50k/100k)
# on wildchat and hermes.
# Manifest: config/experiments/rq2/diff_input_size.yaml
# Analyze:
#   - analyze/diff_input_size.py  (retention/compliance vs context length)
#   - analyze/compact_rate.py     (compaction rate stats, same manifest)

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../_common.sh"

run_eval wildchat10k
run_eval wildchat50k
run_eval wildchat100k
run_eval hermes10k
run_eval hermes50k
run_eval hermes100k

run_analyze diff_input_size \
  --manifest_path "${REPO_ROOT}/config/experiments/rq2/diff_input_size.yaml"

run_analyze compact_rate \
  --manifest_path "${REPO_ROOT}/config/experiments/rq2/diff_input_size.yaml"
