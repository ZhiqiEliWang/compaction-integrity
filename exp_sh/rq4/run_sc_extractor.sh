#!/usr/bin/env bash
# RQ4: SC-extractor evaluation. Uses a separate pipeline
# (compaction_integrity.scripts.eval_sc_extractor) whose configs live under
# config/experiments/rq4/.
# The final analysis reports extractor retention by the five SC categories.

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../_common.sh"

run_sc_extractor_eval sc_extractor_hermes100k
run_sc_extractor_eval sc_extractor_hermes100k-non-explicit
run_sc_extractor_eval sc_extractor_openresearcher100k
run_sc_extractor_eval sc_extractor_wildchat100k

run_analyze sc_extractor_by_type
