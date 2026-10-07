#!/usr/bin/env bash
# RQ1 external validation: COMPINT-SWE-Natural n126, the same four conditions on
# native GitHub-issue agent trajectories, author-written SCs only.
# Dataset swe_natural_sc_100k_n126 is built by dataset/swe_natural_curation/dataset.py build
# (composition: studies/swe_natural/README.md).
# The SC is native to the issue text, so case 2 uses the ablated arm.
# Manifest: config/experiments/rq1/swe_natural_sc_n126.yaml
# Analyze:
#   - analyze/main_exp.py            (summary table, SC-type retention stats)

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../_common.sh"

ANALYSIS_ROOT="${ANALYSIS_ROOT:-/data/compaction_integrity/analysis}"
MANIFEST="${REPO_ROOT}/config/experiments/rq1/swe_natural_sc_n126.yaml"

run_eval swe_natural_sc_100k_n126

# Separate output dirs: the analyze defaults point at the rq1/main.yaml results.
run_analyze main_exp \
  --manifest_path "${MANIFEST}" \
  --output_dir "${ANALYSIS_ROOT}/swe_natural_sc_n126"
