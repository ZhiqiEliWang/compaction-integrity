#!/usr/bin/env bash
# SC annotation of one Open-SWE batch, studies/swe_natural/generated/sc_candidates.open_swe_<batch>.csv.
# Usage: annotate_open_swe.sh <batch> [hydra overrides...]     e.g. annotate_open_swe.sh 251_500
#
# Batches are consecutive task ranges in pool order of open_swe_rebench_1000
# (open_swe_250 = tasks 1-250). Cut each batch so it shares no task with earlier ones:
#   python -m compaction_integrity.dataset.swe_natural_curation.dataset candidates \
#     --pool_path /data/compaction_integrity/default_ds/open_swe_rebench_1000/stitched_dataset \
#     --exclude studies/swe_natural/generated/sc_candidates.open_swe_{250,251_500}.csv \
#     --max_tasks 250 --output studies/swe_natural/generated/sc_candidates.open_swe_501_750.csv
# Every batch builds against open_swe_rebench_1000 (dataset.py build --pool_path).
# Rerun with the same batch to resume from the last finished partition.

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../_common.sh"

batch="$1"
shift

python -m compaction_integrity.dataset.swe_natural_curation.annotate_sc --config-name annotate \
  input_csv="${REPO_ROOT}/studies/swe_natural/generated/sc_candidates.open_swe_${batch}.csv" \
  run_dir="/data/compaction_integrity/annotation_runs/open_swe_${batch}" \
  export_dir="${REPO_ROOT}/studies/swe_natural/annotations/open_swe_${batch}" \
  "$@"
