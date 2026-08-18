#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

EMBEDDING_CACHE_ROOT="/data/compaction_integrity/topic_cohesive_embedding_cache"

# WildChat / Hermes use the topic-cohesive stitching path with the shared
# embedding+KNN cache.
run_topic_cohesive_dataset() {
  local ds_name="$1"

  python -m compaction_integrity.scripts.generate_dataset \
    --config-name "${ds_name}" \
    output_dir="/data/compaction_integrity/default_ds/${ds_name}" \
    stitching_method=topic_cohesive \
    +embedding_cache.path="${EMBEDDING_CACHE_ROOT}" \
    +ds_name="${ds_name}"
}

# OpenResearcher rows already meet the target token length; we only trim them
# rather than stitching, so no embedding cache is needed.
run_openresearcher_dataset() {
  local ds_name="$1"

  python -m compaction_integrity.scripts.generate_dataset \
    --config-name "${ds_name}" \
    output_dir="/data/compaction_integrity/default_ds/${ds_name}"
}


run_topic_cohesive_dataset wildchat_cat_100k
run_topic_cohesive_dataset wildchat_cat_50k
run_topic_cohesive_dataset wildchat_cat_10k

run_topic_cohesive_dataset hermes_cat_100k
run_topic_cohesive_dataset hermes_cat_50k
run_topic_cohesive_dataset hermes_cat_10k

run_openresearcher_dataset openresearcher_cat_100k
run_openresearcher_dataset openresearcher_cat_50k
run_openresearcher_dataset openresearcher_cat_10k
