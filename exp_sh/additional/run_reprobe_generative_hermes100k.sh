#!/usr/bin/env bash
# Generative-vs-MCQ reprobe for the completed HermesAgent RQ1 run.
#
# Reuses the source run's full and compacted contexts. Only gpt-oss-120b probe
# generations are recomputed; the original MCQ columns and retention verdicts
# remain available in the new generative result pickle.

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../_common.sh"

SOURCE_RUN_ID="hermes_cat_100k__gpt_oss_120b_anthropic_prompt__gpt_oss_120b__llm_gpt_5_4__n50__full__19f0a3c590"

python -m compaction_integrity.scripts.reprobe_generative \
  +test=false \
  +overwrite=true \
  +results_root=/data/compaction_integrity \
  +tool_contract=sssc_native_tools_v2 \
  '+sssc_ids=[1,2,3,6,7,10]' \
  "+source_runs=[{run_id:${SOURCE_RUN_ID},label:hermes_cat_100k_gpt_oss_120b_anthropic_prompt}]" \
  '+probe={name:gpt_oss_120b_generative,model:openai/gpt-oss-120b,provider:vllm,kwargs:{batch_size:32,max_tokens:4096}}' \
  '+cases={full_with_sssc:true,full_without_sssc:true,compacted:true,compacted_post_sssc:true}'
