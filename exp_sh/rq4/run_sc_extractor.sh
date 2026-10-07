#!/usr/bin/env bash
# RQ4: the SC-aware extractor (prompts.build_sc_extraction_prompts) on HermesAgent,
# OpenResearcher and WildChat. Each user turn is read whole, one extractor call per turn.
#   1. SC extraction + GPT-5.4 retention judge per dataset, and retention by SC type.
#   2. Compliance of the extractor condition C(H) ⊕ S_t (Eq. 9) on every cached RQ1 run
#      in rq1/main.yaml; cases 1-4 are reused.
#   3. Table: retention, compliance per condition, effective retention.
# Every step resumes from its result pickles, so a crashed run is continued by re-running.

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../_common.sh"

run_sc_extractor_eval sc_extractor_hermes100k
run_sc_extractor_eval sc_extractor_openresearcher100k
run_sc_extractor_eval sc_extractor_wildchat100k
run_analyze sc_extractor_by_type

python -m compaction_integrity.scripts.eval_sc_extractor_compliance \
  --config-name sc_extractor_compliance
run_analyze sc_extractor_compliance
