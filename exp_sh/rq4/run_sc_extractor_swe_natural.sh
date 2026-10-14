#!/usr/bin/env bash
# RQ4: the SWE-Natural extractor (prompts.build_swe_natural_sc_extraction_prompts) on
# SWE-Natural n126, where each SC sits inside the issue handed to the agent. The issue
# is read in 5-sentence windows, one extractor call per window.
#   1. SC extraction + GPT-5.4 retention judge.
#   2. Compliance of the extractor condition C(H) ⊕ S_t (Eq. 9) on every cached RQ1 run
#      in rq1/swe_natural_sc_n126.yaml; cases 1-4 are reused.
#   3. Table: retention, compliance per condition, effective retention.
# Every step resumes from its result pickles, so a crashed run is continued by re-running.

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../_common.sh"

ANALYSIS_ROOT="${ANALYSIS_ROOT:-/data/compaction_integrity/analysis}"

run_sc_extractor_eval sc_extractor_swe_natural_n126

python -m compaction_integrity.scripts.eval_sc_extractor_compliance \
  --config-name sc_extractor_compliance_swe_natural
run_analyze sc_extractor_compliance \
  --config "${REPO_ROOT}/config/tasks/sc_extractor/sc_extractor_compliance_swe_natural.yaml" \
  --output_dir "${ANALYSIS_ROOT}/sc_extractor_compliance_swe_natural"
