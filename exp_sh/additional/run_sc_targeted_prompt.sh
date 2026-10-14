#!/usr/bin/env bash
# Compare the Anthropic compaction prompt with the same prompt plus explicit
# session-constraint preservation, on gpt-oss-120b and Qwen3-30B.

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../_common.sh"

run_eval prompt_targeting_100k

python -m compaction_integrity.scripts.evaluate_prompt_targeting
