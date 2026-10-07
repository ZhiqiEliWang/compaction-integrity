#!/usr/bin/env bash
# Agentic inbox: collect the shared pre-compaction checkpoints, then run both paired
# continuations, in the background with nohup. Arguments are Hydra overrides.
# Config: config/tasks/agentic_inbox/config.yaml
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../_common.sh"
cd "${REPO_ROOT}"

GPU=${GPU:-0}
SEEDS=${SEEDS:-20}
LOG_DIR=${LOG_DIR:-logs/agentic_inbox}
mkdir -p "$LOG_DIR"

CUDA_VISIBLE_DEVICES="$GPU" nohup python -u -m compaction_integrity.agentic_inbox.run \
  seeds="$SEEDS" hydra.run.dir="$LOG_DIR/hydra" hydra.output_subdir=null "$@" \
  > "$LOG_DIR/run.log" 2>&1 &
pid=$!
echo "$pid" > "$LOG_DIR/run.pid"
echo "GPU $GPU, PID $pid, log $LOG_DIR/run.log"
