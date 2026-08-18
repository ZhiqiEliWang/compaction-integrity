## generate_dataset.py
This script prepares context datasets for compaction experiments. It samples conversations from the specified dataset and stitches or trims them to the desired length (e.g., 100k tokens or 1M tokens per conversation history). The prepared dataset is stored in HuggingFace Dataset format.

## print_run_ids.py
This script prints the canonical `run_id` values implied by a `run_compaction` Hydra config, without executing the compactor. Use it to fill experiment manifests before or after launching runs.
