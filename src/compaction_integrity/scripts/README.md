# Entry points

Run each as `python -m compaction_integrity.scripts.<name>` from the repository root.
The `exp_sh/` runners call them with the configs used in the paper.

| Script | What it does |
| --- | --- |
| `generate_dataset.py` | Builds the long-context datasets: samples conversations and stitches or truncates them to the target length (Hydra configs in `config/tasks/generate_dataset/`). |
| `evaluation.py` | The four-condition evaluation (retention and compliance) for every compactor and prober in a `config/tasks/eval/` config. The result schema is documented at the top of the file. |
| `print_eval_run_ids.py` | Prints the `run_id`s and result paths an eval config would produce, for filling experiment manifests. |
| `eval_sc_extractor.py` | The SC-aware extractor (RQ4): runs the extractor over each context and judges the retention of its list `S_t`. |
| `eval_sc_extractor_compliance.py` | Probes the extractor condition `C(H) ⊕ S_t` on cached RQ1 runs. |
| `rejudge_retention.py` | Re-judges retention with a second LLM judge on stored compacted contexts. |
| `reprobe_mcq.py` | Re-probes completed runs with a different downstream prober. |
| `reprobe_generative.py` | Re-probes completed runs with free tool-using generation instead of the A/B probe. |
| `evaluate_prompt_targeting.py` | Compares the baseline and SC-targeted compaction prompts. |
| `judge_robustness/` | Samples and judges the human-agreement and judge-family studies (Appendix C.2); see `studies/judge_robustness/README.md`. |
