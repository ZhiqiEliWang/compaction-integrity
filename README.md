# Lost in Context Compaction

**When LLM Systems Drop Side Constraints**

<p align="center">
  <img src="docs/static/fig1-failure-mode.png" width="90%" alt="Three panels from an inbox-cleanup session: the user asks the agent to suggest deletions and adds the side constraint &quot;Don't act until I tell you to&quot;, and the agent only suggests; the compactor's summary drops the constraint; after &quot;Please continue working through the rest of my inbox.&quot; the agent calls delete_email and deletes 10 emails.">
</p>

When the context window fills, LLM agents compact their history to keep working.
Constraints the user set earlier in the session, such as *"don't action until I
tell you to,"* can be silently dropped along the way. We call them side
constraints (SCs). This repository contains **COMPINT**, which measures how well
compactors preserve SCs, and an SC-aware extractor that runs alongside the
compactor to keep them.

## What's in this repository

| Component | Code |
|---|---|
| COMPINT: 15 hand-written SCs with forced-choice probes, injected into long WildChat, Hermes Agent and OpenResearcher contexts | [`sssc.py`](src/compaction_integrity/sssc.py), [`scripts/evaluation.py`](src/compaction_integrity/scripts/evaluation.py) |
| SWE-Natural: SCs already stated in real GitHub issues, and the LLM annotation pipeline that finds them | [`dataset/swe_natural_curation/`](src/compaction_integrity/dataset/swe_natural_curation/) |
| SC-aware extractor | [`scripts/eval_sc_extractor.py`](src/compaction_integrity/scripts/eval_sc_extractor.py) |
| Agentic inbox: one SC in a live tool-using agent | [`agentic_inbox/`](src/compaction_integrity/agentic_inbox/README.md) |

## Setup

```bash
pip install -e .                  # core
pip install -e ".[vllm]"          # local-weight serving (torch, vllm)
pip install -e ".[llmlingua]"     # the LLMLingua-2 baseline
pip install -e ".[agentic]"       # the agentic inbox
```

Fill in [`api_keys.py`](src/compaction_integrity/api_keys.py) (OpenAI for the
GPT-5.4 judge and OpenAI models; Gemini only for the multi-judge checks), then
hide your edits from git:

```bash
git update-index --skip-worktree src/compaction_integrity/api_keys.py
```

Run everything from the repository root. Results default to
`/data/compaction_integrity`; change `results_root` in `config/tasks/eval/*.yaml`
or pass `--results_root` to the analysis scripts.

## Data

```bash
bash exp_sh/populate_dataset.sh       # WildChat, Hermes Agent, OpenResearcher at 10k / 50k / 100k
python -m compaction_integrity.scripts.generate_dataset --config-name wildchat_cat_220k  # and the other *_cat_220k
```

SWE-Natural is built from `nvidia/Open-SWE-Traces` in batches of 250 tasks:

```bash
D=compaction_integrity.dataset.swe_natural_curation.dataset
POOL=/data/compaction_integrity/default_ds/open_swe_rebench_1000/stitched_dataset
python -m $D open_swe_screen                  # stream and screen the trajectories
python -m $D subset --n_rows 1000 --dataset_name open_swe_rebench_1000
python -m $D candidates --pool_path $POOL --max_tasks 250 \
    --output studies/swe_natural/generated/sc_candidates.open_swe_<batch>.csv
bash exp_sh/swe_natural/annotate_open_swe.sh <batch>  # LLM annotation: label, adjudicate, type, probe
python -m $D build --pool_path $POOL --dataset_name <name> \
    --annotations studies/swe_natural/annotations/open_swe_<batch>/sc_annotations.broad_keeps.probe_ready.csv
```

For later batches, pass the earlier candidate sheets to `candidates --exclude`
and the built dataset to `build --prepend`, so earlier rows keep their position.

## Reproducing the paper

| Result | Command |
|---|---|
| Retention and compliance by compactor and dataset | `bash exp_sh/rq1/run_main.sh` |
| GPT-5.4-mini at 220k | `bash exp_sh/rq1/run_gpt_5_4_mini.sh` |
| SWE-Natural | `bash exp_sh/rq1/run_swe_natural_sc_n126.sh` |
| Input length | `bash exp_sh/rq2/run_diff_input_size.sh` |
| Injection position | `bash exp_sh/rq3/run_injection_positions.sh` |
| SC type | `bash exp_sh/rq3/run_diff_sc_type.sh` |
| Strength × explicitness | `bash exp_sh/rq3/run_diff_prefix.sh` |
| Repeating the SC | `bash exp_sh/rq3/run_repeat.sh` |
| SC-aware extractor | `bash exp_sh/rq4/run_sc_extractor.sh`<br>`bash exp_sh/rq4/run_sc_extractor_swe_natural.sh` |

Each runner evaluates its configs, then writes tables and figures under
`/data/compaction_integrity/analysis/` by default. Runs are resume-safe: completed runs are skipped.
Retention, compliance and effective retention are defined in the paper; the
per-row result schema is documented at the top of
[`scripts/evaluation.py`](src/compaction_integrity/scripts/evaluation.py).

## License

MIT — see [LICENSE](LICENSE).
