# Lost in Compaction

**Evaluating Side-Constraint Loss under Context Compaction**

Zhiqi Wang, Yichi Zhang, Dongwon Lee, Yuchen Yang — The Pennsylvania State University

[![arXiv](https://img.shields.io/badge/arXiv-2608.11242-b31b1b.svg)](https://arxiv.org/abs/2608.11242)
[![Project page](https://img.shields.io/badge/project-page-2b5aa0.svg)](https://zhiqieliwang.github.io/compaction-integrity/)

This repository is COMPINT, the evaluation suite from the paper: dataset
construction, experiment runners, analysis code, and the SC-aware extractor.

A **side constraint (SC)** is a clause in a user's prompt that (1) is not part of
the user's task, (2) is meant to constrain the model's behavior for the rest of
the session, and (3) has no intended use outside it — *"show me the draft before
sending anything from now on."* When the context window fills and a compactor
summarizes the history, SCs are silently dropped: the agent resumes the task
while violating how the user asked it to proceed.

Across the compactors we tested, only **17% of injected SCs survive compaction
on average**, and most compactors leave the model *less* compliant than running
the same task with no compaction at all. An SC-aware extractor running alongside
the compactor recovers **over 90% retention** on all three scenarios.

---

## Metrics

**Retention** (Eq. 6) — is the SC semantically present in the compacted summary?
Binary, judged by GPT-5.4 as LLM-as-a-judge. Computed for the compaction
condition only.

**Compliance** (Eq. 8) — does the model actually behave according to the SC? Each
SC carries a hand-written forced-choice probe whose two options are the
complying and non-complying behavior. Graded by exact A/B letter match, so it
does not depend on a judge. A/B order is randomized per row via `swap_seed`,
held identical across all four conditions so grades are directly comparable.

**Effect Retention** (Eq. 9) — compliance gain from compaction, baseline-
corrected and normalized against the upper bound:

```
ER = (c_comp - c_lctx) / (c_ub - c_lctx)
```

`ER = 1` is lossless behavioral preservation, `ER = 0` is complete loss.
Reading case 3 against the 1/2 bracket separates *"the compactor dropped the
SC"* from *"the model never followed it anyway"*; case 4 separates loss of the
**text** from loss of **attention** to it.

Default setting for the main experiments: framing fixed to **direct +
preferential** (`explicitness: true`, `hard: false`), SC injected once at the
top, 50 contexts × 15 SCs = **750 instances** per compaction condition.

---

## Setup

```bash
pip install -e .                  # core
pip install -e ".[vllm]"          # + local-weight serving (torch, vllm)
pip install -e ".[llmlingua]"     # + the LLMLingua-2 baseline
```

API keys live in [`api_keys.py`](src/compaction_integrity/api_keys.py), tracked
with empty placeholders. Fill it in, then stop git from seeing your edits:

```bash
git update-index --skip-worktree src/compaction_integrity/api_keys.py
```

`openai_api_key` is required for the GPT-5.4 retention judge and any OpenAI
compactor/prober; `gemini_api_key` only for the multi-judge robustness checks.

**Paths.** Configs and analysis scripts default to a `results_root` of
`/data/compaction_integrity`. Change `results_root:` in the relevant
`config/tasks/eval/*.yaml`, or pass `--results_root` to analysis scripts. Model
caches default to `/data/huggingface_cache`, overridable per run via the `env:`
block (`hf_home`, `vllm_cache_root`). All scripts assume the repo root as CWD.

---

## Quickstart

Build the long-context datasets. WildChat and Hermes Agent rows are stitched
topic-cohesively to the target length; OpenResearcher trajectories are natively
long and only truncated at a turn boundary:

```bash
bash exp_sh/populate_dataset.sh          # 10k / 50k / 100k for all three
```

The 220k variants used by the GPT-5.4-mini supplement are not in that script:

```bash
python -m compaction_integrity.scripts.generate_dataset --config-name wildchat_cat_220k
```

Then reproduce the headline table:

```bash
bash exp_sh/rq1/run_main.sh
```

Runs are resume-safe — re-running skips completed `run_id`s, and the two
full-context conditions are cached across compactors since they do not depend on
the compactor.

---

## Reproducing the paper

Each runner first invokes `scripts.evaluation` (or `eval_sc_extractor` for RQ4)
over the relevant configs in `config/tasks/eval/`, then runs the matching
analysis against the manifest in `config/experiments/<dir>/`.

```bash
# RQ1 — Table 2: retention + compliance across compactors and datasets
bash exp_sh/rq1/run_main.sh              # 100k, all three datasets
bash exp_sh/rq1/run_gpt_5_4_mini.sh      # 220k GPT-5.4-mini supplement

# RQ2 — compaction-system factors
bash exp_sh/rq2/run_diff_input_size.sh   # Figure 4: 10k / 50k / 100k

# RQ3 — SC-side factors  (note: three of these live under rq2/)
bash exp_sh/rq2/run_injection_positions.sh  # Figure 5: top / middle / bottom
bash exp_sh/rq2/run_repeat.sh               # Appendix E: repeat 1..30
bash exp_sh/rq2/run_diff_sc_type.sh         # Figure 6: by SC type
bash exp_sh/rq3/run_diff_prefix.sh          # Table 3: strength x explicitness

# RQ4 — Table 4: the SC-aware extractor
bash exp_sh/rq4/run_sc_extractor.sh
```

**The SC-aware extractor** (§5.4) decomposes the context into turns, processes
only *user* turns with a small model (Qwen3.5-9B), and maintains a running list `S_t` of
detected SCs that is appended to the compacted context — `C(H_t) ⊕ S_t`. It is
training-free and needs no change to the compactor or the agent LLM. Reported
retention: 95.6% on Hermes Agent, 95.1% on OpenResearcher, 90.3% on WildChat.
Results land under `{results_root}/sc_extractor_runs/`, the per-SC-type table
under `{results_root}/analysis/sc_extractor_by_type/`.

Run one extractor config directly:

```bash
python -m compaction_integrity.scripts.eval_sc_extractor \
    --config-name sc_extractor_wildchat100k
```

---

## Robustness checks

These reuse cached contexts from completed RQ1 runs — no compaction or probing
is recomputed — so they are cheap relative to a full run.

| Appendix | What it answers | Command |
|---|---|---|
| **B.3** | Does an SC-targeted compaction prompt fix the loss? *(No — still <40% on WildChat.)* | `exp_sh/additional/run_sc_targeted_prompt.sh` |
| **B.4** | Is the loss just source/compactor family mismatch? *(No.)* | `analyze/style_match.py` |
| **C.2** | Does a second-family judge agree with GPT-5.4? | `exp_sh/additional/run_rejudge.sh` |
| **C.2** | Does the judge favor its own model family? Do humans agree? | `exp_sh/additional/run_family_probe.sh` |
| **D** | Does the result hold in free generation, not just MCQ? | `exp_sh/additional/run_reprobe_generative_hermes100k.sh` |
| **D** | Does it hold across downstream probers? | `exp_sh/additional/run_different_probe.sh` |

[`judge_robustness_eval/`](judge_robustness_eval/) holds the human-agreement
study behind Appendix C.2: a blind 50-row sample labeled by hand, then nested
inside a larger population sample so human, GPT-5.4, and Gemini verdicts are
directly comparable on the same rows. `README_sampling.txt` documents the
sampling procedure.

---

## Layout

```
config/
  tasks/generate_dataset/  # generate_dataset.py inputs
  tasks/eval/              # evaluation.py inputs (one yaml per dataset/variant)
  tasks/reprobe_mcq/       # prober-swap inputs
  experiments/rq[1-4]/     # post-hoc run manifests consumed by analyze/*.py
  experiments/additional/  # manifests for the robustness checks
exp_sh/
  populate_dataset.sh      # build the stitched/truncated datasets
  _common.sh               # shared helpers sourced by the runners
  rq[1-4]/*.sh             # per-RQ runners (eval + analyze)
  additional/*.sh          # robustness-check runners
src/compaction_integrity/
  scripts/evaluation.py    # the four-condition runner; result schema at the top
  scripts/eval_sc_extractor.py  # the SC-aware extractor (RQ4)
  analyze/                 # analysis + plotting, one module per experiment
  compactors/              # compaction transforms C
  dataset/                 # loading, stitching, normalization, SC injection
  runtime/                 # OpenAI / vLLM / vLLM-serve adapters
  sssc.py                  # the 15 SCs and their probes
  prompts.py               # every prompt used in the project
judge_robustness_eval/     # human-agreement study for the retention judge
```

The compaction transform `C` is kept separate from the strategy layer, so
compactors are swappable and the extractor plugs in without touching either.

The full schema of `evaluation_results.pkl` — every column, per condition — is
documented at the top of
[`scripts/evaluation.py`](src/compaction_integrity/scripts/evaluation.py).

---

## Citation

```bibtex
@misc{wang2026lostcompactionevaluatingsideconstraint,
      title={Lost in Compaction: Evaluating Side-Constraint Loss under Context Compaction}, 
      author={Zhiqi Wang and Yichi Zhang and Dongwon Lee and Yuchen Yang},
      year={2026},
      eprint={2608.11242},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2608.11242}, 
}
```

## License

MIT — see [LICENSE](LICENSE).
