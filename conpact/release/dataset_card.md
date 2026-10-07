---
pretty_name: conpact
language:
- en
license: other
license_name: mixed-source
task_categories:
- text-generation
tags:
- long-context
- context-compaction
- instruction-following
- benchmark
size_categories:
- 1K<n<10K
configs:
- config_name: wildchat
  data_files:
  - split: test
    path: wildchat/test-*.parquet
- config_name: hermes
  data_files:
  - split: test
    path: hermes/test-*.parquet
- config_name: openresearcher
  data_files:
  - split: test
    path: openresearcher/test-*.parquet
- config_name: swe_natural
  data_files:
  - split: test
    path: swe_natural/test-*.parquet
- config_name: haystacks
  data_files:
  - split: wildchat
    path: haystacks/wildchat-*.parquet
  - split: hermes
    path: haystacks/hermes-*.parquet
  - split: openresearcher
    path: haystacks/openresearcher-*.parquet
---

# conpact

Do side constraints survive context compaction?

A **side constraint (SC)** is a user instruction meant to hold for the rest of a
session, e.g. *"Don't send any emails on my behalf, draft them and let me send
them myself."* Each instance is a ~100k-token history with one SC in it. Compact
the history with your system, then measure whether the SC is still there
(**retention**) and whether a model still follows it (**compliance**).

This is the COMPINT benchmark from
[Lost in Context Compaction](https://arxiv.org/abs/2608.11242). To evaluate a compactor,
or to build instances at other lengths and SC positions, use the `conpact`
package (`pip install conpact`).

## Configs

| Config | Rows | Content |
|---|---|---|
| `wildchat` | 750 | 50 multi-turn chat histories × 15 SCs |
| `hermes` | 750 | 50 agent trajectories × 15 SCs |
| `openresearcher` | 750 | 50 long-horizon research trajectories × 15 SCs |
| `swe_natural` | 126 | SWE agent trajectories whose GitHub issue already states an SC |
| `haystacks` | 50 / 50 / 30 | SC-free ~220k-token histories (splits `wildchat`, `hermes`, `openresearcher`) that the package cuts to length |

The first three configs place the SC at the start of the first user turn as
*"For the rest of this session. …"*: the inputs behind the paper's main results.
SWE-Natural SCs are the issue authors' own words, where they wrote them.

```python
from datasets import load_dataset
ds = load_dataset("ZhiqiEliWang/conpact", "wildchat", split="test")
```

## Protocol

The probe model receives `[system_prompt, developer_prompt, *context, user_turn]`
(as system, developer, …, user messages) and must answer with one letter.

| Condition | `context` | `user_turn` |
|---|---|---|
| full + SC | `messages` | `probe_prompt` |
| full (no SC) | `messages_without_sc` | `probe_prompt` |
| compaction | compacted `messages` | `probe_prompt` |
| upper bound | compacted `messages` | `post_sc_probe_prompt` |

- **Retention rate (%)**: share of compacted contexts in which an LLM judge finds `sc_text`.
- **Compliance rate (%)**: share of parseable answers equal to `compliant_letter`, per condition.
- **Effective retention (%)**: (compaction − full) / (upper bound − full), on compliance rates.

The paper uses gpt-oss-120b as the probe model and GPT-5.4 as the judge, both at
low reasoning effort; the package implements this setup.

## Fields

| Field | Description |
|---|---|
| `id`, `source`, `haystack_id` | instance, source and history ids |
| `messages` | the history with the SC: the compactor's input |
| `messages_without_sc` | the same history without the SC |
| `sc_text`, `sc_rendered` | the SC as written (the judge's reference), and with its framing as it appears in `messages` |
| `sc_id`, `sc_type` | SC id and type: Action, Information, Process, Preference or Output |
| `sc_message_indices` | indices of the turns in `messages` that carry the SC |
| `position`, `repeat`, `strict`, `direct` | placement and framing of the SC |
| `probe`, `correct_answer`, `incorrect_answer` | the probe request and its compliant / non-compliant behaviors |
| `probe_prompt`, `post_sc_probe_prompt` | the rendered probe, without and with the SC restated |
| `compliant_letter`, `swap_seed` | the compliant option (`A`/`B`) and the seed of the A/B order |
| `system_prompt`, `developer_prompt` | the probe model's system and developer messages |
| `target_length`, `token_length` | nominal length and actual tokens of `messages_without_sc` (gpt-oss tokenizer) |
| `meta` | JSON with source-specific fields (SWE-Natural: repo, issue, annotation) |

Tool calls and results are flattened into assistant text. Agent histories start
with the trace's own system message. 8 of the 50 `hermes` histories are
47k–90k tokens, kept as evaluated in the paper.

## Sources and licenses

| Config | Source | License |
|---|---|---|
| `wildchat` | [allenai/WildChat](https://huggingface.co/datasets/allenai/WildChat) | ODC-BY |
| `hermes` | [lambda/hermes-agent-reasoning-traces](https://huggingface.co/datasets/lambda/hermes-agent-reasoning-traces) | Apache-2.0 |
| `openresearcher` | [OpenResearcher/OpenResearcher-Dataset](https://huggingface.co/datasets/OpenResearcher/OpenResearcher-Dataset) | MIT |
| `swe_natural` | [nebius/SWE-rebench-openhands-trajectories](https://huggingface.co/datasets/nebius/SWE-rebench-openhands-trajectories), [nvidia/Open-SWE-Traces](https://huggingface.co/datasets/nvidia/Open-SWE-Traces) | CC-BY-4.0 |

Histories keep their source's license. The SCs, probes and SWE-Natural
annotations are MIT.

## Citation

```bibtex
@misc{wang2026lostcompactionevaluatingsideconstraint,
      title={Lost in Context Compaction: When LLM Systems Drop Side Constraints},
      author={Zhiqi Wang and Yichi Zhang and Dongwon Lee and Nathalie Baracaldo and Yuchen Yang},
      year={2026},
      eprint={2608.11242},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2608.11242},
}
```
