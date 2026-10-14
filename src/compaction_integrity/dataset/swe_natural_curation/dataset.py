"""COMPINT-SWE-Natural: build the natural-side-constraint eval set end to end.

Four stages, with annotation in between:

  1. ``screen``      upstream trajectories -> candidate pool (tokenized, windowed)
  2. ``candidates``  pool -> blank annotation sheet, one row per cue-bearing paragraph
     ---- annotation (annotate_sc.py): fill the sheet, then type it (``broad_clauses_json``) ----
  3. ``verify``      typed annotations -> per-clause substring/echo report
  4. ``build``       typed annotations -> eval dataset with both context arms

``screen`` admits a task if its GitHub issue contains one of eight negations
(the original screen) or, by default, any of six positively-stated cue
families; ``--cues original`` reproduces the negation-only pool.

Pool ordering is the reproducibility contract, because annotation sheets refer
to pool rows by ``source_row_index``. Rows are ordered by admission tier —
tasks matching the original negation pattern first, then tasks admitted only by
the refined cues — and by descending token count within a tier. This matches
how ``swe_natural_90k_110k_v2`` was assembled (refined-cue tasks appended after
the negation pool), so a single ``screen`` run reproduces it row for row.
``--verify_against`` asserts it against a pool on disk.

Typical use::

    python -m compaction_integrity.dataset.swe_natural_curation.dataset screen \\
        --dataset_name swe_natural_90k_110k_v2
    python -m compaction_integrity.dataset.swe_natural_curation.dataset candidates --exclude <reviewed sheets>
    # ... annotation ...

    # Open-SWE-Traces: screen, cut a task-source subset, then the same stages
    python -m compaction_integrity.dataset.swe_natural_curation.dataset open_swe_screen
    python -m compaction_integrity.dataset.swe_natural_curation.dataset subset --n_rows 1000 \\
        --dataset_name open_swe_rebench_1000
    python -m compaction_integrity.dataset.swe_natural_curation.dataset candidates \\
        --pool_path <subset> --exclude <reviewed sheets> --max_tasks 500
    # ... annotation: dataset/swe_natural_curation/annotate_sc.py ...
    python -m compaction_integrity.dataset.swe_natural_curation.dataset verify
    python -m compaction_integrity.dataset.swe_natural_curation.dataset build
"""

import argparse
import json
import re
from pathlib import Path

import matplotlib
import pandas as pd
import pyarrow.parquet as pq
from datasets import Dataset, concatenate_datasets
from huggingface_hub import hf_hub_download

from compaction_integrity.tokenization import (
    count_tokens_messages,
    count_tokens_messages_batch,
    count_tokens_text,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


# Sources

TRAJECTORY_REPO = "nebius/SWE-rebench-openhands-trajectories"
TRAJECTORY_FILE = "trajectories.parquet"

ISSUE_REPO = "nebius/SWE-rebench"
ISSUE_FILES = (
    "data/filtered-00000-of-00001.parquet",
    "data/test-00000-of-00002.parquet",
    "data/test-00001-of-00002.parquet",
)

# The harness wraps the GitHub issue body in this block. Everything outside it is
# OpenHands boilerplate that also contains deontic words, so cue positions are
# only meaningful inside the block.
# OpenHands uses <issue_description>; SWE-agent and mini-swe-agent (Open-SWE-Traces)
# use <pr_description>.
ISSUE_BLOCK = re.compile(
    r"<(issue_description|pr_description)>(?P<issue>.*?)</\1>", re.DOTALL
)
PARAGRAPH = re.compile(r"\S.*?(?=\n\s*\n|\Z)", re.DOTALL)
FENCED_CODE = re.compile(r"```.*?(?:```|\Z)", re.DOTALL)
# Issue templates ship instructions as HTML comments ("<!-- Please fill in ...
# -->"); the author never wrote them, so cues inside them are not constraints.
HTML_COMMENT = re.compile(r"<!--.*?(?:-->|$)", re.DOTALL)
# swe-rebench-v2 tasks append a generated block after the author's issue: signature,
# location, inputs/outputs and description of each API the reference patch adds or
# changes. It is not author text, so it is not a source of SCs (it stays in
# issue_text as context). In the Open-SWE pool it always opens with this heading and
# runs to the end of the issue.
INTERFACE_SPEC = "New interfaces introduced:"


def _in_interface_spec(start: int, issue_text: str) -> bool:
    """Whether issue_text[start:] lies in the generated interface-spec block."""
    spec = issue_text.find(INTERFACE_SPEC)
    return 0 <= spec <= start


# Cues

# The original deontic screen over the GitHub issue body: eight phrases, matched
# as raw substrings against the raw issue text. Kept verbatim because the shipped
# pool's admission tier and its `keyword_hits` / `keyword_token_frac` columns are
# defined by it, and `viz_sc` excludes its terms from the SC word cloud.
NEGATION_PATTERN = r"do not|don't|must not|should not|cannot|can't|won't|never"

# Cue groups, ordered by review priority. The first six target side-constraint
# language (preservation, compatibility, optionality, directive prohibition,
# method preference, scope); chosen against 12 example constraint sentences and
# 208 reviewed paragraphs, they fire on 36% of kept vs 8% of dropped paragraphs
# and on 1.5% of issue prose, half the rate of the original negation screen.
# `original negation` is that screen's eight phrases, word-bounded (the original
# matched substrings, so `never` also fired on "whenever").
_GAP = r"(?:\w+ ){0,3}"
CUE_FAMILIES: dict[str, dict[str, str]] = {
    "preserve": {
        "preserve": r"\bpreserv\w*",
        "unchanged/untouched": r"\bun(?:changed|touched)\b",
        "leave as-is": rf"\b(?:left|leave) {_GAP}(?:untouched|alone|as[- ]is|unchanged)\b",
        "keep same": rf"\bkeep {_GAP}(?:current|existing|old|original|same|working|intact|closer)\b",
        "should still/remain": r"\b(?:should|must|would) (?:still|continue to|remain)\b",
    },
    "compat": {
        "backward compat": r"\bbackwards?[- ]?compat\w*",
    },
    "optional": {
        "opt-in/out": r"\bopt[- ]?(?:in|out)\b",
        "optional": r"\boptional(?:ly)?\b",
        "give option": r"\b(?:give|as|offer) (?:\w+ )?option\b",
        "unless asked": rf"\b(?:unless|only (?:when|if)) {_GAP}(?:ask\w*|request\w*|explicit\w*|specif\w*|enabl\w*|configur\w*|pass\w*)\b",
        "not required": r"\b(?:not (?:be )?(?:required|necessary)|no need to|need not)\b",
    },
    "prohibition": {
        "negated action": r"\b(?:do not|don'?t|should not|shouldn'?t|must not|mustn'?t) (?:\w+ ){0,2}(?:require|duplicat|chang|modif|touch|break|remov|rewrit|retry|happen|report|recommend|want|alter|overwrit)\w*",
        "don't think should": r"\bdon'?t think (?:\w+ ){0,6}(?:should|want|need|good|necessar\w*)\b",
    },
    "method": {
        "recommend": r"\brecommend\w*",
        "prefer": r"\b(?:prefer(?:ably|red|ence)?|i'?d rather)\b",
        "ideally/if possible": r"\b(?:ideally|if possible|where possible)\b",
    },
    "scope": {
        "out of scope": r"\b(?:out of scope|not part of (?:this|the))\b",
        "separate PR": r"\bseparate (?:pr|issue|pull request)\b",
        "unrelated": r"\bunrelated\b",
    },
    "original negation": {
        "do not": r"\bdo not\b",
        "don't": r"\bdon'?t\b",
        "must not": r"\bmust not\b",
        "should not": r"\bshould not\b",
        "cannot": r"\bcannot\b",
        "can't": r"\bcan'?t\b",
        "won't": r"\bwon'?t\b",
        "never": r"\bnever\b",
    },
}
# The six refined groups, used for task admission alongside NEGATION_PATTERN.
SC_CUE_FAMILIES = tuple(
    family for family in CUE_FAMILIES if family != "original negation"
)
FAMILY_RANK = {family: rank for rank, family in enumerate(CUE_FAMILIES)}
COMPILED = {
    family: {
        label: re.compile(pattern, re.IGNORECASE) for label, pattern in cues.items()
    }
    for family, cues in CUE_FAMILIES.items()
}


# Defaults

DEFAULT_SAVE_ROOT = Path("/data/compaction_integrity/default_ds")
DEFAULT_SCREEN_DIR = Path("outputs/swe_natural_screen")
DEFAULT_MIN_TOKENS = 90_000
DEFAULT_MAX_TOKENS = 110_000

DEFAULT_POOL_PATH = Path(
    "/data/compaction_integrity/default_ds/swe_natural_90k_110k_v2/stitched_dataset"
)
DEFAULT_CANDIDATE_SHEET = Path("studies/swe_natural/generated/sc_candidates.csv")
# Later batches are appended, so earlier rows keep their sc_id and row position.
DEFAULT_ANNOTATIONS = (
    Path("studies/swe_natural/annotations/sc_annotations.broad_keeps.typed.csv"),
    Path("studies/swe_natural/annotations/sc_annotations.broad_keeps.typed_v2.csv"),
)
# Sheets the annotations were drawn from; `verify` reads paragraph offsets here.
DEFAULT_SHEETS = (
    Path("outputs/swe_natural_screen_expanded/manual_sc_annotations_full.csv"),
    Path("studies/swe_natural/annotations/sc_candidates.v2.csv"),
)
DEFAULT_VERIFICATION = Path("studies/swe_natural/generated/span_verification.csv")
DEFAULT_DATASET_NAME = "swe_natural_sc_100k"
DEFAULT_SEAM_REPORT = Path("studies/swe_natural/generated/ablation_seams.csv")

METADATA_COLUMNS = ["trajectory_id", "instance_id", "repo", "resolved", "exit_status"]
SCREEN_BATCH = 32
BUILD_BATCH = 8


# Shared helpers


def _load_issues() -> pd.Series:
    frames = []
    for filename in ISSUE_FILES:
        path = hf_hub_download(ISSUE_REPO, filename, repo_type="dataset")
        frames.append(
            pq.ParquetFile(path)
            .read(columns=["instance_id", "problem_statement"])
            .to_pandas()
        )
    issues = pd.concat(frames, ignore_index=True).drop_duplicates("instance_id")
    return issues.set_index("instance_id")["problem_statement"]


def _to_openai_messages(messages: list[dict[str, object]]) -> list[dict[str, str]]:
    """Flatten to role/content. Tool calls move into content so no context is lost."""
    flattened = []
    for message in messages:
        content = message["content"] or ""
        tool_calls = message["tool_calls"]
        if tool_calls:
            content += json.dumps(tool_calls)
        flattened.append({"role": message["role"], "content": content})
    return flattened


def _issue_index(messages: list[dict[str, str]]) -> int:
    return next(
        index for index, message in enumerate(messages) if message["role"] == "user"
    )


def _prose(issue_text: str) -> str:
    """Issue text with fenced code and template comments blanked out."""
    return HTML_COMMENT.sub(" ", FENCED_CODE.sub(" ", issue_text))


def _matching_families(text: str) -> tuple[list[str], list[str]]:
    """Cue families and individual cue labels firing on a piece of text."""
    families: list[str] = []
    labels: list[str] = []
    for family, cues in COMPILED.items():
        hit = [label for label, regex in cues.items() if regex.search(text)]
        if hit:
            families.append(family)
            labels.extend(hit)
    return families, labels


def _refined_families(issue_text: str) -> list[str]:
    prose = _prose(issue_text)
    return [
        family
        for family in SC_CUE_FAMILIES
        if any(regex.search(prose) for regex in COMPILED[family].values())
    ]


def _keyword_position(
    messages: list[dict[str, str]],
    total_tokens: int,
    regex: re.Pattern[str],
) -> dict[str, float]:
    """Locate the first in-issue negation hit, in the issue and in the full context."""
    user_index = _issue_index(messages)
    user_content = messages[user_index]["content"]
    block = ISSUE_BLOCK.search(user_content)
    if block is None:
        issue_text = user_content
        issue_offset = 0
    else:
        issue_text = block.group("issue")
        issue_offset = block.start("issue")
    hit = regex.search(issue_text)
    if hit is None:
        return {
            "n_keyword_hits": 0,
            "issue_char_frac": float("nan"),
            "keyword_token_position": float("nan"),
            "keyword_token_frac": float("nan"),
        }

    tokens_before_user = count_tokens_messages(messages[:user_index])
    tokens_into_user = count_tokens_text(user_content[: issue_offset + hit.start()])
    token_position = tokens_before_user + tokens_into_user

    return {
        "n_keyword_hits": len(regex.findall(issue_text)),
        "issue_char_frac": hit.start() / len(issue_text),
        "keyword_token_position": token_position,
        "keyword_token_frac": token_position / total_tokens,
    }


def _write_manifest(output_dir: Path, manifest: dict[str, object]) -> None:
    """Same shape as scripts.generate_dataset writes, so consumers need one path."""
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "generation_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )


# Stage 1: screen


def _admit(
    statements: pd.Series,
    instance_ids,
    cues: str,
) -> tuple[set[str], set[str]]:
    """Admit a task by the original negation pattern or (if enabled) refined cues.

    Returns the admitted tasks and the subset matching the original pattern,
    which is their admission tier and so fixes their position in the pool.
    """
    negation_regex = re.compile(NEGATION_PATTERN, re.IGNORECASE)
    negation: set[str] = set()
    admitted: set[str] = set()
    for instance_id in instance_ids:
        statement = statements[instance_id]
        if negation_regex.search(statement):
            negation.add(instance_id)
            admitted.add(instance_id)
        elif cues == "expanded" and _refined_families(statement):
            admitted.add(instance_id)
    return admitted, negation


def _measure(
    parquet_file: pq.ParquetFile,
    metadata: pd.DataFrame,
    candidates: pd.DataFrame,
    statements: pd.Series,
    known_tokens: dict[str, int],
    min_tokens: int,
    max_tokens: int,
) -> tuple[pd.DataFrame, dict[str, list[dict[str, str]]]]:
    """Token-count every admitted trajectory, keeping messages for in-window ones.

    Trajectories whose count is already in ``known_tokens`` are not tokenized
    again; their messages are still decoded when the known count is in window,
    so pool construction never needs a previously saved pool.
    """
    regex = re.compile(NEGATION_PATTERN, re.IGNORECASE)
    wanted = set(candidates["trajectory_id"])
    records: list[dict[str, object]] = []
    kept: dict[str, list[dict[str, str]]] = {}
    row_offset = 0
    for row_group in range(parquet_file.num_row_groups):
        group_size = parquet_file.metadata.row_group(row_group).num_rows
        group = metadata.iloc[row_offset : row_offset + group_size]
        row_offset += group_size

        local_indices = [
            index
            for index, trajectory_id in enumerate(group["trajectory_id"])
            if trajectory_id in wanted
            and (
                trajectory_id not in known_tokens
                or min_tokens <= known_tokens[trajectory_id] <= max_tokens
            )
        ]
        if not local_indices:
            continue

        column = parquet_file.read_row_group(row_group, columns=["trajectory"])
        trajectories = column.column("trajectory").take(local_indices).to_pylist()
        message_lists = [_to_openai_messages(trajectory) for trajectory in trajectories]

        to_tokenize = [
            position
            for position, index in enumerate(local_indices)
            if group.iloc[index]["trajectory_id"] not in known_tokens
        ]
        fresh: dict[int, int] = {}
        for start in range(0, len(to_tokenize), SCREEN_BATCH):
            batch = to_tokenize[start : start + SCREEN_BATCH]
            counts = count_tokens_messages_batch([message_lists[p] for p in batch])
            fresh.update(zip(batch, counts))

        for position, (local_index, messages) in enumerate(
            zip(local_indices, message_lists)
        ):
            meta = group.iloc[local_index]
            trajectory_id = meta["trajectory_id"]
            if position in fresh:
                total_tokens = fresh[position]
                statement = statements[meta["instance_id"]]
                records.append(
                    {
                        "trajectory_id": trajectory_id,
                        "instance_id": meta["instance_id"],
                        "repo": meta["repo"],
                        "resolved": meta["resolved"],
                        "exit_status": meta["exit_status"],
                        "n_turns": sum(1 for m in messages if m["role"] == "assistant"),
                        "n_user_turns": sum(1 for m in messages if m["role"] == "user"),
                        "gpt_oss_tokens": total_tokens,
                        "keyword_hits": ",".join(
                            sorted({m.lower() for m in regex.findall(statement)})
                        ),
                        "issue_chars": len(statement),
                        **_keyword_position(messages, total_tokens, regex),
                    }
                )
            else:
                total_tokens = known_tokens[trajectory_id]
            if min_tokens <= total_tokens <= max_tokens:
                kept[trajectory_id] = messages

        print(
            f"  row group {row_group + 1}/{parquet_file.num_row_groups}: "
            f"{len(records):,} newly tokenized, {len(kept):,} in window"
        )

    return pd.DataFrame.from_records(records), kept


def _select(
    measured: pd.DataFrame,
    negation: set[str],
    min_tokens: int,
    max_tokens: int,
) -> pd.DataFrame:
    """Longest trajectory per task inside the window, ordered by admission tier.

    Tier keeps every negation-screened task ahead of every refined-cue-only task,
    so a fresh run reproduces an existing pool row for row and its
    ``source_row_index`` values stay valid.
    """
    in_window = measured[measured["gpt_oss_tokens"].between(min_tokens, max_tokens)]
    tiered = in_window.assign(
        tier=(~in_window["instance_id"].isin(negation)).astype(int)
    )
    return (
        tiered.sort_values(
            ["tier", "gpt_oss_tokens"], ascending=[True, False], kind="stable"
        )
        .drop_duplicates("instance_id")
        .reset_index(drop=True)
    )


def _pool_rows(
    selected: pd.DataFrame,
    kept: dict[str, list[dict[str, str]]],
    statements: pd.Series,
) -> list[dict[str, object]]:
    regex = re.compile(NEGATION_PATTERN, re.IGNORECASE)
    rows: list[dict[str, object]] = []
    for record in selected.to_dict("records"):
        messages = kept[record["trajectory_id"]]
        statement = statements[record["instance_id"]]
        tokens = int(record["gpt_oss_tokens"])
        position = _keyword_position(messages, tokens, regex)
        rows.append(
            {
                "messages": messages,
                "token_length": tokens,
                "source_conversation_count": 1,
                "trajectory_id": record["trajectory_id"],
                "instance_id": record["instance_id"],
                "repo": record["repo"],
                "resolved": record["resolved"],
                "exit_status": record["exit_status"],
                "n_turns": sum(1 for m in messages if m["role"] == "assistant"),
                "keyword_hits": ",".join(
                    sorted({m.lower() for m in regex.findall(statement)})
                ),
                "n_keyword_hits": position["n_keyword_hits"],
                "issue_char_frac": position["issue_char_frac"],
                "keyword_token_frac": position["keyword_token_frac"],
            }
        )
    return rows


def _plot_lengths(
    frame: pd.DataFrame, min_tokens: int, max_tokens: int, output_dir: Path
) -> Path:
    fig, axis = plt.subplots(figsize=(8, 4))
    axis.hist(frame["gpt_oss_tokens"], bins=80, color="#2b5aa0")
    axis.set_xlabel("context length (gpt-oss-20b tokens)")
    axis.set_ylabel("SC candidate trajectories")
    axis.set_title(
        f"{TRAJECTORY_REPO}: SC candidates by context length (n={len(frame):,})",
        fontsize=10,
    )
    for threshold, style in (
        (64_000, "--"),
        (min_tokens, "-"),
        (100_000, ":"),
        (max_tokens, "-"),
    ):
        axis.axvline(threshold, color="#b31b1b", linestyle=style, linewidth=1)
    axis.margins(x=0.01)

    fig.tight_layout()
    output_path = output_dir / "sc_candidate_length_distribution"
    fig.savefig(output_path.with_suffix(".png"), dpi=160)
    fig.savefig(output_path.with_suffix(".pdf"))
    plt.close(fig)
    return output_path.with_suffix(".png")


def _plot_positions(frame: pd.DataFrame, output_dir: Path) -> Path:
    fig, (left, middle, right) = plt.subplots(1, 3, figsize=(13, 3.8))

    left.hist(frame["issue_char_frac"].dropna(), bins=40, color="#2b5aa0")
    left.set_xlabel("relative position within the issue body")
    left.set_ylabel("SC candidates")
    left.set_title("Where in the issue text", fontsize=10)

    middle.hist(frame["keyword_token_position"].dropna(), bins=40, color="#2b5aa0")
    middle.set_xlabel("absolute token offset in context")
    middle.set_title("Where in the full context (tokens)", fontsize=10)

    right.hist(
        frame["keyword_token_frac"].dropna(), bins=40, range=(0, 1), color="#2b5aa0"
    )
    right.set_xlabel("fraction of full context before the keyword")
    right.set_xlim(0, 1)
    right.set_title("Where in the full context (relative)", fontsize=10)

    fig.suptitle(
        "First in-issue negation hit: position within the issue and within the trajectory",
        fontsize=11,
    )
    fig.tight_layout()
    output_path = output_dir / "sc_candidate_keyword_position"
    fig.savefig(output_path.with_suffix(".png"), dpi=160)
    fig.savefig(output_path.with_suffix(".pdf"))
    plt.close(fig)
    return output_path.with_suffix(".png")


def _summarize_screen(frame: pd.DataFrame) -> str:
    stats = frame["gpt_oss_tokens"].describe(percentiles=[0.5, 0.9, 0.95, 0.99])
    lines = [
        f"gpt_oss_tokens  median={stats['50%']:>9,.0f} p95={stats['95%']:>9,.0f} max={stats['max']:>9,.0f}",
        "",
        f"SC candidates   {len(frame):>7,} trajectories  {frame['instance_id'].nunique():>6,} tasks",
    ]
    for threshold in (64_000, 80_000, 100_000, 120_000):
        long_frame = frame[frame["gpt_oss_tokens"] >= threshold]
        per_task = long_frame.drop_duplicates("instance_id")
        resolved = per_task[per_task["resolved"] == 1]
        lines.append(
            f"  >= {threshold // 1000:>3}K tokens  {len(long_frame):>7,} traj  "
            f"tasks {per_task['instance_id'].nunique():>6,}  "
            f"resolved tasks {len(resolved):>5,}"
        )

    lines.append("")
    lines.append("negation-hit position (first in-issue hit)")
    for column in ("issue_char_frac", "keyword_token_frac", "keyword_token_position"):
        if column not in frame:
            continue
        values = frame[column].dropna()
        if values.empty:
            continue
        stats = values.describe(percentiles=[0.5, 0.9, 0.99])
        lines.append(
            f"  {column:22} median={stats['50%']:>10,.4f} p90={stats['90%']:>10,.4f} "
            f"p99={stats['99%']:>10,.4f} max={stats['max']:>10,.4f}"
        )
    return "\n".join(lines)


def _known_token_counts(paths) -> dict[str, int]:
    """Exact counts from earlier screen.csv files, so re-runs skip the tokenizer."""
    known: dict[str, int] = {}
    for path in paths:
        frame = pd.read_csv(path, usecols=["trajectory_id", "gpt_oss_tokens"])
        known.update(zip(frame["trajectory_id"], frame["gpt_oss_tokens"]))
    return known


def _all_counts(
    new_screen: pd.DataFrame,
    candidates: pd.DataFrame,
    known_tokens: dict[str, int],
    metadata: pd.DataFrame,
) -> pd.DataFrame:
    """Freshly measured counts plus reused ones, back in upstream row order.

    The row order matters: it is what breaks a tie when two trajectories of one
    task have the same length, so reusing counts selects exactly what a
    from-scratch run selects.
    """
    if not known_tokens:
        return new_screen
    reused = candidates[candidates["trajectory_id"].isin(known_tokens.keys())].copy()
    reused["gpt_oss_tokens"] = reused["trajectory_id"].map(known_tokens)
    upstream = {
        trajectory_id: index
        for index, trajectory_id in enumerate(metadata["trajectory_id"])
    }
    return pd.concat([reused, new_screen], ignore_index=True).sort_values(
        "trajectory_id", key=lambda column: column.map(upstream), kind="stable"
    )


def _assert_pool_order(rows: list[dict[str, object]], existing_path: Path) -> None:
    """The pool is append-only: an existing pool must still be its prefix."""
    existing = Dataset.load_from_disk(str(existing_path)).to_pandas()
    existing_ids = existing["trajectory_id"].tolist()
    fresh_ids = [row["trajectory_id"] for row in rows]
    assert fresh_ids[: len(existing_ids)] == existing_ids, (
        f"pool order diverged from {existing_path}; source_row_index in existing "
        "annotation sheets would no longer be valid"
    )
    print(f"verified: first {len(existing_ids)} rows match {existing_path}")


def _screen_manifest(
    args: argparse.Namespace, selected: pd.DataFrame, n_rows: int
) -> dict[str, object]:
    expanded = args.cues == "expanded"
    return {
        "dataset": TRAJECTORY_REPO,
        "test": False,
        "num_convo": n_rows,
        "target_size": f"{args.min_tokens // 1000}k-{args.max_tokens // 1000}k",
        "stitching_method": "none",
        "min_tokens": args.min_tokens,
        "max_tokens": args.max_tokens,
        "screen_pattern": NEGATION_PATTERN,
        "refined_cue_groups": list(SC_CUE_FAMILIES) if expanded else [],
        "admission": (
            "original negation pattern OR any refined cue group"
            if expanded
            else "original negation pattern"
        ),
        "row_order": "admission tier (negation first), then descending tokens",
        "n_negation_tasks": int((~selected["tier"].astype(bool)).sum()),
        "n_refined_only_tasks": int(selected["tier"].astype(bool).sum()),
    }


def run_screen(args: argparse.Namespace) -> None:
    statements = _load_issues()
    parquet_file = pq.ParquetFile(
        hf_hub_download(TRAJECTORY_REPO, TRAJECTORY_FILE, repo_type="dataset")
    )
    metadata = parquet_file.read(columns=METADATA_COLUMNS).to_pandas()

    admitted, negation = _admit(statements, metadata["instance_id"].unique(), args.cues)
    candidates = metadata[metadata["instance_id"].isin(admitted)]
    known_tokens = _known_token_counts(args.measured)
    print(
        f"tasks {metadata['instance_id'].nunique():,} -> admitted {len(admitted):,} "
        f"({len(negation):,} by negation, {len(admitted) - len(negation):,} by refined cues only); "
        f"trajectories {len(candidates):,}, reusing "
        f"{len(set(candidates['trajectory_id']) & known_tokens.keys()):,} token counts"
    )

    new_screen, kept = _measure(
        parquet_file,
        metadata,
        candidates,
        statements,
        known_tokens,
        args.min_tokens,
        args.max_tokens,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    new_screen.to_csv(args.output_dir / "screen.csv", index=False)

    measured = _all_counts(new_screen, candidates, known_tokens, metadata)
    selected = _select(measured, negation, args.min_tokens, args.max_tokens)
    rows = _pool_rows(selected, kept, statements)
    if args.verify_against is not None:
        _assert_pool_order(rows, args.verify_against)

    name = args.dataset_name or (
        f"swe_natural_{args.min_tokens // 1000}k_{args.max_tokens // 1000}k"
    )
    output_dir = args.save_root / name
    save_path = output_dir / "stitched_dataset"
    Dataset.from_list(rows).save_to_disk(str(save_path))
    _write_manifest(output_dir, _screen_manifest(args, selected, len(rows)))

    print()
    if not new_screen.empty:
        print(_summarize_screen(new_screen))
        print()
        print(
            "length plot:  ",
            _plot_lengths(
                new_screen, args.min_tokens, args.max_tokens, args.output_dir
            ),
        )
        print("position plot:", _plot_positions(new_screen, args.output_dir))
    print(
        f"window [{args.min_tokens:,}, {args.max_tokens:,}]: {len(rows)} tasks "
        f"({int((~selected['tier'].astype(bool)).sum())} negation, "
        f"{int(selected['tier'].astype(bool).sum())} refined-cue only)"
    )
    print("saved pool:", save_path)


# Stage 1b: screen Open-SWE-Traces

# Second trajectory source. Same admission (negation OR refined cues over the issue)
# and the same window / longest-per-task selection as `screen`, but the issue is read
# from the trajectory itself (no separate issue table) and shards are streamed and
# deleted, since the full dataset (~43 GB) is too large to keep on disk.
# Path layout: data/<harness>/<teacher_model>/<task_source>/train-*.parquet.
# `task_source` matters for the claim: swe-rebench-v2 issues are raw GitHub text,
# scale-swe issues are rewritten into a templated form.
OPEN_SWE_REPO = "nvidia/Open-SWE-Traces"
OPEN_SWE_COLUMNS = ["instance_id", "repo", "language", "trajectory_id", "resolved", "messages"]
DEFAULT_OPEN_SWE_SCREEN_DIR = Path("outputs/open_swe_screen")
DEFAULT_OPEN_SWE_DOWNLOAD_DIR = Path("/data/compaction_integrity/open_swe_shards")
# Observed chars/token over three harnesses is 2.4-5.7; only trajectories whose
# character count could land in the token window are tokenized.
CHARS_PER_TOKEN_RANGE = (2.0, 7.0)


def _open_swe_shards(patterns: list[str]) -> list[str]:
    from fnmatch import fnmatch
    from huggingface_hub import HfApi

    files = HfApi().list_repo_files(OPEN_SWE_REPO, repo_type="dataset")
    return sorted(
        name
        for name in files
        if name.endswith(".parquet")
        and any(fnmatch(name, f"data/{pattern}") for pattern in patterns)
    )


def _screen_open_swe_shard(
    job: tuple[str, str, int, int, Path],
) -> tuple[list[dict[str, object]], dict[str, list[dict[str, str]]]]:
    filename, cues, min_tokens, max_tokens, download_dir = job
    _, harness, teacher_model, task_source, _ = filename.split("/")
    path = Path(
        hf_hub_download(
            OPEN_SWE_REPO, filename, repo_type="dataset", local_dir=download_dir
        )
    )
    table = pq.read_table(path, columns=OPEN_SWE_COLUMNS)
    path.unlink()

    negation_regex = re.compile(NEGATION_PATTERN, re.IGNORECASE)
    min_chars = min_tokens * CHARS_PER_TOKEN_RANGE[0]
    max_chars = max_tokens * CHARS_PER_TOKEN_RANGE[1]
    records: list[dict[str, object]] = []
    kept: dict[str, list[dict[str, str]]] = {}
    for batch in table.to_batches(max_chunksize=SCREEN_BATCH):
        to_tokenize: list[tuple[dict[str, object], list[dict[str, str]]]] = []
        for row in batch.to_pylist():
            messages = _to_openai_messages(row["messages"])
            issue_text = ISSUE_BLOCK.search(
                messages[_issue_index(messages)]["content"]
            ).group("issue")
            negation = bool(negation_regex.search(issue_text))
            refined = _refined_families(issue_text)
            if not (negation or (cues == "expanded" and refined)):
                continue
            chars = sum(len(message["content"]) for message in messages)
            record = {
                "trajectory_id": row["trajectory_id"],
                "instance_id": row["instance_id"],
                "repo": row["repo"],
                "language": row["language"],
                "harness": harness,
                "teacher_model": teacher_model,
                "task_source": task_source,
                "resolved": row["resolved"],
                "exit_status": None,
                "negation": negation,
                "refined_cue_families": ",".join(refined),
                "keyword_hits": ",".join(
                    sorted({m.lower() for m in negation_regex.findall(issue_text)})
                ),
                "issue_chars": len(issue_text),
                "n_turns": sum(1 for m in messages if m["role"] == "assistant"),
                "chars": chars,
                "gpt_oss_tokens": None,
            }
            records.append(record)
            if min_chars <= chars <= max_chars:
                to_tokenize.append((record, messages))
        counts = count_tokens_messages_batch([messages for _, messages in to_tokenize])
        for (record, messages), count in zip(to_tokenize, counts):
            record["gpt_oss_tokens"] = count
            if min_tokens <= count <= max_tokens:
                kept[record["trajectory_id"]] = messages
    print(
        f"  {filename}: {table.num_rows:,} rows, {len(records):,} admitted, "
        f"{len(kept):,} in window",
        flush=True,
    )
    return records, kept


def run_open_swe_screen(args: argparse.Namespace) -> None:
    from multiprocessing import Pool

    shards = _open_swe_shards(args.subsets)
    print(f"{len(shards)} shards from {OPEN_SWE_REPO} matching {args.subsets}")
    jobs = [
        (shard, args.cues, args.min_tokens, args.max_tokens, args.download_dir)
        for shard in shards
    ]
    records: list[dict[str, object]] = []
    kept: dict[str, list[dict[str, str]]] = {}
    with Pool(args.workers) as pool:
        for shard_records, shard_kept in pool.imap(_screen_open_swe_shard, jobs):
            records.extend(shard_records)
            kept.update(shard_kept)

    screen = pd.DataFrame.from_records(records)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    screen.to_csv(args.output_dir / "screen.csv", index=False)

    negation = set(screen.loc[screen["negation"], "instance_id"])
    selected = _select(screen, negation, args.min_tokens, args.max_tokens)
    existing = set(
        Dataset.load_from_disk(str(args.existing_pool))["instance_id"]
    )
    statements = pd.Series(
        {
            record["instance_id"]: ISSUE_BLOCK.search(
                messages[_issue_index(messages)]["content"]
            ).group("issue")
            for record in selected.to_dict("records")
            for messages in [kept[record["trajectory_id"]]]
        }
    )
    rows = _pool_rows(selected, kept, statements)
    for row, record in zip(rows, selected.to_dict("records")):
        for column in ("language", "harness", "teacher_model", "task_source"):
            row[column] = record[column]
        row["in_swe_natural_pool"] = record["instance_id"] in existing

    name = args.dataset_name or (
        f"open_swe_{args.min_tokens // 1000}k_{args.max_tokens // 1000}k"
    )
    output_dir = args.save_root / name
    save_path = output_dir / "stitched_dataset"
    Dataset.from_list(rows).save_to_disk(str(save_path))
    _write_manifest(
        output_dir,
        {
            "dataset": OPEN_SWE_REPO,
            "subsets": args.subsets,
            "test": False,
            "num_convo": len(rows),
            "target_size": f"{args.min_tokens // 1000}k-{args.max_tokens // 1000}k",
            "stitching_method": "none",
            "min_tokens": args.min_tokens,
            "max_tokens": args.max_tokens,
            "screen_pattern": NEGATION_PATTERN,
            "refined_cue_groups": list(SC_CUE_FAMILIES) if args.cues == "expanded" else [],
            "row_order": "admission tier (negation first), then descending tokens",
            "n_negation_tasks": int((~selected["tier"].astype(bool)).sum()),
            "n_refined_only_tasks": int(selected["tier"].astype(bool).sum()),
        },
    )

    pool_frame = pd.DataFrame.from_records(
        [{key: value for key, value in row.items() if key != "messages"} for row in rows]
    )
    print()
    print(
        f"trajectories {len(screen):,} admitted over {screen['instance_id'].nunique():,} tasks; "
        f"{int(screen['gpt_oss_tokens'].notna().sum()):,} tokenized"
    )
    print(
        f"window [{args.min_tokens:,}, {args.max_tokens:,}]: {len(rows)} tasks "
        f"({int((~selected['tier'].astype(bool)).sum())} negation, "
        f"{int(selected['tier'].astype(bool).sum())} refined-cue only; "
        f"{int(pool_frame['in_swe_natural_pool'].sum())} already in {args.existing_pool})"
    )
    print()
    print(
        pool_frame.groupby(["task_source", "harness", "teacher_model"])
        .agg(tasks=("instance_id", "size"), resolved=("resolved", lambda s: int((s == 1).sum())))
        .to_string()
    )
    print()
    print(pool_frame["language"].value_counts().to_string())
    print("saved pool:", save_path)


# Stage 1c: subset a screened pool

# The first n parent rows from one task source, minus tasks already in other pools,
# in parent order. A larger subset therefore extends a smaller one row for row, and
# every source_row_index cut from the smaller one stays valid in the larger:
# open_swe_rebench_500 is `subset --n_rows 500`, the first 500 rows of
# open_swe_rebench_1000.
DEFAULT_OPEN_SWE_POOL = DEFAULT_SAVE_ROOT / "open_swe_90k_110k" / "stitched_dataset"


def _id_key(instance_id: str) -> str:
    """Case- and punctuation-insensitive task id, for matching across sources."""
    return re.sub(r"[^a-z0-9]", "", instance_id.lower())


def run_subset(args: argparse.Namespace) -> None:
    parent = Dataset.load_from_disk(str(args.parent_pool))
    excluded = {
        _id_key(instance_id)
        for path in args.exclude_pool
        for instance_id in Dataset.load_from_disk(str(path))["instance_id"]
    }
    rows = [
        index
        for index, (task_source, instance_id) in enumerate(
            zip(parent["task_source"], parent["instance_id"])
        )
        if task_source == args.task_source and _id_key(instance_id) not in excluded
    ][: args.n_rows]

    output_dir = args.save_root / args.dataset_name
    save_path = output_dir / "stitched_dataset"
    parent.select(rows).save_to_disk(str(save_path))
    parent_manifest = json.loads(
        (args.parent_pool.parent / "generation_manifest.json").read_text()
    )
    excluded_names = ", ".join(path.parent.name for path in args.exclude_pool)
    _write_manifest(
        output_dir,
        {
            **parent_manifest,
            "num_convo": len(rows),
            "parent_pool": str(args.parent_pool),
            "subset": f"task_source == {args.task_source}, minus tasks in {excluded_names} "
            f"(case/punct-insensitive id match), first {args.n_rows} in parent pool order",
            "parent_row_index": rows,
        },
    )
    print(f"{len(rows)} rows (parent rows {rows[0]}-{rows[-1]}) -> {save_path}")


# Stage 2: candidates

# Filled in at annotation. `decision` is keep/reject/uncertain; a strict COMPINT SC is
# also yes in all three criterion columns (not the main task, applies across the
# session, observable compliance condition). `sc_text` must be an exact,
# character-for-character substring of `candidate_paragraph`. If one paragraph
# carries several distinct SCs, duplicate the row, give each copy one SC, and
# make its `annotation_id` unique.
MANUAL_COLUMNS = (
    "decision",
    "not_main_task",
    "session_scoped",
    "observable",
    "sc_text",
    "sc_occurrence_in_paragraph",
    "sc_type",
    "exclusion_reason",
    "notes",
    "probe",
    "correct_answer",
    "incorrect_answer",
)


def _candidate_rows(pool: Dataset, exclude: set[str]) -> list[dict[str, object]]:
    negation_regex = re.compile(NEGATION_PATTERN, re.IGNORECASE)
    records: list[dict[str, object]] = []
    for source_row_index, row in enumerate(pool):
        messages = row["messages"]
        issue_index = _issue_index(messages)
        block = ISSUE_BLOCK.search(messages[issue_index]["content"])
        issue_text = block.group("issue")
        fenced = [(m.start(), m.end()) for m in FENCED_CODE.finditer(issue_text)]

        for paragraph_index, match in enumerate(PARAGRAPH.finditer(issue_text)):
            annotation_id = f"{row['instance_id']}::p{paragraph_index}"
            if annotation_id in exclude:
                continue
            if any(start <= match.start() < end for start, end in fenced):
                continue
            # Author-written issue text only: skip the generated interface-spec block.
            if _in_interface_spec(match.start(), issue_text):
                continue
            paragraph = match.group(0).rstrip()
            families, labels = _matching_families(HTML_COMMENT.sub("", paragraph))
            if not families:
                continue

            start = match.start()
            end = start + len(paragraph)
            records.append(
                {
                    **{column: "" for column in MANUAL_COLUMNS},
                    "cue_family": families[0],
                    "matched_cues": ",".join(labels),
                    "n_cue_families": len(families),
                    "annotation_id": annotation_id,
                    "source_row_index": source_row_index,
                    "trajectory_id": row["trajectory_id"],
                    "instance_id": row["instance_id"],
                    "repo": row["repo"],
                    "resolved": row["resolved"],
                    "exit_status": row["exit_status"],
                    "token_length": row["token_length"],
                    "n_turns": row["n_turns"],
                    "n_user_turns": sum(1 for m in messages if m["role"] == "user"),
                    "keyword_hits": row["keyword_hits"],
                    "n_keyword_hits": row["n_keyword_hits"],
                    "keyword_token_frac": row["keyword_token_frac"],
                    "paragraph_index": paragraph_index,
                    "paragraph_char_start": start,
                    "paragraph_char_end": end,
                    "paragraph_keyword_hits": ",".join(
                        sorted({m.lower() for m in negation_regex.findall(paragraph)})
                    ),
                    "candidate_paragraph": paragraph,
                    "issue_text": issue_text,
                    "issue_message_index": issue_index,
                    "issue_message_char_start": block.start("issue"),
                    "issue_message_char_end": block.end("issue"),
                    "candidate_message_char_start": block.start("issue") + start,
                    "candidate_message_char_end": block.start("issue") + end,
                }
            )
    return records


def run_candidates(args: argparse.Namespace) -> None:
    if args.output.exists():
        raise SystemExit(
            f"{args.output} already exists; move it or choose --output so review work is not overwritten."
        )

    pool = Dataset.load_from_disk(str(args.pool_path))
    exclude: set[str] = set()
    for path in args.exclude:
        exclude.update(pd.read_csv(path, usecols=["annotation_id"]).annotation_id)

    frame = pd.DataFrame.from_records(_candidate_rows(pool, exclude))
    if args.max_tasks is not None:
        # Records are in pool order: keep the next max_tasks unreviewed tasks.
        tasks = frame["instance_id"].drop_duplicates().iloc[: args.max_tasks]
        frame = frame[frame["instance_id"].isin(tasks)]
    # Most-specific cue family first, so review can stop early once precision drops.
    frame = frame.sort_values(
        by=["cue_family", "n_cue_families", "instance_id", "paragraph_index"],
        key=lambda column: (
            column.map(FAMILY_RANK) if column.name == "cue_family" else column
        ),
        ascending=[True, False, True, True],
    ).reset_index(drop=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)

    print(
        f"{len(frame)} candidate paragraphs over {frame['instance_id'].nunique()} tasks "
        f"(pool {len(pool)} tasks, {len(exclude)} paragraphs already reviewed)"
    )
    print()
    print(
        frame.groupby("cue_family", sort=False)
        .agg(paragraphs=("annotation_id", "size"), tasks=("instance_id", "nunique"))
        .to_string()
    )
    print()
    print("saved:", args.output)


# Stage 3: verify

# COMPINT injects the SC, so its no-SC condition is just the raw context. Here
# the SC is already in the GitHub issue, so the no-SC condition is built by
# *deleting* the annotated span. That subtraction is only well defined if the
# span still occurs verbatim in the saved pool, and only clean if it occurs once:
# the agent frequently re-quotes the issue, and an echo the deletion misses
# leaves the SC standing in the arm that is supposed to lack it.
OFFSET_COLUMNS = (
    "paragraph_index",
    "paragraph_char_start",
    "paragraph_char_end",
    "issue_message_index",
    "candidate_message_char_start",
    "candidate_message_char_end",
)


def _verify(
    annotations: pd.DataFrame,
    offsets: pd.DataFrame,
    pool: Dataset,
) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for row in annotations.to_dict("records"):
        offset = offsets.loc[row["annotation_id"]]
        paragraph = row["candidate_paragraph"]
        entry = pool[int(row["source_row_index"])]
        messages = entry["messages"]
        issue_index = int(offset["issue_message_index"])
        issue_message = messages[issue_index]["content"]
        paragraph_slice = issue_message[
            int(offset["candidate_message_char_start"]) : int(
                offset["candidate_message_char_end"]
            )
        ]

        for clause_index, clause in enumerate(json.loads(row["broad_clauses_json"])):
            text = clause["text"]
            occurrences = [
                (index, message["role"], message["content"].count(text))
                for index, message in enumerate(messages)
                if text in message["content"]
            ]
            total = sum(count for _, _, count in occurrences)
            echoes = [item for item in occurrences if item[0] != issue_index]
            records.append(
                {
                    "annotation_id": row["annotation_id"],
                    "instance_id": row["instance_id"],
                    "source_row_index": int(row["source_row_index"]),
                    "sc_type": row["sc_type"],
                    "clause_index": clause_index,
                    "clause_chars": len(text),
                    "paragraph_matches_dataset": paragraph_slice == paragraph,
                    "paragraph_in_issue_text": paragraph in row["issue_text"],
                    "clause_at_recorded_offsets": paragraph[
                        clause["start"] : clause["end"]
                    ]
                    == text,
                    "clause_in_paragraph": text in paragraph,
                    "occurrences_in_issue_message": issue_message.count(text),
                    "occurrences_in_context": total,
                    "echo_message_indices": ",".join(
                        str(index) for index, _, _ in echoes
                    ),
                    "echo_roles": ",".join(sorted({role for _, role, _ in echoes})),
                    "clause_text": text,
                }
            )
    return pd.DataFrame.from_records(records)


def _summarize_verification(report: pd.DataFrame) -> str:
    exact = report[
        [
            "paragraph_matches_dataset",
            "paragraph_in_issue_text",
            "clause_at_recorded_offsets",
            "clause_in_paragraph",
        ]
    ]
    echoed = report[report["occurrences_in_context"] > 1]
    lines = [
        f"clauses {len(report):,} over {report['annotation_id'].nunique()} annotations "
        f"and {report['instance_id'].nunique()} tasks",
        "",
        "strict-substring checks (all must be True for the subtraction to be defined)",
    ]
    for column in exact.columns:
        lines.append(f"  {column:28} {int(exact[column].sum()):>3}/{len(report)} True")
    lines += [
        "",
        f"clauses occurring once in the trajectory   {int((report['occurrences_in_context'] == 1).sum()):>3}/{len(report)}",
        f"clauses echoed outside the issue message   {len(echoed):>3}/{len(report)} "
        f"({echoed['annotation_id'].nunique()} annotations)",
    ]
    for row in echoed.to_dict("records"):
        lines.append(
            f"  {row['annotation_id']:<52} x{row['occurrences_in_context']} "
            f"roles={row['echo_roles']} msgs={row['echo_message_indices']}"
        )
    return "\n".join(lines)


def _read_offsets(sheets) -> pd.DataFrame:
    """Paragraph offsets for every reviewed candidate, across all sheets."""
    frames = [
        pd.read_csv(path, usecols=("annotation_id", *OFFSET_COLUMNS)) for path in sheets
    ]
    return (
        pd.concat(frames, ignore_index=True)
        .drop_duplicates("annotation_id")
        .set_index("annotation_id")
    )


def _read_annotations(paths) -> pd.DataFrame:
    annotations = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    duplicated = annotations["annotation_id"][annotations["annotation_id"].duplicated()]
    assert duplicated.empty, f"annotation_id in more than one batch: {list(duplicated)}"
    return annotations


# Clause-level sheets (one clause and one probe per row, e.g. the Open-SWE batch)
# carry annotation_id / clause_index / sc_text / sc_type / status / probe / qa_note.
# Each `ready` row becomes its own eval row; trajectory keys come from the
# candidate sheet the clauses were cut from, and the `[kind]` tag that opens
# qa_note is the kind label.
QA_KIND = re.compile(r"^\[(?P<kind>[^\]]+)\]")


def _read_clause_annotations(paths, sheet: Path) -> pd.DataFrame:
    clauses = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    clauses = clauses[clauses["status"] == "ready"].reset_index(drop=True)
    keys = pd.read_csv(
        sheet,
        usecols=["annotation_id", "source_row_index", "trajectory_id", "instance_id",
                 "candidate_paragraph", "issue_text"],
    )
    clauses = clauses.merge(keys, on="annotation_id", how="left", validate="many_to_one")
    clauses["broad_kind"] = clauses["qa_note"].str.extract(QA_KIND)["kind"]
    clauses["broad_constraint_text"] = clauses["sc_text"]
    clauses["broad_clauses_json"] = [
        json.dumps([{"text": text}]) for text in clauses["sc_text"]
    ]
    return clauses


def run_verify(args: argparse.Namespace) -> None:
    annotations = _read_annotations(args.annotations)
    offsets = _read_offsets(args.sheets)
    pool = Dataset.load_from_disk(str(args.pool_path))

    report = _verify(annotations, offsets, pool)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(args.output, index=False)

    print(_summarize_verification(report))
    print()
    print("report:", args.output)


# Stage 4: build

PROBE_PATTERN = re.compile(
    r"^\\textbf\{User:\}\s*``(?P<probe>.*?)''\s*\\newline"
    r"\s*\\compl\{\}\s*(?P<correct>.*?)\s*\\newline"
    r"\s*\\viol\{\}\s*(?P<incorrect>.*?)$",
    re.DOTALL,
)
LATEX_WRAPPER = re.compile(r"\\(?:texttt|textbf|textit|emph)\{(.*?)\}", re.DOTALL)
LATEX_ESCAPE = re.compile(r"\\([%_&#$])")

SENTINEL = "\x00"
# These trajectories carry non-breaking spaces from the GitHub issue bodies.
BLANK = "[ \\t\\xa0]"
SEAM_RADIUS = 90

SOURCE_COLUMNS = ("repo", "resolved", "exit_status", "n_turns", "keyword_token_frac")

# The screen flattened each assistant turn's tool_calls into its text content, so
# the surviving `tool` messages have no call to answer and every chat template
# rejects them ("tool role, but there was no previous assistant message with a
# tool call"). Tool observations are inputs to the agent, so they render as user
# turns — the same plain-chat shape the stitched agent datasets already use.
ROLE_MAP = {"tool": "user"}


def _plain(text: str) -> str:
    """Strip the LaTeX the probes are authored in; the prober sees plain text."""
    text = LATEX_WRAPPER.sub(r"\1", text)
    text = LATEX_ESCAPE.sub(r"\1", text)
    return text.replace("``", '"').replace("''", '"').strip()


def _parse_probe(raw: str) -> dict[str, str]:
    match = PROBE_PATTERN.match(raw.strip())
    return {
        "probe": _plain(match.group("probe")),
        "correct_answer": _plain(match.group("correct")),
        "incorrect_answer": _plain(match.group("incorrect")),
    }


def _delete_spans(
    messages: list[dict[str, str]],
    clauses: list[str],
    issue_index: int,
) -> tuple[list[dict[str, str]], list[dict[str, object]]]:
    """Remove every verbatim clause occurrence and report each removal site.

    Removals are marked with a sentinel first so the whitespace left at the seam
    can be closed without touching indentation elsewhere in the message — these
    trajectories are full of code blocks.
    """
    ablated: list[dict[str, str]] = []
    removals: list[dict[str, object]] = []
    for index, message in enumerate(messages):
        content = message["content"]
        removed = sum(content.count(clause) for clause in clauses)
        if removed:
            for clause in clauses:
                content = content.replace(clause, SENTINEL)
            # A clause cut mid-sentence orphans the punctuation that followed it;
            # leaving it in would advertise the deletion to the prober.
            content = re.sub(rf"{SENTINEL}{BLANK}*[,;:]", SENTINEL, content)
            # Likewise the brackets around a clause that was a whole aside.
            content = re.sub(rf"\({BLANK}*{SENTINEL}{BLANK}*\)", SENTINEL, content)
            content = re.sub(rf"{BLANK}*{SENTINEL}{BLANK}*(\n)", r"\1", content)
            content = re.sub(rf"(\n){BLANK}*{SENTINEL}{BLANK}*", r"\1", content)
            content = re.sub(rf"{BLANK}*{SENTINEL}{BLANK}*", " ", content)
            content = re.sub(rf"{BLANK}+\n", "\n", content)
            # Cutting every item out of a list leaves bare markers behind,
            # including issue-template checkboxes.
            content = re.sub(
                rf"(?m)^{BLANK}*(?:[-*+]|\d+[.)])(?:{BLANK}*\[[ xX]\])?{BLANK}*\r?$\n?",
                "",
                content,
            )
            content = re.sub(r"\n{3,}", "\n\n", content).strip()
            removals.append(
                {
                    "message_index": index,
                    "role": message["role"],
                    "is_issue_message": index == issue_index,
                    "n_spans": removed,
                    "seam_before": _seam(message["content"], clauses),
                    "seam_after": _seam_after(content, message["content"], clauses),
                }
            )
        ablated.append({"role": message["role"], "content": content})
    return ablated, removals


def _seam(content: str, clauses: list[str]) -> str:
    start = min(content.index(clause) for clause in clauses if clause in content)
    end = max(
        content.index(clause) + len(clause) for clause in clauses if clause in content
    )
    return content[max(0, start - SEAM_RADIUS) : end + SEAM_RADIUS]


def _seam_after(ablated: str, original: str, clauses: list[str]) -> str:
    """Same window in the ablated text, anchored on the prefix before the cut."""
    start = min(original.index(clause) for clause in clauses if clause in original)
    prefix = original[max(0, start - SEAM_RADIUS) : start]
    # The cut also takes an opening bracket or a list marker left bare.
    prefix = re.sub(
        rf"(?:\(|(?m:^){BLANK}*(?:[-*+]|\d+[.)])(?:{BLANK}*\[[ xX]\])?){BLANK}*\Z",
        "",
        prefix,
    ).strip()
    anchor = ablated.find(prefix[-60:]) if prefix else -1
    if anchor < 0:
        return ablated[: 2 * SEAM_RADIUS]
    return ablated[anchor : anchor + 2 * SEAM_RADIUS]


def _build(
    annotations: pd.DataFrame,
    pool: Dataset,
    first_sc_id: int = 0,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    rows: list[dict[str, object]] = []
    seams: list[dict[str, object]] = []
    for sc_id, record in enumerate(annotations.to_dict("records"), start=first_sc_id):
        entry = pool[int(record["source_row_index"])]
        messages = [
            {
                "role": ROLE_MAP.get(message["role"], message["role"]),
                "content": message["content"],
            }
            for message in entry["messages"]
        ]
        issue_index = _issue_index(messages)
        clauses = [clause["text"] for clause in json.loads(record["broad_clauses_json"])]
        ablated, removals = _delete_spans(messages, clauses, issue_index)
        issue_removed = sum(
            removal["n_spans"] for removal in removals if removal["is_issue_message"]
        )
        echo_removed = sum(
            removal["n_spans"]
            for removal in removals
            if not removal["is_issue_message"]
        )
        seams.extend(
            {"annotation_id": record["annotation_id"], **removal}
            for removal in removals
        )

        # Experiment contract: the no-SC arm must contain no clause, and every
        # clause must have been cut from the issue message itself.
        residual = sum(
            message["content"].count(clause)
            for message in ablated
            for clause in clauses
        )
        assert residual == 0, f"{record['annotation_id']}: {residual} clause spans left"
        assert issue_removed == len(clauses), record["annotation_id"]

        rows.append(
            {
                "messages": messages,
                "messages_without_sc": ablated,
                "source_conversation_count": 1,
                "sc_id": sc_id,
                "sc_text": record["broad_constraint_text"],
                "sc_type": record["sc_type"],
                "sc_kind": record["broad_kind"],
                "sc_clauses": clauses,
                "sc_issue_message_index": issue_index,
                "n_spans_removed": issue_removed + echo_removed,
                "n_echo_spans_removed": echo_removed,
                "annotation_id": record["annotation_id"],
                "instance_id": record["instance_id"],
                "trajectory_id": record["trajectory_id"],
                **_parse_probe(record["probe"]),
                **{column: entry[column] for column in SOURCE_COLUMNS},
            }
        )

    for start in range(0, len(rows), BUILD_BATCH):
        batch = rows[start : start + BUILD_BATCH]
        with_counts = count_tokens_messages_batch([row["messages"] for row in batch])
        without_counts = count_tokens_messages_batch(
            [row["messages_without_sc"] for row in batch]
        )
        for row, with_count, without_count in zip(batch, with_counts, without_counts):
            row["token_length"] = with_count
            row["token_length_without_sc"] = without_count
    return rows, seams


def _summarize_build(rows: list[dict[str, object]]) -> str:
    frame = pd.DataFrame(
        [
            {
                key: row[key]
                for key in (
                    "annotation_id",
                    "sc_type",
                    "token_length",
                    "token_length_without_sc",
                    "n_spans_removed",
                    "n_echo_spans_removed",
                    "resolved",
                )
            }
            for row in rows
        ]
    )
    frame["token_delta"] = frame["token_length_without_sc"] - frame["token_length"]
    echoed = frame[frame["n_echo_spans_removed"] > 0]
    lines = [
        frame.to_string(index=False),
        "",
        f"rows {len(frame)}  tasks {len({row['instance_id'] for row in rows})}  "
        f"resolved {int((frame['resolved'] == 1).sum())}",
        f"tokens with SC   min {frame['token_length'].min():,} "
        f"median {int(frame['token_length'].median()):,} max {frame['token_length'].max():,}",
        f"token delta      min {frame['token_delta'].min()} max {frame['token_delta'].max()}",
        f"rows with echoes {len(echoed)} ({echoed['n_echo_spans_removed'].sum()} echo spans removed)",
    ]
    return "\n".join(lines)


def run_build(args: argparse.Namespace) -> None:
    if args.clause_sheet is None:
        annotations = _read_annotations(args.annotations)
    else:
        annotations = _read_clause_annotations(args.annotations, args.clause_sheet)
    spec = [
        _in_interface_spec(issue.find(paragraph), issue)
        for paragraph, issue in zip(annotations["candidate_paragraph"], annotations["issue_text"])
    ]
    print(f"dropped {sum(spec)} SC(s) cut from interface-spec blocks (author-written text only)")
    annotations = annotations[[not s for s in spec]].reset_index(drop=True)
    pool = Dataset.load_from_disk(str(args.pool_path))
    pool_manifest = json.loads(
        (args.pool_path.parent / "generation_manifest.json").read_text()
    )
    for record in annotations.to_dict("records"):
        entry = pool[int(record["source_row_index"])]
        assert entry["trajectory_id"] == record["trajectory_id"], record["annotation_id"]

    # --prepend: an already-built eval dataset whose rows come first, unchanged,
    # so its sc_ids stay valid; the new rows continue the sc_id sequence.
    base = None if args.prepend is None else Dataset.load_from_disk(str(args.prepend))
    rows, seams = _build(annotations, pool, 0 if base is None else len(base))
    args.seam_report.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame.from_records(seams).to_csv(args.seam_report, index=False)

    output_dir = args.save_root / args.dataset_name
    save_path = output_dir / "stitched_dataset"
    dataset = Dataset.from_list(rows)
    manifest: dict[str, object] = {
        "dataset": pool_manifest["dataset"],
        "test": False,
        "num_convo": len(rows),
        "target_size": pool_manifest["target_size"],
        "stitching_method": "none",
        "source_dataset": str(args.pool_path),
        "annotations": [str(path) for path in args.annotations],
        "unique_tasks": len({row["instance_id"] for row in rows}),
        "no_sc_arm": "annotated clause spans deleted verbatim in every role",
        "interface_specs": "excluded: author-written issue text only",
        "role_normalization": ROLE_MAP,
    }
    if base is not None:
        base_manifest = json.loads(
            (args.prepend.parent / "generation_manifest.json").read_text()
        )
        dataset = concatenate_datasets([base, dataset.cast(base.features)])
        manifest = {
            "dataset": [base_manifest["dataset"], pool_manifest["dataset"]],
            "test": False,
            "num_convo": len(dataset),
            "target_size": pool_manifest["target_size"],
            "stitching_method": "none",
            "parts": [
                {"rows": [0, len(base) - 1], "prepended_dataset": str(args.prepend), **base_manifest},
                {"rows": [len(base), len(dataset) - 1], **manifest},
            ],
            "unique_tasks": len(set(dataset["instance_id"])),
        }
    dataset.save_to_disk(str(save_path))
    _write_manifest(output_dir, manifest)

    print(_summarize_build(rows))
    print()
    print("saved:", save_path)
    print("seams:", args.seam_report)


# CLI


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    subparsers = parser.add_subparsers(dest="stage", required=True)

    screen = subparsers.add_parser(
        "screen", help="screen upstream trajectories into a tokenized candidate pool"
    )
    screen.add_argument(
        "--cues",
        choices=("expanded", "original"),
        default="expanded",
        help="'original' reproduces the negation-only pool (default: expanded)",
    )
    screen.add_argument(
        "--measured",
        type=Path,
        nargs="*",
        default=[],
        help="screen.csv files with exact token counts to reuse instead of re-tokenizing",
    )
    screen.add_argument("--min_tokens", type=int, default=DEFAULT_MIN_TOKENS)
    screen.add_argument("--max_tokens", type=int, default=DEFAULT_MAX_TOKENS)
    screen.add_argument("--save_root", type=Path, default=DEFAULT_SAVE_ROOT)
    screen.add_argument(
        "--dataset_name", default=None, help="default: swe_natural_<min>k_<max>k"
    )
    screen.add_argument("--output_dir", type=Path, default=DEFAULT_SCREEN_DIR)
    screen.add_argument(
        "--verify_against",
        type=Path,
        default=None,
        help="assert the fresh pool's rows still start with this pool's rows, in order",
    )
    screen.set_defaults(func=run_screen)

    open_swe = subparsers.add_parser(
        "open_swe_screen",
        help=f"screen {OPEN_SWE_REPO} (streamed shard by shard) into a candidate pool",
    )
    open_swe.add_argument(
        "--subsets",
        nargs="+",
        default=["*/*/*/*"],
        help="glob(s) under data/, e.g. 'openhands/*/swe-rebench-v2/*' (default: all)",
    )
    open_swe.add_argument("--cues", choices=("expanded", "original"), default="expanded")
    open_swe.add_argument("--min_tokens", type=int, default=DEFAULT_MIN_TOKENS)
    open_swe.add_argument("--max_tokens", type=int, default=DEFAULT_MAX_TOKENS)
    open_swe.add_argument("--workers", type=int, default=16)
    open_swe.add_argument(
        "--download_dir", type=Path, default=DEFAULT_OPEN_SWE_DOWNLOAD_DIR
    )
    open_swe.add_argument("--save_root", type=Path, default=DEFAULT_SAVE_ROOT)
    open_swe.add_argument(
        "--dataset_name", default=None, help="default: open_swe_<min>k_<max>k"
    )
    open_swe.add_argument("--output_dir", type=Path, default=DEFAULT_OPEN_SWE_SCREEN_DIR)
    open_swe.add_argument(
        "--existing_pool",
        type=Path,
        default=DEFAULT_POOL_PATH,
        help="flag tasks already in this pool (in_swe_natural_pool)",
    )
    open_swe.set_defaults(func=run_open_swe_screen)

    subset = subparsers.add_parser(
        "subset", help="first n rows of a screened pool from one task source, in pool order"
    )
    subset.add_argument("--parent_pool", type=Path, default=DEFAULT_OPEN_SWE_POOL)
    subset.add_argument("--task_source", default="swe-rebench-v2")
    subset.add_argument(
        "--exclude_pool",
        type=Path,
        nargs="*",
        default=[DEFAULT_POOL_PATH],
        help="drop tasks already in these pools (default: the nebius swe_natural pool)",
    )
    subset.add_argument("--n_rows", type=int, required=True)
    subset.add_argument("--save_root", type=Path, default=DEFAULT_SAVE_ROOT)
    subset.add_argument("--dataset_name", required=True)
    subset.set_defaults(func=run_subset)

    candidates = subparsers.add_parser(
        "candidates", help="export a blank annotation sheet from the pool"
    )
    candidates.add_argument("--pool_path", type=Path, default=DEFAULT_POOL_PATH)
    candidates.add_argument(
        "--exclude",
        type=Path,
        nargs="*",
        default=[],
        help="sheets whose annotation_ids were already reviewed; skip those paragraphs",
    )
    candidates.add_argument(
        "--max_tasks",
        type=int,
        default=None,
        help="keep only the first N tasks (pool order) that still have candidates",
    )
    candidates.add_argument("--output", type=Path, default=DEFAULT_CANDIDATE_SHEET)
    candidates.set_defaults(func=run_candidates)

    verify = subparsers.add_parser(
        "verify", help="check every annotated clause is a verbatim, ideally unique, substring"
    )
    verify.add_argument(
        "--annotations", type=Path, nargs="+", default=list(DEFAULT_ANNOTATIONS)
    )
    verify.add_argument("--sheets", type=Path, nargs="+", default=list(DEFAULT_SHEETS))
    verify.add_argument("--pool_path", type=Path, default=DEFAULT_POOL_PATH)
    verify.add_argument("--output", type=Path, default=DEFAULT_VERIFICATION)
    verify.set_defaults(func=run_verify)

    build = subparsers.add_parser(
        "build", help="build the eval dataset, both context arms"
    )
    build.add_argument(
        "--annotations", type=Path, nargs="+", default=list(DEFAULT_ANNOTATIONS)
    )
    build.add_argument("--pool_path", type=Path, default=DEFAULT_POOL_PATH)
    build.add_argument("--save_root", type=Path, default=DEFAULT_SAVE_ROOT)
    build.add_argument("--dataset_name", default=DEFAULT_DATASET_NAME)
    build.add_argument("--seam_report", type=Path, default=DEFAULT_SEAM_REPORT)
    build.add_argument(
        "--clause_sheet",
        type=Path,
        default=None,
        help="candidate sheet for clause-level annotations (one clause + probe per row)",
    )
    build.add_argument(
        "--prepend",
        type=Path,
        default=None,
        help="built eval dataset whose rows are kept first, unchanged",
    )
    build.set_defaults(func=run_build)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
