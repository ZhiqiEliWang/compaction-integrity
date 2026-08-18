"""Matched-vs-mismatched analysis: does a compactor's retention advantage come
from *compaction ability* or from stylistic match between the source-trajectory
author model and the compactor's own model family?

Reviewer concern (paraphrased): the RQ1 matrix compares compactors across
heterogeneous trajectory styles (wildchat / hermes / openresearcher) whose
source models differ, so observed model/dataset differences might be driven by
source-trajectory <-> compactor mismatch rather than compaction ability.

We can test this *without generating any new trajectory*, because each source
dataset has a known author family and one of them was authored by the headline
compactor family:

    wildchat        -> OpenAI (GPT-3.5/4)      matched compactor: gpt_5.4_mini
    hermes (GLM)    -> GLM-5.1 (ZhipuAI)       matched compactor: (none in set)
    openresearcher  -> GPT-OSS-120B            matched compactor: gpt_oss_120b  <- MATCHED

So `openresearcher x gpt_oss_120b` is a genuine *matched* cell already present in
the existing runs, while `gpt_oss_120b` is *mismatched* on wildchat and hermes.

Core estimand -- a difference-in-differences "home-field" effect that nets out
(a) how good the compactor generally is and (b) how hard the dataset generally
is:

    home_field(c) = [ R(c, home_d) - mean_{c' in peers} R(c', home_d) ]
                  - [ mean_{d != home_d} ( R(c, d) - mean_{c'} R(c', d) ) ]

where peers = the other LLM compactors. If style-match were driving the
leaderboard, matched compactors would show a large positive home_field effect.
A ~0 (or negative) effect is evidence the advantage is compaction ability, not
stylistic home-field.

Usage:
    python -m compaction_integrity.analyze.style_match \
        --manifest config/experiments/rq1/main.yaml \
        --results_root /data/compaction_integrity \
        --output_dir /data/compaction_integrity/analysis/style_match
"""

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from compaction_integrity.analyze.utils import (
    extract_compactor_name,
    extract_dataset_config,
    fmt_compactor_label,
    fmt_dataset_label,
    load_manifest_results,
    tee_stdout,
)

# ----------------------------------------------------------------------------
# Provenance: author family of each source dataset, and model family of each
# compactor. `matched` == a cell where the compactor's family authored the
# source trajectories.
# ----------------------------------------------------------------------------

# keyed by extract_dataset_config(dataset) -> e.g. "wildchat_cat"
DATASET_AUTHOR_FAMILY: dict[str, str] = {
    "wildchat_cat": "openai",
    "hermes_cat": "glm",          # GLM-5.1 subset of hermes-agent-reasoning-traces
    "openresearcher_cat": "gpt-oss",
}

# keyed by compactor_name (config key). Non-LLM baselines map to None (they have
# no author family and so are never matched/mismatched).
COMPACTOR_MODEL_FAMILY: dict[str, str | None] = {
    "gpt_oss_120b_anthropic_prompt": "gpt-oss",
    "gpt_oss_120b_anthropic_sc_targeted_prompt": "gpt-oss",
    "gpt_oss_120b_pi_mono_prompt": "gpt-oss",
    "qwen30b_anthropic_prompt": "qwen",
    "qwen30b_anthropic_sc_targeted_prompt": "qwen",
    "gemma_4_anthropic_prompt": "gemma",
    "gpt_5.4_mini_pi_mono_prompt": "openai",
    "gpt_5_4_mini_pi_mono_prompt": "openai",
    "recent_5": None,
    "llmlingua2_t500": None,
}

# Which families have a matched source dataset in this experiment (used to pick
# the "home" dataset for the home-field test).
FAMILY_HOME_DATASET: dict[str, str] = {
    author: cfg for cfg, author in DATASET_AUTHOR_FAMILY.items()
}


def _load(manifest_path: Path, results_root: Path) -> pd.DataFrame:
    df = load_manifest_results(
        manifest_path=manifest_path,
        results_root=results_root,
        result_file_name="evaluation_results.pkl",
    )
    df = df.loc[df["compaction_status"] == "success"].copy()
    df["compactor_name"] = df["compactor"].map(extract_compactor_name)
    df["dataset_config"] = df["dataset"].map(extract_dataset_config)
    df["compactor_label"] = df["compactor_name"].map(fmt_compactor_label)
    df["dataset_label"] = df["dataset_config"].map(fmt_dataset_label)
    df["compactor_family"] = df["compactor_name"].map(COMPACTOR_MODEL_FAMILY)
    df["dataset_family"] = df["dataset_config"].map(DATASET_AUTHOR_FAMILY)
    df["retention"] = df["retention"].astype(float)
    df["matched"] = (
        df["compactor_family"].notna()
        & (df["compactor_family"] == df["dataset_family"])
    )
    return df


def _cell_table(df: pd.DataFrame) -> pd.DataFrame:
    """Mean retention per (compactor, dataset) cell."""
    tab = (
        df.groupby(["compactor_name", "dataset_config"])["retention"]
        .mean()
        .unstack("dataset_config")
    )
    return tab


def _additive_residuals(tab: pd.DataFrame) -> pd.DataFrame:
    """Two-way additive-model residuals: R(c,d) - rowmean(c) - colmean(d) + grand.
    Positive residual on a matched cell == the confound the reviewer worries
    about."""
    grand = np.nanmean(tab.values)
    row = tab.mean(axis=1)
    col = tab.mean(axis=0)
    resid = tab.sub(row, axis=0).sub(col, axis=1) + grand
    return resid


def _home_field_did(
    df: pd.DataFrame,
    family: str,
    n_boot: int = 2000,
    seed: int = 42,
) -> dict[str, float] | None:
    """Difference-in-differences home-field effect for a model `family`, with a
    conversation-clustered bootstrap CI.

    Compares (family - peers) on the home dataset against (family - peers) on
    the away datasets. Peers = other *LLM* compactors (families != this one,
    excluding non-LLM baselines)."""
    home_cfg = FAMILY_HOME_DATASET.get(family)
    if home_cfg is None or home_cfg not in set(df["dataset_config"]):
        return None

    fam = df["compactor_family"]
    focal = df[fam == family]
    peers = df[fam.notna() & (fam != family)]  # other LLM compactors only
    if focal.empty or peers.empty:
        return None

    def did(focal_df: pd.DataFrame, peers_df: pd.DataFrame) -> float:
        def gap(d_cfg: str) -> float:
            f = focal_df.loc[focal_df["dataset_config"] == d_cfg, "retention"]
            p = peers_df.loc[peers_df["dataset_config"] == d_cfg, "retention"]
            if len(f) == 0 or len(p) == 0:
                return np.nan
            return f.mean() - p.mean()

        home_gap = gap(home_cfg)
        away_cfgs = [d for d in df["dataset_config"].unique() if d != home_cfg]
        away_gaps = [gap(d) for d in away_cfgs]
        away_gaps = [g for g in away_gaps if not np.isnan(g)]
        if np.isnan(home_gap) or not away_gaps:
            return np.nan
        return home_gap - float(np.mean(away_gaps))

    point = did(focal, peers)

    # Cluster bootstrap over source conversations within each dataset.
    rng = np.random.default_rng(seed)
    # unique conversation ids per dataset
    conv_key = ["dataset_config", "source_row_index"]
    focal_g = {k: v for k, v in focal.groupby(conv_key)}
    peers_g = {k: v for k, v in peers.groupby(conv_key)}
    convs_by_ds = (
        df.drop_duplicates(conv_key)
        .groupby("dataset_config")["source_row_index"]
        .apply(list)
        .to_dict()
    )

    boots = []
    for _ in range(n_boot):
        f_parts, p_parts = [], []
        for ds, convs in convs_by_ds.items():
            picked = rng.choice(convs, size=len(convs), replace=True)
            for cid in picked:
                key = (ds, cid)
                if key in focal_g:
                    f_parts.append(focal_g[key])
                if key in peers_g:
                    p_parts.append(peers_g[key])
        if not f_parts or not p_parts:
            continue
        val = did(pd.concat(f_parts), pd.concat(p_parts))
        if not np.isnan(val):
            boots.append(val)

    lo, hi = (np.nan, np.nan)
    if boots:
        lo, hi = np.percentile(boots, [2.5, 97.5])
    return {
        "family": family,
        "home_dataset": home_cfg,
        "home_field_did": point,
        "ci_lo": float(lo),
        "ci_hi": float(hi),
        "n_boot": len(boots),
    }


def _plot_heatmaps(tab: pd.DataFrame, resid: pd.DataFrame, df: pd.DataFrame, out: Path) -> None:
    matched_cells = set(
        map(tuple, df.loc[df["matched"], ["compactor_name", "dataset_config"]].drop_duplicates().values)
    )
    col_labels = [fmt_dataset_label(c) for c in tab.columns]
    row_labels = [fmt_compactor_label(r) for r in tab.index]

    for name, data, cmap, center in [
        ("retention", tab, "viridis", None),
        ("interaction_residual", resid, "RdBu_r", 0.0),
    ]:
        fig, ax = plt.subplots(figsize=(1.6 + 1.4 * data.shape[1], 1.0 + 0.5 * data.shape[0]))
        arr = data.values.astype(float)
        vmax = np.nanmax(np.abs(arr)) if center == 0.0 else np.nanmax(arr)
        vmin = -vmax if center == 0.0 else np.nanmin(arr)
        im = ax.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
        ax.set_xticks(range(data.shape[1]), col_labels, rotation=30, ha="right")
        ax.set_yticks(range(data.shape[0]), row_labels)
        for i, r in enumerate(data.index):
            for j, c in enumerate(data.columns):
                v = arr[i, j]
                if np.isnan(v):
                    continue
                is_match = (r, c) in matched_cells
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=8, color="black" if is_match else "dimgray",
                        fontweight="bold" if is_match else "normal")
                if is_match:
                    ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                                               edgecolor="lime", lw=2.5))
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(f"{name.replace('_', ' ')} (green box = matched source/compactor)")
        fig.tight_layout()
        fig.savefig(out / f"heatmap_{name}.pdf", bbox_inches="tight")
        fig.savefig(out / f"heatmap_{name}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, default=Path("config/experiments/rq1/main.yaml"))
    ap.add_argument("--results_root", type=Path, default=Path("/data/compaction_integrity"))
    ap.add_argument("--output_dir", type=Path,
                    default=Path("/data/compaction_integrity/analysis/style_match"))
    ap.add_argument("--n_boot", type=int, default=2000)
    args = ap.parse_args()

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    with tee_stdout(out / "style_match.log"):
        df = _load(args.manifest, args.results_root)

        print("=" * 78)
        print("PROVENANCE MAP (source author family -> matched compactor family)")
        for cfg, fam in DATASET_AUTHOR_FAMILY.items():
            print(f"  {cfg:<20} authored by {fam}")
        print()

        n_items = df.groupby(["compactor_name", "dataset_config"]).size().unstack("dataset_config")
        print("per-cell item counts:\n", n_items, "\n")

        matched = df.loc[df["matched"], ["compactor_name", "dataset_config"]].drop_duplicates()
        print("MATCHED cells present in this manifest:")
        if matched.empty:
            print("  (none)")
        for _, r in matched.iterrows():
            print(f"  {r['compactor_name']}  x  {r['dataset_config']}")
        print()

        tab = _cell_table(df)
        resid = _additive_residuals(tab)
        print("mean retention per cell R(compactor, dataset):")
        print(tab.round(3), "\n")
        print("two-way interaction residual  (positive on a matched cell == the confound):")
        print(resid.round(3), "\n")

        # residuals on matched vs mismatched cells
        rows = []
        for c in tab.index:
            for d in tab.columns:
                fam_c = COMPACTOR_MODEL_FAMILY.get(c)
                if fam_c is None:
                    continue  # skip non-LLM baselines
                rows.append({
                    "compactor": c, "dataset": d,
                    "matched": DATASET_AUTHOR_FAMILY.get(d) == fam_c,
                    "residual": resid.loc[c, d],
                    "retention": tab.loc[c, d],
                })
        cell_df = pd.DataFrame(rows)
        m_res = cell_df.loc[cell_df["matched"], "residual"]
        mm_res = cell_df.loc[~cell_df["matched"], "residual"]
        print(f"matched-cell residuals:    mean={m_res.mean():+.3f}  values={m_res.round(3).tolist()}")
        print(f"mismatched-cell residuals: mean={mm_res.mean():+.3f}  n={len(mm_res)}")
        print("  -> if matched-cell residuals are not systematically positive, the")
        print("     style-match confound is not driving the leaderboard.\n")

        print("=" * 78)
        print("HOME-FIELD DIFFERENCE-IN-DIFFERENCES (per model family with a home dataset)")
        did_rows = []
        for fam in sorted(set(v for v in COMPACTOR_MODEL_FAMILY.values() if v)):
            res = _home_field_did(df, fam, n_boot=args.n_boot)
            if res is None:
                continue
            did_rows.append(res)
            print(f"  {fam:<8} home={res['home_dataset']:<18} "
                  f"DiD={res['home_field_did']:+.3f}  "
                  f"95% CI=[{res['ci_lo']:+.3f}, {res['ci_hi']:+.3f}]  "
                  f"(nboot={res['n_boot']})")
        print("  DiD > 0  => family does relatively better on its OWN-authored source")
        print("             (supports the reviewer's style-match confound).")
        print("  CI spanning 0 => no detectable home-field advantage.\n")

        cell_df.to_csv(out / "cell_residuals.csv", index=False)
        pd.DataFrame(did_rows).to_csv(out / "home_field_did.csv", index=False)
        tab.to_csv(out / "retention_matrix.csv")
        resid.to_csv(out / "interaction_residual_matrix.csv")
        _plot_heatmaps(tab, resid, df, out)
        print(f"All outputs written to {out}")


if __name__ == "__main__":
    main()
