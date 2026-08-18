"""Per-constraint-type retention analysis for the SC extractor."""

from __future__ import annotations

from pathlib import Path

import hydra
import pandas as pd
from omegaconf import DictConfig, OmegaConf

from compaction_integrity.analyze.utils import tee_stdout


SC_TYPE_ORDER = ["Action", "Information", "Process", "Preference", "Output"]
SC_TYPE_BY_ID = {
    **{sssc_id: "Action" for sssc_id in range(1, 4)},
    **{sssc_id: "Information" for sssc_id in range(4, 7)},
    **{sssc_id: "Process" for sssc_id in range(7, 10)},
    **{sssc_id: "Preference" for sssc_id in range(10, 13)},
    **{sssc_id: "Output" for sssc_id in range(13, 16)},
}


def _load_results(cfg: DictConfig) -> tuple[pd.DataFrame, list[str]]:
    results_root = Path(str(cfg.results_root))
    expected_attrs = OmegaConf.to_container(cfg.sssc_attrs, resolve=True)
    frames: list[pd.DataFrame] = []
    dataset_labels: list[str] = []

    for run_cfg in cfg.runs:
        run = OmegaConf.to_container(run_cfg, resolve=True)
        run_id = str(run["run_id"])
        dataset_label = str(run["dataset_label"])
        run_dir = results_root / "sc_extractor_runs" / run_id
        frame = pd.read_pickle(run_dir / "judgment_results.pkl")
        frame = frame.loc[frame["sssc_attrs"].apply(lambda attrs: attrs == expected_attrs)].copy()
        frame["run_id"] = run_id
        frame["dataset_label"] = dataset_label
        frame["sc_type"] = frame["sssc_id"].map(SC_TYPE_BY_ID)
        frames.append(frame)
        dataset_labels.append(dataset_label)

    return pd.concat(frames, ignore_index=True), dataset_labels


def _aggregate(results: pd.DataFrame) -> pd.DataFrame:
    judged = results.loc[results["retention"].notna()].copy()
    judged["retained"] = judged["retention"].astype(bool).astype(int)
    return (
        judged.groupby(["sc_type", "dataset_label"], as_index=False, sort=False)
        .agg(
            retained_count=("retained", "sum"),
            judged_count=("retained", "count"),
            retention_rate=("retained", "mean"),
        )
    )


def _wide_table(stats: pd.DataFrame, dataset_labels: list[str]) -> pd.DataFrame:
    table = (
        stats.pivot(index="sc_type", columns="dataset_label", values="retention_rate")
        .reindex(index=SC_TYPE_ORDER, columns=dataset_labels)
    )
    table["Average"] = table[dataset_labels].mean(axis=1)
    return table


def _to_latex(table: pd.DataFrame) -> str:
    lines = [
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        "SC type & " + " & ".join(table.columns) + " \\\\",
        r"\midrule",
    ]
    for sc_type, row in table.iterrows():
        values = " & ".join(f"{100 * float(value):.1f}\\%" for value in row)
        lines.append(f"{sc_type} & {values} \\\\")
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return "\n".join(lines) + "\n"


@hydra.main(
    version_base=None,
    config_path="../../../config/experiments/rq4",
    config_name="sc_extractor_by_type",
)
def main(cfg: DictConfig) -> None:
    output_dir = Path(str(cfg.output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)

    with tee_stdout(output_dir / "sc_extractor_by_type.log"):
        results, dataset_labels = _load_results(cfg)
        stats = _aggregate(results)
        table = _wide_table(stats, dataset_labels)

        stats.to_csv(output_dir / "retention_by_sc_type_long.csv", index=False)
        table.to_csv(output_dir / "retention_by_sc_type.csv", index_label="SC type")
        (output_dir / "retention_by_sc_type.tex").write_text(
            _to_latex(table), encoding="utf-8"
        )

        print(table.map(lambda value: f"{100 * float(value):.1f}%").to_string())


if __name__ == "__main__":
    main()
