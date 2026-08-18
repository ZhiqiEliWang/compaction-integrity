"""Lean dataset loader for evaluation.py.

Unlike GeneratedDatasetLoader, this loader 
It only reads the `messages` column and exposes the user-turn structure needed
for SSSC injection.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from datasets import Dataset


Message = dict[str, str]
Position = Literal["top", "middle", "bottom"]


@dataclass(frozen=True, slots=True)
class EvalDatasetRow:
    source_row_index: int
    messages: list[Message]

    def user_turn_indices(self) -> list[int]:
        return [i for i, m in enumerate(self.messages) if m["role"] == "user"]

    def position_index(self, position: Position) -> int:
        idxs = self.user_turn_indices()
        if not idxs:
            raise ValueError(
                f"Row {self.source_row_index} has no user turns to inject into."
            )
        if position == "top":
            return idxs[0]
        if position == "middle":
            return idxs[len(idxs) // 2]
        if position == "bottom":
            return idxs[-1]
        raise ValueError(f"Unsupported position={position!r}")


@dataclass(frozen=True, slots=True)
class EvalDatasetLoader:
    dataset: Dataset

    @classmethod
    def load(
        cls,
        dataset_path: str | Path,
        test_mode: bool,
        num_rows: int | None,
    ) -> "EvalDatasetLoader":
        ds = Dataset.load_from_disk(str(Path(dataset_path)))
        if test_mode:
            ds = ds.select(list(range(min(5, len(ds)))))
        elif num_rows is not None:
            if num_rows > len(ds):
                raise ValueError(
                    f"Requested {num_rows} rows, but dataset only has {len(ds)} rows."
                )
            ds = ds.select(list(range(num_rows)))
        return cls(dataset=ds)

    def rows(self) -> list[EvalDatasetRow]:
        return [
            EvalDatasetRow(
                source_row_index=i,
                messages=[
                    {"role": m["role"], "content": m["content"]}
                    for m in row["messages"]
                ],
            )
            for i, row in enumerate(self.dataset)
        ]
