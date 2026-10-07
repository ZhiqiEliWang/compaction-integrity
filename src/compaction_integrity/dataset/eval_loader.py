"""Lean dataset loader for evaluation.py.

Reads only the `messages` column and exposes the user-turn structure needed for
SSSC injection.

Natural-SC datasets (COMPINT-SWE-Natural) are the exception: their SC is already
part of the history, so the row also carries the ablated no-SC arm and the SC
itself. `EvalDatasetRow.sssc` being set is what marks a row as natural — the
runner then probes `messages` / `messages_without_sssc` as given instead of
injecting one of the `SSSCS`.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from datasets import Dataset


Message = dict[str, str]
Position = Literal["top", "middle", "bottom"]

# Present only on natural-SC datasets; see dataset/swe_natural_curation/dataset.py build.
NATURAL_SC_COLUMN = "messages_without_sc"


@dataclass(frozen=True, slots=True)
class EvalDatasetRow:
    source_row_index: int
    messages: list[Message]
    system_prompt: str | None = None
    sssc: dict[str, Any] | None = None
    messages_without_sssc: list[Message] | None = None

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


def _messages(raw: list[dict[str, Any]]) -> list[Message]:
    return [{"role": m["role"], "content": m["content"]} for m in raw]


def _split_system(messages: list[Message]) -> tuple[str | None, list[Message]]:
    """Agent traces carry their own system prompt as the first message; the
    probe prompt puts it back at the top, so it is lifted out of the context."""
    if messages[0]["role"] == "system":
        return messages[0]["content"], messages[1:]
    return None, messages


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
        if NATURAL_SC_COLUMN in self.dataset.column_names:
            return self._natural_rows()
        return [
            EvalDatasetRow(
                source_row_index=i,
                messages=_messages(row["messages"]),
            )
            for i, row in enumerate(self.dataset)
        ]

    def _natural_rows(self) -> list[EvalDatasetRow]:
        rows: list[EvalDatasetRow] = []
        for i, row in enumerate(self.dataset):
            system_prompt, messages = _split_system(_messages(row["messages"]))
            _, without = _split_system(_messages(row[NATURAL_SC_COLUMN]))
            rows.append(
                EvalDatasetRow(
                    source_row_index=i,
                    messages=messages,
                    system_prompt=system_prompt,
                    sssc={
                        "id": int(row["sc_id"]),
                        "type": str(row["sc_type"]),
                        "sssc": str(row["sc_text"]),
                        "probe": str(row["probe"]),
                        "correct_answer": str(row["correct_answer"]),
                        "incorrect_answer": str(row["incorrect_answer"]),
                    },
                    messages_without_sssc=without,
                )
            )
        return rows
