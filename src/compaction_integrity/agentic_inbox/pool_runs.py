"""Pool COMPINT-Inbox report runs that share one configuration into a single run directory.

Sessions are renumbered per (condition, compactor) in the order the sources are given, and
each copied trace records where it came from (`source_run`, `source_seed`), also listed in
`manifest.csv`. The pooled directory reads like any run directory, so `compare_runs.py`
scores it directly.

Usage::

    python -m compaction_integrity.agentic_inbox.pool_runs --sources <run1> <run2>
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import pandas as pd

RUNS_ROOT = Path("/data/compaction_integrity/agentic_inbox_report_runs")
DEFAULT_COMPACTORS = ["gpt_oss_120b_pi_mono_prompt", "none"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sources", nargs="+", required=True)
    parser.add_argument("--compactors", nargs="+", default=DEFAULT_COMPACTORS)
    parser.add_argument("--out_dir", type=Path, default=RUNS_ROOT / "pooled")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=False)
    next_seed: dict[tuple[str, str], int] = defaultdict(int)
    manifest = []
    for source in args.sources:
        for path in sorted((RUNS_ROOT / source).glob("*.jsonl"), key=lambda p: json.loads(p.read_text())["seed"]):
            trace = json.loads(path.read_text())
            if trace["compactor"] not in args.compactors:
                continue
            key = (trace["condition"], trace["compactor"])
            seed = next_seed[key]
            next_seed[key] += 1
            trace.update(source_run=source, source_seed=trace["seed"], seed=seed)
            out_path = args.out_dir / f"{trace['condition']}__{trace['compactor']}__seed{seed}.jsonl"
            out_path.write_text(json.dumps(trace))
            manifest.append({"trace": out_path.name, "source_run": source, "source_file": path.name})

    pd.DataFrame(manifest).to_csv(args.out_dir / "manifest.csv", index=False)
    for (condition, compactor), n in sorted(next_seed.items()):
        print(f"{condition:18s} {compactor:28s} {n} sessions")
    print(f"pooled into {args.out_dir}")


if __name__ == "__main__":
    main()
