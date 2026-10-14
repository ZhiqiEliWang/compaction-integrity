#!/usr/bin/env python3
"""Synthetic tests for deterministic plumbing, not semantic model accuracy."""
import argparse
import copy
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("pipeline", ROOT / "scripts/pipeline.py")
p = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p)


class PipelineTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="annotation_package_test_")
        self.root = Path(self.temp.name)
        self.source = self.root / "source.csv"
        candidate = "Keep zero values 🚀 unchanged.\r\nLeave unrelated comments in place."
        description = "I cannot parse this file."
        issue = "Add interpolation support.\r\n" + candidate + "\r\n" + description
        self.rows = [
            {"annotation_id": "one::p0", "instance_id": "one", "candidate_paragraph": candidate, "issue_text": issue, "decision": "old_manual_value", "metadata": "001"},
            {"annotation_id": "one::p1", "instance_id": "one", "candidate_paragraph": description, "issue_text": issue, "decision": "", "metadata": "002"},
            {"annotation_id": "two::p0", "instance_id": "two", "candidate_paragraph": "Please ping me when you start.", "issue_text": "Fix the cache.\nPlease ping me when you start.", "decision": "", "metadata": "003"},
        ]
        self.header = list(self.rows[0])
        p.write_csv(self.source, self.header, self.rows)
        self.run = self.root / "run"
        self.source_hash = p.sha(self.source)
        p.prepare(argparse.Namespace(input=self.source, run=self.run, calibration_ids=None, seed=20260905, mode="full"))
        self.gold = {}
        for row in self.rows:
            texts = row["candidate_paragraph"].split("\r\n") if row["annotation_id"] == "one::p0" else [row["candidate_paragraph"]]
            kept = row["annotation_id"] != "one::p1"
            clauses = []
            if kept:
                for text in texts:
                    start = row["candidate_paragraph"].index(text)
                    clauses.append({"text": text, "start": start, "end": start+len(text),
                        "gates": {"is_directive": "yes", "not_main_task": "yes", "session_scoped": "no", "no_external_use": "no", "generic": "yes"},
                        "strict_sc": "drop", "sc_types": [], "observable": "yes", "observability_note": "File or notification output is observable.",
                        "broad_task_constraint": "keep", "broad_kind": "scope", "broad_reason": "An ancillary guardrail.",
                        "evidence": [{"source": "candidate_paragraph", "quote": text}]})
            self.gold[row["annotation_id"]] = {"annotation_id": row["annotation_id"], "rubric_version": p.VERSION,
                "primary_task": "Complete the requested feature.", "candidate_role": "author_directive" if kept else "description",
                "broad_task_constraint": "keep" if kept else "drop", "strict_sc": "drop", "broad_exclusion_reason": "none" if kept else "not_a_constraint",
                "strict_exclusion_reason": "not_session_scoped" if kept else "not_a_constraint",
                "primary_clause_index": 0 if kept else -1, "strict_primary_clause_index": 0 if kept else -1,
                "clauses": clauses, "rationale": "Synthetic test fixture.", "needs_review": False, "review_reason": ""}

    def tearDown(self):
        self.temp.cleanup()

    def fill_annotations(self, stage, output_dir, records):
        for src in sorted((self.run / "partitions" / stage).glob("part_*.jsonl")):
            expected = p.input_map(src)
            out = self.run / output_dir / src.name
            p.write_jsonl(out, [records[i] for i in expected])
            p.validate(src, out, "annotation")

    def fill_types(self, stage, out_dir, type_name):
        for src in sorted((self.run / "partitions" / stage).glob("part_*.jsonl")):
            rows = p.input_map(src)
            p.write_jsonl(self.run / out_dir / src.name, [{"annotation_id": i, "sc_type": type_name, "rationale": "Synthetic primary type.", "needs_review": False} for i in rows])

    def complete_annotation_and_types(self):
        self.fill_annotations("production", "pass_A", self.gold)
        self.fill_annotations("production", "pass_B", self.gold)
        summary = p.review(argparse.Namespace(run=self.run, stage="production"))
        self.assertEqual(summary["broad"]["agreed"], 3)
        self.fill_annotations("adjudication", "adjudication", self.gold)
        p.merge(argparse.Namespace(run=self.run))
        p.prepare_types(argparse.Namespace(run=self.run))
        self.fill_types("types", "type_A", "Process")
        self.fill_types("types", "type_B", "Output")
        self.assertEqual(p.review_types(argparse.Namespace(run=self.run))["review_count"], 2)
        self.fill_types("type_adjudication", "type_adjudication", "Process")
        p.merge_types(argparse.Namespace(run=self.run))

    def test_complete_pipeline_with_revision_and_hold(self):
        self.fill_annotations("calibration", "calibration_A", self.gold)
        self.fill_annotations("calibration", "calibration_B", self.gold)
        p.review(argparse.Namespace(run=self.run, stage="calibration"))
        p.write_text(self.run / "calibration_report.md", "# Synthetic calibration test\n\nFixture outputs test plumbing only.\n")
        p.write_jsonl(self.run / "dispatch_log.jsonl", [{"worker": "synthetic_fixture", "model": "none", "note": "No LLM was used in this smoke test."}])
        controls = {}
        key = p.indexed(p.read_jsonl(self.run / "frozen/package/references/calibration/controls_key.jsonl"))
        for src in (self.run / "partitions/controls").glob("part_*.jsonl"):
            for i, s in p.input_map(src).items():
                r = copy.deepcopy(self.gold["one::p0"])
                r["annotation_id"] = i
                broad, strict = key[i]["expected_broad_task_constraint"], key[i]["expected_strict_sc"]
                r["broad_task_constraint"], r["strict_sc"] = broad, strict
                r["broad_exclusion_reason"] = "none" if broad == "keep" else "not_a_constraint"
                r["strict_exclusion_reason"] = "none" if strict == "keep" else "not_session_scoped" if broad == "keep" else "not_a_constraint"
                if broad == "drop":
                    r["clauses"] = []
                    r["primary_clause_index"] = r["strict_primary_clause_index"] = -1
                else:
                    c = copy.deepcopy(r["clauses"][0])
                    c.update(text=s["candidate_paragraph"], start=0, end=len(s["candidate_paragraph"]), strict_sc=strict)
                    c["evidence"] = [{"source": "candidate_paragraph", "quote": s["candidate_paragraph"]}]
                    if strict == "keep":
                        c["gates"] = {k: "yes" for k in c["gates"]}
                        c["sc_types"] = [key[i]["expected_primary_type"]]
                    r["clauses"] = [c]
                controls[i] = r
        self.fill_annotations("controls", "controls", controls)
        score = p.score_controls(argparse.Namespace(run=self.run))
        self.assertEqual(score["broad_task_constraint"], {"correct": 20, "total": 20})
        self.complete_annotation_and_types()
        p.prepare_probes(argparse.Namespace(run=self.run))
        for src in sorted((self.run / "partitions/probes").glob("part_*.jsonl")):
            probes = []
            for i, s in p.input_map(src).items():
                hold = i == "two::p0"
                probes.append({"annotation_id": i, "target_clause_index": 0, "sc_text": s["final"]["clauses"][0]["text"], "status": "hold" if hold else "ready",
                    "user": "Continue the feature work." if not hold else "", "compliant": "Preserve zeros." if not hold else "", "violating": "Change zeros." if not hold else "",
                    "probe_note": "Synthetic contrast; the other clause is untested.", "context_assumptions": "Source-only fixture."})
            p.write_jsonl(self.run / "probe_drafts" / src.name, probes)
        v1 = self.run / "probes_v1.jsonl"
        p.collect_probes(argparse.Namespace(run=self.run, output=v1))
        ri1, review1 = self.run / "review_input_v1.jsonl", self.run / "review_v1.jsonl"
        p.prepare_probe_review(argparse.Namespace(run=self.run, probes=v1, output=ri1))
        p.write_jsonl(review1, [{"annotation_id": i, "disposition": "targeted_revision" if i == "one::p0" else "hold", "issues": ["Test review finding."], "reason": "Test disposition."} for i in p.input_map(ri1)])
        revision_input = self.run / "revision_input.jsonl"
        p.prepare_revision(argparse.Namespace(input=ri1, review=review1, output=revision_input))
        revision = copy.deepcopy(next(iter(p.input_map(v1).values())))
        revision["user"] = "Recommend the next implementation step."
        revfile = self.run / "revisions.jsonl"
        p.write_jsonl(revfile, [revision])
        v2 = self.run / "probes_v2.jsonl"
        result = p.apply_revision(argparse.Namespace(run=self.run, probes=v1, input=revision_input, revisions=revfile, output=v2))
        self.assertEqual(result, {"updated": 1, "unchanged": 1})
        self.assertEqual(p.input_map(v1)["two::p0"], p.input_map(v2)["two::p0"])
        ri2, review2 = self.run / "review_input_v2.jsonl", self.run / "review_v2.jsonl"
        p.prepare_probe_review(argparse.Namespace(run=self.run, probes=v2, output=ri2))
        p.write_jsonl(review2, [{"annotation_id": i, "disposition": "ready" if i == "one::p0" else "hold", "issues": [], "reason": "Synthetic quality disposition."} for i in p.input_map(ri2)])
        with self.assertRaisesRegex(ValueError, "stale"):
            p.finalize(argparse.Namespace(run=self.run, probes=v2, review_input=ri1, review=review1))
        result = p.finalize(argparse.Namespace(run=self.run, probes=v2, review_input=ri2, review=review2))
        self.assertEqual((result["source_rows"], result["broad_keep_rows"], result["broad_kept_clauses"], result["probe_ready_rows"], result["held_probe_rows"]), (3, 2, 3, 1, 1))
        self.assertEqual(result["new_column_mapping"]["decision"], "ann_decision")
        _, exported = p.read_csv(self.run / "all_candidates.annotated.csv")
        self.assertEqual(exported[0]["decision"], "old_manual_value")
        self.assertEqual(exported[0]["ann_decision"], "keep")
        self.assertIn("\r\n", exported[0]["candidate_paragraph"])
        self.assertEqual(p.sha(self.source), self.source_hash)

    def test_invalid_annotations_and_exact_spans(self):
        src = next((self.run / "partitions/production").glob("part_*.jsonl"))
        data = [copy.deepcopy(self.gold[i]) for i in p.input_map(src)]
        for mode in ("extra", "missing", "span", "gate", "schema"):
            bad = copy.deepcopy(data)
            if mode == "extra":
                bad.append(copy.deepcopy(bad[0]))
            elif mode == "missing":
                bad.pop()
            elif mode == "schema":
                bad[0]["extra_key"] = "invalid"
            else:
                record = next(r for r in bad if r["clauses"])
                if mode == "span":
                    record["clauses"][0]["text"] += " changed"
                else:
                    record["clauses"][0]["gates"]["not_main_task"] = "no"
            path = self.root / f"invalid_{mode}.jsonl"
            p.write_jsonl(path, bad)
            with self.assertRaises(ValueError):
                p.validate(src, path, "annotation")

    def test_deduplication_and_context_conflict(self):
        previous = self.root / "previous.csv"
        p.write_csv(previous, self.header, self.rows[:1])
        out = self.root / "new_only.csv"
        result = p.deduplicate(argparse.Namespace(input=self.source, previous=previous, output=out, allow_not_superset=False))
        self.assertEqual(result["new_only_rows"], 2)
        self.assertEqual(p.read_csv(out)[1], self.rows[1:])
        modified = copy.deepcopy(self.rows)
        modified[0]["issue_text"] += "\nChanged context."
        modified[1]["issue_text"] += "\nChanged context."
        changed = self.root / "changed.csv"
        p.write_csv(changed, self.header, modified)
        with self.assertRaisesRegex(ValueError, "changed source"):
            p.deduplicate(argparse.Namespace(input=changed, previous=previous, output=self.root / "bad.csv", allow_not_superset=False))

    def test_freeze_and_prepare_guards(self):
        with self.assertRaisesRegex(ValueError, "already exists"):
            p.prepare(argparse.Namespace(input=self.source, run=self.run, calibration_ids=None, seed=20260905, mode="full"))
        with self.assertRaisesRegex(ValueError, "outside"):
            p.prepare(argparse.Namespace(input=self.source, run=ROOT / "accidental_run", calibration_ids=None, seed=20260905, mode="full"))
        frozen = self.run / "frozen/package/SC_RUBRIC.md"
        frozen.write_text(frozen.read_text() + "\nModified fixture.")
        with self.assertRaisesRegex(ValueError, "frozen file changed"):
            p.run_info(self.run)


if __name__ == "__main__":
    unittest.main(verbosity=2)
