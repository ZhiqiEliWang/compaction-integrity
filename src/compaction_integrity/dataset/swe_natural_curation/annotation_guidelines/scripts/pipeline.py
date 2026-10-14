#!/usr/bin/env python3
"""Deterministic annotation plumbing; all semantic judgments come from workers.

Python 3.9+, standard library only. Run --help for commands. Files are created
exclusively: retain old attempts and use a new output path for reruns.
"""
import argparse
import csv
import hashlib
import json
import random
import shutil
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

PACKAGE = Path(__file__).resolve().parents[1]
VERSION = "1.0-broad-default"
TYPES = {"Action", "Information", "Process", "Preference", "Output"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_jsonl(path):
    result = []
    with Path(path).open(encoding="utf-8-sig", newline="") as f:
        for n, line in enumerate(f, 1):
            if line.strip():
                try:
                    result.append(json.loads(line))
                except ValueError as e:
                    raise ValueError(f"{path}:{n}: {e}") from e
    return result


def write_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="") as f:
        f.write(text)


def write_json(path, obj):
    write_text(path, json.dumps(obj, ensure_ascii=False, indent=2) + "\n")


def write_jsonl(path, objs):
    write_text(path, "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in objs))


def indexed(records):
    result = {}
    for r in records:
        aid = r["annotation_id"]
        require(aid not in result, f"duplicate ID {aid}")
        result[aid] = r
    return result


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        header, rows = reader.fieldnames, list(reader)
    require(header and len(header) == len(set(header)), "missing or duplicate CSV headers")
    require({"annotation_id", "instance_id", "candidate_paragraph", "issue_text"} <= set(header), "required source columns missing")
    require(all(None not in r and all(v is not None for v in r.values()) for r in rows), "ragged CSV rows")
    require(all(r["annotation_id"] and r["instance_id"] and r["candidate_paragraph"] for r in rows), "blank ID/instance/candidate")
    indexed(rows)
    require(all(r["candidate_paragraph"] in r["issue_text"] for r in rows), "candidate missing from full issue")
    group_rows(rows)
    return header, rows


def write_csv(path, header, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header, lineterminator="\r\n")
        writer.writeheader()
        writer.writerows(rows)


def group_rows(rows):
    groups = {}
    for r in rows:
        iid = r["instance_id"]
        if iid not in groups:
            groups[iid] = {"instance_id": iid, "issue_text": r["issue_text"], "candidates": []}
        require(groups[iid]["issue_text"] == r["issue_text"], f"inconsistent issue text: {iid}")
        groups[iid]["candidates"].append({"annotation_id": r["annotation_id"], "candidate_paragraph": r["candidate_paragraph"]})
    return list(groups.values())


def flatten(groups):
    return indexed([{**c, "instance_id": g["instance_id"], "issue_text": g["issue_text"]} for g in groups for c in g["candidates"]])


def input_map(path):
    records = read_jsonl(path)
    return flatten(records) if records and "candidates" in records[0] else indexed(records)


def pack(groups, seed, target=15, maximum=20, chars=50000):
    groups = list(groups)
    random.Random(seed).shuffle(groups)
    parts, current, n, size = [], [], 0, 0
    for g in groups:
        count = len(g["candidates"])
        length = len(g["issue_text"]) + sum(len(c["candidate_paragraph"]) for c in g["candidates"])
        if current and (n >= target or n + count > maximum or size + length > chars):
            parts.append(current)
            current, n, size = [], 0, 0
        current.append(g)
        n += count
        size += length
        if count > maximum or length > chars:
            parts.append(current)
            current, n, size = [], 0, 0
    if current:
        parts.append(current)
    return parts


def make_parts(run, kind, groups, seed, target=15, maximum=20, chars=50000):
    parts = []
    for n, batch in enumerate(pack(groups, seed, target, maximum, chars), 1):
        p = Path(run) / "partitions" / kind / f"part_{n:03d}.jsonl"
        write_jsonl(p, batch)
        ids = list(flatten(batch))
        parts.append({"input": str(p), "annotation_ids": ids, "candidate_count": len(ids), "sha256": sha(p)})
    return parts


def schema_check(value, schema, root=None, where="$ "):
    """Checks every JSON Schema keyword used in our four bundled schemas."""
    root = schema if root is None else root
    if "$ref" in schema:
        node = root
        for key in schema["$ref"].removeprefix("#/").split("/"):
            node = node[key]
        return schema_check(value, node, root, where)
    if "const" in schema:
        require(value == schema["const"], f"{where}: wrong constant")
    if "enum" in schema:
        require(value in schema["enum"], f"{where}: invalid enum {value!r}")
    tests = {"object": lambda v: isinstance(v, dict), "array": lambda v: isinstance(v, list),
             "string": lambda v: isinstance(v, str), "integer": lambda v: type(v) is int,
             "boolean": lambda v: type(v) is bool}
    if "type" in schema:
        require(tests[schema["type"]](value), f"{where}: wrong type")
    if isinstance(value, dict):
        require(set(schema.get("required", [])) <= set(value), f"{where}: missing keys")
        if schema.get("additionalProperties") is False:
            require(set(value) <= set(schema.get("properties", {})), f"{where}: extra keys")
        for k, spec in schema.get("properties", {}).items():
            if k in value:
                schema_check(value[k], spec, root, where + "." + k)
    if isinstance(value, list):
        require(len(value) >= schema.get("minItems", 0), f"{where}: too few items")
        if schema.get("uniqueItems"):
            require(len(value) == len({json.dumps(x, sort_keys=True) for x in value}), f"{where}: duplicate items")
        for i, v in enumerate(value):
            schema_check(v, schema.get("items", {}), root, f"{where}[{i}]")
    if isinstance(value, str):
        require(len(value) >= schema.get("minLength", 0), f"{where}: empty string")
    if type(value) is int and "minimum" in schema:
        require(value >= schema["minimum"], f"{where}: below minimum")


def label(values):
    return "drop" if "no" in values else "keep" if all(v == "yes" for v in values) else "uncertain"


def aggregate(clauses, field):
    vals = [c[field] for c in clauses]
    return "keep" if "keep" in vals else "uncertain" if "uncertain" in vals else "drop"


def primary(clauses, field):
    for status in ("keep", "uncertain"):
        for i, c in enumerate(clauses):
            if c[field] == status:
                return i
    return 0 if clauses else -1


def validate_record(r, source):
    clauses = r["clauses"]
    previous = -1
    for c in clauses:
        s, e = c["start"], c["end"]
        require(0 <= s < e <= len(source["candidate_paragraph"]), "offset bounds")
        require(source["candidate_paragraph"][s:e] == c["text"], "not exact source slice")
        require(s >= previous, "clauses out of source order")
        previous = s
        require(c["broad_task_constraint"] == label([c["gates"][k] for k in ("is_directive", "not_main_task")]), "broad gate aggregation")
        require(c["strict_sc"] == label(list(c["gates"].values())), "strict gate aggregation")
        require(c["strict_sc"] != "keep" or bool(c["sc_types"]), "strict keep missing types")
        require(c["strict_sc"] != "drop" or not c["sc_types"], "strict drop has clause types")
        require(c["broad_task_constraint"] != "keep" or c["broad_kind"].strip(), "broad keep missing kind")
        for ev in c["evidence"]:
            require(ev["quote"] in source[ev["source"]], "evidence quote missing")
    for field, idx, reason in (("broad_task_constraint", "primary_clause_index", "broad_exclusion_reason"),
                              ("strict_sc", "strict_primary_clause_index", "strict_exclusion_reason")):
        require(r[field] == aggregate(clauses, field), f"row {field} aggregation")
        require(r[idx] == primary(clauses, field), f"invalid {idx}")
        if r[field] == "keep":
            require(r[reason] == "none", "keep exclusion reason")
        elif r[field] == "uncertain":
            require(r[reason] == "uncertain", "uncertain exclusion reason")
        else:
            require(r[reason] not in ("none", "uncertain"), "drop exclusion reason")
        if not clauses:
            require(r[reason] == "not_a_constraint", "empty clause exclusion reason")
    if r["needs_review"]:
        require(r["review_reason"].strip(), "review flag missing reason")
    if "uncertain" in (r["broad_task_constraint"], r["strict_sc"]) or any(c["observable"] != "yes" for c in clauses if c["broad_task_constraint"] == "keep"):
        require(r["needs_review"], "uncertainty/observability must be flagged")


def validate(input_path, output_path, kind):
    expected = input_map(input_path)
    out = indexed(read_jsonl(output_path))
    require(set(expected) == set(out), f"ID coverage: missing={sorted(set(expected)-set(out))}, extra={sorted(set(out)-set(expected))}")
    schema_name = {"annotation": "ANNOTATION", "types": "TYPE", "probes": "PROBE", "probe-review": "PROBE_REVIEW"}[kind]
    schema = read_json(PACKAGE / "schemas" / f"{schema_name}_SCHEMA.json")
    for aid, r in out.items():
        try:
            schema_check(r, schema)
            if kind == "annotation":
                validate_record(r, expected[aid])
            if kind == "probes":
                src = expected[aid]
                clauses = src["final"]["clauses"]
                i = r["target_clause_index"]
                require(i < len(clauses) and clauses[i]["broad_task_constraint"] == "keep", "target not broad kept")
                require(r["sc_text"] == clauses[i]["text"], "probe SC changed")
                require(src["candidate_paragraph"][clauses[i]["start"]:clauses[i]["end"]] == r["sc_text"], "probe source slice")
                fields = [r[k].strip() for k in ("user", "compliant", "violating")]
                require(all(fields) if r["status"] == "ready" else not any(fields), "ready/hold field policy")
                if r["status"] == "ready":
                    require(r["compliant"] != r["violating"], "identical probe options")
                    require(not any(m in r[k] for k in ("user", "compliant", "violating") for m in ("\\compl{}", "\\viol{}")), "macros belong only in display export")
            if kind == "probe-review":
                draft = expected[aid]["draft"]
                require(not (r["disposition"] == "ready" and draft["status"] == "hold"), "held draft marked ready")
        except ValueError as e:
            raise ValueError(f"{aid}: {e}") from e
    return out


def checked_dir(run, input_stage, output_dir, kind):
    result = {}
    inputs = sorted((Path(run) / "partitions" / input_stage).glob("part_*.jsonl"))
    expected_names = {p.name for p in inputs}
    actual_names = {p.name for p in (Path(run) / output_dir).glob("part_*.jsonl")}
    require(actual_names == expected_names, f"partition file coverage mismatch: {output_dir}")
    for p in inputs:
        require((Path(run) / output_dir / p.name).is_file(), f"missing worker output {output_dir}/{p.name}")
        for aid, r in validate(p, Path(run) / output_dir / p.name, kind).items():
            require(aid not in result, f"duplicate across partitions: {aid}")
            result[aid] = r
    return result


def run_info(run):
    run = Path(run)
    m = read_json(run / "run_manifest.json")
    for relative, digest in m["frozen_digests"].items():
        require(sha(run / relative) == digest, f"frozen file changed: {relative}")
    return m


def source_rows(run):
    run_info(run)
    return read_csv(Path(run) / "frozen/source.csv")


def prepare(args):
    source, run = args.input.resolve(), args.run.resolve()
    require(not run.exists(), "run directory already exists; resume it or choose a new run")
    require(not PACKAGE.is_relative_to(run) and not run.is_relative_to(PACKAGE), "run must be outside the package")
    source_digest = sha(source)
    header, rows = read_csv(source)
    groups = group_rows(rows)
    source_map = indexed(rows)
    historical = PACKAGE / "references/calibration/calibration_input.jsonl"
    mode, selected = "seeded_random", []
    if args.calibration_ids:
        ids = read_json(args.calibration_ids)
        require(isinstance(ids, list) and len(ids) == len(set(ids)) and set(ids) <= set(source_map), "invalid calibration ID list")
        selected, mode = [source_map[i] for i in ids], "explicit_ID_selection"
    elif historical.is_file():
        old = flatten(read_jsonl(historical))
        if all(aid in source_map and all(source_map[aid][k] == r[k] for k in ("instance_id", "candidate_paragraph", "issue_text")) for aid, r in old.items()):
            selected, mode = [source_map[i] for i in old], "historical_25_exact_match"
    if not selected:
        selected = random.Random(args.seed).sample(rows, min(25, len(rows)))
    run.mkdir(parents=True)
    (run / "frozen").mkdir()
    shutil.copy2(source, run / "frozen/source.csv")
    require(sha(run / "frozen/source.csv") == source_digest and sha(source) == source_digest, "source changed during preparation")
    shutil.copytree(PACKAGE, run / "frozen/package", ignore=shutil.ignore_patterns("__pycache__", "runs"))
    parts = {"production": make_parts(run, "production", groups, args.seed),
             "calibration": make_parts(run, "calibration", group_rows(selected), args.seed + 1)}
    controls = PACKAGE / "references/calibration/controls_input.jsonl"
    if controls.is_file():
        control_groups = read_jsonl(controls)
        require(not set(flatten(control_groups)) & set(source_map), "control IDs collide with production")
        parts["controls"] = make_parts(run, "controls", control_groups, args.seed + 3)
    ann_keys = [k for k in header if k.split(" (")[0] in {"decision", "not_main_task", "session_scoped", "observable", "sc_text", "sc_type", "exclusion_reason", "note", "notes", "probe", "correct_answer", "incorrect_answer"}]
    existing = [{"annotation_id": r["annotation_id"], "existing": {k: r[k] for k in ann_keys if r[k]}} for r in rows if any(r[k] for k in ann_keys)]
    write_jsonl(run / "manual_annotations_present.jsonl", existing)
    write_json(run / "partition_manifest.json", parts)
    write_json(run / "calibration_selection.json", {"mode": mode, "seed": args.seed, "annotation_ids": [r["annotation_id"] for r in selected]})
    write_json(run / "run_manifest.json", {
        "run_id": run.name, "created_at_UTC": datetime.now(timezone.utc).isoformat(), "mode": args.mode,
        "primary_label": "broad_task_constraint", "record_strict_label": True, "rubric_version": VERSION,
        "seed": args.seed, "independent_annotation_passes": 2, "independent_type_passes": 2,
        "max_concurrent_agents_including_coordinator": 4, "probe_unit": "row", "replace_original": False,
        "input_csv": str(source), "input_sha256": source_digest, "source_header": header,
        "source_count": len(rows), "issue_count": len(groups), "model_metadata": "unavailable; coordinator records actual exposed identifiers in dispatch_log.jsonl",
        "frozen_digests": {str(p.relative_to(run)): sha(p) for p in (run / "frozen").rglob("*") if p.is_file()}
    })
    return {"run": str(run), "rows": len(rows), "issues": len(groups), "partition_counts": {k: len(v) for k, v in parts.items()}, "calibration_selection": mode, "existing_annotation_rows": len(existing)}


def agreement(a, b, field):
    ids = list(a)
    either = sum("keep" in (a[i][field], b[i][field]) for i in ids)
    both = sum(a[i][field] == b[i][field] == "keep" for i in ids)
    return {"agreed": sum(a[i][field] == b[i][field] for i in ids), "total": len(ids), "both_keep": both,
            "either_keep": either, "positive_specific_agreement": both / either if either else None,
            "pass_A_counts": dict(Counter(a[i][field] for i in ids)), "pass_B_counts": dict(Counter(b[i][field] for i in ids))}


def review(args):
    run = args.run
    m = run_info(run)
    stage = args.stage
    a_dir, b_dir = ("calibration_A", "calibration_B") if stage == "calibration" else ("pass_A", "pass_B")
    a, b = checked_dir(run, stage, a_dir, "annotation"), checked_dir(run, stage, b_dir, "annotation")
    src = {}
    for p in sorted((run / "partitions" / stage).glob("part_*.jsonl")):
        src.update(input_map(p))
    require(set(src) == set(a) == set(b), "pass coverage mismatch")
    reasons = defaultdict(list)
    for aid in src:
        for f in ("broad_task_constraint", "strict_sc", "candidate_role", "primary_clause_index", "strict_primary_clause_index", "broad_exclusion_reason", "strict_exclusion_reason"):
            if a[aid][f] != b[aid][f]:
                reasons[aid].append(f + "_disagreement")
        for f in ("text", "start", "end", "gates", "broad_task_constraint", "strict_sc", "sc_types", "observable"):
            if [c[f] for c in a[aid]["clauses"]] != [c[f] for c in b[aid]["clauses"]]:
                reasons[aid].append("clause_" + f + "_disagreement")
        if any(r["needs_review"] or "uncertain" in (r["broad_task_constraint"], r["strict_sc"]) for r in (a[aid], b[aid])):
            reasons[aid].append("uncertainty_or_review_flag")
        for f in ("broad_task_constraint", "strict_sc"):
            if "keep" in (a[aid][f], b[aid][f]):
                reasons[aid].append("proposed_" + f + "_positive")
    negatives = [i for i in src if a[i]["broad_task_constraint"] == b[i]["broad_task_constraint"] == "drop"]
    sample = random.Random(m["seed"]).sample(negatives, min(20, len(negatives)))
    for aid in sample:
        reasons[aid].append("seeded_agreed_broad_negative_review")
    duplicates = defaultdict(list)
    for aid, r in src.items():
        duplicates[(r["instance_id"], r["candidate_paragraph"])].append(aid)
    for ids in duplicates.values():
        if len(ids) > 1:
            for data in (a, b):
                if len({(data[i]["broad_task_constraint"], data[i]["strict_sc"]) for i in ids}) > 1:
                    for i in ids:
                        reasons[i].append("same_issue_duplicate_inconsistency")
    if stage == "production":
        for r in read_jsonl(run / "manual_annotations_present.jsonl"):
            reasons[r["annotation_id"]].append("existing_annotation_requires_researcher_comparison")
    queue = [{**src[i], "review_reasons": list(dict.fromkeys(reasons[i])), "pass_A": a[i], "pass_B": b[i]} for i in src if reasons[i]]
    out_kind = "calibration_adjudication" if stage == "calibration" else "adjudication"
    groups = {}
    for q in queue:
        iid = q["instance_id"]
        groups.setdefault(iid, {"instance_id": iid, "issue_text": q["issue_text"], "candidates": []})["candidates"].append({k: v for k, v in q.items() if k not in ("instance_id", "issue_text")})
    parts = make_parts(run, out_kind, groups.values(), m["seed"] + 10, 12, 16)
    write_jsonl(run / ("calibration_review_queue.jsonl" if stage == "calibration" else "annotation_review_queue.jsonl"), queue)
    summary = {"review_count": len(queue), "review_partitions": parts, "negative_sample": sample,
               "broad": agreement(a, b, "broad_task_constraint"), "strict": agreement(a, b, "strict_sc")}
    write_json(run / f"{stage}_comparison.json", summary)
    return summary


def merge(args):
    run = args.run
    _, rows = source_rows(run)
    a, b = checked_dir(run, "production", "pass_A", "annotation"), checked_dir(run, "production", "pass_B", "annotation")
    adj = checked_dir(run, "adjudication", "adjudication", "annotation")
    queue = indexed(read_jsonl(run / "annotation_review_queue.jsonl"))
    require(set(adj) == set(queue), "adjudication queue coverage")
    require(set(a) == set(b) == set(indexed(rows)), "complete production coverage")
    detail = [{"annotation_id": r["annotation_id"], "source_row_index": r.get("source_row_index", str(i)),
               "final": adj.get(r["annotation_id"], a[r["annotation_id"]]),
               "provenance": "model_adjudication" if r["annotation_id"] in adj else "independent_A_B_agreement",
               "review_reasons": queue.get(r["annotation_id"], {}).get("review_reasons", []),
               "pass_A": a[r["annotation_id"]], "pass_B": b[r["annotation_id"]]} for i, r in enumerate(rows)]
    write_jsonl(run / "annotations_detailed.jsonl", detail)
    write_jsonl(run / "adjudication_log.jsonl", [{"annotation_id": i, "final": r, "review_reasons": queue[i]["review_reasons"], "provenance": "model_adjudication"} for i, r in adj.items()])
    return {"records": len(detail), "broad": dict(Counter(r["final"]["broad_task_constraint"] for r in detail)), "strict": dict(Counter(r["final"]["strict_sc"] for r in detail))}


def broad_inputs(run):
    _, rows = source_rows(run)
    details = indexed(read_jsonl(Path(run) / "annotations_detailed.jsonl"))
    require(set(details) == set(indexed(rows)), "detailed coverage")
    result = []
    for row in rows:
        d = details[row["annotation_id"]]
        if d["final"]["broad_task_constraint"] == "keep":
            kept = [c for c in d["final"]["clauses"] if c["broad_task_constraint"] == "keep"]
            result.append({"annotation_id": row["annotation_id"], "instance_id": row["instance_id"], "candidate_paragraph": row["candidate_paragraph"], "issue_text": row["issue_text"],
                           "primary_task": d["final"]["primary_task"], "final": d["final"], "broad_clauses": kept})
    return result


def row_parts(run, kind, rows, size=8):
    parts = []
    for i in range(0, len(rows), size):
        path = Path(run) / "partitions" / kind / f"part_{i//size+1:03d}.jsonl"
        batch = rows[i:i+size]
        write_jsonl(path, batch)
        parts.append({"input": str(path), "annotation_ids": [r["annotation_id"] for r in batch], "sha256": sha(path)})
    write_json(Path(run) / f"{kind}_manifest.json", parts)
    return parts


def prepare_types(args):
    rows = broad_inputs(args.run)
    write_jsonl(args.run / "type_input.jsonl", rows)
    return {"records": len(rows), "parts": row_parts(args.run, "types", rows)}


def review_types(args):
    run_info(args.run)
    a, b = checked_dir(args.run, "types", "type_A", "types"), checked_dir(args.run, "types", "type_B", "types")
    src = input_map(args.run / "type_input.jsonl")
    require(set(src) == set(a) == set(b), "complete type coverage")
    queue = [{**src[i], "type_pass_A": a[i], "type_pass_B": b[i]} for i in src if a[i]["sc_type"] != b[i]["sc_type"] or a[i]["needs_review"] or b[i]["needs_review"]]
    write_jsonl(args.run / "type_review_queue.jsonl", queue)
    return {"records": len(src), "review_count": len(queue), "parts": row_parts(args.run, "type_adjudication", queue)}


def merge_types(args):
    src = input_map(args.run / "type_input.jsonl")
    a, b = checked_dir(args.run, "types", "type_A", "types"), checked_dir(args.run, "types", "type_B", "types")
    adj = checked_dir(args.run, "type_adjudication", "type_adjudication", "types")
    queue = input_map(args.run / "type_review_queue.jsonl")
    require(set(a) == set(b) == set(src) and set(adj) == set(queue), "type merge coverage")
    rows = []
    for i, s in src.items():
        final = adj.get(i, a[i])
        rows.append({**s, "sc_type": final["sc_type"], "type_rationale": final["rationale"], "type_needs_review": final["needs_review"],
                     "type_provenance": "fresh_model_adjudication" if i in adj else "independent_A_B_agreement", "type_pass_A": a[i], "type_pass_B": b[i]})
    write_jsonl(args.run / "typed_records.jsonl", rows)
    return {"records": len(rows), "counts": dict(Counter(r["sc_type"] for r in rows)), "review_ids": [r["annotation_id"] for r in rows if r["type_needs_review"]]}


def prepare_probes(args):
    run_info(args.run)
    rows = read_jsonl(args.run / "typed_records.jsonl")
    require(set(indexed(rows)) == set(indexed(broad_inputs(args.run))), "probe input broad coverage")
    return {"records": len(rows), "parts": row_parts(args.run, "probes", rows)}


def collect_probes(args):
    records = checked_dir(args.run, "probes", "probe_drafts", "probes")
    require(set(records) == set(input_map(args.run / "typed_records.jsonl")), "draft coverage")
    write_jsonl(args.output, records.values())
    return {"records": len(records), "statuses": dict(Counter(r["status"] for r in records.values()))}


def prepare_probe_review(args):
    rows = read_jsonl(args.run / "typed_records.jsonl")
    drafts = validate(args.run / "typed_records.jsonl", args.probes, "probes")
    joined = [{**r, "draft": drafts[r["annotation_id"]]} for r in rows]
    write_jsonl(args.output, joined)
    return {"records": len(joined), "input_for_reviewer": str(args.output)}


def prepare_revision(args):
    src = input_map(args.input)
    reviews = validate(args.input, args.review, "probe-review")
    selected = [{**s, "review": reviews[i]} for i, s in src.items() if reviews[i]["disposition"] in ("targeted_revision", "substantial_revision")]
    write_jsonl(args.output, selected)
    return {"revision_count": len(selected), "annotation_ids": [r["annotation_id"] for r in selected]}


def apply_revision(args):
    src = input_map(args.run / "typed_records.jsonl")
    drafts = validate(args.run / "typed_records.jsonl", args.probes, "probes")
    revisions = validate(args.input, args.revisions, "probes")
    require(set(revisions) <= set(src), "revision unknown IDs")
    for r in read_jsonl(args.input):
        require(r["review"]["disposition"] in ("targeted_revision", "substantial_revision"), "unapproved revision set")
        require(r["draft"] == drafts[r["annotation_id"]], "revision input is stale")
    write_jsonl(args.output, [revisions.get(i, drafts[i]) for i in src])
    return {"updated": len(revisions), "unchanged": len(src)-len(revisions)}


def score_controls(args):
    run_info(args.run)
    records = checked_dir(args.run, "controls", "controls", "annotation")
    key = indexed(read_jsonl(args.run / "frozen/package/references/calibration/controls_key.jsonl"))
    require(set(key) == set(records), "control coverage mismatch")
    result = {f: {"correct": sum(records[i][f] == key[i]["expected_"+f] for i in key), "total": len(key)} for f in ("broad_task_constraint", "strict_sc")}
    write_json(args.run / "control_scores.json", result)
    return result


def latex_probe(probe):
    if probe["status"] == "hold":
        return ""
    return "\\textbf{User:} ``" + probe["user"] + "'' \\newline\n\\compl{} " + probe["compliant"] + " \\newline\n\\viol{} " + probe["violating"]


def finalize(args):
    run = args.run
    header, source = source_rows(run)
    m = run_info(run)
    require((run / "dispatch_log.jsonl").is_file(), "missing coordinator dispatch_log.jsonl")
    if m["mode"] == "full":
        require((run / "calibration_report.md").is_file(), "missing coordinator calibration_report.md")
        require((run / "calibration_comparison.json").is_file(), "missing calibration comparison")
        if list((run / "partitions/controls").glob("part_*.jsonl")):
            require((run / "control_scores.json").is_file(), "missing control scoring")
    details = indexed(read_jsonl(run / "annotations_detailed.jsonl"))
    typed = input_map(run / "typed_records.jsonl")
    require(set(typed) == set(indexed(broad_inputs(run))), "final type coverage")
    probes = validate(run / "typed_records.jsonl", args.probes, "probes")
    review_input = input_map(args.review_input)
    reviews = validate(args.review_input, args.review, "probe-review")
    require(set(reviews) == set(probes), "final review coverage")
    for i in probes:
        require(review_input[i]["draft"] == probes[i], "final review is stale")
        require(reviews[i]["disposition"] in ("ready", "hold"), "revisions remain before finalization")
    derived = {}
    pending = []
    for i, d in details.items():
        r = d["final"]
        schema_check(r, read_json(PACKAGE / "schemas/ANNOTATION_SCHEMA.json"))
        validate_record(r, indexed(source)[i])
        idx = r["primary_clause_index"]
        c = r["clauses"][idx] if idx >= 0 else None
        t, p = typed.get(i, {}), probes.get(i, {})
        kept = [c for c in r["clauses"] if c["broad_task_constraint"] == "keep"]
        covering = indexed(source)[i]["candidate_paragraph"][min(c["start"] for c in kept):max(c["end"] for c in kept)] if kept else ""
        derived[i] = {"decision": r["broad_task_constraint"], "broad_task_constraint": r["broad_task_constraint"], "strict_sc": r["strict_sc"],
                      "sc_text": c["text"] if c and r["broad_task_constraint"] == "keep" else "", "broad_constraint_text": covering,
                      "broad_clauses_json": json.dumps(kept, ensure_ascii=False), "sc_type": t.get("sc_type", ""),
                      "type_rationale": t.get("type_rationale", ""), "type_needs_review": str(t.get("type_needs_review", "")).lower(),
                      "type_provenance": t.get("type_provenance", ""), "annotation_provenance": d["provenance"],
                      "rubric_version": VERSION, "primary_clause_index": str(idx), "strict_primary_clause_index": str(r["strict_primary_clause_index"]),
                      "not_main_task": c["gates"]["not_main_task"] if c else "", "session_scoped": c["gates"]["session_scoped"] if c else "",
                      "observable": c["observable"] if c else "", "broad_kind": "; ".join(dict.fromkeys(c["broad_kind"] for c in kept)),
                      "exclusion_reason": r["broad_exclusion_reason"], "strict_exclusion_reason": r["strict_exclusion_reason"], "rationale": r["rationale"],
                      "probe": latex_probe(p) if p else "", "probe_status": reviews[i]["disposition"] if i in reviews else "",
                      "probe_target_clause_index": str(p["target_clause_index"]) if p else "", "probe_sc_text": p.get("sc_text", ""),
                      "probe_note": p.get("probe_note", ""), "probe_context_assumptions": p.get("context_assumptions", ""),
                      "correct_answer": p.get("compliant", ""), "incorrect_answer": p.get("violating", "")}
        flags = []
        if r["needs_review"]:
            flags.append(r["review_reason"])
        if t.get("type_needs_review"):
            flags.append("unresolved type ambiguity")
        if i in reviews and reviews[i]["disposition"] == "hold":
            flags.append("held probe: " + reviews[i]["reason"])
        if flags:
            pending.append({"annotation_id": i, "review_reasons": flags})
    mapping = {}
    used = set(header)
    keys = list(next(iter(derived.values())).keys()) if derived else []
    for k in keys:
        name = k
        while name in used:
            name = "ann_" + name
        mapping[k] = name
        used.add(name)
    out_header = header + list(mapping.values())
    rows = [{**r, **{mapping[k]: v for k, v in derived[r["annotation_id"]].items()}} for r in source]
    broad = [r for r in rows if derived[r["annotation_id"]]["broad_task_constraint"] == "keep"]
    ready = [r for r in broad if derived[r["annotation_id"]]["probe_status"] == "ready"]
    paths = [("all_candidates.annotated.csv", rows), ("sc_annotations.broad_keeps.typed.csv", broad), ("sc_annotations.broad_keeps.probe_ready.csv", ready)]
    for name, data in paths:
        write_csv(run / name, out_header, data)
        saved_header, saved = read_csv(run / name)
        require(saved_header == out_header and saved == data, f"CSV roundtrip failed {name}")
        for r in saved:
            orig = indexed(source)[r["annotation_id"]]
            require(all(r[k] == orig[k] for k in header), "source cells changed")
            d = derived[r["annotation_id"]]
            if d["broad_task_constraint"] == "keep":
                for k in ("sc_text", "broad_constraint_text", "probe_sc_text"):
                    require(d[k] and d[k] in orig["candidate_paragraph"] and d[k] in orig["issue_text"], f"non-substring {k}")
                require(d["sc_type"] in TYPES, "missing broad type")
    write_jsonl(run / "probes_final.jsonl", probes.values())
    write_jsonl(run / "review_queue.jsonl", pending)
    comparison = read_json(run / "production_comparison.json")
    live = Path(m["input_csv"])
    live_changed = not live.is_file() or sha(live) != m["input_sha256"]
    result = {"valid": True, "source_rows": len(source), "broad_keep_rows": len(broad), "probe_ready_rows": len(ready), "held_probe_rows": len(broad)-len(ready),
              "broad_kept_clauses": sum(len(x["broad_clauses"]) for x in typed.values()), "pending_review_rows": len(pending),
              "source_cells_preserved": True, "live_source_changed": live_changed, "new_column_mapping": mapping,
              "output_sha256": {name: sha(run / name) for name, _ in paths}}
    write_json(run / "final_validation.json", result)
    qtext = "# Probe quality review\n\nReady means suitable for a pilot; no downstream experiment was run.\n\n"
    for i, r in reviews.items():
        qtext += f"## {i}\n\nDisposition: `{r['disposition']}`. {r['reason']}\n\n"
        if r["issues"]:
            qtext += "\n".join("- " + x for x in r["issues"]) + "\n\n"
    write_text(run / "probe_quality_review.md", qtext)
    report = ("# Annotation run report\n\nDefault selection: broader task constraints. All judgments are provisional model annotations unless researcher review is separately recorded.\n\n"
              + f"Source rows: {len(source)}. Broader keeps: {len(broad)}. Ready probes: {len(ready)}. Held probes: {len(broad)-len(ready)}. Pending review rows: {len(pending)}.\n\n"
              + "Broad/strict independent agreement and type counts:\n\n```json\n" + json.dumps({"broad": comparison["broad"], "strict": comparison["strict"], "types": dict(Counter(r["sc_type"] for r in typed.values())),
                  "final_broad": dict(Counter(d["final"]["broad_task_constraint"] for d in details.values())), "final_strict": dict(Counter(d["final"]["strict_sc"] for d in details.values()))}, indent=2) + "\n```\n\n"
              + f"Original source changed since freezing: {live_changed}. Original fields remain exact strings in all new exports; collisions use the mapping in final_validation.json.\n\n"
              + "See calibration_report.md, control_scores.json when present, dispatch_log.jsonl, retained pass/adjudication files, review_queue.jsonl, and final_validation.json. Review flags and holds remain explicit. Probes are row-level and each targets one kept clause; untested clauses remain in detailed evidence. No retention, compliance, or no-SC/upper-bound experiment is claimed.\n\n"
              + "Control scores (constructed anchors, outside production):\n\n```json\n" + json.dumps(read_json(run / "control_scores.json") if (run / "control_scores.json").is_file() else {"status": "not run"}, indent=2) + "\n```\n\n"
              + "Worker/model identifiers and any independence exceptions must be recorded by the coordinator in dispatch_log.jsonl. This helper does not invent execution provenance.\n")
    write_text(run / "run_report.md", report)
    return result


def deduplicate(args):
    header, rows = read_csv(args.input)
    _, previous = read_csv(args.previous)
    new, old = indexed(rows), indexed(previous)
    if not args.allow_not_superset:
        require(set(old) <= set(new), "new file is not a superset of previous IDs")
    for aid in set(old) & set(new):
        require(all(old[aid][k] == new[aid][k] for k in ("instance_id", "candidate_paragraph", "issue_text")), f"overlapping ID changed source: {aid}")
    kept = [r for r in rows if r["annotation_id"] not in old]
    write_csv(args.output, header, kept)
    require(read_csv(args.output) == (header, kept), "dedup roundtrip")
    report = {"new_source": str(args.input), "previous_source": str(args.previous), "new_sha256": sha(args.input), "previous_sha256": sha(args.previous),
              "new_rows": len(rows), "previous_rows": len(previous), "overlap": len(set(new) & set(old)), "new_only_rows": len(kept), "key": "annotation_id with exact source consistency"}
    write_json(args.output.with_suffix(".deduplication.json"), report)
    return report


def package_manifest(args):
    files = {str(p.relative_to(PACKAGE)): sha(p) for p in sorted(PACKAGE.rglob("*")) if p.is_file() and "__pycache__" not in p.parts and p.name != "PACKAGE_MANIFEST.json"}
    result = {"package_version": VERSION, "primary_label": "broad_task_constraint", "file_count": len(files), "sha256": files}
    write_json(PACKAGE / "PACKAGE_MANIFEST.json", result)
    return {"file_count": len(files), "path": str(PACKAGE / "PACKAGE_MANIFEST.json")}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    def command(name, fields=()):
        q = sub.add_parser(name)
        for f in fields:
            q.add_argument("--"+f.replace("_", "-"), type=Path, required=True)
        return q
    q = command("prepare", ("input", "run"))
    q.add_argument("--seed", type=int, default=20260905)
    q.add_argument("--mode", choices=("full", "calibration"), default="full")
    q.add_argument("--calibration-ids", type=Path)
    command("review", ("run",)).add_argument("--stage", choices=("calibration", "production"), default="production")
    for name in ("merge", "prepare-types", "review-types", "merge-types", "prepare-probes", "score-controls"):
        command(name, ("run",))
    for kind in ("annotation", "types", "probes", "probe-review"):
        command("validate-"+kind, ("input", "output"))
    command("collect-probes", ("run", "output"))
    command("prepare-probe-review", ("run", "probes", "output"))
    command("prepare-revision", ("input", "review", "output"))
    command("apply-revision", ("run", "probes", "input", "revisions", "output"))
    command("finalize", ("run", "probes", "review_input", "review"))
    command("deduplicate", ("input", "previous", "output")).add_argument("--allow-not-superset", action="store_true")
    command("package-manifest")
    args = p.parse_args()
    try:
        if args.command.startswith("validate-"):
            records = validate(args.input, args.output, args.command.removeprefix("validate-"))
            result = {"valid": True, "records": len(records)}
        else:
            result = globals()[args.command.replace("-", "_")](args)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    except (ValueError, KeyError, OSError, TypeError, IndexError) as e:
        print(json.dumps({"valid": False, "error": str(e)}, ensure_ascii=False), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
