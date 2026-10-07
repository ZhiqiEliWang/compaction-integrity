#!/usr/bin/env python3
"""Read-only package completeness, links, schemas, and optional digest audit."""
import hashlib
import json
import re
from pathlib import Path

root = Path(__file__).resolve().parents[1]
required = ["README.md", "ORCHESTRATION_PROMPT.md", "SC_RUBRIC.md", "EVALUATION_PROMPT.md", "ADJUDICATION_PROMPT.md",
            "CALIBRATION_PROMPT.md", "TYPE_RUBRIC.md", "TYPE_EVALUATION_PROMPT.md", "TYPE_ADJUDICATION_PROMPT.md",
            "PROBE_GENERATION_PROMPT.md", "PROBE_QUALITY_PROMPT.md", "PROBE_REVISION_PROMPT.md", "VALIDATION_EXPORT_PROMPT.md",
            "DEDUPLICATION_PROMPT.md", "DISPATCH_TEMPLATES.md", "scripts/pipeline.py", "tests/test_pipeline.py"]
errors = []
for name in required:
    if not (root / name).is_file():
        errors.append("missing " + name)
for name in ("ANNOTATION", "TYPE", "PROBE", "PROBE_REVIEW"):
    json.loads((root / "schemas" / f"{name}_SCHEMA.json").read_text())
for path in root.glob("*.md"):
    body = path.read_text(encoding="utf-8")
    if len(re.findall(r"^```", body, re.M)) % 2:
        errors.append("unclosed code fence " + path.name)
    for target in re.findall(r"\[[^\]]+\]\(([^)]+)\)", body):
        if not target.startswith(("http:", "https:", "#", "/")):
            if not (path.parent / target.split("#")[0]).exists():
                errors.append(f"broken link {path.name}: {target}")
orchestration = (root / "ORCHESTRATION_PROMPT.md").read_text()
if "PRIMARY_LABEL=broad_task_constraint" not in orchestration or "PRIMARY_LABEL=strict_sc" in orchestration:
    errors.append("incorrect top-level selection default")
definitions = (root / "TYPE_EVALUATION_PROMPT.md").read_text()
for type_name in ("Action", "Information", "Process", "Preference", "Output"):
    if f"**{type_name}:**" not in definitions:
        errors.append("missing judge definition " + type_name)
files = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(root.rglob("*")) if p.is_file() and "__pycache__" not in p.parts and p.name != "PACKAGE_MANIFEST.json"}
manifest = root / "PACKAGE_MANIFEST.json"
if manifest.is_file() and json.loads(manifest.read_text())["sha256"] != files:
    errors.append("package files do not match PACKAGE_MANIFEST.json")
print(json.dumps({"valid": not errors, "files": len(files), "manifest_checked": manifest.is_file(), "errors": errors}, indent=2))
raise SystemExit(1 if errors else 0)
