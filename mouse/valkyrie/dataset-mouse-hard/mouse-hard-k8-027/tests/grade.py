#!/usr/bin/env python3
"""Exact-match letter grader. argv: answer_file expected.json verifier_dir.

Parses the LAST `Answer: X` in the reply (also accepts a bare single letter as the whole reply).
Writes reward.txt (1 or 0) and details.json. A missing or unparseable answer scores 0.
"""
import json
import re
import sys
from pathlib import Path

answer_file, expected_file, vdir = sys.argv[1], sys.argv[2], Path(sys.argv[3])
vdir.mkdir(parents=True, exist_ok=True)
exp = json.loads(Path(expected_file).read_text())
letters = exp["letters"]

try:
    text = Path(answer_file).read_text(errors="replace")
except FileNotFoundError:
    text = ""

pred = None
# one line of the form "Answer: X" (markdown bold/brackets tolerated), nothing else on the line
hits = re.findall(r"^[\s*_#>`]*answer[\s*_`]*[:：][\s*_`(\[]*([A-Za-z])[\s*_`)\].]*$", text, re.IGNORECASE | re.MULTILINE)
hits = [h.upper() for h in hits if h.upper() in letters]
if hits:
    pred = hits[-1]
else:
    bare = text.strip().strip("*`.()[] ").upper()
    if len(bare) == 1 and bare in letters:
        pred = bare

correct = pred == exp["answer"]
(vdir / "reward.txt").write_text("1" if correct else "0")
(vdir / "details.json").write_text(json.dumps({"predicted": pred, "expected": exp["answer"], "correct": correct, "parsed": pred is not None}))
print(f"predicted={pred} expected={exp['answer']} correct={correct}")
