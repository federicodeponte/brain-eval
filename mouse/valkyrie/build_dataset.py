#!/usr/bin/env python3
"""Turn tasks.jsonl (id, k, prompt, answer, ...) into a Valkyrie dataset folder.

Each task becomes a Harbor-format folder that Valkyrie runs with no benchmark service:
  <out>/<task_id>/task.toml
  <out>/<task_id>/instruction.md      prompt + answer-format line (what the agent sees)
  <out>/<task_id>/tests/test.sh       verifier entrypoint (uploaded only at grading time)
  <out>/<task_id>/tests/grade.py      exact-match letter grader
  <out>/<task_id>/tests/expected.json answer key (never visible to the agent)
  <out>/<task_id>/solution/solve.sh   oracle: writes the right answer (sanity check)

Usage: python3 build_dataset.py tasks.jsonl ./mouse-mcq [--image python:3.12-slim]
"""
import argparse
import json
import re
import shutil
import string
from pathlib import Path

HERE = Path(__file__).resolve().parent
ANSWER_FILE = "/app/answer.txt"
ID_OK = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")

SUFFIX = (
    "\n\n---\n"
    "Reply with your reasoning if you like, but the LAST line of your reply must be exactly\n"
    "`Answer: X` where X is one letter from {letters}.\n"
    f"(If you are working in a shell, write that final line to {ANSWER_FILE}.)\n"
)

TEST_SH = """#!/bin/bash
mkdir -p /logs/verifier
python3 /tests/grade.py "{answer_file}" /tests/expected.json /logs/verifier
"""

SOLVE_SH = """#!/bin/bash
mkdir -p /app
echo "Answer: {answer}" > {answer_file}
"""


def safe_id(raw: str) -> str:
    tid = re.sub(r"[^A-Za-z0-9._-]", "-", str(raw)).strip("-.") or "task"
    if not tid[0].isalnum():
        tid = "t" + tid
    tid = tid[:128]
    assert ID_OK.fullmatch(tid), tid
    return tid


def toml_str(s: str) -> str:
    return json.dumps(s)  # JSON string literals are valid TOML basic strings


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("tasks_jsonl")
    ap.add_argument("out_dir")
    ap.add_argument("--image", default="python:3.12-slim")
    ap.add_argument("--agent-timeout", type=float, default=1800)
    ap.add_argument("--name-prefix", default="vals-mouse")
    args = ap.parse_args()

    out = Path(args.out_dir)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    grade_src = (HERE / "grade.py").read_text()
    seen = set()
    n = 0
    for line in Path(args.tasks_jsonl).read_text().splitlines():
        if not line.strip():
            continue
        t = json.loads(line)
        tid = safe_id(t["id"])
        assert tid not in seen, f"duplicate task id {tid}"
        seen.add(tid)
        k = int(t["k"])
        letters = string.ascii_uppercase[:k]
        answer = str(t["answer"]).strip().upper()
        assert answer in letters, f"{tid}: answer {answer!r} not in {letters}"

        d = out / tid
        (d / "tests").mkdir(parents=True)
        (d / "solution").mkdir()
        meta = {key: v for key, v in t.items() if key not in ("prompt", "answer") and isinstance(v, (str, int, float, bool))}
        meta_lines = "\n".join(f"{key} = {toml_str(v) if isinstance(v, str) else json.dumps(v)}" for key, v in meta.items())
        (d / "task.toml").write_text(
            f'schema_version = "1.4"\n\n'
            f"[task]\nname = {toml_str(args.name_prefix + '/' + tid)}\n\n"
            f"[metadata]\n{meta_lines}\n\n"
            f"[agent]\ntimeout_sec = {args.agent_timeout}\n\n"
            f"[verifier]\ntimeout_sec = 120.0\n\n"
            f"[environment]\ndocker_image = {toml_str(args.image)}\nworkdir = \"/app\"\n"
            f"cpus = 1\nmemory_mb = 1024\n"
        )
        (d / "instruction.md").write_text(t["prompt"].rstrip() + SUFFIX.format(letters=", ".join(letters)))
        (d / "tests" / "test.sh").write_text(TEST_SH.format(answer_file=ANSWER_FILE))
        (d / "tests" / "grade.py").write_text(grade_src)
        (d / "tests" / "expected.json").write_text(json.dumps({"answer": answer, "letters": letters}))
        (d / "solution" / "solve.sh").write_text(SOLVE_SH.format(answer=answer, answer_file=ANSWER_FILE))
        for p in (d / "tests" / "test.sh", d / "solution" / "solve.sh"):
            p.chmod(0o755)
        n += 1
    print(f"wrote {n} tasks to {out}")


if __name__ == "__main__":
    main()
