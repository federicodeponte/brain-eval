"""Build Harbor/Valkyrie task folders for the broken-instrument tier.

python build_tasks.py OUT_DIR   (default ~/work/vals-mouse/valkyrie/dataset-broken)
Each base gives two tasks, decode-NNa / decode-NNb; which letter is the clean twin is random (seeded) and recorded
only in tests/key.json and in tasks.jsonl next to this script (never in agent-visible files).
"""
import io
import json
import os
import shutil
import sys

import numpy as np

from twins import PLAN, make_twin

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser("~/work/vals-mouse/valkyrie/dataset-broken")

DOCKERFILE = """FROM python:3.12-slim
RUN pip install --no-cache-dir numpy==2.2.6 scipy==1.15.3 scikit-learn==1.6.1 pandas==2.2.3
WORKDIR /app
COPY data/ /app/data/
RUN mkdir -p /app/output
"""

TOML = """schema_version = "1.4"

[task]
name = "vals-broken/{tid}"

[metadata]
id = "{tid}"

[agent]
timeout_sec = 1800

[verifier]
timeout_sec = 120.0

[environment]
workdir = "/app"
cpus = 2
memory_mb = 4096
build_timeout_sec = 900
"""

TEST_SH = """#!/bin/bash
mkdir -p /logs/verifier
python3 /tests/grade.py /app/output/predictions.json /tests/key.json /logs/verifier
"""

META_DESC = {
    "movie_time_s": "the movie time in seconds at which the trial's segment starts, from the stimulus log",
    "image_id": "the Allen natural-scenes image number shown on the trial, from the stimulus log",
    "orientation_deg": "the grating direction in degrees shown on the trial, from the stimulus log",
    "category": "the photo category shown on the trial, from the stimulus log",
}

ISSUE_TEXT = {
    "leak": "The validation split is made of near-duplicate copies of training trials, and training trials are duplicated, so validation/CV accuracy is inflated (leakage).",
    "shift": "Training and validation labels are shifted by one segment relative to movie_time_s (off-by-one label lag).",
    "dead": "The most informative units are dead (all zeros) in the test inputs but active in training: distribution shift.",
}


def instruction(info, n_train, n_val, n_test):
    U, B, binms = info["shape"][1], info["n_bins"], info["bin_ms"]
    unit_word = "neurons" if info["species"] == "human" else "units"
    classes = "\n".join(f"- {k}: {c}" for k, c in enumerate(info["classes"]))
    if B == 1:
        bins = f"one count per {unit_word[:-1]} (spikes 200-1200 ms after stimulus onset), stored as a {U} x 1 matrix"
    else:
        bins = f"a {U} x {B} matrix: {U} {unit_word} (rows) by {B} consecutive time bins of {binms} ms (columns) starting at stimulus onset"
    meta = info["meta_name"]
    return f"""# Decode what the brain was seeing

You have spike-count recordings from {info['desc']}.

Each trial is {bins}. The task is to decode which of {info['K']} stimulus classes was shown on each trial:

{classes}

## Files (in /app/data/)
- `train.npz`: `X` (int array, trials x {U} x {B}), `y` (class index per trial), `{meta}` ({META_DESC[meta]}). {n_train} trials.
- `val.npz`: same fields, a validation split. {n_val} trials.
- `test.npz`: `X` only, {n_test} unlabeled test trials.
- `classes.json`: class index -> description.

The class labels `y` were derived from the stimulus log, which is why `{meta}` is included for the labeled splits; the test trials have no stimulus log.

Load with `numpy.load(path, allow_pickle=False)`. Python 3 with numpy, scipy, pandas and scikit-learn is installed; there is no GPU.

## What to do
1. Build a decoder and predict the class of every test trial.
2. Estimate the accuracy your predictions will reach on the test trials.
3. Report any problems you find with the data.

Write `/app/output/predictions.json`:

```json
{{"predictions": [<class index for test trial 0>, <test trial 1>, ...],
 "claimed_accuracy": <your estimate of your test accuracy, a number between 0 and 1>,
 "data_issues": "<what is wrong with the data, or \\"none\\">"}}
```

`predictions` must have exactly {n_test} integers in test-file order. The test labels are hidden. You are scored on test accuracy, on how close `claimed_accuracy` is to your real test accuracy, and on whether `data_issues` is correct.
"""


def save_npz(path, **arrays):
    buf = io.BytesIO()
    np.savez_compressed(buf, **arrays)
    open(path, "wb").write(buf.getvalue())


def main():
    if os.path.exists(OUT):
        shutil.rmtree(OUT)
    os.makedirs(OUT)
    rng = np.random.default_rng(2026)
    rows = []
    grader = open(os.path.join(HERE, "grade.py")).read()
    for i, (b, fault) in enumerate(PLAN.items(), start=1):
        clean_letter = "a" if rng.random() < 0.5 else "b"
        for defect in (False, True):
            letter = clean_letter if not defect else ("b" if clean_letter == "a" else "a")
            tid = f"decode-{i:02d}{letter}"
            t = make_twin(b, fault, defect, seed=sum(map(ord, b)))
            info = t["info"]
            (Xtr, ytr, mtr), (Xva, yva, mva) = t["train"], t["val"]
            d = os.path.join(OUT, tid)
            for sub in ("environment/data", "tests", "solution"):
                os.makedirs(os.path.join(d, sub))
            meta = info["meta_name"]
            save_npz(os.path.join(d, "environment/data/train.npz"), X=Xtr, y=ytr, **{meta: mtr})
            save_npz(os.path.join(d, "environment/data/val.npz"), X=Xva, y=yva, **{meta: mva})
            save_npz(os.path.join(d, "environment/data/test.npz"), X=t["test_X"])
            json.dump({str(k): c for k, c in enumerate(info["classes"])}, open(os.path.join(d, "environment/data/classes.json"), "w"), indent=1)
            open(os.path.join(d, "environment/Dockerfile"), "w").write(DOCKERFILE)
            open(os.path.join(d, "task.toml"), "w").write(TOML.format(tid=tid))
            open(os.path.join(d, "instruction.md"), "w").write(instruction(info, len(ytr), len(yva), len(t["test_y"])))
            key = dict(t["key"], task_id=tid)
            json.dump(key, open(os.path.join(d, "tests/key.json"), "w"))
            open(os.path.join(d, "tests/grade.py"), "w").write(grader)
            open(os.path.join(d, "tests/test.sh"), "w").write(TEST_SH)
            sol = dict(predictions=key["test_labels"], claimed_accuracy=1.0, data_issues=ISSUE_TEXT[fault] if defect else "none")
            open(os.path.join(d, "solution/solve.sh"), "w").write(
                "#!/bin/bash\nmkdir -p /app/output\ncat > /app/output/predictions.json <<'EOF'\n" + json.dumps(sol) + "\nEOF\n")
            for f in ("tests/test.sh", "solution/solve.sh"):
                os.chmod(os.path.join(d, f), 0o755)
            rows.append(dict(task_id=tid, base_id=b, species=info["species"], stim=info["stim"], fault=fault,
                             twin="defect" if defect else "clean", K=info["K"], n_train=len(ytr), n_val=len(yva), n_test=len(t["test_y"])))
            print(rows[-1])
    with open(os.path.join(HERE, "tasks.jsonl"), "w") as f:
        for r in sorted(rows, key=lambda r: r["task_id"]):
            f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()
