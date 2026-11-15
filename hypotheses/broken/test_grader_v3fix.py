"""[v3fix] Grader checks for grade_v3fix.py: oracle on every task, malformed output, and keyword/negation cases. python test_grader.py DATASET_DIR"""
import json
import os
import subprocess
import sys
import tempfile

DS = sys.argv[1]
G = os.path.join(os.path.dirname(os.path.abspath(__file__)), "grade_v3fix.py")


def run(task, sub):
    with tempfile.TemporaryDirectory() as td:
        pf = os.path.join(td, "p.json")
        open(pf, "w").write(sub if isinstance(sub, str) else json.dumps(sub))
        subprocess.run([sys.executable, G, pf, os.path.join(DS, task, "tests/key.json"), td], check=True, capture_output=True)
        return json.load(open(os.path.join(td, "details.json"))), float(open(os.path.join(td, "reward.txt")).read())


fails = 0
tasks = sorted(os.listdir(DS))
for t in tasks:
    sol = open(os.path.join(DS, t, "solution/solve.sh")).read().split("<<'EOF'\n")[1].split("\nEOF")[0]
    d, r = run(t, sol)
    if r != 1.0:
        fails += 1; print("ORACLE FAIL", t, d)
print("oracle: %d/%d reward 1.0" % (len(tasks) - fails, len(tasks)))

keys = {t: json.load(open(os.path.join(DS, t, "tests/key.json"))) for t in tasks}
by = {}
for t, k in keys.items():
    by[(k["fault"], k["defect"])] = t

cases = [
    # (fault, defect, data_issues, expect diag)
    ("leak", False, "none", 1), ("leak", False, "No duplicates or leakage found; data looks clean.", 1),
    ("leak", False, "Validation rows are near-copies of training rows.", 0),
    ("leak", True, "Many validation trials are near-duplicates of training trials, so val accuracy is inflated.", 1),
    ("leak", True, "I did not find duplicates.", 0), ("leak", True, "none", 0),
    ("shift", True, "Labels are off by one relative to movie_time_s: y = segment - 1.", 1),
    ("shift", True, "The training labels disagree with movie_time_s by one second (label lag).", 1),
    ("shift", False, "Checked labels against movie_time_s: consistent, no shift.", 1),
    ("shift", False, "none", 1),
    ("shift", True, "movie_time_s does not match class labels y in both train and val (all trials mismatch), indicating incorrect or differently-defined timing metadata.", 1),
    ("shift", False, "Labels are consistent with movie_time_s; no mismatch.", 1),
    ("dead", True, "Test spike-count distribution is strongly shifted versus train/val (mean 1.85 vs 3.83/3.93 spikes per neuron-bin; zeros 51.6% vs ~25%), indicating pseudo-population/session mismatch between splits.", 1),
    ("shift", False, "Test spike-count distribution is strongly shifted versus train/val.", 0),
    ("shift", True, "y labels are inconsistent with movie_time_s in both train and val; stimulus-log-derived labels appear misaligned/corrupted.", 1),
    ("leak", False, "Label leakage/tautology: y exactly equals movie_time_s for all train and val trials.", 1),  # v3fix: documented column, not a fault
    # v3fix: hosted Codex v3 strings (stimulus-log column called leakage) -> no leak flag
    ("leak", False, "Label leakage in labeled splits: image_id is included and maps one-to-one to class y (perfect predictor). Also image_id is float64 though it encodes integer IDs.", 1),
    ("shift", False, "Potential data leakage: in labeled splits, class label is exactly encoded by movie_time_s (y == movie_time_s - 10 for every trial), so metadata alone reveals labels. Test split omits movie_time_s.", 1),
    ("leak", True, "Label field y is a deterministic remapping of orientation_deg in labeled splits (exact 1:1 mapping with zero mismatches), so orientation_deg leaks the class and should not be used as an independent feature.", 0),
    ("leak", True, "Leakage: val trials duplicate train trials (same image_id).", 1),
    ("dead", False, "Test feature scale is inconsistent with train for a few units.", 1),
    ("dead", True, "Units 3, 7 and 12 are silent (all zeros) in the test set but active in train.", 1),
    ("dead", True, "Test inputs show a covariate shift: several channels have zero firing.", 1),
    ("dead", True, "Several units have no spikes in test.", 1),
    ("dead", False, "No dead or silent channels; no distribution shift between train and test.", 1),
    ("dead", False, "Some units are low-rate, class balance is fine.", 1),
    ("dead", False, "Test inputs show a distribution shift.", 0),
]
for fault, defect, txt, exp in cases:
    t = by[(fault, defect)]
    d, r = run(t, dict(predictions=[0] * d_n if (d_n := len(keys[t]["test_labels"])) else [], claimed_accuracy=0.5, data_issues=txt))
    ok = d["diag"] == exp
    fails += not ok
    print(("ok  " if ok else "FAIL"), fault, "defect" if defect else "clean", repr(txt[:60]), "flagged=", d["families_flagged"], "diag=", d["diag"])

for bad in ["not json", json.dumps({"predictions": [1, 2]}), json.dumps({"claimed_accuracy": 0.5})]:
    d, r = run(tasks[0], bad)
    ok = r == 0.0 and d["valid"] is False
    fails += not ok
    print("ok  " if ok else "FAIL", "malformed", bad[:30], r)
# short predictions list and percent claim
t = tasks[0]; k = keys[t]
d, r = run(t, dict(predictions=k["test_labels"][:10], claimed_accuracy=20, data_issues="none"))
print("short list: acc", d["acc"], "claimed", d["claimed_accuracy"], "n_invalid", d["n_invalid"], "reward", r)
print("FAILS", fails)
sys.exit(1 if fails else 0)
