"""v4 grader checks: python test_grader.py DATASET_DIR
oracle = 1.0 on every task; keyword/negation/group cases per family; extra-group penalty; fix component; malformed."""
import json
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from grade import diagnose  # noqa: E402

DS = sys.argv[1]
G = os.path.join(HERE, "grade.py")


def run(keyfile, sub):
    with tempfile.TemporaryDirectory() as td:
        pf = os.path.join(td, "p.json")
        open(pf, "w").write(sub if isinstance(sub, str) else json.dumps(sub))
        subprocess.run([sys.executable, G, pf, keyfile, td], check=True, capture_output=True)
        return json.load(open(os.path.join(td, "details.json"))), float(open(os.path.join(td, "reward.txt")).read())


fails = 0
tasks = sorted(t for t in os.listdir(DS) if os.path.isdir(os.path.join(DS, t)))
for t in tasks:
    sol = open(os.path.join(DS, t, "solution/solve.sh")).read().split("<<'EOF'\n")[1].split("\nEOF")[0]
    d, r = run(os.path.join(DS, t, "tests/key.json"), sol)
    if r != 1.0:
        fails += 1; print("ORACLE FAIL", t, d)
print("oracle: %d/%d reward 1.0" % (len(tasks) - fails, len(tasks)))

cases = [  # (planted faults, data_issues, expected diag)
    ([], "none", 1), ([], "No duplicates or leakage found; data looks clean.", 1),
    ([], "Checked labels against movie_time_s: consistent, no shift.", 1),
    ([], "Labels are consistent with movie_time_s; no mismatch.", 1),
    ([], "No dead or silent channels; no distribution shift between train and test.", 1),
    ([], "Some units are low-rate, class balance is fine.", 1),
    ([], "Label leakage/tautology: y exactly equals movie_time_s for all train and val trials.", 0),
    ([], "Test inputs show a distribution shift.", 0),
    ([], "Validation rows are near-copies of training rows.", 0),
    (["leak"], "Many validation trials are near-duplicates of training trials, so val accuracy is inflated.", 1),
    (["leak"], "I did not find duplicates.", 0), (["leak"], "none", 0),
    (["shift"], "Labels are off by one relative to movie_time_s: y = segment - 1.", 1),
    (["shift"], "The training labels disagree with movie_time_s by one second (label lag).", 1),
    (["shift"], "movie_time_s does not match class labels y in both train and val (all trials mismatch), indicating incorrect or differently-defined timing metadata.", 1),
    (["shift"], "y labels are inconsistent with movie_time_s in both train and val; stimulus-log-derived labels appear misaligned/corrupted.", 1),
    (["shift"], "Labels are shifted by 2 segments relative to the stimulus log.", 1),
    (["shift"], "Units 3 and 7 are silent in test.", 0),
    (["noise"], "About 30% of training labels do not match the stimulus log (label noise).", 1),
    (["noise"], "y disagrees with image_id on 71 of 240 trials; those labels look randomly corrupted.", 1),
    (["noise"], "Some labels are wrong: they are inconsistent with orientation_deg.", 1),
    (["noise"], "No label noise: y matches category on every trial.", 0),
    (["noise"], "Labels are off by one segment relative to movie_time_s.", 0),
    (["dead"], "Units 3, 7 and 12 are silent (all zeros) in the test set but active in train.", 1),
    (["dead"], "Test inputs show a covariate shift: several channels have zero firing.", 1),
    (["dead"], "Several units have no spikes in test.", 1),
    (["dead"], "Test spike-count distribution is strongly shifted versus train/val (mean 1.85 vs 3.83/3.93 spikes per neuron-bin; zeros 51.6% vs ~25%), indicating pseudo-population/session mismatch between splits.", 1),
    (["perm"], "The unit order in the test file appears permuted relative to training: per-unit firing rates only match after reordering columns.", 1),
    (["perm"], "Channel map mismatch: test columns are shuffled.", 1),
    (["perm"], "Test firing rates per unit differ strongly from train.", 1),
    (["perm"], "Training labels are noisy.", 0),
    (["mix"], "Training rows form two clusters: part of the data seems to come from a different recording session with different units in the same columns.", 1),
    (["mix"], "About 40% of train and val trials look like they come from another population of neurons (bimodal per-unit rates).", 1),
    (["mix"], "none", 0),
    (["leak", "dead"], "Val trials are near-copies of train trials. Also, a third of the units are all zeros in test.", 1),
    (["leak", "dead"], "Val trials are near-copies of train trials.", 0.5),
    (["shift", "perm"], "Labels lag movie_time_s by one segment; test channel order is permuted.", 1),
    (["shift"], "Labels lag movie_time_s by one segment. Validation rows are duplicates of training rows.", 0.75),
    (["noise", "dead"], "Label noise in ~30% of trials; dead units in test.", 1),
    # hosted v4 smoke strings (run ddb9b286): stimulus-log column called "leakage" is neither credit nor false alarm
    (["leak"], "orientation_deg is included only for labeled splits and is deterministically mapped to label; this metadata leakage is unavailable for test and should not be used for decoding from neural counts. also class 135 deg is absent (7 classes are not evenly spaced around 360).", 0),
    ([], "orientation_deg is included only for labeled splits and is deterministically mapped to label; this metadata leakage is unavailable for test and should not be used for decoding from neural counts.", 1),
    (["noise"], "train labels inconsistent with movie_time_s; val labels inconsistent with movie_time_s; label is directly derivable from movie_time_s in labeled splits; this metadata would leak class if used as a feature, but test lacks movie_time_s", 1),
    (["leak"], "Val trials are near-copies of train trials, which leaks into the image_id-matched validation score.", 1),
]
for faults, txt, exp in cases:
    dg, dd = diagnose(txt, faults)
    ok = abs(dg - exp) < 1e-9
    fails += not ok
    print(("ok  " if ok else "FAIL"), "+".join(faults) or "clean", repr(txt[:70]), "fam=", sorted(dd["fam"]), "gen=", sorted(dd["gen"]), "diag=", dg)

# fix component + reward composition on a real faulted task with fix_applies
fixt = next((t for t in tasks if json.load(open(os.path.join(DS, t, "tests/key.json")))["fix_applies"]), None)
if fixt:
    kf = os.path.join(DS, fixt, "tests/key.json"); k = json.load(open(kf)); lab = k["test_labels"]; n = len(lab)
    wrong = [(v + 1) % k["K"] for v in lab]
    for target_acc, exp_fix in ((1.0, 1.0), (k["fix_target"] - k["fix_tol"], 1.0), (max(0.0, k["fix_target"] - k["fix_tol"] - 0.25), 0.0)):
        m = int(round(target_acc * n)); preds = lab[:m] + wrong[m:]
        d, r = run(kf, dict(predictions=preds, claimed_accuracy=m / n, data_issues="none"))
        ok = d["fix"] is not None and (abs(d["fix"] - exp_fix) < 0.06) and abs(r - (d["skill"] + d["calib"] + d["diag"] + d["fix"]) / 4) < 1e-3
        fails += not ok
        print("ok  " if ok else "FAIL", "fix", fixt, "acc", d["acc"], "fix", d["fix"], "reward", r)
nofix = next((t for t in tasks if not json.load(open(os.path.join(DS, t, "tests/key.json")))["fix_applies"]), None)
d, r = run(os.path.join(DS, nofix, "tests/key.json"), dict(predictions=[0], claimed_accuracy=0.5, data_issues="none"))
ok = d["fix"] is None and abs(r - (d["skill"] + d["calib"] + d["diag"]) / 3) < 1e-3
fails += not ok
print("ok  " if ok else "FAIL", "no-fix task uses 3 components", nofix, r)
for bad in ["not json", json.dumps({"predictions": [1, 2]}), json.dumps({"claimed_accuracy": 0.5})]:
    d, r = run(os.path.join(DS, tasks[0], "tests/key.json"), bad)
    ok = r == 0.0 and d["valid"] is False
    fails += not ok
    print("ok  " if ok else "FAIL", "malformed", bad[:30], r)
print("FAILS", fails)
sys.exit(1 if fails else 0)
