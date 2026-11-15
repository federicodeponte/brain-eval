# Hard variant: which family unit made this 5R1 coda? Shots and tests from disjoint recording dates.
import json, random, numpy as np, pandas as pd, sys
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
d = pd.read_csv(sys.argv[1], encoding="utf-8-sig")
ici = ["ICI1", "ICI2", "ICI3", "ICI4"]; U = ["A", "F", "J", "U"]; L = "ABCD"
d = d[(d.CodaType == "5R1") & d.Unit.isin(U) & (d[ici] > 0).all(axis=1)].copy()
rng = np.random.default_rng(3)
shots, tests, pool = [], [], []
for u in U:
    s = d[d.Unit == u]; dates = s.Date.unique().tolist(); rng.shuffle(dates)
    # put ~half of dates in test, rest in train
    td = set(dates[: max(1, len(dates) // 2)])
    te = s[s.Date.isin(td)].sample(10, random_state=1); tr = s[~s.Date.isin(td)]
    shots.append(tr.sample(min(20, len(tr)), random_state=2)); pool.append(tr); tests.append(te)
shots = pd.concat(shots).sample(frac=1, random_state=4); tests = pd.concat(tests).sample(frac=1, random_state=5); pool = pd.concat(pool)
fmt = lambda r: ", ".join(f"{r[k]*1000:.0f}" for k in ici)
head = ("Sperm whales off Dominica live in stable family units. All codas below are the same coda type (5R1: five evenly spaced clicks), "
        "given as 4 inter-click intervals in milliseconds. Units may differ subtly in tempo and rhythm.\n"
        "Family units: A = unit A, B = unit F, C = unit J, D = unit U.\n\nLabeled examples:\n" +
        "\n".join(f"- ICIs (ms): {fmt(r)} -> {L[U.index(r.Unit)]}" for _, r in shots.iterrows()))
out = []
for i, (_, r) in enumerate(tests.iterrows()):
    out.append({"id": f"whale-hard-{i:03d}", "hypothesis": "speak-whale-unit", "prompt": head + f"\n\nNew coda ICIs (ms): {fmt(r)}\nWhich family unit produced it? Answer with a single letter.",
                "answer": L[U.index(r.Unit)], "choices": {L[j]: f"unit {u}" for j, u in enumerate(U)},
                "meta": {"codaNUM2018": int(r.codaNUM2018), "date": r.Date, "unit": r.Unit}})
with open("tasks_hard.jsonl", "w") as f:
    for o in out: f.write(json.dumps(o) + "\n")
open("examples/whale-hard-000.txt", "w").write(out[0]["prompt"] + f"\n\n[answer: {out[0]['answer']}]\n")
X = lambda df: df[ici].values * 1000; y = lambda df: np.array(df.Unit.tolist()); yt = y(tests)
print("items", len(out), {l: sum(o["answer"] == l for o in out) for l in L}, "chars", max(len(o["prompt"]) for o in out))
for name, m, tr in [("knn5_shots", KNeighborsClassifier(5), shots), ("knn15_pool", KNeighborsClassifier(15), pool), ("rf_pool", RandomForestClassifier(300, random_state=0), pool)]:
    print(name, round(float((m.fit(X(tr), y(tr)).predict(X(tests)) == yt).mean()), 3))
