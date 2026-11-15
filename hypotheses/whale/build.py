import json, random, numpy as np, pandas as pd, sys
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
R = random.Random(7)
d = pd.read_csv(sys.argv[1], encoding="utf-8-sig")
CLS = ["1+1+3", "5R1", "5R2", "5R3"]
L = "ABCD"
d = d[d.CodaType.isin(CLS) & (d.nClicks == 5)].copy()
ici = ["ICI1", "ICI2", "ICI3", "ICI4"]
d = d[(d[ici] > 0).all(axis=1)]
d = d.sample(frac=1, random_state=7).reset_index(drop=True)
NSHOT, NTEST = 6, 10
shots, tests, pool = [], [], []
for c in CLS:
    s = d[d.CodaType == c]
    shots.append(s.iloc[:NSHOT]); tests.append(s.iloc[NSHOT:NSHOT + NTEST]); pool.append(s.iloc[NSHOT + NTEST:])
shots = pd.concat(shots).sample(frac=1, random_state=1); tests = pd.concat(tests).sample(frac=1, random_state=2); pool = pd.concat(pool)
def fmt(r): return ", ".join(f"{r[k]*1000:.0f}" for k in ici)
shot_txt = "\n".join(f"- ICIs (ms): {fmt(r)} -> {L[CLS.index(r.CodaType)]}" for _, r in shots.iterrows())
head = ("Sperm whales off Dominica communicate with codas: short bursts of clicks. A coda is described by its "
        "inter-click intervals (ICIs), the gaps in milliseconds between consecutive clicks. All codas below have 5 clicks (4 ICIs).\n"
        "Coda types: A = 1+1+3, B = 5R1, C = 5R2, D = 5R3.\n\nLabeled examples:\n" + shot_txt)
out = []
for i, (_, r) in enumerate(tests.iterrows()):
    p = head + f"\n\nNew coda ICIs (ms): {fmt(r)}\nWhich coda type is it? Answer with a single letter."
    out.append({"id": f"whale-{i:03d}", "hypothesis": "speak-whale", "prompt": p, "answer": L[CLS.index(r.CodaType)],
                "choices": {L[j]: c for j, c in enumerate(CLS)},
                "meta": {"codaNUM2018": int(r.codaNUM2018), "clan": r.Clan, "unit": r.Unit, "coda_type": r.CodaType, "duration_s": float(r.Duration)}})
with open("tasks.jsonl", "w") as f:
    for o in out: f.write(json.dumps(o) + "\n")
for k in (0, 1): open(f"examples/{out[k]['id']}.txt", "w").write(out[k]["prompt"] + f"\n\n[answer: {out[k]['answer']}]\n")
X = lambda df: df[ici].values * 1000; y = lambda df: df.CodaType.values
Xt, yt = X(tests), y(tests)
res = {}
for k in (1, 3):
    res[f"knn{k}_shots_only"] = (KNeighborsClassifier(k).fit(X(shots), y(shots)).predict(Xt) == yt).mean()
res["logreg_shots_only"] = (LogisticRegression(max_iter=5000).fit(X(shots), y(shots)).predict(Xt) == yt).mean()
res["knn5_full_pool"] = (KNeighborsClassifier(5).fit(X(pool), y(pool)).predict(Xt) == yt).mean()
res["rf_full_pool"] = (RandomForestClassifier(300, random_state=0).fit(X(pool), y(pool)).predict(Xt) == yt).mean()
# majority/no-data baseline
res["chance"] = 0.25
print("items", len(out), "letters", {l: sum(o["answer"] == l for o in out) for l in L}, "max prompt chars", max(len(o["prompt"]) for o in out))
print({k: round(float(v), 3) for k, v in res.items()})
