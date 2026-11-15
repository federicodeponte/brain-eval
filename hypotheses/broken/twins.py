"""Clean/defect twins from a base recording, plus the naive-pipeline fault validation.

make_twin(base_id, fault, defect: bool, seed) -> dict(train=(X,y,meta), val=(X,y,meta), test_X, test_y, key)
Faults:
  leak  : val split = jittered near-copies of train trials, and every train trial appears twice (jittered).
          Naive val / CV is inflated; hidden test trials are untouched and disjoint.
  shift : train+val labels lag by one stimulus segment (label k given to segment k+1, cyclic); meta (movie time) stays
          correct, so labels disagree with movie_time_s by one second. Hidden test labels are correct.
  dead  : the most informative third of the units (ANOVA F on train+val) are all-zero in the test inputs only.
Run validation: python twins.py  -> validation.json + table on stdout
"""
import json
import os
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neighbors import NearestCentroid
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
BASES = os.path.join(HERE, "bases")

# base -> fault (balanced: 4 each; shift needs an ordered stimulus, so it uses the movie bases)
PLAN = {
    "m-nm1-a": "shift", "m-nm1-b": "shift", "m-nm1-c": "shift", "m-nm3-a": "shift",
    "m-ns-a": "leak", "m-dg": "leak", "h-houses-b": "leak", "h-houses": "leak",
    "m-ns-b": "dead", "m-ns-c": "dead", "h-cars-b": "dead", "h-fruit": "dead",
}


def load_base(base_id):
    d = np.load(os.path.join(BASES, base_id + ".npz"), allow_pickle=False)
    info = json.load(open(os.path.join(BASES, base_id + ".json")))
    return {k: d[k] for k in d.files}, info


def jitter(X, rng):
    noise = rng.choice([-1, 0, 1], size=X.shape, p=[0.05, 0.9, 0.05])
    return np.clip(X + noise, 0, None).astype(X.dtype)


def fstat(X, y):
    v = np.log1p(X.sum(2).astype(float))
    gm = v.mean(0)
    cls = np.unique(y)
    ssb = sum((y == c).sum() * (v[y == c].mean(0) - gm) ** 2 for c in cls)
    ssw = sum(((v[y == c] - v[y == c].mean(0)) ** 2).sum(0) for c in cls)
    return (ssb / (len(cls) - 1)) / (ssw / (len(v) - len(cls)) + 1e-12)


def make_twin(base_id, fault, defect, seed=0):
    d, info = load_base(base_id)
    rng = np.random.default_rng(seed)
    X, y, sp, meta = d["X"], d["y"], d["split"], d["meta"]
    K = int(info["K"])
    tr, va, te = (sp == "train"), (sp == "val"), (sp == "test")
    Xtr, ytr, mtr = X[tr], y[tr], meta[tr]
    Xva, yva, mva = X[va], y[va], meta[va]
    Xte, yte = X[te].copy(), y[te]
    extra = {}
    if defect and fault == "leak":
        n_va = len(yva)
        src = rng.choice(len(ytr), n_va, replace=len(ytr) < n_va)
        Xva, yva, mva = jitter(Xtr[src], rng), ytr[src], mtr[src]
        dup = np.arange(len(ytr))  # every train trial appears twice
        Xtr = np.concatenate([Xtr, jitter(Xtr[dup], rng)]); ytr = np.concatenate([ytr, ytr[dup]]); mtr = np.concatenate([mtr, mtr[dup]])
        perm = rng.permutation(len(ytr)); Xtr, ytr, mtr = Xtr[perm], ytr[perm], mtr[perm]
        extra = dict(val_copied_from_train=True, train_duplicates=int(len(dup)))
    elif defect and fault == "shift":
        ytr = (ytr - 1) % K
        yva = (yva - 1) % K
        extra = dict(label_lag="label = true segment - 1 (cyclic) in train and val")
    elif defect and fault == "dead":
        Fv = fstat(np.concatenate([Xtr, Xva]), np.concatenate([ytr, yva]))
        n_dead = int(np.ceil(X.shape[1] / 3))
        dead = sorted(np.argsort(-Fv)[:n_dead].tolist())
        Xte[:, dead, :] = 0
        extra = dict(dead_units=dead)
    # shuffle test order so the class sequence is not visible
    p = rng.permutation(len(yte)); Xte, yte = Xte[p], yte[p]
    pv = rng.permutation(len(yva)); Xva, yva, mva = Xva[pv], yva[pv], mva[pv]
    key = dict(base_id=base_id, fault=fault, defect=bool(defect), K=K, chance=1.0 / K,
               test_labels=[int(v) for v in yte], **extra)
    return dict(train=(Xtr, ytr, mtr), val=(Xva, yva, mva), test_X=Xte, test_y=yte, key=key, info=info)


def feats(X):
    return np.log1p(X.reshape(len(X), -1).astype(float))


def models():
    return {
        "logreg": lambda: make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=3000)),
        "centroid": lambda: make_pipeline(StandardScaler(), NearestCentroid()),
    }


def naive_eval(t):
    (Xtr, ytr, _), (Xva, yva, _) = t["train"], t["val"]
    Xall, yall = np.concatenate([Xtr, Xva]), np.concatenate([ytr, yva])
    out = {}
    for name, mk in models().items():
        m = mk().fit(feats(Xtr), ytr)
        val = m.score(feats(Xva), yva)
        cv = cross_val_score(mk(), feats(Xall), yall, cv=StratifiedKFold(5, shuffle=True, random_state=0)).mean()
        m2 = mk().fit(feats(Xall), yall)
        test = m2.score(feats(t["test_X"]), t["test_y"])
        out[name] = dict(val=round(float(val), 3), cv=round(float(cv), 3), test=round(float(test), 3))
    return out


def aware_eval(t):
    """What a careful analyst could get: undo the fault using only agent-visible data."""
    (Xtr, ytr, mtr), (Xva, yva, mva) = t["train"], t["val"]
    k = t["key"]; fault = k["fault"]
    Xte = t["test_X"]
    if fault == "leak" and k["defect"]:
        # drop near-duplicates: keep train rows whose nearest other train row is not a near-copy, ignore val
        A = Xtr.reshape(len(Xtr), -1).astype(float)
        D = ((A[:, None] - A[None]) ** 2).sum(-1); np.fill_diagonal(D, np.inf)
        keep = np.ones(len(A), bool)
        for i in np.argsort(D.min(1)):
            j = int(np.argmin(D[i]))
            if keep[i] and keep[j] and D[i, j] <= 0.5 * A.shape[1] * 0.2 + 1:
                keep[j] = False
        Xall, yall = Xtr[keep], ytr[keep]
    else:
        Xall, yall = np.concatenate([Xtr, Xva]), np.concatenate([ytr, yva])
        mall = np.concatenate([mtr, mva])
    if fault == "shift" and k["defect"]:
        seg0 = int(t["info"]["seg0"]); yall = (mall.astype(float) - seg0).round().astype(int)
    keepu = np.ones(Xall.shape[1], bool)
    if fault == "dead" and k["defect"]:
        keepu = Xte.reshape(len(Xte), Xte.shape[1], -1).sum((0, 2)) > 0
    mk = models()["logreg"]
    cv = cross_val_score(mk(), feats(Xall[:, keepu]), yall, cv=StratifiedKFold(5, shuffle=True, random_state=0)).mean()
    test = mk().fit(feats(Xall[:, keepu]), yall).score(feats(Xte[:, keepu]), t["test_y"])
    return dict(cv=round(float(cv), 3), test=round(float(test), 3))


if __name__ == "__main__":
    rows = []
    for b, fault in PLAN.items():
        for defect in (False, True):
            t = make_twin(b, fault, defect, seed=hash(b) % 1000 if False else sum(map(ord, b)))
            r = dict(base=b, fault=fault, twin="defect" if defect else "clean", K=t["key"]["K"],
                     n_train=len(t["train"][1]), n_val=len(t["val"][1]), n_test=len(t["test_y"]),
                     naive=naive_eval(t), aware=aware_eval(t))
            rows.append(r)
            n = r["naive"]["logreg"]; c = r["naive"]["centroid"]
            print(f"{b:9s} {fault:5s} {r['twin']:6s} K={r['K']:2d} LR val {n['val']:.2f} cv {n['cv']:.2f} test {n['test']:.2f} gap {n['cv']-n['test']:+.2f} | "
                  f"NC val {c['val']:.2f} cv {c['cv']:.2f} test {c['test']:.2f} | aware cv {r['aware']['cv']:.2f} test {r['aware']['test']:.2f}")
            sys.stdout.flush()
    json.dump(rows, open(os.path.join(HERE, "validation.json"), "w"), indent=1)
