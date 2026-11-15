import sys, numpy as np, itertools
sys.path.insert(0, sys.argv[1])
from pseudo import *
from sklearn.linear_model import LogisticRegression
def nc(R, K, nclass):
    Xs = np.sqrt(R.astype(float)); E = Xs[:-1].reshape(nclass, K, -1).mean(1); q = Xs[-1]
    d = [1 - np.corrcoef(q, e)[0, 1] for e in E]; return int(np.argmin(d))
for first in ["1cars", "houses", "fruit"]:
  for win in ["w200_1200", "w200_1700"]:
    S, keep = load_group(first, win)
    nn = sum(s["X"].shape[1] for s in S)
    for M in [40, 80]:
        neurons = select_neurons(S, keep, M)
        for K in [3, 5]:
            rng = np.random.default_rng(1); hit = 0; lrh = 0; n = 120
            for it in range(n):
                a = it % 4; plan = [c for c in keep for _ in range(K)] + [keep[a]]
                R, tri = sample_rows(S, neurons, plan, rng)
                hit += nc(R, K, 4) == a
            # LR on pseudo trials from pool (train many pseudo trials, test fresh)
            tr = []; ytr = []
            for j in range(400):
                c = j % 4; R, _ = sample_rows(S, neurons, [keep[c]], rng); tr.append(R[0]); ytr.append(c)
            te = []; yte = []
            for j in range(200):
                c = j % 4; R, _ = sample_rows(S, neurons, [keep[c]], rng); te.append(R[0]); yte.append(c)
            clf = LogisticRegression(C=0.05, max_iter=3000).fit(np.sqrt(np.array(tr, float)), ytr)
            lr = (clf.predict(np.sqrt(np.array(te, float))) == np.array(yte)).mean()
            print(f"{first:7s} {win} sessions={len(S)} totalN={nn} M={M} K={K} NC={hit/n:.2f} LR={lr:.2f}", flush=True)
