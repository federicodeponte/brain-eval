import glob, numpy as np, itertools, sys
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
rng = np.random.default_rng(0)
CATS = ["houses", "landscapes", "mobility", "phones", "smallAnimal"]
def nc_fewshot(X, y, img, k=3, n=300):
    cl = np.unique(y); hit = 0
    for _ in range(n):
        q = rng.integers(len(y)); ex = []
        for c in cl:
            pool = np.where((y == c) & (np.arange(len(y)) != q) & (img != img[q]))[0]
            ex.append(X[rng.choice(pool, k, replace=False)].mean(0))
        E = np.array(ex); xq = X[q]
        d = [1 - np.corrcoef(xq, e)[0, 1] if xq.std() > 0 and e.std() > 0 else 2 for e in E]
        hit += cl[int(np.argmin(d))] == y[q]
    return hit / n
win = sys.argv[1] if len(sys.argv) > 1 else "w200_1200"
rows = []
for fn in sorted(glob.glob("data004/s*.npz")):
    d = np.load(fn); X = np.sqrt(d[win].astype(float)); cat = d["cat"]
    best = None
    for keep in itertools.combinations(sorted(set(cat)), 4):
        m = np.isin(cat, keep)
        if m.sum() < 40: continue
        lr = cross_val_score(make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=2000)), X[m], cat[m], cv=StratifiedKFold(5, shuffle=True, random_state=0)).mean()
        if best is None or lr > best[0]: best = (lr, keep)
    cs = sorted(set(cat)); DROP = {"houses":"landscapes","1cars":"4spatial","fruit":"space"}
    keep = tuple(c for c in cs if c != DROP[cs[0]])
    m = np.isin(cat, keep)
    lr_fixed = cross_val_score(make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=2000)), X[m], cat[m], cv=StratifiedKFold(5, shuffle=True, random_state=0)).mean()
    nc = nc_fewshot(X[m], cat[m], d["img"][m])
    rows.append((fn, X.shape[1], m.sum(), lr_fixed, nc, best[0], best[1]))
rows.sort(key=lambda r: -r[3])
for r in rows: print(f"{r[0]:20s} n={r[1]:3d} T={r[2]:3d} LRfix={r[3]:.2f} NC3={r[4]:.2f} LRbest={r[5]:.2f} best4={r[6]}")
a = np.array([r[3] for r in rows]); print("median LRfix", np.median(a), ">=.6:", (a >= .6).sum())
