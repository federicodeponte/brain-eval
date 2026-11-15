# pseudo-population construction for DANDI 000004: neurons pooled across sessions sharing an image set
import glob, numpy as np
from scipy.stats import f_oneway
DROP = {"houses": "landscapes", "1cars": "4spatial", "fruit": "space"}
def load_group(first_cat, win, seed=0):
    rng = np.random.default_rng(seed)
    S = []
    for fn in sorted(glob.glob("data004/s*.npz"), key=lambda x: int(x.split("/s")[1][:-4])):
        d = np.load(fn); cs = sorted(set(d["cat"]))
        if cs[0] != first_cat: continue
        keep = [c for c in cs if c != DROP[first_cat]]
        cat = d["cat"]; m = np.isin(cat, keep)
        # stratified 50/50 split of trials: selection half vs item half
        sel = np.zeros(len(cat), bool)
        for c in keep:
            ix = rng.permutation(np.where(cat == c)[0]); sel[ix[: len(ix) // 2]] = True
        S.append(dict(fn=fn, path=str(d["path"]), X=np.asarray(d[win]), cat=cat, img=d["img"], keep=keep, sel=sel & m, pool=(~sel) & m, uloc=d["uloc"]))
    return S, keep
def select_neurons(S, keep, M):
    sc = []
    for si, s in enumerate(S):
        X = np.sqrt(s["X"][s["sel"]].astype(float)); y = s["cat"][s["sel"]]
        for u in range(X.shape[1]):
            g = [X[y == c, u] for c in keep]
            if X[:, u].std() == 0 or X[:, u].mean() < 0.5: continue
            F, p = f_oneway(*g); sc.append((p, si, u))
    sc.sort(); return [(si, u) for _, si, u in sc[:M]]
def sample_rows(S, neurons, plan, rng, avoid_img=None):
    """plan: list of categories, one per row. Returns rows (len(plan) x M) using distinct pool trials per session, and image ids used."""
    sess = sorted(set(si for si, _ in neurons))
    tri = {}
    for si in sess:
        s = S[si]; used = set(); chosen = []
        for c in plan:
            cand = [t for t in np.where(s["pool"] & (s["cat"] == c))[0] if t not in used and s["img"][t] not in used]
            t = rng.choice(cand); used.add(t); used.add(s["img"][t]); chosen.append(t)
        tri[si] = chosen
    R = np.array([[S[si]["X"][tri[si][r], u] for si, u in neurons] for r in range(len(plan))])
    return R, tri
