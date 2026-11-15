"""Broken-instrument decoding gym: procedurally generated environments from real public spike recordings.

python gen.py --seed N [--faults leak,dead|clean] [--out DIR]        -> one Harbor task folder (DIR/gym-NNNNN)
python gen.py --validate --seeds 0:200 [--faults ...]                -> naive / aware / clean-ref stats per env (jsonl)

Deterministic per seed. Each environment randomizes: base recording (Allen Visual Coding Neuropixels session
732592105: natural movie one / three, natural scenes, drifting / static gratings, cortical area, unit subset;
or DANDI 000004 human MTL: category set, neuron count), stimulus subset, split sizes, time binning, which fault(s),
fault strength, file names, array key names, axis order and unit order. The hidden key (test labels, faults, reference
accuracies) goes to tests/key.json only.

Faults (all detectable from the visible files by a careful analyst):
  leak  : part of val = jittered near-copies of train trials; part of train duplicated with jitter
  shift : train+val labels lag the stimulus log by L segments (movies only; stimulus-log column is correct)
  dead  : the most informative 20-35% of units are all-zero in the test inputs only
  noise : 25-35% of train+val trials have their labels shuffled among themselves (stimulus-log column is correct)
  mix   : 35-50% of train+val rows come from a different set of units (another recording) in the same columns
  perm  : the unit (column) order of the test file is permuted for 50-100% of units
"""
import argparse
import glob
import io
import json
import os
import pickle
import shutil
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MD = os.path.expanduser("~/work/vals-mouse/data/")
HD = os.path.expanduser("~/work/vals-hyp/human-neurons/data004/")
SESSION = 732592105
CACHE = os.path.join(HERE, "cache_allen.pkl")
AREAS = {"VISp": "primary visual cortex (VISp)", "VISal": "anterolateral visual area (VISal)",
         "VISpm": "posteromedial visual area (VISpm)", "VISrl": "rostrolateral visual area (VISrl)"}
TRAIN_SIDE = ["leak", "shift", "noise", "mix"]
TEST_SIDE = ["dead", "perm"]
ALL_FAULTS = TRAIN_SIDE + TEST_SIDE
FIXABLE = {"leak", "shift", "noise", "mix", "perm"}
COMBOS = [("leak", "dead"), ("shift", "dead"), ("noise", "dead"), ("mix", "dead"), ("leak", "perm"),
          ("shift", "perm"), ("noise", "perm"), ("leak", "shift"), ("leak", "noise")]


# ------------------------------------------------------------------ data access
def allen():
    if os.path.exists(CACHE):
        return pickle.load(open(CACHE, "rb"))
    import h5py
    import pandas as pd
    u = pd.read_csv(MD + "units.csv", usecols=["id", "ecephys_channel_id", "isi_violations", "amplitude_cutoff", "presence_ratio"])
    c = pd.read_csv(MD + "channels.csv", usecols=["id", "ecephys_probe_id", "ecephys_structure_acronym"]).rename(columns={"id": "ecephys_channel_id"})
    p = pd.read_csv(MD + "probes.csv", usecols=["id", "ecephys_session_id"]).rename(columns={"id": "ecephys_probe_id"})
    u = u.merge(c, on="ecephys_channel_id").merge(p, on="ecephys_probe_id")
    m = (u.ecephys_session_id == SESSION) & (u.isi_violations < 0.5) & (u.amplitude_cutoff < 0.1) & (u.presence_ratio > 0.95)
    area_of = {int(i): a for i, a in zip(u[m].id, u[m].ecephys_structure_acronym) if a in AREAS}
    F = h5py.File(MD + f"session_{SESSION}.nwb", "r")
    ids = F["units/id"][:]; st = F["units/spike_times"][:]; sti = F["units/spike_times_index"][:]
    starts = np.concatenate([[0], sti[:-1]])
    spk = {int(i): st[s:e].copy() for i, s, e in zip(ids, starts, sti) if int(i) in area_of}
    stim = {}
    for s in ("natural_movie_one", "natural_movie_three", "natural_scenes", "drifting_gratings", "static_gratings"):
        g = F[f"intervals/{s}_presentations"]
        d = {"start_time": g["start_time"][:], "stimulus_block": g["stimulus_block"][:]}
        for k in ("frame", "orientation"):
            if k in g:
                d[k] = g[k][:]
        stim[s] = d
    out = dict(spk=spk, area_of=area_of, stim=stim)
    pickle.dump(out, open(CACHE, "wb"))
    return out


def counts(spk, onsets, nbins, binms, uids):
    edges = np.arange(nbins + 1) * binms / 1000.0
    X = np.zeros((len(onsets), len(uids), nbins), np.int32)
    for k, uid in enumerate(uids):
        e = onsets[:, None] + edges[None, :]
        X[:, k] = np.diff(np.searchsorted(spk[uid], e), axis=1)
    return X


def split_per_class(rng, y, ntr, nva, nte):
    sp = np.array(["drop"] * len(y), dtype=object)
    for c in np.unique(y):
        ix = rng.permutation(np.where(y == c)[0])
        sp[ix[:ntr]] = "train"; sp[ix[ntr:ntr + nva]] = "val"; sp[ix[ntr + nva:ntr + nva + nte]] = "test"
    return sp


# ------------------------------------------------------------------ base recordings
def mouse_units(A, rng, nunits):
    area = rng.choice([a for a in AREAS if sum(1 for v in A["area_of"].values() if v == a) >= 2 * nunits])
    pool = sorted(u for u, a in A["area_of"].items() if a == area)
    pick = rng.permutation(pool)
    return str(area), sorted(pick[:nunits].tolist()), sorted(pick[nunits:2 * nunits].tolist())


def mouse_base(rng, kind):
    """Returns dict X, Xalt, y, split, meta, info. Xalt = same trials recorded on a disjoint unit set (for mix)."""
    A = allen()
    nunits = int(rng.integers(16, 31))
    area, uids, alt = mouse_units(A, rng, nunits)
    S = A["stim"]
    if kind == "movie":
        stim = str(rng.choice(["natural_movie_one", "natural_movie_three"], p=[0.6, 0.4]))
        g = S[stim]; t0 = g["start_time"]; fr = g["frame"].astype(int); blk = g["stimulus_block"]
        nframes = fr.max() + 1
        reps = [i0 for i0 in np.where(fr == 0)[0] if i0 + nframes <= len(fr) and np.all(fr[i0:i0 + nframes] == np.arange(nframes))]
        R = len(reps)
        seglen = float(rng.choice([0.5, 1.0], p=[0.3, 0.7]))
        K = int(rng.integers(6, 11))
        dur = nframes / 30.0
        seg0 = float(rng.integers(0, int(dur - K * seglen)) )
        nte = max(2, int(round(R * rng.uniform(0.22, 0.3)))); nva = max(1, int(round(R * rng.uniform(0.12, 0.18))))
        lab = np.array(["test"] * nte + ["val"] * nva + ["train"] * (R - nte - nva))
        # interleave blocks: permute repeats inside each block, then deal the labels round-robin
        order = []
        for b in sorted(set(blk[reps])):
            order.append(list(rng.permutation([r for r in range(R) if blk[reps[r]] == b])))
        seq = []
        while any(order):
            for o in order:
                if o:
                    seq.append(o.pop())
        rep_lab = dict(zip(seq, rng.permutation(lab)))
        binms, nbins = (100, int(seglen * 10)) if rng.random() < 0.6 else (50, int(seglen * 20))
        on, y, sp, meta = [], [], [], []
        for r in range(R):
            for k in range(K):
                f0 = int(round((seg0 + k * seglen) * 30))
                on.append(t0[reps[r] + f0]); y.append(k); sp.append(rep_lab[r]); meta.append(seg0 + k * seglen)
        on = np.array(on)
        mv = "natural movie one (30 s clip)" if stim == "natural_movie_one" else "natural movie three (120 s clip)"
        info = dict(kind="movie", stim=stim, K=K, seg0=seg0, seglen=seglen, meta_name="movie_time_s",
                    desc=f"while the mouse watched {mv}, replayed {R} times",
                    classes=[f"movie time {seg0 + k * seglen:g}-{seg0 + (k + 1) * seglen:g} s ({seglen:g}-second segment starting at {seg0 + k * seglen:g} s)" for k in range(K)])
        y = np.array(y); sp = np.array(sp); meta = np.array(meta, float)
    elif kind == "scenes":
        g = S["natural_scenes"]; t0 = g["start_time"]; fr = g["frame"].astype(int)
        K = int(rng.integers(5, 11))
        imgs = sorted(rng.choice(np.arange(118), K, replace=False).tolist())
        m = np.isin(fr, imgs); on = t0[m]; img = fr[m]
        y = np.array([imgs.index(i) for i in img])
        sp = split_per_class(rng, y, int(rng.integers(20, 29)), int(rng.integers(6, 11)), int(rng.integers(10, 16)))
        keep = sp != "drop"; on, y, sp, meta = on[keep], y[keep], sp[keep].astype(str), img[keep].astype(float)
        binms, nbins = [(50, 5), (25, 10), (125, 2), (250, 1)][int(rng.integers(4))]
        info = dict(kind="scenes", stim="natural_scenes", K=K, images=imgs, meta_name="image_id",
                    desc="while the mouse was shown flashed natural photographs (250 ms each, random order)",
                    classes=[f"natural scene image #{i} of the Allen natural-scenes set" for i in imgs])
    else:  # gratings
        stim = str(rng.choice(["drifting_gratings", "static_gratings"]))
        g = S[stim]; t0 = g["start_time"]; ori = g["orientation"]
        m = ~np.isnan(ori); on = t0[m]; ori = ori[m]
        allo = sorted(set(ori.tolist()))
        K = int(rng.integers(max(4, len(allo) - 2), len(allo) + 1))
        oris = sorted(rng.choice(allo, K, replace=False).tolist())
        m = np.isin(ori, oris); on, ori = on[m], ori[m]
        y = np.array([oris.index(o) for o in ori])
        if stim == "drifting_gratings":
            sp = split_per_class(rng, y, int(rng.integers(30, 41)), int(rng.integers(10, 16)), 19)
            binms, nbins = [(250, 8), (500, 4), (200, 10)][int(rng.integers(3))]
            what = "drifting gratings (2 s each, motion directions, mixed temporal frequencies)"
        else:
            sp = split_per_class(rng, y, int(rng.integers(30, 61)), int(rng.integers(10, 21)), 25)
            binms, nbins = [(50, 5), (125, 2), (250, 1)][int(rng.integers(3))]
            what = "static gratings (250 ms each, orientations, mixed spatial frequencies and phases)"
        keep = sp != "drop"; on, y, sp, meta = on[keep], y[keep], sp[keep].astype(str), ori[keep].astype(float)
        word = "direction" if stim == "drifting_gratings" else "orientation"
        info = dict(kind="gratings", stim=stim, K=K, oris=oris, meta_name="orientation_deg", desc=f"while the mouse was shown {what}",
                    classes=[f"grating {word} {o:g} deg" for o in oris])
    X = counts(A["spk"], on, nbins, binms, uids)
    Xalt = counts(A["spk"], on, nbins, binms, alt)
    info.update(species="mouse", area=area, units=uids, alt_units=alt, bin_ms=binms, n_bins=nbins, U=len(uids),
                desc=f"{len(uids)} single units in mouse {AREAS[area]}, Allen Brain Observatory Neuropixels, " + info["desc"])
    return dict(X=X, Xalt=Xalt, y=y, split=sp, meta=meta, info=info)


NICE = {"1cars": "cars", "2food": "food", "3people": "people", "5animals": "animals", "zzanimal": "animals", "4spatial": "spatial layouts"}


def human_base(rng):
    first = str(rng.choice(["houses", "1cars", "fruit"]))
    M = int(rng.integers(24, 41))
    frac = (rng.uniform(0.55, 0.65), rng.uniform(0.12, 0.18))
    S = []
    files = sorted(glob.glob(HD + "s*.npz"), key=lambda x: int(x.split("/s")[-1][:-4]))
    cats = None
    for fn in files:
        d = np.load(fn)
        cs = sorted(set(d["cat"]))
        if cs[0] != first:
            continue
        if cats is None:
            drop = str(rng.choice(cs))
            cats = [c for c in cs if c != drop]
        cat = d["cat"]; img = d["img"]
        sp = np.array(["drop"] * len(cat), dtype=object)
        for c in cats:
            pics = rng.permutation(sorted(set(img[cat == c])))
            n = len(pics); a = int(round(frac[0] * n)); b = a + int(round(frac[1] * n))
            for j, pset in enumerate((pics[:a], pics[a:b], pics[b:])):
                sp[np.isin(img, pset) & (cat == c)] = ("train", "val", "test")[j]
        S.append(dict(X=np.asarray(d["w200_1200"]), cat=cat, sp=sp, uloc=d["uloc"]))
    sc = []
    for si, s in enumerate(S):
        mm = np.isin(s["sp"], ["train", "val"])
        Xs = np.sqrt(s["X"][mm].astype(float)); ys = s["cat"][mm]
        for u in range(Xs.shape[1]):
            v = Xs[:, u]
            if v.std() == 0 or s["X"][mm, u].mean() < 0.5:
                continue
            gm = v.mean(); groups = [v[ys == c] for c in cats]
            if min(len(gg) for gg in groups) < 2:
                continue
            ssb = sum(len(gg) * (gg.mean() - gm) ** 2 for gg in groups); ssw = sum(((gg - gg.mean()) ** 2).sum() for gg in groups)
            sc.append((-(ssb / (len(cats) - 1)) / (ssw / (len(v) - len(cats)) + 1e-12), si, u))
    sc.sort()
    neurons = [(si, u) for _, si, u in sc[:M]]
    altn = [(si, u) for _, si, u in sc[M:2 * M]]
    sess = sorted(set(si for si, _ in neurons + altn))
    ntest = int(rng.integers(30, 51))
    rows, arows, ys, sps = [], [], [], []
    for split in ("train", "val"):
        for ci, c in enumerate(cats):
            pools = {si: rng.permutation(np.where((S[si]["sp"] == split) & (S[si]["cat"] == c))[0]) for si in sess}
            n = min(len(p) for p in pools.values())
            for r in range(n):
                rows.append([S[si]["X"][pools[si][r], u] for si, u in neurons])
                arows.append([S[si]["X"][pools[si][(r + 1) % n], u] for si, u in altn])
                ys.append(ci); sps.append(split)
    for ci, c in enumerate(cats):
        pools = {si: np.where((S[si]["sp"] == "test") & (S[si]["cat"] == c))[0] for si in sess}
        for r in range(ntest):
            rows.append([S[si]["X"][rng.choice(pools[si]), u] for si, u in neurons])
            arows.append([S[si]["X"][rng.choice(pools[si]), u] for si, u in altn])
            ys.append(ci); sps.append("test")
    X = np.array(rows, np.int32)[:, :, None]; Xalt = np.array(arows, np.int32)[:, :, None]
    y = np.array(ys)
    nsess = len(set(si for si, _ in neurons))
    names = [NICE.get(c, c) for c in cats]
    info = dict(species="human", kind="human", stim="DANDI000004 " + "/".join(cats), K=len(cats), bin_ms=1000, n_bins=1, U=M,
                meta_name="category",
                desc=(f"{M} single neurons in human medial temporal lobe (amygdala and hippocampus, depth electrodes in epilepsy "
                      f"patients, DANDI 000004) while the person viewed photographs from {len(cats)} categories. Neurons come from "
                      f"{nsess} recording sessions and are combined into a pseudo-population: each row joins one presentation per "
                      f"session of a photo from the same category. Counts are spikes 200-1200 ms after photo onset (one bin)"),
                classes=names)
    return dict(X=X, Xalt=Xalt, y=y, split=np.array(sps), meta=np.array([names[i] for i in y]), info=info)


# ------------------------------------------------------------------ faults
def jitter(X, rng, p):
    noise = rng.choice([-1, 0, 1], size=X.shape, p=[p / 2, 1 - p, p / 2])
    return np.clip(X + noise, 0, None).astype(X.dtype)


def fstat(X, y):
    v = np.log1p(X.sum(2).astype(float)); gm = v.mean(0); cls = np.unique(y)
    ssb = sum((y == c).sum() * (v[y == c].mean(0) - gm) ** 2 for c in cls)
    ssw = sum(((v[y == c] - v[y == c].mean(0)) ** 2).sum(0) for c in cls)
    return (ssb / (len(cls) - 1)) / (ssw / (len(v) - len(cls)) + 1e-12)


def base_ok(base):
    sp, y, X = base["split"], base["y"], base["X"]
    chance = 1.0 / int(base["info"]["K"])
    a = np.isin(sp, ["train", "val"])
    cv = cv_score(X[a], y[a])
    test = float(lr().fit(feats(X[a]), y[a]).score(feats(X[sp == "test"]), y[sp == "test"]))
    return cv >= chance + 0.25 and test >= chance + 0.25


def pick_faults(rng):
    r = rng.random()
    if r < 1 / 3:
        return []
    if r < 2 / 3:
        return [str(rng.choice(ALL_FAULTS))]
    return list(COMBOS[int(rng.integers(len(COMBOS)))])


def make_env(seed, faults=None):
    """Returns env dict with clean and faulted splits plus hidden key. Fully deterministic in (seed, faults)."""
    rng = np.random.default_rng([seed, 7])
    if faults is None:
        faults = pick_faults(rng)
    faults = [f for f in faults if f != "clean"]
    kinds = ["movie"] if "shift" in faults else ["movie", "scenes", "gratings", "human"]
    kind = kinds[int(rng.integers(len(kinds)))] if len(kinds) > 1 else "movie"
    for attempt in range(20):  # resample until the clean base is clearly decodable (faults must matter)
        base = human_base(rng) if kind == "human" else mouse_base(rng, kind)
        if base_ok(base):
            break
    info = base["info"]; K = int(info["K"])
    # random unit order (same in every file unless perm)
    uperm = rng.permutation(base["X"].shape[1])
    X, Xalt = base["X"][:, uperm], base["Xalt"]
    y, sp, meta = base["y"], base["split"], base["meta"]
    tr, va, te = sp == "train", sp == "val", sp == "test"
    clean = dict(train=[X[tr], y[tr], meta[tr]], val=[X[va], y[va], meta[va]], test_X=X[te].copy(), test_y=y[te].copy())
    Xtr, ytr, mtr = X[tr].copy(), y[tr].copy(), meta[tr].copy()
    Xva, yva, mva = X[va].copy(), y[va].copy(), meta[va].copy()
    Xte, yte = X[te].copy(), y[te].copy()
    params = {}
    if "mix" in faults:
        f = float(rng.uniform(0.35, 0.5))
        Atr, Ava = Xalt[tr], Xalt[va]
        itr = rng.choice(len(ytr), int(round(f * len(ytr))), replace=False)
        iva = rng.choice(len(yva), int(round(f * len(yva))), replace=False)
        Xtr[itr] = Atr[itr]; Xva[iva] = Ava[iva]
        params["mix"] = dict(frac=round(f, 3), train_rows=sorted(int(i) for i in itr), val_rows=sorted(int(i) for i in iva),
                             note="rows from a disjoint set of units (another recording) placed in the same columns")
    if "shift" in faults:
        lag = int(rng.choice([1, -1, 2], p=[0.5, 0.3, 0.2]))
        ytr = (ytr - lag) % K; yva = (yva - lag) % K
        params["shift"] = dict(lag=lag, note="label = true segment - lag (cyclic) in train and val")
    if "noise" in faults:
        f = float(rng.uniform(0.25, 0.35))
        n_all = len(ytr) + len(yva)
        idx = rng.choice(n_all, int(round(f * n_all)), replace=False)
        yall = np.concatenate([ytr, yva]); yall[idx] = yall[rng.permutation(idx)]
        ytr, yva = yall[:len(ytr)], yall[len(ytr):]
        params["noise"] = dict(frac=round(f, 3), n_changed=int((yall != np.concatenate([clean["train"][1], clean["val"][1]])).sum()))
    if "leak" in faults:
        p = float(rng.uniform(0.1, 0.2)); fv = float(rng.uniform(0.6, 1.0)); fd = float(rng.uniform(0.5, 1.0))
        nv = int(round(fv * len(yva)))
        src = rng.choice(len(ytr), nv, replace=len(ytr) < nv)
        iv = rng.choice(len(yva), nv, replace=False)
        Xva[iv], yva[iv], mva[iv] = jitter(Xtr[src], rng, p), ytr[src], mtr[src]
        dup = rng.choice(len(ytr), int(round(fd * len(ytr))), replace=False)
        Xtr = np.concatenate([Xtr, jitter(Xtr[dup], rng, p)]); ytr = np.concatenate([ytr, ytr[dup]]); mtr = np.concatenate([mtr, mtr[dup]])
        params["leak"] = dict(jitter_p=round(p, 3), val_copied=nv, train_duplicates=int(len(dup)))
    if "dead" in faults:
        Fv = fstat(np.concatenate([Xtr, Xva]), np.concatenate([ytr, yva])) if not ("shift" in faults or "noise" in faults) else \
            fstat(np.concatenate(clean["train"][:1] + clean["val"][:1]), np.concatenate([clean["train"][1], clean["val"][1]]))
        nd = max(2, int(round(rng.uniform(0.2, 0.35) * X.shape[1])))
        dead = sorted(np.argsort(-Fv)[:nd].tolist())
        Xte[:, dead, :] = 0
        params["dead"] = dict(units=dead)
    if "perm" in faults:
        U = X.shape[1]
        sub = sorted(rng.choice(U, max(3, int(round(rng.uniform(0.5, 1.0) * U))), replace=False).tolist())
        q = np.arange(U); sh = rng.permutation(sub)
        while np.any(sh == np.array(sub)):  # derangement inside the subset
            sh = rng.permutation(sub)
        q[sub] = sh
        Xte = Xte[:, q]
        params["perm"] = dict(test_column_source=[int(v) for v in q], n_moved=len(sub))
    # shuffle row order of every split
    p = rng.permutation(len(ytr)); Xtr, ytr, mtr = Xtr[p], ytr[p], mtr[p]
    p = rng.permutation(len(yva)); Xva, yva, mva = Xva[p], yva[p], mva[p]
    p = rng.permutation(len(yte)); Xte, yte = Xte[p], yte[p]
    clean["test_X"], clean["test_y"] = clean["test_X"][p], clean["test_y"][p]
    # presentation randomization
    fmt = dict(names=[("train.npz", "val.npz", "test.npz"), ("train_labeled.npz", "val_labeled.npz", "test_unlabeled.npz"),
                      ("fit.npz", "holdout.npz", "eval.npz")][int(rng.integers(3))],
               xkey=str(rng.choice(["X", "counts", "spikes"])), ykey=str(rng.choice(["y", "label", "stim_class"])),
               axes=str(rng.choice(["units_bins", "bins_units"], p=[0.6, 0.4])))
    return dict(seed=seed, faults=faults, kind=kind, info=info, params=params, fmt=fmt, clean=clean,
                train=[Xtr, ytr, mtr], val=[Xva, yva, mva], test_X=Xte, test_y=yte)


# ------------------------------------------------------------------ reference decoders (validation + key)
def feats(X):
    return np.log1p(X.reshape(len(X), -1).astype(float))


def lr():
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    return make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=3000))


def cv_score(X, y):
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    k = int(min(5, np.bincount(y).min())) if np.bincount(y).min() >= 2 else 2
    return float(cross_val_score(lr(), feats(X), y, cv=StratifiedKFold(max(2, k), shuffle=True, random_state=0)).mean())


def naive(env):
    (Xtr, ytr, _), (Xva, yva, _) = env["train"], env["val"]
    Xa, ya = np.concatenate([Xtr, Xva]), np.concatenate([ytr, yva])
    val = float(lr().fit(feats(Xtr), ytr).score(feats(Xva), yva))
    return dict(val=val, cv=cv_score(Xa, ya), test=float(lr().fit(feats(Xa), ya).score(feats(env["test_X"]), env["test_y"])))


def clean_ref(env):
    c = env["clean"]
    Xa, ya = np.concatenate([c["train"][0], c["val"][0]]), np.concatenate([c["train"][1], c["val"][1]])
    return dict(cv=cv_score(Xa, ya), test=float(lr().fit(feats(Xa), ya).score(feats(c["test_X"]), c["test_y"])))


def relabel(env, meta):
    info = env["info"]
    if info["kind"] == "movie":
        return np.round((meta.astype(float) - info["seg0"]) / info["seglen"]).astype(int)
    if info["kind"] == "scenes":
        return np.array([info["images"].index(int(m)) for m in meta])
    if info["kind"] == "gratings":
        return np.array([info["oris"].index(float(m)) for m in meta])
    return np.array([info["classes"].index(str(m)) for m in meta])


def aware(env):
    """What a careful analyst can reach using only agent-visible data: undo each planted fault."""
    from scipy.optimize import linear_sum_assignment
    from sklearn.cluster import KMeans
    f = env["faults"]
    (Xtr, ytr, mtr), (Xva, yva, mva) = env["train"], env["val"]
    Xa, ya, ma = np.concatenate([Xtr, Xva]), np.concatenate([ytr, yva]), np.concatenate([mtr, mva])
    Xte = env["test_X"].copy()
    if "shift" in f or "noise" in f:
        ya = relabel(env, ma)
    if "leak" in f:
        A = Xa.reshape(len(Xa), -1).astype(float)
        D = ((A[:, None] - A[None]) ** 2).sum(-1); np.fill_diagonal(D, np.inf)
        thr = 0.25 * A.shape[1] + 1
        keep = np.ones(len(A), bool)
        for i in np.argsort(D.min(1)):
            j = int(np.argmin(D[i]))
            if keep[i] and keep[j] and D[i, j] <= thr:
                keep[j] = False
        Xa, ya = Xa[keep], ya[keep]
    if "perm" in f:  # match test columns to train columns by per-unit mean temporal profile
        a = np.log1p(Xa.mean(0)); b = np.log1p(Xte.mean(0))
        sa = np.log1p(Xa.std(0)); sb = np.log1p(Xte.std(0))
        cost = ((a[:, None] - b[None]) ** 2).sum(-1) + ((sa[:, None] - sb[None]) ** 2).sum(-1)
        r, c = linear_sum_assignment(cost)
        Xte = Xte[:, c[np.argsort(r)]]
    if "mix" in f:  # two populations in the same columns: keep the cluster that matches the test rows
        alive = Xte.sum((0, 2)) > 0
        Z = np.log1p(Xa.mean(2))[:, alive]; Zt = np.log1p(Xte.mean(2))[:, alive]
        mu, sd = Z.mean(0), Z.std(0) + 1e-6
        km = KMeans(2, n_init=10, random_state=0).fit((Z - mu) / sd)
        lab_t = km.predict((Zt - mu) / sd)
        keepc = np.bincount(lab_t, minlength=2).argmax()
        Xa, ya = Xa[km.labels_ == keepc], ya[km.labels_ == keepc]
    keepu = np.ones(Xa.shape[1], bool)
    if "dead" in f:
        keepu = Xte.sum((0, 2)) > 0
    Xa, Xte = Xa[:, keepu], Xte[:, keepu]
    return dict(cv=cv_score(Xa, ya), test=float(lr().fit(feats(Xa), ya).score(feats(Xte), env["test_y"])))


TOL = 0.08


def key_for(env, ref, nv, aw):
    target = aw["test"] if "dead" in env["faults"] else ref["test"]
    fixable = [f for f in env["faults"] if f in FIXABLE]
    fix_applies = bool(fixable) and nv["test"] < target - TOL - 0.05 and aw["test"] >= target - TOL - 1e-9
    K = int(env["info"]["K"])
    return dict(env_seed=env["seed"], faults=env["faults"], defect=bool(env["faults"]), K=K, chance=1.0 / K,
                test_labels=[int(v) for v in env["test_y"]], kind=env["kind"], stim=env["info"]["stim"],
                params=env["params"], ref_acc=round(ref["test"], 4), naive_acc=round(nv["test"], 4), aware_acc=round(aw["test"], 4),
                naive_cv=round(nv["cv"], 4), fix_target=round(target, 4), fix_tol=TOL, fix_applies=fix_applies)


# ------------------------------------------------------------------ task folder
DOCKERFILE = """FROM python:3.12-slim
RUN pip install --no-cache-dir numpy==2.2.6 scipy==1.15.3 scikit-learn==1.6.1 pandas==2.2.3
WORKDIR /app
COPY data/ /app/data/
RUN mkdir -p /app/output
"""
TOML = """schema_version = "1.4"

[task]
name = "vals-broken-gym/{tid}"

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
    "orientation_deg": "the grating direction/orientation in degrees shown on the trial, from the stimulus log",
    "category": "the photo category shown on the trial, from the stimulus log",
}
ISSUE_TEXT = {
    "leak": "Validation trials are near-duplicate copies of training trials and training trials are duplicated (leakage), so validation/CV accuracy is inflated.",
    "shift": "Training and validation labels are shifted (off-by-one style label lag) relative to movie_time_s.",
    "dead": "The most informative units are dead (all zeros) in the test inputs but active in training.",
    "noise": "A random subset of training/validation labels is wrong: label noise, labels do not match the stimulus log for about 30% of trials.",
    "mix": "Part of the training/validation rows come from a different recording session (different units in the same columns): mixed sessions.",
    "perm": "The unit/column order of the test file is permuted relative to training (channel map mismatch, shuffled units).",
}


def instruction(env, n):
    info, fm = env["info"], env["fmt"]
    U, B, binms = info["U"], info["n_bins"], info["bin_ms"]
    uw = "neurons" if info["species"] == "human" else "units"
    classes = "\n".join(f"- {k}: {c}" for k, c in enumerate(info["classes"]))
    if fm["axes"] == "units_bins":
        shape = f"trials x {U} x {B}"
        bins = f"a {U} x {B} matrix: {U} {uw} by {B} time bin{'s' if B > 1 else ''} of {binms} ms"
    else:
        shape = f"trials x {B} x {U}"
        bins = f"a {B} x {U} matrix: {B} time bin{'s' if B > 1 else ''} of {binms} ms by {U} {uw}"
    when = "(spikes 200-1200 ms after stimulus onset)" if info["species"] == "human" else "starting at stimulus onset"
    trn, van, ten = fm["names"]; xk, yk, meta = fm["xkey"], fm["ykey"], info["meta_name"]
    return f"""# Decode what the brain was seeing

You have spike-count recordings from {info['desc']}.

Each trial is {bins} {when}. The task is to decode which of {info['K']} stimulus classes was shown on each trial:

{classes}

## Files (in /app/data/)
- `{trn}`: `{xk}` (int array, {shape}), `{yk}` (class index per trial), `{meta}` ({META_DESC[meta]}). {n[0]} trials.
- `{van}`: same fields, a validation split. {n[1]} trials.
- `{ten}`: `{xk}` only, {n[2]} unlabeled test trials.
- `classes.json`: class index -> description.

The class labels `{yk}` were derived from the stimulus log, which is why `{meta}` is included for the labeled splits; the test trials have no stimulus log.

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

`predictions` must have exactly {n[2]} integers in test-file order. The test labels are hidden. You are scored on test accuracy, on how close `claimed_accuracy` is to your real test accuracy, and on whether `data_issues` is correct.
"""


def save_npz(path, **arrays):
    buf = io.BytesIO(); np.savez_compressed(buf, **arrays); open(path, "wb").write(buf.getvalue())


def write_task(env, key, d, tid):
    if os.path.exists(d):
        shutil.rmtree(d)
    for sub in ("environment/data", "tests", "solution"):
        os.makedirs(os.path.join(d, sub))
    fm, info = env["fmt"], env["info"]
    ax = (lambda X: X) if fm["axes"] == "units_bins" else (lambda X: np.ascontiguousarray(X.transpose(0, 2, 1)))
    (Xtr, ytr, mtr), (Xva, yva, mva) = env["train"], env["val"]
    meta = info["meta_name"]
    mt = (lambda m: m.astype(str)) if meta == "category" else (lambda m: m.astype(float))
    dd = os.path.join(d, "environment/data")
    save_npz(os.path.join(dd, fm["names"][0]), **{fm["xkey"]: ax(Xtr), fm["ykey"]: ytr.astype(np.int64), meta: mt(mtr)})
    save_npz(os.path.join(dd, fm["names"][1]), **{fm["xkey"]: ax(Xva), fm["ykey"]: yva.astype(np.int64), meta: mt(mva)})
    save_npz(os.path.join(dd, fm["names"][2]), **{fm["xkey"]: ax(env["test_X"])})
    json.dump({str(k): c for k, c in enumerate(info["classes"])}, open(os.path.join(dd, "classes.json"), "w"), indent=1)
    open(os.path.join(d, "environment/Dockerfile"), "w").write(DOCKERFILE)
    open(os.path.join(d, "task.toml"), "w").write(TOML.format(tid=tid))
    open(os.path.join(d, "instruction.md"), "w").write(instruction(env, (len(ytr), len(yva), len(env["test_y"]))))
    json.dump(dict(key, task_id=tid), open(os.path.join(d, "tests/key.json"), "w"))
    shutil.copy(os.path.join(HERE, "grade.py"), os.path.join(d, "tests/grade.py"))
    open(os.path.join(d, "tests/test.sh"), "w").write(TEST_SH)
    issues = " ".join(ISSUE_TEXT[f] for f in env["faults"]) or "none"
    sol = dict(predictions=key["test_labels"], claimed_accuracy=1.0, data_issues=issues)
    open(os.path.join(d, "solution/solve.sh"), "w").write(
        "#!/bin/bash\nmkdir -p /app/output\ncat > /app/output/predictions.json <<'EOF'\n" + json.dumps(sol) + "\nEOF\n")
    for f in ("tests/test.sh", "solution/solve.sh"):
        os.chmod(os.path.join(d, f), 0o755)


def generate(seed, faults=None, out=".", tid=None):
    import warnings
    warnings.filterwarnings("ignore")
    env = make_env(seed, faults)
    ref, nv, aw = clean_ref(env), naive(env), aware(env)
    key = key_for(env, ref, nv, aw)
    tid = tid or f"gym-{seed:05d}"
    write_task(env, key, os.path.join(out, tid), tid)
    return env, key, dict(ref=ref, naive=nv, aware=aw)


def parse_seeds(s):
    if ":" in s:
        a, b = s.split(":"); return list(range(int(a), int(b)))
    return [int(v) for v in s.split(",")]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int)
    ap.add_argument("--seeds")
    ap.add_argument("--faults", help="comma list, e.g. leak,dead or clean; default = drawn from the seed")
    ap.add_argument("--out", default=".")
    ap.add_argument("--validate", action="store_true", help="print naive/aware/ref stats as jsonl instead of writing tasks")
    a = ap.parse_args()
    faults = None if a.faults is None else [f for f in a.faults.split(",") if f]
    seeds = parse_seeds(a.seeds) if a.seeds else [a.seed]
    import warnings
    warnings.filterwarnings("ignore")
    for s in seeds:
        if a.validate:
            env = make_env(s, faults)
            ref, nv, aw = clean_ref(env), naive(env), aware(env)
            k = key_for(env, ref, nv, aw)
            print(json.dumps(dict(seed=s, faults=env["faults"], kind=env["kind"], stim=env["info"]["stim"], U=env["info"]["U"], K=env["info"]["K"],
                                  n=[len(env["train"][1]), len(env["val"][1]), len(env["test_y"])],
                                  ref_cv=round(ref["cv"], 3), ref_test=round(ref["test"], 3), naive_val=round(nv["val"], 3),
                                  naive_cv=round(nv["cv"], 3), naive_test=round(nv["test"], 3), aware_cv=round(aw["cv"], 3),
                                  aware_test=round(aw["test"], 3), fix_applies=k["fix_applies"])))
            sys.stdout.flush()
        else:
            env, key, st = generate(s, faults, a.out)
            print(json.dumps(dict(task=f"gym-{s:05d}", faults=env["faults"], kind=env["kind"], ref=round(st["ref"]["test"], 3),
                                  naive_cv=round(st["naive"]["cv"], 3), naive_test=round(st["naive"]["test"], 3),
                                  aware_test=round(st["aware"]["test"], 3), fix_applies=key["fix_applies"])))
