"""Extract base recordings for the broken-instrument tier.

Writes bases/<base_id>.npz with X (n, U, B) int counts, y (n,) class index, split (n,) in {train,val,test},
meta (n,) stimulus descriptor per trial (float or str), plus bases/<base_id>.json with the description.
Splits are fixed here so every downstream twin shares the same hidden test trials.

Mouse: Allen Visual Coding Neuropixels session 732592105 (local NWB). Human: DANDI 000004 per-session npz.
Run: ~/work/vals-mouse/.venv/bin/python extract.py
"""
import glob
import json
import os

import h5py
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "bases")
os.makedirs(OUT, exist_ok=True)
MD = os.path.expanduser("~/work/vals-mouse/data/")
HD = os.path.expanduser("~/work/vals-hyp/human-neurons/data004/")
SESSION = 732592105


# ---------------------------------------------------------------- mouse
def good_units():
    u = pd.read_csv(MD + "units.csv", usecols=["id", "ecephys_channel_id", "isi_violations", "amplitude_cutoff", "presence_ratio"])
    c = pd.read_csv(MD + "channels.csv", usecols=["id", "ecephys_probe_id", "ecephys_structure_acronym"]).rename(columns={"id": "ecephys_channel_id"})
    p = pd.read_csv(MD + "probes.csv", usecols=["id", "ecephys_session_id"]).rename(columns={"id": "ecephys_probe_id"})
    u = u.merge(c, on="ecephys_channel_id").merge(p, on="ecephys_probe_id")
    m = (u.ecephys_session_id == SESSION) & (u.ecephys_structure_acronym == "VISp") & (u.isi_violations < 0.5) & (u.amplitude_cutoff < 0.1) & (u.presence_ratio > 0.95)
    return set(int(x) for x in u[m].id.values)


F = h5py.File(MD + f"session_{SESSION}.nwb", "r")
GOOD = good_units()
ids = F["units/id"][:]
st = F["units/spike_times"][:]
sti = F["units/spike_times_index"][:]
starts = np.concatenate([[0], sti[:-1]])
SPK = {int(i): st[s:e] for i, s, e in zip(ids, starts, sti) if int(i) in GOOD}
UIDS = sorted(SPK)
print("good VISp units", len(UIDS))


def counts(onsets, nbins, binms, uids):
    edges = np.arange(nbins + 1) * binms / 1000.0
    X = np.zeros((len(onsets), len(uids), nbins), np.int32)
    for k, uid in enumerate(uids):
        e = onsets[:, None] + edges[None, :]
        X[:, k] = np.diff(np.searchsorted(SPK[uid], e), axis=1)
    return X


def pick_units(rng, n):
    return sorted(rng.choice(UIDS, n, replace=False).tolist())


def save(base_id, X, y, split, meta, info):
    np.savez_compressed(os.path.join(OUT, base_id + ".npz"), X=X, y=y, split=split, meta=meta)
    info = dict(info, base_id=base_id, n=int(len(y)), shape=list(X.shape),
                n_split={s: int((split == s).sum()) for s in ("train", "val", "test")})
    json.dump(info, open(os.path.join(OUT, base_id + ".json"), "w"), indent=1)
    print(base_id, info["n_split"], X.shape)


def movie_base(base_id, stim, seg0, K, nunits, rep_split, seed):
    rng = np.random.default_rng(seed)
    g = F[f"intervals/{stim}_presentations"]
    t0 = g["start_time"][:]
    fr = g["frame"][:].astype(int)
    blk = g["stimulus_block"][:]
    nframes = fr.max() + 1
    rs = np.where(fr == 0)[0]
    reps = [(t0[i0 + np.arange(seg0, seg0 + K) * 30], blk[i0]) for i0 in rs
            if i0 + nframes <= len(fr) and np.all(fr[i0:i0 + nframes] == np.arange(nframes))]
    R = len(reps)
    # assign repeats to splits, interleaved across stimulus blocks
    blocks = sorted(set(b for _, b in reps))
    order = []
    for b in blocks:
        order += list(rng.permutation([r for r in range(R) if reps[r][1] == b]))
    lab = (["train"] * rep_split[0] + ["val"] * rep_split[1] + ["test"] * rep_split[2])
    # round-robin over blocks so each split mixes blocks
    per_block = [list(rng.permutation([r for r in range(R) if reps[r][1] == b])) for b in blocks]
    seq = []
    while any(per_block):
        for pb in per_block:
            if pb:
                seq.append(pb.pop())
    rep_lab = {}
    tot = sum(rep_split)
    assert tot == R, (R, rep_split)
    # spread: take seq and map with lab pattern interleaved
    pattern = []
    for i in range(tot):
        pattern.append(lab[i])
    pattern = list(rng.permutation(pattern))
    for r, s in zip(seq, pattern):
        rep_lab[r] = s
    uids = pick_units(rng, nunits)
    on, y, sp, meta = [], [], [], []
    for r in range(R):
        for k in range(K):
            on.append(reps[r][0][k]); y.append(k); sp.append(rep_lab[r]); meta.append(float(seg0 + k))
    X = counts(np.array(on), 10, 100, uids)
    info = dict(species="mouse", stim=stim, kind="movie", K=K, seg0=seg0, bin_ms=100, n_bins=10, units=uids,
                rep_split=list(rep_split), rep_labels={int(r): s for r, s in rep_lab.items()},
                meta_name="movie_time_s",
                desc=(f"{len(uids)} single units in mouse primary visual cortex (VISp), Allen Brain Observatory Neuropixels, "
                      f"while the mouse watched a natural movie that was replayed {R} times"),
                classes=[f"movie seconds {seg0 + k}-{seg0 + k + 1} (1-second segment starting at movie time {seg0 + k} s)" for k in range(K)])
    save(base_id, X, np.array(y), np.array(sp), np.array(meta), info)


def split_per_class(rng, y, ntr, nva, nte):
    sp = np.array(["drop"] * len(y), dtype=object)
    for c in np.unique(y):
        ix = rng.permutation(np.where(y == c)[0])
        sp[ix[:ntr]] = "train"; sp[ix[ntr:ntr + nva]] = "val"; sp[ix[ntr + nva:ntr + nva + nte]] = "test"
    return sp


def scenes_base(base_id, K, nunits, seed):
    rng = np.random.default_rng(seed)
    g = F["intervals/natural_scenes_presentations"]
    t0 = g["start_time"][:]; fr = g["frame"][:].astype(int)
    imgs = sorted(rng.choice(np.arange(118), K, replace=False).tolist())
    m = np.isin(fr, imgs)
    on = t0[m]; img = fr[m]
    y = np.array([imgs.index(i) for i in img])
    sp = split_per_class(rng, y, 25, 10, 15)
    keep = sp != "drop"
    uids = pick_units(rng, nunits)
    X = counts(on[keep], 5, 50, uids)
    info = dict(species="mouse", stim="natural_scenes", kind="images", K=K, images=imgs, bin_ms=50, n_bins=5, units=uids,
                meta_name="image_id",
                desc=(f"{len(uids)} single units in mouse primary visual cortex (VISp), Allen Brain Observatory Neuropixels, "
                      f"while the mouse was shown flashed natural photographs (250 ms each, random order)"),
                classes=[f"natural scene image #{i} of the Allen natural-scenes set" for i in imgs])
    save(base_id, X, y[keep], sp[keep].astype(str), img[keep].astype(float), info)


def gratings_base(base_id, stim, nunits, per_class, nbins, binms, seed):
    rng = np.random.default_rng(seed)
    g = F[f"intervals/{stim}_presentations"]
    t0 = g["start_time"][:]; ori = g["orientation"][:]
    m = ~np.isnan(ori)
    on = t0[m]; ori = ori[m]
    oris = sorted(set(ori.tolist()))
    y = np.array([oris.index(o) for o in ori])
    sp = split_per_class(rng, y, *per_class)
    keep = sp != "drop"
    uids = pick_units(rng, nunits)
    X = counts(on[keep], nbins, binms, uids)
    what = "drifting gratings (2 s each, 8 motion directions, mixed temporal frequencies)" if stim == "drifting_gratings" else \
        "static gratings (250 ms each, 6 orientations, mixed spatial frequencies and phases)"
    info = dict(species="mouse", stim=stim, kind="gratings", K=len(oris), oris=oris, bin_ms=binms, n_bins=nbins, units=uids,
                meta_name="orientation_deg",
                desc=(f"{len(uids)} single units in mouse primary visual cortex (VISp), Allen Brain Observatory Neuropixels, "
                      f"while the mouse was shown {what}"),
                classes=[f"grating {'direction' if stim == 'drifting_gratings' else 'orientation'} {o:g} deg" for o in oris])
    save(base_id, X, y[keep], sp[keep].astype(str), ori[keep].astype(float), info)


# ---------------------------------------------------------------- human
DROP = {"houses": "landscapes", "1cars": "4spatial", "fruit": "space"}
NICE = {"1cars": "cars", "2food": "food", "3people": "people", "5animals": "animals", "zzanimal": "animals"}


def human_base(base_id, first_cat, M, seed, frac=(0.6, 0.15, 0.25), n_test_per_class=50):
    rng = np.random.default_rng(seed)
    S = []
    for fn in sorted(glob.glob(HD + "s*.npz"), key=lambda x: int(x.split("/s")[-1][:-4])):
        d = np.load(fn)
        cs = sorted(set(d["cat"]))
        if cs[0] != first_cat:
            continue
        keep = [c for c in cs if c != DROP[first_cat]]
        cat = d["cat"]; img = d["img"]
        sp = np.array(["drop"] * len(cat), dtype=object)
        for c in keep:  # split by picture so a picture never spans splits
            pics = rng.permutation(sorted(set(img[cat == c])))
            n = len(pics); a = int(round(frac[0] * n)); b = a + int(round(frac[1] * n))
            for j, pset in enumerate((pics[:a], pics[a:b], pics[b:])):
                sp[np.isin(img, pset) & (cat == c)] = ("train", "val", "test")[j]
        S.append(dict(X=np.asarray(d["w200_1200"]), cat=cat, sp=sp, uloc=d["uloc"]))
    classes = keep
    # select M units by one-way ANOVA F on train+val trials only
    sc = []
    for si, s in enumerate(S):
        mm = np.isin(s["sp"], ["train", "val"])
        Xs = np.sqrt(s["X"][mm].astype(float)); ys = s["cat"][mm]
        for u in range(Xs.shape[1]):
            v = Xs[:, u]
            if v.std() == 0 or s["X"][mm, u].mean() < 0.5:
                continue
            gm = v.mean(); groups = [v[ys == c] for c in classes]
            ssb = sum(len(gg) * (gg.mean() - gm) ** 2 for gg in groups); ssw = sum(((gg - gg.mean()) ** 2).sum() for gg in groups)
            Fst = (ssb / (len(classes) - 1)) / (ssw / (len(v) - len(classes)) + 1e-12)
            sc.append((-Fst, si, u))
    sc.sort()
    neurons = [(si, u) for _, si, u in sc[:M]]
    sess = sorted(set(si for si, _ in neurons))
    rows, ys, sps = [], [], []
    for split in ("train", "val"):
        for ci, c in enumerate(classes):
            # no trial reuse within a split: rows = min available
            pools = {si: rng.permutation(np.where((S[si]["sp"] == split) & (S[si]["cat"] == c))[0]) for si in sess}
            n = min(len(p) for p in pools.values())
            for r in range(n):
                rows.append([S[si]["X"][pools[si][r], u] for si, u in neurons]); ys.append(ci); sps.append(split)
    for ci, c in enumerate(classes):  # test rows: independent draws from test-only trials
        pools = {si: np.where((S[si]["sp"] == "test") & (S[si]["cat"] == c))[0] for si in sess}
        for r in range(n_test_per_class):
            rows.append([S[si]["X"][rng.choice(pools[si]), u] for si, u in neurons]); ys.append(ci); sps.append("test")
    X = np.array(rows, np.int32)[:, :, None]  # one 1000 ms bin
    y = np.array(ys)
    regions = [str(S[si]["uloc"][u]) for si, u in neurons]
    info = dict(species="human", stim="DANDI000004 " + "/".join(classes), kind="images", K=len(classes), bin_ms=1000, n_bins=1,
                n_sessions=len(sess), regions=regions, meta_name="category",
                desc=(f"{M} single neurons in human medial temporal lobe (amygdala and hippocampus, depth electrodes in epilepsy patients, "
                      f"DANDI 000004) while the person viewed photographs from {len(classes)} categories. Neurons come from {len(sess)} "
                      f"recording sessions and are combined into a pseudo-population: each row joins one presentation per session of a "
                      f"photo from the same category. Counts are spikes 200-1200 ms after photo onset (one bin)"),
                classes=[NICE.get(c, c) for c in classes])
    save(base_id, X, y, np.array(sps), np.array([NICE.get(classes[i], classes[i]) for i in y]), info)


if __name__ == "__main__":
    movie_base("m-nm1-a", "natural_movie_one", 0, 10, 20, (12, 3, 5), 1)
    movie_base("m-nm1-b", "natural_movie_one", 10, 10, 20, (12, 3, 5), 2)
    movie_base("m-nm1-c", "natural_movie_one", 20, 10, 20, (12, 3, 5), 3)
    movie_base("m-nm3-a", "natural_movie_three", 60, 10, 25, (6, 1, 3), 4)
    scenes_base("m-ns-a", 8, 25, 5)
    scenes_base("m-ns-b", 8, 25, 6)
    gratings_base("m-dg", "drifting_gratings", 30, (40, 15, 19), 8, 250, 7)
    scenes_base("m-ns-c", 8, 25, 9)
    human_base("h-houses-b", "houses", 30, 15)
    human_base("h-cars-b", "1cars", 30, 12)
    human_base("h-houses", "houses", 40, 13)
    human_base("h-fruit", "fruit", 40, 14)
