import os, glob, json, numpy as np, mne, scipy.signal as ss
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.neighbors import KNeighborsClassifier
mne.set_log_level("ERROR")
D = os.path.expanduser("~/work/vals-hyp/_data/mne/physionet-sleep-data"); OUT = os.path.expanduser("~/work/vals-hyp/sleep")
FS = 50; L = "ABCD"
NAMES = ["Wake", "Light sleep (N1 or N2)", "Deep sleep (N3, slow-wave)", "REM sleep"]
MAP = {"Sleep stage W": 0, "Sleep stage 1": 1, "Sleep stage 2": 1, "Sleep stage 3": 2, "Sleep stage 4": 2, "Sleep stage R": 3}

subjects = {}
for psg in sorted(glob.glob(D + "/SC4*PSG.edf")):
    sid = os.path.basename(psg)[:5]
    hyp = glob.glob(D + f"/{sid}*Hypnogram.edf")
    if not hyp: continue
    raw = mne.io.read_raw_edf(psg, include=["EEG Fpz-Cz"], preload=True)
    ann = mne.read_annotations(hyp[0]); raw.set_annotations(ann, emit_warning=False)
    sf = raw.info["sfreq"]; x = raw.get_data()[0] * 1e6  # microvolts
    ep = []; lab = []
    for a in ann:
        if a["description"] not in MAP: continue
        for k in range(int(a["duration"] // 30)):
            s = int((a["onset"] + 30 * k) * sf)
            if s + int(30 * sf) <= len(x): ep.append(s); lab.append(MAP[a["description"]])
    ep = np.array(ep); lab = np.array(lab)
    sl = np.where(lab != 0)[0]
    keep = (ep >= ep[sl[0]] - int(1800 * sf)) & (ep <= ep[sl[-1]] + int(1800 * sf))  # wake within 30 min of the sleep period
    segs = np.stack([ss.resample_poly(x[s:s + int(30 * sf)], FS, int(sf)) for s in ep[keep]])
    subjects[sid] = (np.round(segs).astype(int), lab[keep])
    print(sid, np.bincount(lab[keep], minlength=4))

sids = sorted(subjects); rng = np.random.default_rng(11)
test_s = sids[:4]; train_s = sids[4:]
print("item subjects", test_s, "train subjects", train_s)


def feats(y):
    f, P = ss.welch(y.astype(float), FS, nperseg=FS * 4)
    bands = [(0.5, 2), (2, 4), (4, 8), (8, 12), (12, 16), (16, 25)]
    bp = np.array([P[(f >= a) & (f < b)].sum() for a, b in bands]); tot = bp.sum()
    return list(np.log(bp + 1e-9)) + list(bp / tot) + [np.log(y.std() + 1e-9), np.abs(np.diff(y)).mean() / (y.std() + 1e-9)]


items = []
for c in range(4):
    pool = [(s, i) for s in test_s for i in np.where(subjects[s][1] == c)[0]]
    for j in rng.choice(len(pool), 10, replace=False): items.append((c,) + pool[j])
rng.shuffle(items)
shots = []
for c in range(4):
    s = train_s[c % len(train_s)]; i = rng.choice(np.where(subjects[s][1] == c)[0]); shots.append((c, s, i))


def ser(v): return " ".join(map(str, v))


head = (f"Below is a raw 30-second scalp EEG recording (channel Fpz-Cz, healthy adult, overnight home recording), sampled at {FS} Hz, "
        "values in microvolts as integers, in time order.\n")
q = "Which sleep stage was the person in during these 30 seconds?\n" + "\n".join(f"{L[i]}) {NAMES[i]}" for i in range(4)) + "\nAnswer with a single letter."


def mk(fewshot, tag):
    ex = ""
    if fewshot:
        ex = "Labeled reference epochs from other people (same channel and format):\n"
        for c, s, i in shots: ex += f"[Example, label {L[c]}) {NAMES[c]}]\n{ser(subjects[s][0][i])}\n"
        ex += "\nNow the epoch to classify:\n"
    rows = []
    for k, (c, s, i) in enumerate(items):
        rows.append({"id": f"sleep-{tag}-{k:03d}", "hypothesis": "sleep", "prompt": head + ex + ser(subjects[s][0][i]) + "\n\n" + q,
                     "answer": L[c], "choices": {L[j]: NAMES[j] for j in range(4)},
                     "meta": {"subject_night": s, "epoch_index": int(i), "label": NAMES[c], "fs": FS, "variant": tag}})
    return rows


zs = mk(False, "zs"); fsr = mk(True, "fs")
for fn, rows in [("tasks.jsonl", zs), ("tasks_fewshot.jsonl", fsr)]:
    with open(f"{OUT}/{fn}", "w") as f:
        for r in rows: f.write(json.dumps(r) + "\n")
    print(fn, len(rows), "max chars", max(len(r["prompt"]) for r in rows), "answers", {a: sum(r["answer"] == a for r in rows) for a in L})
open(f"{OUT}/examples/example_zeroshot.txt", "w").write(zs[0]["prompt"] + "\n\n[ANSWER] " + zs[0]["answer"])
open(f"{OUT}/examples/example_fewshot.txt", "w").write(fsr[1]["prompt"] + "\n\n[ANSWER] " + fsr[1]["answer"])

X = np.array([feats(e) for s in train_s for e in subjects[s][0]]); Y = np.concatenate([subjects[s][1] for s in train_s])
Xq = np.array([feats(subjects[s][0][i]) for c, s, i in items]); Yq = np.array([c for c, s, i in items])
lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, class_weight="balanced")).fit(X, Y)
res = {"LR_bandpower_heldout_subjects": float((lr.predict(Xq) == Yq).mean())}
pr = lr.predict(Xq); print("LR per-class", [round(float((pr[Yq == k] == k).mean()), 2) for k in range(4)])
Xs = np.array([feats(subjects[s][0][i]) for c, s, i in shots]); Ys = np.array([c for c, s, i in shots])
sc = StandardScaler().fit(Xs); knn = KNeighborsClassifier(1).fit(sc.transform(Xs), Ys)
res["1NN_bandpower_prompt_examples_only"] = float((knn.predict(sc.transform(Xq)) == Yq).mean())
print(json.dumps({k: round(v, 3) for k, v in res.items()}))
json.dump(res, open(f"{OUT}/baseline.json", "w"), indent=1)
