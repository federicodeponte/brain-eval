import json, glob, os, re, numpy as np, soundfile as sf
from scipy.signal import butter, sosfiltfilt
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
L = "ABCD"
CLS = ["small", "medium", "large", "outdoor"]
NAMES = {"small": "small enclosed room (bathroom, shower, bedroom, closet-sized room, car)",
         "medium": "medium indoor room (classroom, office, kitchen, living room, bar, restaurant, shop)",
         "large": "large indoor hall (atrium, auditorium, gym, supermarket, station, ballroom, theater, library hall)",
         "outdoor": "outdoors (street, forest, field, parking lot)"}
def label(n):
    s = n.lower()
    if re.search(r"outside|outdoor|streetsof|parkinglot|forest|field|backyard|frontyard|driveway|bikepath|porch|balcony|playground|skatepark", s) and "subway" not in s: return "outdoor"
    if re.search(r"bathroom|shower|\bcar\b|_car|bedroom|babysroom|pantry|closet|laundry|elevator|basementstorage|examinationroom", s) and "master" not in s: return "small"
    if re.search(r"atrium|auditorium|gym(?!_weight)|supermarket|supermerket|superMarket|station|ballroom|movietheater|publi[c]?library|departmentstore|mallfoodcourt|swimmingpool|cafeteria|dininghall|largebrick|infinitecorridor", s): return "large"
    if re.search(r"classroom|clasroom|office|kitchen|livingroom|diningroom|bar|restaurant|pizzeria|coffeeshop|icecream|sandwich|samdwich|toystore|bookstore|lounge|drycleaners|waitingroom|billiards|conferenceroom|meetingroom|artgallery|computerroom|dramaroom|masterbedroom|homefoyer|exercise", s): return "medium"
    return None
files = sorted(glob.glob("raw/mit/Audio/*.wav")); recs = []
for f in files:
    n = os.path.basename(f); c = label(n)
    if c is None: continue
    x, sr = sf.read(f); x = x if x.ndim == 1 else x[:, 0]
    recs.append(dict(f=f, n=n, c=c, x=x.astype(float), sr=sr))
def edc_db(x):
    e = np.cumsum((x ** 2)[::-1])[::-1]; return 10 * np.log10(e / e[0] + 1e-12)
def feats(x, sr):
    i0 = int(np.argmax(np.abs(x))); x = x[i0:]
    out = []
    for band in [None, (250, 500), (1000, 2000), (4000, 8000)]:
        y = x if band is None else sosfiltfilt(butter(4, band, btype="band", fs=sr, output="sos"), x)
        d = edc_db(y); t = np.arange(len(d)) / sr
        def tt(lv): k = np.argmax(d <= lv); return t[k] if k > 0 else t[-1]
        out += [tt(-10) * 6, (tt(-25) - tt(-5)) * 3]  # EDT, T20
    n50 = int(0.05 * sr); n3 = int(0.003 * sr)
    drr = 10 * np.log10((x[:n3] ** 2).sum() / ((x[n3:] ** 2).sum() + 1e-12) + 1e-12)
    c50 = 10 * np.log10((x[:n50] ** 2).sum() / ((x[n50:] ** 2).sum() + 1e-12) + 1e-12)
    return np.array(out + [drr, c50])
for r in recs: r["F"] = feats(r["x"], r["sr"])
rng = np.random.default_rng(9)
by = {c: [r for r in recs if r["c"] == c] for c in CLS}
print({c: len(v) for c, v in by.items()})
tests, pool = [], []
for c in CLS:
    v = list(by[c]); rng.shuffle(v); tests += v[:10]; pool += v[10:]
order = rng.permutation(len(tests)); tests = [tests[i] for i in order]
def text_curve(r):
    x = r["x"]; sr = r["sr"]; i0 = int(np.argmax(np.abs(x))); x = x[i0:]
    d = edc_db(x); step = int(0.02 * sr); n = min(len(d), int(2.0 * sr))
    pts = [f"{max(d[k], -80):.1f}" for k in range(0, n, step)]
    env = x[: int(0.06 * sr)] ** 2; seg = int(0.002 * sr)
    e = [10 * np.log10(env[k:k + seg].sum() / env[:seg].sum() + 1e-12) for k in range(0, len(env) - seg + 1, seg)]
    return pts, [f"{v:.1f}" for v in e]
legend = "\n".join(f"{L[i]} = {NAMES[c]}" for i, c in enumerate(CLS))
items = []
for i, r in enumerate(tests):
    pts, early = text_curve(r)
    wavname = f"audio/room-{i:03d}.wav"; sf.write(wavname, r["x"], r["sr"])
    p = ("A room impulse response (RIR) was recorded by playing a click-like sound and recording what came back at the microphone. "
         "Below are two summaries of it, both measured from the direct sound (time 0).\n\n"
         f"1) Schroeder energy-decay curve in dB (remaining energy relative to total), every 20 ms from 0 to {len(pts)*20-20} ms:\n" + ", ".join(pts) +
         "\n\n2) Early energy envelope in dB relative to the first 2 ms, every 2 ms from 0 to 58 ms:\n" + ", ".join(early) +
         f"\n\nWhere was it recorded?\n{legend}\n\nAnswer with a single letter.")
    items.append({"id": f"room-{i:03d}", "hypothesis": "hear-the-room", "prompt": p, "answer": L[CLS.index(r["c"])],
                  "choices": {L[j]: NAMES[c] for j, c in enumerate(CLS)},
                  "meta": {"source_file": r["n"], "class": r["c"], "audio_path": wavname, "sr": r["sr"],
                           "edt_s": round(float(r["F"][0]), 3), "t20_s": round(float(r["F"][1]), 3), "drr_db": round(float(r["F"][-2]), 2), "c50_db": round(float(r["F"][-1]), 2)}})
with open("tasks.jsonl", "w") as f:
    for o in items: f.write(json.dumps(o) + "\n")
for o in items[:2]: open(f"examples/{o['id']}.txt", "w").write(o["prompt"] + f"\n\n[answer: {o['answer']}]\n")
Xp = np.array([r["F"] for r in pool]); yp = np.array([r["c"] for r in pool]); Xt = np.array([r["F"] for r in tests]); yt = np.array([r["c"] for r in tests])
print("items", len(items), {l: sum(o["answer"] == l for o in items) for l in L}, "max chars", max(len(o["prompt"]) for o in items), "pool", len(pool))
lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, C=1.0)).fit(Xp, yp)
kn = make_pipeline(StandardScaler(), KNeighborsClassifier(5)).fit(Xp, yp)
print("logreg(EDT/T20 x4 bands + DRR + C50)", round(float((lr.predict(Xt) == yt).mean()), 3), "| knn5", round(float((kn.predict(Xt) == yt).mean()), 3))
# single-feature rule: broadband T20 thresholds fit on pool
t = Xp[:, 1]; med = {c: np.median(t[yp == c]) for c in CLS}
pred = [min(CLS, key=lambda c: abs(np.log(max(f[1], 1e-3)) - np.log(med[c]))) for f in Xt]
print("T20-only nearest class median", round(float((np.array(pred) == yt).mean()), 3), {c: round(float(v), 2) for c, v in med.items()})
from sklearn.metrics import confusion_matrix
lrb = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, C=1.0, class_weight="balanced")).fit(Xp, yp)
print("logreg balanced", round(float((lrb.predict(Xt) == yt).mean()), 3))
print(CLS); print(confusion_matrix(yt, lrb.predict(Xt), labels=CLS))
for c in CLS: print(c, sorted(set(re.sub(r"^h\d+_|_?\d*txt?s.*", "", r["n"]) for r in by[c]))[:40])
