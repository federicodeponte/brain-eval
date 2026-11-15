import os, glob, json, numpy as np, scipy.signal as ss
D = os.path.expanduser("~/work/vals-hyp/_data/mcd/rgb2"); OUT = os.path.expanduser("~/work/vals-hyp/pulse")
WIN = 20; FS_OUT = 10; L = "ABCD"


def bp(x, fs, lo=0.7, hi=3.5):
    b, a = ss.butter(3, [lo / (fs / 2), hi / (fs / 2)], "band")
    return ss.filtfilt(b, a, ss.detrend(x))


def hr_fft(x, fs, lo=0.7, hi=3.5):
    x = bp(x, fs, lo, hi)
    n = 8192; f = np.fft.rfftfreq(n, 1 / fs); P = np.abs(np.fft.rfft(x * np.hanning(len(x)), n)) ** 2
    m = (f >= lo) & (f <= hi); return 60 * f[m][P[m].argmax()]


def hr_peaks(x, fs):
    x = bp(x, fs)
    pk, _ = ss.find_peaks(x, distance=int(fs * 0.33), prominence=np.std(x) * 0.5)
    return 60 / np.median(np.diff(pk) / fs) if len(pk) > 3 else np.nan


def pos(rgb, fs):
    l = int(1.6 * fs); H = np.zeros(len(rgb))
    for t in range(len(rgb) - l + 1):
        C = rgb[t:t + l].T; Cn = C / C.mean(1, keepdims=True)
        S = np.array([[0, 1, -1], [-2, 1, 1]]) @ Cn
        h = S[0] + (S[0].std() / (S[1].std() + 1e-9)) * S[1]
        H[t:t + l] += h - h.mean()
    return H


def chrom(rgb):
    n = rgb / rgb.mean(0)
    X = 3 * n[:, 0] - 2 * n[:, 1]; Y = 1.5 * n[:, 0] + n[:, 1] - 1.5 * n[:, 2]
    return X - (X.std() / Y.std()) * Y


cands = []
for fn in sorted(glob.glob(D + "/*.npz")):
    d = np.load(fn); rgb = d["rgb"].astype(float); fps = float(d["fps"]); ppg = d["ppg"][:, 0].astype(float)
    n = min(len(rgb), len(ppg)); pid, step = os.path.basename(fn)[:-4].split("_")
    for s0 in [30, 70, 110]:
        a, b = int(s0 * fps), int((s0 + WIN) * fps)
        if b > n or np.isnan(rgb[a:b]).any(): continue
        g1 = hr_fft(ppg[a:b], fps); g2 = hr_peaks(ppg[a:b], fps)
        if not np.isfinite(g2) or abs(g1 - g2) > 5: continue  # ground truth must agree across two methods
        t = np.arange(a, b) / fps; tt = np.arange(0, WIN, 1 / FS_OUT) + s0
        ds = np.stack([np.interp(tt, t, rgb[a:b, k]) for k in range(3)], 1)
        cands.append(dict(pid=pid, step=step, s0=s0, hr=(g1 + g2) / 2, rgb=ds))
hrs = np.array([c["hr"] for c in cands])
print(len(cands), "windows; HR pct", np.percentile(hrs, [0, 25, 50, 75, 100]).round(1))
EDGES = [0, 68, 80, 92, 999]; NAMES = ["under 68 bpm", "68-80 bpm", "80-92 bpm", "over 92 bpm"]


def bucket(h): return int(np.clip(np.searchsorted(EDGES, h, side="right") - 1, 0, 3))


for c in cands: c["b"] = bucket(c["hr"])
print("bucket counts", np.bincount([c["b"] for c in cands], minlength=4))


def margin(h): return min(abs(h - e) for e in EDGES[1:4])


rng = np.random.default_rng(3); order = sorted(rng.permutation(len(cands)), key=lambda i: -cands[i]["b"]); used = set(); items = []; cnt = [0] * 4
for i in order:  # one window per subject, avoid GT within 2 bpm of a bucket edge
    c = cands[i]
    if c["pid"] in used or cnt[c["b"]] >= 10 or margin(c["hr"]) < 2: continue
    items.append(c); used.add(c["pid"]); cnt[c["b"]] += 1
print("items per bucket", cnt)
shots = []
for k in range(4):
    for i in order:
        c = cands[i]
        if c["b"] == k and c["pid"] not in used and margin(c["hr"]) >= 3:
            shots.append(c); used.add(c["pid"]); break
print("shots", [(s["pid"], round(s["hr"])) for s in shots])


def ser(r): return "\n".join(f"{x[0]:.2f} {x[1]:.2f} {x[2]:.2f}" for x in r)


head = (f"Below is a {WIN}-second recording from a webcam video of a seated person's face. For each video frame, the mean Red, Green and Blue "
        f"pixel values (0-255) of the skin region of the face were computed; the series is resampled to {FS_OUT} samples per second. One line per sample: R G B.\n")
q = "What was this person's heart rate during the recording?\n" + "\n".join(f"{L[i]}) {NAMES[i]}" for i in range(4)) + "\nAnswer with a single letter."


def mk(fewshot, tag):
    rows = []; ex = ""
    if fewshot:
        ex = "Labeled reference recordings from other people (same camera and format):\n"
        for s in shots: ex += f"[Example, heart rate {s['hr']:.0f} bpm, label {L[s['b']]}]\n{ser(s['rgb'])}\n"
        ex += "\nNow the recording to classify:\n"
    its = list(items); np.random.default_rng(5).shuffle(its)
    for k, c in enumerate(its):
        rows.append({"id": f"pulse-{tag}-{k:03d}", "hypothesis": "pulse", "prompt": head + ex + ser(c["rgb"]) + "\n\n" + q, "answer": L[c["b"]],
                     "choices": {L[i]: NAMES[i] for i in range(4)},
                     "meta": {"subject": c["pid"], "state": c["step"], "start_s": c["s0"], "hr_bpm": round(c["hr"], 1), "variant": tag,
                              "source": "MCD-rPPG FullHDwebcam (frontal)"}})
    return rows, its


zs, its = mk(False, "zs"); fsr, _ = mk(True, "fs")
for fn, rows in [("tasks.jsonl", zs), ("tasks_fewshot.jsonl", fsr)]:
    with open(f"{OUT}/{fn}", "w") as f:
        for r in rows: f.write(json.dumps(r) + "\n")
    print(fn, len(rows), "max chars", max(len(r["prompt"]) for r in rows), "answers", {a: sum(r["answer"] == a for r in rows) for a in L})
open(f"{OUT}/examples/example_zeroshot.txt", "w").write(zs[0]["prompt"] + "\n\n[ANSWER] " + zs[0]["answer"])
open(f"{OUT}/examples/example_fewshot.txt", "w").write(fsr[1]["prompt"] + "\n\n[ANSWER] " + fsr[1]["answer"])

# training-free baselines computed only from the 10 Hz numbers that appear in the prompt
Y = np.array([c["b"] for c in its]); gt = np.array([c["hr"] for c in its]); res = {}
for name, f in [("green_fft", lambda r: hr_fft(r[:, 1], FS_OUT)), ("POS_fft", lambda r: hr_fft(pos(r, FS_OUT), FS_OUT)),
                ("CHROM_fft", lambda r: hr_fft(chrom(r), FS_OUT))]:
    h = np.array([f(c["rgb"]) for c in its])
    res[name] = float((np.array([bucket(x) for x in h]) == Y).mean()); res[name + "_MAE_bpm"] = float(np.mean(np.abs(h - gt)))
    res[name + "_within5bpm"] = float((np.abs(h - gt) <= 5).mean())
res["majority"] = float(np.bincount(Y).max() / len(Y))
print(json.dumps({k: round(v, 3) for k, v in res.items()}))
json.dump(res, open(f"{OUT}/baseline.json", "w"), indent=1)
