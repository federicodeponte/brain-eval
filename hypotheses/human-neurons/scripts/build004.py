# build tasks.jsonl from DANDI 000004 sessions: 4-way image-category few-shot decoding
import json, sys, numpy as np, os
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
rng = np.random.default_rng(int(os.environ.get("SEED", 7)))
WIN = os.environ.get("WIN", "w200_1200"); K = int(os.environ.get("K", 3)); NITEMS = int(os.environ.get("N", 40))
sessions = [l.split()[0] for l in open(sys.argv[1]) if l.strip()]
DROP = {"houses": "landscapes", "1cars": "4spatial", "fruit": "space"}
PRETTY = {"1cars": "cars", "2food": "food", "3people": "people", "5animals": "animals", "zzanimal": "animals", "smallAnimal": "small animals", "mobility": "vehicles"}
AB = {"Left Amygdala": "LA", "Right Amygdala": "RA", "Left Hippocampus": "LH", "Right Hippocampus": "RH"}
L = "ABCD"
def corr_nc(Xq, cents):
    d = []
    for c in cents:
        if Xq.std() == 0 or c.std() == 0: d.append(2.0)
        else: d.append(1 - np.corrcoef(Xq, c)[0, 1])
    return int(np.argmin(d))
items, nc_hits, lr_hits = [], 0, 0
answers = np.array([i % 4 for i in range(NITEMS)]); rng.shuffle(answers)
for i in range(NITEMS):
    fn = sessions[i % len(sessions)]
    d = np.load(fn); C = d[WIN]; cat = d["cat"]; img = d["img"]; phase = d["phase"]
    cs = sorted(set(cat)); keep = [c for c in cs if c != DROP[cs[0]]]
    order = list(rng.permutation(keep))          # letter -> category
    qc = order[answers[i]]
    q = rng.choice(np.where(cat == qc)[0])
    ex_idx = {}
    for c in order:
        pool = np.where((cat == c) & (np.arange(len(cat)) != q) & (img != img[q]))[0]
        ex_idx[c] = rng.choice(pool, K, replace=False)
    used = np.concatenate(list(ex_idx.values()))
    assert q not in used and not np.isin(img[used], img[q]).any()
    # in-prompt baseline: nearest centroid (correlation distance) on sqrt counts
    Xs = np.sqrt(C.astype(float))
    cents = [Xs[ex_idx[c]].mean(0) for c in order]
    nc_pred = corr_nc(Xs[q], cents); nc_hits += nc_pred == answers[i]
    # held-out LR: train on all other trials of the session (excluding query image), 4 classes
    tr = np.isin(cat, keep) & (img != img[q])
    lr = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=3000)).fit(Xs[tr], cat[tr])
    lr_hits += lr.predict(Xs[q:q+1])[0] == qc
    regions = [AB.get(r, r) for r in d["uloc"]]
    nN = C.shape[1]
    hdr = "neuron:  " + " ".join(f"{j+1:>3d}" for j in range(nN))
    reg = "region:  " + " ".join(f"{r:>3s}" for r in regions)
    def row(t): return " ".join(f"{v:>3d}" for v in C[t])
    lines = []
    for li, c in enumerate(order):
        name = PRETTY.get(c, c)
        lines.append(f"\nClass {L[li]} = picture of {name}:")
        for k, t in enumerate(ex_idx[c]): lines.append(f"  ex{k+1}:  {row(t)}")
    prompt = (
        f"These are real recordings from {nN} single neurons in the human medial temporal lobe (amygdala and hippocampus), "
        f"recorded with depth electrodes in an epilepsy patient while they looked at photographs. "
        f"Each row is one picture presentation; each number is how many spikes that neuron fired from {WIN[1:].replace('_', ' to ')} ms after the picture appeared. "
        f"Columns are the same neurons in every row (LA/RA = left/right amygdala, LH/RH = left/right hippocampus).\n\n"
        f"{hdr}\n{reg}\n" + "\n".join(lines) +
        f"\n\nQuery trial (a different picture, not one of the examples above):\n  query: {row(q)}\n\n"
        f"Which category of picture was the patient looking at during the query trial? "
        + " ".join(f"{L[li]}) {PRETTY.get(c, c)}" for li, c in enumerate(order)) +
        "\nAnswer with a single letter."
    )
    items.append({"id": f"human-mtl-cat-{i:03d}", "hypothesis": "Frontier LLMs can decode which image category a human was viewing from raw amygdala/hippocampus single-neuron spike counts given a few labeled examples.",
                  "prompt": prompt, "answer": L[answers[i]], "choices": [f"{L[li]}) {PRETTY.get(c, c)}" for li, c in enumerate(order)],
                  "meta": {"dandiset": "000004", "nwb": str(d["path"]), "n_neurons": int(nN), "window": WIN, "k_per_class": K, "query_trial": int(q),
                           "query_phase": str(phase[q]), "query_category": str(qc), "nc_correlation_pred": L[nc_pred], "chars": len(prompt)}})
with open(sys.argv[2], "w") as f:
    for it in items: f.write(json.dumps(it) + "\n")
lens = [it["meta"]["chars"] for it in items]
print(f"items={len(items)} K={K} win={WIN} NC_inprompt={nc_hits/NITEMS:.3f} LR_heldout={lr_hits/NITEMS:.3f} chance=0.25 letters={dict(zip(*np.unique([it['answer'] for it in items], return_counts=True)))} chars max={max(lens)} mean={int(np.mean(lens))}")
