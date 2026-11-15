# Build tasks.jsonl: 4-way image-category decoding from a human MTL pseudo-population (DANDI 000004)
import sys, os, json, numpy as np
sys.path.insert(0, sys.argv[1])
from pseudo import load_group, select_neurons, sample_rows
from sklearn.linear_model import LogisticRegression
FIRST = os.environ.get("GROUP", "1cars"); WIN = os.environ.get("WIN", "w200_1200")
M = int(os.environ.get("M", 40)); K = int(os.environ.get("K", 3)); N = int(os.environ.get("N", 40)); SEED = int(os.environ.get("SEED", 11))
OUT = sys.argv[2]
PRETTY = {"1cars": "cars", "2food": "food", "3people": "people", "5animals": "animals", "houses": "houses", "mobility": "vehicles", "phones": "phones",
          "smallAnimal": "small animals", "fruit": "fruit", "kids": "children", "military": "military", "zzanimal": "animals"}
AB = {"Left Amygdala": "LA", "Right Amygdala": "RA", "Left Hippocampus": "LH", "Right Hippocampus": "RH"}
L = "ABCD"
S, keep = load_group(FIRST, WIN, seed=0)
neurons = select_neurons(S, keep, M)
regions = [AB.get(str(S[si]["uloc"][u]), str(S[si]["uloc"][u])[:3]) for si, u in neurons]
rng = np.random.default_rng(SEED)
answers = np.array([i % 4 for i in range(N)]); rng.shuffle(answers)
qcats = np.array([keep[(i // 4 + i) % 4] for i in range(N)])  # balanced query categories, decorrelated from letters
def ncpred(R):
    Xs = np.sqrt(R.astype(float)); E = Xs[:-1].reshape(4, K, -1).mean(1); q = Xs[-1]
    return int(np.argmin([1 - np.corrcoef(q, e)[0, 1] if q.std() > 0 and e.std() > 0 else 2 for e in E]))
items = []; nch = 0; lrh = 0
for i in range(N):
    a = int(answers[i]); rest = list(rng.permutation([c for c in keep if c != qcats[i]])); order = rest[:a] + [str(qcats[i])] + rest[a:]
    plan = [c for c in order for _ in range(K)] + [order[a]]
    R, tri = sample_rows(S, neurons, plan, rng)
    # check: query trial never reused as an example, and never the same picture, within each session
    for si, ts in tri.items():
        assert ts[-1] not in ts[:-1] and S[si]["img"][ts[-1]] not in set(S[si]["img"][ts[:-1]])
    p = ncpred(R); nch += p == a
    # held-out LR: train on pseudo-trials from pool trials excluding this item's query trials/pictures
    saved = {}
    for si, ts in tri.items():
        s = S[si]; saved[si] = s["pool"].copy(); bad = (np.arange(len(s["cat"])) == ts[-1]) | (s["img"] == s["img"][ts[-1]]); s["pool"] = s["pool"] & ~bad
    Xtr, ytr = [], []
    for j in range(240):
        c = keep[j % 4]; r, _ = sample_rows(S, neurons, [c], rng); Xtr.append(r[0]); ytr.append(c)
    for si in saved: S[si]["pool"] = saved[si]
    clf = LogisticRegression(C=0.05, max_iter=3000).fit(np.sqrt(np.array(Xtr, float)), ytr)
    lrh += clf.predict(np.sqrt(R[-1:].astype(float)))[0] == order[a]
    row = lambda r: " ".join(f"{v:>3d}" for v in r)
    body = []
    for li, c in enumerate(order):
        body.append(f"\nClass {L[li]} = photo of {PRETTY[c]}:")
        for k in range(K): body.append(f"  ex{k+1}:   {row(R[li*K+k])}")
    prompt = (
        f"Below are real spike counts from {M} single neurons recorded in the human medial temporal lobe (amygdala and hippocampus) "
        f"of epilepsy patients with clinical depth electrodes, while they viewed photographs. "
        f"Each number is how many spikes a neuron fired 200-1200 ms after a photo appeared. "
        f"Neurons were pooled from several recording sessions, so each row is a pseudo-trial: every neuron's count comes from one presentation of a photo from that row's category. "
        f"Columns are the same neurons in every row (LA/RA = left/right amygdala, LH/RH = left/right hippocampus).\n\n"
        f"neuron:  " + " ".join(f"{j+1:>3d}" for j in range(M)) + "\n"
        f"region:  " + " ".join(f"{r:>3s}" for r in regions) + "\n" + "\n".join(body) +
        f"\n\nQuery pseudo-trial (new photos, never shown in the example rows above):\n  query:  {row(R[-1])}\n\n"
        "Which category of photo was being viewed during the query pseudo-trial? "
        + " ".join(f"{L[li]}) {PRETTY[c]}" for li, c in enumerate(order)) + "\nAnswer with a single letter.")
    items.append({"id": f"human-mtl-category-{i:03d}",
                  "hypothesis": "Frontier LLMs can decode which category of photo a human was viewing from raw amygdala/hippocampus single-neuron spike counts plus a few labeled examples.",
                  "prompt": prompt, "answer": L[a], "choices": [f"{L[li]}) {PRETTY[c]}" for li, c in enumerate(order)],
                  "meta": {"dandiset": "000004", "group": FIRST, "n_neurons": M, "n_sessions_contributing": len(tri), "window_ms": [200, 1200], "k_per_class": K,
                           "query_category": PRETTY[order[a]], "nc_correlation_pred": L[p],
                           "query_trials": {S[si]["path"]: int(ts[-1]) for si, ts in tri.items()}, "chars": len(prompt)}})
with open(OUT, "w") as f:
    for it in items: f.write(json.dumps(it) + "\n")
lens = [it["meta"]["chars"] for it in items]
ans, cnt = np.unique([it["answer"] for it in items], return_counts=True)
print(json.dumps({"items": N, "M": M, "K": K, "group": FIRST, "sessions": len(S), "chance": 0.25, "nc_inprompt": nch / N, "lr_heldout": lrh / N,
                  "letters": dict(zip(ans.tolist(), cnt.tolist())), "chars_max": max(lens), "chars_mean": int(np.mean(lens)),
                  "regions": dict(zip(*[x.tolist() for x in np.unique(regions, return_counts=True)]))}))
