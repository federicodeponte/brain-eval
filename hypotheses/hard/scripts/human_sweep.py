"""HARD tier, human: decoder sweep over (shots per class, neurons). Run from ~/work/vals-hyp/human-neurons:
. ./env.sh; .venv/bin/python -I ~/work/vals-hyp/hard/scripts/human_sweep.py $PWD/scripts out.json
Nearest centroid on in-prompt examples only (corr on sqrt counts = the existing gate; also raw corr, euclid sqrt)."""
import sys, json, numpy as np
sys.path.insert(0, sys.argv[1])
from pseudo import load_group, select_neurons, sample_rows
S, keep = load_group("1cars", "w200_1200", seed=0)
def cd(x, y):
    x = x - x.mean(); y = y - y.mean(); d = np.linalg.norm(x)*np.linalg.norm(y); return 1 - x@y/d if d > 0 else 2
res = []
for M, K in [(40,1),(40,2),(20,1),(20,2),(20,3),(10,3)]:
    neurons = select_neurons(S, keep, M); rng = np.random.default_rng(500+M+K); n = 400; acc = dict(nc_corr_sqrt=0, nc_corr=0, nc_euclid_sqrt=0, nc_euclid=0)
    for i in range(n):
        order = list(rng.permutation(keep)); a = i % 4
        R, _ = sample_rows(S, neurons, [c for c in order for _ in range(K)] + [order[a]], rng); R = R.astype(float)
        for tag, tf in (('sqrt', np.sqrt), ('', lambda x: x)):
            Xs = tf(R); E = Xs[:-1].reshape(4, K, -1).mean(1); q = Xs[-1]
            acc['nc_corr' + ('_sqrt' if tag else '')] += int(np.argmin([cd(q, e) for e in E])) == a
            acc['nc_euclid' + ('_sqrt' if tag else '')] += int(np.argmin([np.linalg.norm(q-e) for e in E])) == a
    r = dict(neurons=M, shots=K, n=n, chance=0.25, **{k: round(v/n, 3) for k, v in acc.items()}); print(json.dumps(r), flush=True); res.append(r)
json.dump(res, open(sys.argv[2], 'w'), indent=1)
