"""HARD tier, mouse. Run from ~/work/vals-mouse: .venv/bin/python ~/work/vals-hyp/hard/scripts/mouse_hard.py [sweep|build K NEX U]
Units chosen per item from the item's example (non-query) repeats only. No model calls."""
import json, os, sys, string
import numpy as np, pandas as pd, h5py
from collections import Counter
from sklearn.linear_model import LogisticRegression
SESSION = 732592105; B = 10; BINMS = 100.0
D = os.path.expanduser('~/work/vals-mouse/data/'); OUT = os.path.expanduser('~/work/vals-hyp/hard/')
LET = string.ascii_uppercase
u = pd.read_csv(D+'units.csv', usecols=['id','ecephys_channel_id','isi_violations','amplitude_cutoff','presence_ratio'])
c = pd.read_csv(D+'channels.csv', usecols=['id','ecephys_probe_id','ecephys_structure_acronym']).rename(columns={'id':'ecephys_channel_id'})
p = pd.read_csv(D+'probes.csv', usecols=['id','ecephys_session_id']).rename(columns={'id':'ecephys_probe_id'})
u = u.merge(c, on='ecephys_channel_id').merge(p, on='ecephys_probe_id')
good = set(int(x) for x in u[(u.ecephys_session_id==SESSION)&(u.ecephys_structure_acronym=='VISp')&(u.isi_violations<0.5)&(u.amplitude_cutoff<0.1)&(u.presence_ratio>0.95)].id.values)
f = h5py.File(D+f'session_{SESSION}.nwb', 'r')
ids = f['units/id'][:]; st = f['units/spike_times'][:]; sti = f['units/spike_times_index'][:]
starts = np.concatenate([[0], sti[:-1]])
spikes = {int(i): st[s:e] for i, s, e in zip(ids, starts, sti) if int(i) in good}
uid_all = sorted(spikes)
g = f['intervals/natural_movie_three_presentations']; t0 = g['start_time'][:]; fr = g['frame'][:].astype(int)
nframes = fr.max()+1; rs = np.where(fr == 0)[0]; segf = int(B*BINMS/1000*30); nseg = nframes // segf
on = [t0[i0:i0+nframes][np.arange(nseg)*segf] for i0 in rs if len(fr[i0:i0+nframes]) == nframes and np.all(fr[i0:i0+nframes] == np.arange(nframes))]
on = np.array(on); edges = np.arange(B+1)*BINMS/1000
X3 = np.zeros((len(uid_all), len(on), nseg, B), np.int32)
for k, uid in enumerate(uid_all):
    s = spikes[uid]
    for r in range(len(on)): X3[k, r] = np.diff(np.searchsorted(s, on[r][:, None]+edges[None, :]), axis=1)
R3, S3 = X3.shape[1:3]

SEL = {}
def units_for(qr, U):  # split-half PSTH reliability on the 9 non-query repeats (examples are drawn from these)
    if qr not in SEL:
        tr = [r for r in range(R3) if r != qr]
        P = X3[:, tr].reshape(len(uid_all), len(tr), -1).astype(float)
        ev, od = P[:, 0::2].mean(1), P[:, 1::2].mean(1)
        SEL[qr] = np.argsort(-np.array([np.corrcoef(a, b)[0, 1] if a.std() > 0 and b.std() > 0 else -1 for a, b in zip(ev, od)]))
    return SEL[qr][:U]

def corr_d(x, y):
    x = x - x.mean(); y = y - y.mean(); d = np.linalg.norm(x)*np.linalg.norm(y)
    return 1 - (x@y)/d if d > 0 else 1.0
def nc(ex, q, metric='corr', tf=lambda x: x):
    qv = tf(q.ravel().astype(float)); cents = [np.mean([tf(m.ravel().astype(float)) for m in e], 0) for e in ex]
    d = [corr_d(qv, cc) if metric == 'corr' else np.linalg.norm(qv-cc) for cc in cents]
    return int(np.argmin(d))
def lr(ex, q):
    Xtr = np.array([m.ravel() for e in ex for m in e], float); y = np.array([j for j, e in enumerate(ex) for _ in e])
    Z = np.log1p(Xtr); mu = Z.mean(0); sd = Z.std(0)+1e-6
    return int(LogisticRegression(C=1.0, max_iter=2000).fit((Z-mu)/sd, y).predict(((np.log1p(q.ravel())-mu)/sd)[None])[0])

def draw(rng, n, K, NEX, U):
    qreps = np.resize(rng.permutation(R3), n); rng.shuffle(qreps); ans = np.resize(np.arange(K), n); rng.shuffle(ans)
    out = []
    for i in range(n):
        qr, a = int(qreps[i]), int(ans[i]); segs = rng.choice(S3, K, replace=False); us = units_for(qr, U)
        others = [r for r in range(R3) if r != qr]; exr = [sorted(rng.choice(others, NEX, replace=False).tolist()) for _ in range(K)]
        assert all(qr not in e for e in exr)
        out.append(dict(qr=qr, a=a, segs=segs, us=us, exr=exr, ex=[[X3[us, r, segs[j]] for r in exr[j]] for j in range(K)], q=X3[us, qr, segs[a]]))
    return out

def score(items, with_lr=True):
    n = len(items); r = dict(nc_corr=0, nc_euclid=0, nc_corr_sqrt=0)
    if with_lr: r['logreg'] = 0
    for it in items:
        r['nc_corr'] += nc(it['ex'], it['q']) == it['a']; r['nc_euclid'] += nc(it['ex'], it['q'], 'euclid') == it['a']
        r['nc_corr_sqrt'] += nc(it['ex'], it['q'], tf=np.sqrt) == it['a']
        if with_lr: r['logreg'] += lr(it['ex'], it['q']) == it['a']
    return {k: round(v/n, 3) for k, v in r.items()}

def mat_txt(m): return '\n'.join(' '.join(str(int(v)) for v in row) for row in m)
def render(K, ex, q, U, NEX):
    labs = ', '.join(LET[:K])
    s = (f"Below are spike counts recorded from {U} neurons in a mouse's primary visual cortex (VISp) while it watched a nature movie. "
         f"Each trial is a {U}x{B} matrix: one row per neuron (same neuron order in every matrix), one column per {BINMS:g} ms time bin (1 s total). "
         f"There are {K} different movie clips, labeled {labs}; for each clip you get {NEX} example trial{'s' if NEX > 1 else ''} recorded on {'a different viewing' if NEX == 1 else 'different viewings'} of that clip. "
         "Decide which clip the mouse was watching during the final query trial.")
    parts = [s, '']
    for j in range(K):
        for e, m in enumerate(ex[j]): parts += [f'Clip {LET[j]}, example {e+1}:', mat_txt(m), '']
    parts += ['Query trial (one of the clips above, on a new viewing):', mat_txt(q), '',
              f'Which clip ({labs}) was the mouse watching during the query trial? Answer with a single letter.']
    return '\n'.join(parts)

if sys.argv[1] == 'sweep':
    res = []
    for K, NEX, U in [(8,1,40),(8,1,20),(8,2,40),(10,1,40),(12,1,40),(12,1,20),(8,1,10)]:
        it = draw(np.random.default_rng(1000+K*10+NEX+U), 600, K, NEX, U)
        r = dict(K=K, shots=NEX, units=U, n=600, chance=round(1/K, 3), **score(it, with_lr=NEX > 1 or True))
        print(json.dumps(r), flush=True); res.append(r)
    os.makedirs(OUT+'scripts', exist_ok=True); json.dump(res, open(OUT+'mouse_sweep.json', 'w'), indent=1)
else:
    K, NEX, U = map(int, sys.argv[2:5]); name = 'mouse-hard'; n = 30
    sweep = json.load(open(OUT+'mouse_sweep.json'))
    items = draw(np.random.default_rng(7), n, K, NEX, U); sc = score(items)
    rows = []
    for i, it in enumerate(items):
        rows.append(dict(id=f'mouse-hard-k{K}-{i:03d}', k=K, prompt=render(K, it['ex'], it['q'], U, NEX), answer=LET[it['a']], session_id=SESSION,
            unit_ids=[uid_all[x] for x in it['us']], unit_selection='split-half reliability on the 9 non-query repeats (example repeats only)',
            segment_ids=[int(s) for s in it['segs']], query_repeat=it['qr'], example_repeats={LET[j]: it['exr'][j] for j in range(K)},
            stimulus='natural_movie_three', bin_ms=BINMS, n_bins=B, shots=NEX, n_units=U,
            meta=dict(nc_corr_pred=LET[nc(it['ex'], it['q'])], nc_euclid_pred=LET[nc(it['ex'], it['q'], 'euclid')])))
    os.makedirs(OUT+name, exist_ok=True)
    with open(OUT+name+'/tasks.jsonl', 'w') as fh:
        for r in rows: fh.write(json.dumps(r)+'\n')
    rep = dict(set=name, config=dict(K=K, shots=NEX, units=U, bins=B, bin_ms=BINMS, seed=7), items=n, chance=round(1/K, 4),
        decoders_on_set=sc, decoders_600_item_estimate=[s for s in sweep if (s['K'], s['shots'], s['units']) == (K, NEX, U)],
        candidates=sweep, letters=dict(sorted(Counter(r['answer'] for r in rows).items())),
        prompt_chars_max=max(len(r['prompt']) for r in rows), prompt_chars_mean=int(np.mean([len(r['prompt']) for r in rows])),
        reference_existing=dict(k4_3shot_40u='NC 0.817 (v2), Opus 0.87 vs NC 0.87 (v1)', k8_3shot_40u='NC 0.833 (v2), Opus 0.67 vs NC 0.67 (v1)'))
    json.dump(rep, open(OUT+name+'/report.json', 'w'), indent=1); print(json.dumps({k: rep[k] for k in ('config','decoders_on_set','letters','prompt_chars_max')}))
