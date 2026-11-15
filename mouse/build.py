"""Build 'Mind-reading a mouse' eval items from Allen Visual Coding Neuropixels.
Usage: python build.py [--units 40] [--bins 10] [--binms 100] [--stim natural_movie_three]
"""
import argparse, json, os, sys
import numpy as np, pandas as pd, h5py
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

ap = argparse.ArgumentParser()
ap.add_argument('--session', type=int, default=732592105)
ap.add_argument('--stim', default='natural_movie_three')
ap.add_argument('--units', type=int, default=40)
ap.add_argument('--bins', type=int, default=10)
ap.add_argument('--binms', type=float, default=100.0)
ap.add_argument('--n4', type=int, default=60)
ap.add_argument('--n8', type=int, default=30)
ap.add_argument('--nex', type=int, default=3)
ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--out', default='.')
a = ap.parse_args()
rng = np.random.default_rng(a.seed)
D = 'data/'
seg_s = a.bins * a.binms / 1000.0

# ---- good VISp units (default Allen quality filters) ----
u = pd.read_csv(D+'units.csv', usecols=['id','ecephys_channel_id','isi_violations','amplitude_cutoff','presence_ratio'])
c = pd.read_csv(D+'channels.csv', usecols=['id','ecephys_probe_id','ecephys_structure_acronym']).rename(columns={'id':'ecephys_channel_id'})
p = pd.read_csv(D+'probes.csv', usecols=['id','ecephys_session_id']).rename(columns={'id':'ecephys_probe_id'})
u = u.merge(c, on='ecephys_channel_id').merge(p, on='ecephys_probe_id')
good = u[(u.ecephys_session_id==a.session)&(u.ecephys_structure_acronym=='VISp')&(u.isi_violations<0.5)&(u.amplitude_cutoff<0.1)&(u.presence_ratio>0.95)].id.values
good = set(int(x) for x in good)

f = h5py.File(D+f'session_{a.session}.nwb', 'r')
ids = f['units/id'][:]
st = f['units/spike_times'][:]; sti = f['units/spike_times_index'][:]
starts = np.concatenate([[0], sti[:-1]])
spikes = {int(i): st[s:e] for i, s, e in zip(ids, starts, sti) if int(i) in good}
print('good VISp units in nwb:', len(spikes))

# ---- stimulus table ----
g = f[f'intervals/{a.stim}_presentations']
t0 = g['start_time'][:]; fr = g['frame'][:].astype(int)
nframes = fr.max()+1
rep_starts = np.where(fr == 0)[0]
print('frames', nframes, 'repeats', len(rep_starts))
fps = 30.0
nseg = int(np.floor(nframes / fps / seg_s))
seg_frames = int(round(seg_s*fps))
# onset time of each (repeat, segment) = presentation time of first frame of segment
onsets = np.full((len(rep_starts), nseg), np.nan)
for r, i0 in enumerate(rep_starts):
    blk = fr[i0:i0+nframes]; tt = t0[i0:i0+nframes]
    if len(blk) < nframes or not np.all(blk == np.arange(nframes)):
        print('repeat', r, 'incomplete, skipping'); continue
    onsets[r] = tt[np.arange(nseg)*seg_frames]
okr = ~np.isnan(onsets).any(1); onsets = onsets[okr]
R = onsets.shape[0]; print('usable repeats', R, 'segments', nseg, 'seg_s', seg_s)

uid_all = sorted(spikes)
edges_rel = np.arange(a.bins+1)*a.binms/1000.0
X = np.zeros((len(uid_all), R, nseg, a.bins), np.int32)
for k, uid in enumerate(uid_all):
    s = spikes[uid]
    for r in range(R):
        e = (onsets[r][:, None] + edges_rel[None, :])  # nseg x bins+1
        idx = np.searchsorted(s, e)
        X[k, r] = np.diff(idx, axis=1)

# ---- responsiveness: split-half reliability of the full-movie PSTH (odd vs even repeats) ----
psth = X.reshape(len(uid_all), R, -1).astype(float)
ev = psth[:, 0::2].mean(1); od = psth[:, 1::2].mean(1)
rel = np.array([np.corrcoef(ev[i], od[i])[0,1] if ev[i].std()>0 and od[i].std()>0 else -1 for i in range(len(uid_all))])
order = np.argsort(-rel)[:a.units]
sel = [uid_all[i] for i in order]
print('top unit reliability: min %.3f median %.3f max %.3f' % (rel[order].min(), np.median(rel[order]), rel[order].max()))
Xs = X[order]  # U x R x S x B

LET = 'ABCDEFGH'
def mat_txt(m):
    return '\n'.join(' '.join(str(v) for v in row) for row in m)

INSTR = ("Below are spike counts recorded from {U} neurons in a mouse's primary visual cortex (VISp) while it watched a nature movie. "
 "Each trial is a {U}x{B} matrix: one row per neuron (same neuron order in every matrix), one column per {ms:g} ms time bin ({dur:g} s total). "
 "There are {K} different movie clips, labeled {labs}; for each clip you get {n} example trials recorded on different viewings of that clip. "
 "Decide which clip the mouse was watching during the final query trial.")

def make_items(K, n_items, tag):
    items = []
    qreps = np.resize(rng.permutation(R), n_items); rng.shuffle(qreps)
    answers = np.resize(np.arange(K), n_items); rng.shuffle(answers)
    for it in range(n_items):
        segs = rng.choice(nseg, K, replace=False)  # segs[j] shown as letter j
        ans = int(answers[it]); qr = int(qreps[it])
        others = [r for r in range(R) if r != qr]
        ex = {j: sorted(rng.choice(others, a.nex, replace=False).tolist()) for j in range(K)}
        parts = [INSTR.format(U=a.units, B=a.bins, ms=a.binms, dur=seg_s, K=K, labs=', '.join(LET[:K]), n=a.nex), '']
        for j in range(K):
            for e_i, r in enumerate(ex[j]):
                parts += [f'Clip {LET[j]}, example {e_i+1}:', mat_txt(Xs[:, r, segs[j]]), '']
        parts += ['Query trial (one of the clips above, on a new viewing):', mat_txt(Xs[:, qr, segs[ans]]), '',
                  f'Which clip ({", ".join(LET[:K])}) was the mouse watching during the query trial? Answer with a single letter.']
        items.append(dict(id=f'{tag}-{it:03d}', k=K, prompt='\n'.join(parts), answer=LET[ans], session_id=a.session,
            unit_ids=sel, segment_ids=[int(s) for s in segs], query_repeat=qr, example_repeats={LET[j]: ex[j] for j in range(K)},
            stimulus=a.stim, bin_ms=a.binms, n_bins=a.bins))
    return items

def corr_d(x, y):
    x = x - x.mean(); y = y - y.mean(); d = np.linalg.norm(x)*np.linalg.norm(y)
    return 1 - (x@y)/d if d > 0 else 1.0

def baselines(items):
    acc = {'chance': 1/items[0]['k'], 'centroid_corr': 0, 'knn1_corr': 0, 'logreg': 0}
    for itm in items:
        K = itm['k']; segs = itm['segment_ids']; qr = itm['query_repeat']
        Xtr, ytr = [], []
        for j in range(K):
            for r in itm['example_repeats'][LET[j]]:
                Xtr.append(Xs[:, r, segs[j]].ravel().astype(float)); ytr.append(j)
        Xtr = np.array(Xtr); ytr = np.array(ytr)
        q = Xs[:, qr, segs[LET.index(itm['answer'])]].ravel().astype(float)
        truth = LET.index(itm['answer'])
        cents = np.array([Xtr[ytr==j].mean(0) for j in range(K)])
        acc['centroid_corr'] += int(np.argmin([corr_d(q, c_) for c_ in cents]) == truth)
        acc['knn1_corr'] += int(ytr[np.argmin([corr_d(q, x) for x in Xtr])] == truth)
        Z = np.log1p(Xtr); mu = Z.mean(0); sd = Z.std(0)+1e-6
        lr = LogisticRegression(C=1.0, max_iter=2000).fit((Z-mu)/sd, ytr)
        acc['logreg'] += int(lr.predict(((np.log1p(q)-mu)/sd)[None])[0] == truth)
    n = len(items)
    return {k: (v if k=='chance' else v/n) for k, v in acc.items()}

i4 = make_items(4, a.n4, 'k4'); i8 = make_items(8, a.n8, 'k8')
res = {'k4': baselines(i4), 'k8': baselines(i8)}
for k, v in res.items(): print(k, {kk: round(vv, 3) for kk, vv in v.items()})
L4 = [len(x['prompt']) for x in i4]; L8 = [len(x['prompt']) for x in i8]
print('prompt chars k4 max %d mean %d | k8 max %d mean %d' % (max(L4), np.mean(L4), max(L8), np.mean(L8)))
from collections import Counter
print('answer balance k4', dict(sorted(Counter(x['answer'] for x in i4).items())), 'k8', dict(sorted(Counter(x['answer'] for x in i8).items())))
# leakage check
for itm in i4+i8:
    assert all(itm['query_repeat'] not in v for v in itm['example_repeats'].values())
os.makedirs(a.out+'/examples', exist_ok=True)
with open(a.out+'/tasks.jsonl', 'w') as fh:
    for itm in i4+i8: fh.write(json.dumps(itm)+'\n')
open(a.out+'/examples/k4_example.txt','w').write(i4[0]['prompt']+f"\n\n[ANSWER: {i4[0]['answer']}]\n")
open(a.out+'/examples/k8_example.txt','w').write(i8[0]['prompt']+f"\n\n[ANSWER: {i8[0]['answer']}]\n")
json.dump(dict(config=vars(a), n_good_visp=len(spikes), repeats=R, segments=nseg, baselines=res,
    prompt_chars=dict(k4_max=max(L4), k8_max=max(L8)), unit_reliability=[float(rel[i]) for i in order]),
    open(a.out+'/build_report.json','w'), indent=1)
print('wrote', a.out+'/tasks.jsonl', len(i4)+len(i8))
