"""Mouse control sets + v2 main sets (units selected from non-query repeats only).
Run from ~/work/vals-mouse with its .venv: python ~/work/vals-hyp/controls/scripts/mouse_controls.py
No model calls. Outputs: ~/work/vals-hyp/controls/{swap-out,shot-curve,null,planted}/tasks.jsonl + report.json,
~/work/vals-mouse/tasks_v2.jsonl + build_report_v2.json."""
import json, os
import numpy as np, pandas as pd, h5py
from collections import Counter
from sklearn.linear_model import LogisticRegression

SESSION = 732592105; U = 40; B = 10; BINMS = 100.0; NEX = 3
D = os.path.expanduser('~/work/vals-mouse/data/')
CT = os.path.expanduser('~/work/vals-hyp/controls/')
LET = 'ABCDEFGH'

# ---------- data ----------
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

def extract(stim):
    g = f[f'intervals/{stim}_presentations']; t0 = g['start_time'][:]; fr = g['frame'][:].astype(int)
    nframes = fr.max()+1; rs = np.where(fr == 0)[0]; seg_frames = int(B*BINMS/1000*30); nseg = nframes // seg_frames
    on = []
    for i0 in rs:
        blk = fr[i0:i0+nframes]
        if len(blk) == nframes and np.all(blk == np.arange(nframes)): on.append(t0[i0:i0+nframes][np.arange(nseg)*seg_frames])
    on = np.array(on); edges = np.arange(B+1)*BINMS/1000
    X = np.zeros((len(uid_all), len(on), nseg, B), np.int32)
    for k, uid in enumerate(uid_all):
        s = spikes[uid]
        for r in range(len(on)): X[k, r] = np.diff(np.searchsorted(s, on[r][:, None]+edges[None, :]), axis=1)
    return X
X3 = extract('natural_movie_three'); X1 = extract('natural_movie_one')
R3, S3 = X3.shape[1:3]; R1, S1 = X1.shape[1:3]
print('nm3', X3.shape, 'nm1', X1.shape)

# unit selection from the 9 non-query repeats only (split-half PSTH reliability within them)
SEL = {}
def units_for(qr):
    if qr not in SEL:
        tr = [r for r in range(R3) if r != qr]
        P = X3[:, tr].reshape(len(uid_all), len(tr), -1).astype(float)
        ev, od = P[:, 0::2].mean(1), P[:, 1::2].mean(1)
        rel = np.array([np.corrcoef(a, b)[0, 1] if a.std() > 0 and b.std() > 0 else -1 for a, b in zip(ev, od)])
        SEL[qr] = np.argsort(-rel)[:U]
    return SEL[qr]

def mat_txt(m): return '\n'.join(' '.join(str(int(v)) for v in row) for row in m)
def corr_d(x, y):
    x = x - x.mean(); y = y - y.mean(); d = np.linalg.norm(x)*np.linalg.norm(y)
    return 1 - (x@y)/d if d > 0 else 1.0

HEAD = ("Below are spike counts recorded from {U} neurons in a mouse's primary visual cortex (VISp) while it watched a nature movie. "
        "Each trial is a {U}x{B} matrix: one row per neuron (same neuron order in every matrix), one column per {ms:g} ms time bin (1 s total). ")
def head(): return HEAD.format(U=U, B=B, ms=BINMS)

def render(K, ex_mats, q, nshot, swap=False):
    labs = ', '.join(LET[:K])
    if nshot == 0:
        s = head() + f"There are {K} different movie clips, labeled {labs}. No example trials are given for the clips. Decide which clip the mouse was watching during the query trial."
    else:
        s = head() + (f"There are {K} different movie clips, labeled {labs}; for each clip you get {nshot} example trial{'s' if nshot > 1 else ''} recorded on different viewings of that clip. ")
        s += ("The query trial is from a new viewing of some clip. It may be one of the clips above, or a clip that is NOT among them. "
              "Decide which clip the mouse was watching during the query trial, or say that it matches none of them." if swap else
              "Decide which clip the mouse was watching during the final query trial.")
    parts = [s, '']
    for j in range(K):
        for e, m in enumerate(ex_mats[j]): parts += [f'Clip {LET[j]}, example {e+1}:', mat_txt(m), '']
    if swap:
        parts += ['Query trial (new viewing):', mat_txt(q), '',
                  f'Which clip ({labs}) was the mouse watching during the query trial? If it matches none of them, answer E (none of these). '
                  f'Options: {", ".join(LET[:K])}, or E) none of these. Answer with a single letter.']
    else:
        parts += ['Query trial (one of the clips above, on a new viewing):' if nshot else f'Query trial (one of clips {labs}):', mat_txt(q), '',
                  f'Which clip ({labs}) was the mouse watching during the query trial? Answer with a single letter.']
    return '\n'.join(parts)

def nc_pred(ex_mats, q, tf=lambda x: x):
    d = [corr_d(tf(q.ravel().astype(float)), np.mean([tf(m.ravel().astype(float)) for m in ex], 0)) for ex in ex_mats]
    return int(np.argmin(d)), float(np.min(d))

def lr_pred(ex_mats, q):
    Xtr = np.array([m.ravel() for ex in ex_mats for m in ex], float); y = np.array([j for j, ex in enumerate(ex_mats) for _ in ex])
    Z = np.log1p(Xtr); mu = Z.mean(0); sd = Z.std(0)+1e-6
    return int(LogisticRegression(C=1.0, max_iter=2000).fit((Z-mu)/sd, y).predict(((np.log1p(q.ravel())-mu)/sd)[None])[0])

def write(name, items, report, path=None):
    path = path or CT+name+'/tasks.jsonl'; os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as fh:
        for it in items: fh.write(json.dumps(it)+'\n')
    report['letters'] = dict(sorted(Counter(i['answer'] for i in items).items())); report['items'] = len(items)
    report['prompt_chars_max'] = max(len(i['prompt']) for i in items)
    json.dump(report, open(os.path.join(os.path.dirname(path), 'report.json' if path.startswith(CT) else 'build_report_v2.json'), 'w'), indent=1)
    print(name, json.dumps(report))

def base_draw(rng, n, K):
    qreps = np.resize(rng.permutation(R3), n); rng.shuffle(qreps)
    ans = np.resize(np.arange(K), n); rng.shuffle(ans)
    return [(int(qreps[i]), int(ans[i]), rng.choice(S3, K, replace=False)) for i in range(n)]

# ---------- v2 main sets ----------
rng = np.random.default_rng(0)
v2 = []; rep = {}
for K, n in ((4, 60), (8, 30)):
    acc = dict(centroid_corr=0, logreg=0)
    for it, (qr, a, segs) in enumerate(base_draw(rng, n, K)):
        us = units_for(qr); others = [r for r in range(R3) if r != qr]
        exr = [sorted(rng.choice(others, NEX, replace=False).tolist()) for _ in range(K)]
        ex = [[X3[us, r, segs[j]] for r in exr[j]] for j in range(K)]; q = X3[us, qr, segs[a]]
        acc['centroid_corr'] += nc_pred(ex, q)[0] == a; acc['logreg'] += lr_pred(ex, q) == a
        assert all(qr not in e for e in exr)
        v2.append(dict(id=f'v2-k{K}-{it:03d}', k=K, prompt=render(K, ex, q, NEX), answer=LET[a], session_id=SESSION,
            unit_ids=[uid_all[i] for i in us], unit_selection='split-half reliability on the 9 non-query repeats',
            segment_ids=[int(s) for s in segs], query_repeat=qr, example_repeats={LET[j]: exr[j] for j in range(K)},
            stimulus='natural_movie_three', bin_ms=BINMS, n_bins=B))
    rep[f'k{K}'] = dict(n=n, chance=1/K, **{k: v/n for k, v in acc.items()})
write('v2', v2, dict(baselines=rep, note='units selected per item from non-query repeats'), os.path.expanduser('~/work/vals-mouse/tasks_v2.jsonl'))

# ---------- 1. swap-out (K=4 + E) ----------
def swap_items(rng, n_normal, n_swap, tag):
    out = []; K = 4
    n = n_normal + n_swap
    kinds = ['normal']*n_normal + ['swap_nm1']*(n_swap//2) + ['swap_nm3']*(n_swap - n_swap//2)
    kinds = list(np.array(kinds)[rng.permutation(n)])
    qreps = np.resize(rng.permutation(R3), n); rng.shuffle(qreps)
    slot = np.zeros(n, int)
    for grp in (['normal'], ['swap_nm1', 'swap_nm3']):
        ix = [i for i, k_ in enumerate(kinds) if k_ in grp]; v = np.resize(np.arange(K), len(ix)); rng.shuffle(v); slot[ix] = v
    draws = [(int(qreps[i]), int(slot[i]), rng.choice(S3, K, replace=False)) for i in range(n)]
    for it, ((qr, a, segs), kind) in enumerate(zip(draws, kinds)):
        us = units_for(qr); others = [r for r in range(R3) if r != qr]
        exr = [sorted(rng.choice(others, NEX, replace=False).tolist()) for _ in range(K)]
        ex = [[X3[us, r, segs[j]] for r in exr[j]] for j in range(K)]; q = X3[us, qr, segs[a]]
        info = dict(swap_type=kind, swapped_slot=None, replacement=None)
        if kind == 'swap_nm1':
            s1 = int(rng.integers(S1)); r1 = sorted(rng.choice(R1, NEX, replace=False).tolist())
            ex[a] = [X1[us, r, s1] for r in r1]; info.update(swapped_slot=LET[a], replacement=dict(stimulus='natural_movie_one', segment=s1, repeats=r1))
        elif kind == 'swap_nm3':
            s3 = int(rng.choice([s for s in range(S3) if s not in segs]))
            ex[a] = [X3[us, r, s3] for r in exr[a]]; info.update(swapped_slot=LET[a], replacement=dict(stimulus='natural_movie_three', segment=s3, repeats=exr[a]))
        ansL = LET[a] if kind == 'normal' else 'E'
        out.append(dict(item=dict(id=f'{tag}-{it:03d}', k=5, prompt=render(K, ex, q, NEX, swap=True), answer=ansL, session_id=SESSION,
            query_clip=dict(stimulus='natural_movie_three', segment=int(segs[a]), repeat=qr), segment_ids=[int(s) for s in segs],
            example_repeats={LET[j]: exr[j] for j in range(K)}, unit_ids=[uid_all[i] for i in us], meta=dict(info)),
            ex=ex, q=q, a=a, ans=ansL))
    return out

def own_other_tau(ex):  # in-prompt rule: midpoint between median LOO own-class and median nearest-other distances
    own, oth = [], []
    for j, e in enumerate(ex):
        for i in range(len(e)):
            x = e[i].ravel().astype(float)
            own.append(corr_d(x, np.mean([m.ravel() for k, m in enumerate(e) if k != i], 0).astype(float)))
            oth.append(min(corr_d(x, np.mean([m.ravel() for m in e2], 0).astype(float)) for k2, e2 in enumerate(ex) if k2 != j))
    return (np.median(own) + np.median(oth)) / 2

def thr_acc(rows, tau=None, tf=lambda x: x):
    ok = 0
    for r in rows:
        j, d = nc_pred(r['ex'], r['q'], tf); t = own_other_tau(r['ex']) if tau is None else tau
        ok += (('E' if d > t else LET[j]) == r['ans'])
    return ok / len(rows)

cal = swap_items(np.random.default_rng(101), 200, 200, 'cal')
taus = np.linspace(0.1, 0.9, 81); cal_acc = [thr_acc(cal, t) for t in taus]; tau_star = float(taus[int(np.argmax(cal_acc))])
cal_sq = [thr_acc(cal, t, np.sqrt) for t in taus]; tau_sq = float(taus[int(np.argmax(cal_sq))])
sw = swap_items(np.random.default_rng(1), 20, 20, 'swap-out')
norm = [r for r in sw if r['item']['meta']['swap_type'] == 'normal']; swp = [r for r in sw if r['item']['meta']['swap_type'] != 'normal']
rep = dict(chance_uniform=0.2, always_E=0.5, never_E_best=sum(nc_pred(r['ex'], r['q'])[0] == r['a'] for r in norm)/len(sw),
    nc_no_threshold=sum((LET[nc_pred(r['ex'], r['q'])[0]] == r['ans']) for r in sw)/len(sw),
    nc_threshold_inprompt=thr_acc(sw), nc_threshold_inprompt_normal=thr_acc(norm), nc_threshold_inprompt_swap=thr_acc(swp),
    tau_calibrated=tau_star, calibration_items=len(cal), calibration_acc=max(cal_acc),
    nc_threshold_calibrated=thr_acc(sw, tau_star), nc_threshold_calibrated_normal=thr_acc(norm, tau_star), nc_threshold_calibrated_swap=thr_acc(swp, tau_star),
    nc_threshold_inprompt_on_calibration=thr_acc(cal),
    sqrt_tau_calibrated=tau_sq, sqrt_calibration_acc=max(cal_sq), nc_sqrt_threshold_calibrated=thr_acc(sw, tau_sq),
    nc_sqrt_threshold_calibrated_normal=thr_acc(norm, tau_sq), nc_sqrt_threshold_calibrated_swap=thr_acc(swp, tau_sq),
    nc_sqrt_calibrated_swap_nm1=thr_acc([r for r in swp if r['item']['meta']['swap_type']=='swap_nm1'], tau_sq),
    nc_sqrt_calibrated_swap_nm3=thr_acc([r for r in swp if r['item']['meta']['swap_type']=='swap_nm3'], tau_sq),
    swap_types=dict(Counter(r['item']['meta']['swap_type'] for r in sw)),
    note='threshold rule: predict E if correlation distance to the nearest centroid > tau. in-prompt tau = midpoint of median leave-one-out own-class and nearest-other distances among the examples; calibrated tau tuned on 400 separate items (seed 101).')
for r in sw:
    j, d = nc_pred(r['ex'], r['q']); r['item']['meta'].update(nc_pred=LET[j], nc_min_dist=round(d, 4))
write('swap-out', [r['item'] for r in sw], rep)

# ---------- 3. shot curve (K=4; 0/1/3/5 shots, nested examples, same queries) ----------
rng = np.random.default_rng(3); SH = (0, 1, 3, 5); K = 4
base = base_draw(rng, 40, K); items = []; acc = {s: 0 for s in SH}; acc_lr = {s: 0 for s in SH if s >= 3}
for it, (qr, a, segs) in enumerate(base):
    us = units_for(qr); others = [r for r in range(R3) if r != qr]
    exr5 = [rng.permutation(others)[:5].tolist() for _ in range(K)]; q = X3[us, qr, segs[a]]
    for s in SH:
        exr = [sorted(e[:s]) for e in exr5]; ex = [[X3[us, r, segs[j]] for r in exr[j]] for j in range(K)]
        if s: acc[s] += nc_pred(ex, q)[0] == a
        if s in acc_lr: acc_lr[s] += lr_pred(ex, q) == a
        items.append(dict(id=f'shot{s}-{it:03d}', k=K, shots=s, base_item=it, prompt=render(K, ex, q, s), answer=LET[a], session_id=SESSION,
            segment_ids=[int(x) for x in segs], query_repeat=qr, example_repeats={LET[j]: exr[j] for j in range(K)}, unit_ids=[uid_all[i] for i in us]))
acc[0] = 0.25
write('shot-curve', items, dict(chance=0.25, nc_by_shots={s: acc[s]/40 if s else 0.25 for s in SH}, lr_by_shots={s: v/40 for s, v in acc_lr.items()},
    items_per_level=40, note='0-shot has no examples; decoder = chance by construction. examples nested (1 in 3 in 5).'))

# ---------- 4a. null (no clip structure) and 4b. planted ----------
rng = np.random.default_rng(4); K = 4
draws = base_draw(rng, 40, K); items = []; nc = 0
for it, (qr, a, segs) in enumerate(draws):
    us = units_for(qr); kind = 'shuffled_labels' if it % 2 == 0 else 'poisson'
    if kind == 'poisson':
        lam = X3[us].mean(axis=(1, 2))  # unit x bin mean rate, no clip structure
        ex = [[rng.poisson(lam) for _ in range(NEX)] for _ in range(K)]; q = rng.poisson(lam)
    else:  # every example and the query are real trials from independently drawn clips; labels carry no clip identity
        used = rng.choice(S3, K*NEX+1, replace=False); rr = rng.integers(R3, size=K*NEX+1)
        ex = [[X3[us, rr[j*NEX+e], used[j*NEX+e]] for e in range(NEX)] for j in range(K)]; q = X3[us, rr[-1], used[-1]]
    nc += nc_pred(ex, q)[0] == a
    items.append(dict(id=f'null-{it:03d}', k=K, null_type=kind, prompt=render(K, ex, q, NEX), answer=LET[a], session_id=SESSION, unit_ids=[uid_all[i] for i in us]))
write('null', items, dict(chance=0.25, nc=nc/40, expected='chance (0.25); answer letters are arbitrary'))

rng = np.random.default_rng(5); draws = base_draw(rng, 40, K); items = []; nc = 0
for it, (qr, a, segs) in enumerate(draws):
    us = units_for(qr); lam = X3[us].mean(axis=(1, 2))
    blocks = rng.permutation(U)[:K*4].reshape(K, 4); b0 = rng.integers(0, B-2, size=K)
    def pat(j):
        m = np.zeros((U, B)); m[np.ix_(blocks[j], range(b0[j], b0[j]+3))] = 6; return m
    ex = [[rng.poisson(lam + pat(j)) for _ in range(NEX)] for j in range(K)]; q = rng.poisson(lam + pat(a))
    nc += nc_pred(ex, q)[0] == a
    items.append(dict(id=f'planted-{it:03d}', k=K, prompt=render(K, ex, q, NEX), answer=LET[a], session_id=SESSION, unit_ids=[uid_all[i] for i in us],
        meta=dict(planted={LET[j]: dict(units_rows_1based=(blocks[j]+1).tolist(), bins_1based=list(range(int(b0[j])+1, int(b0[j])+4)), extra_rate=6) for j in range(K)})))
write('planted', items, dict(chance=0.25, nc=nc/40, expected='~1.0', note='Poisson background at real unit x bin mean rates; each clip has its own planted block (4 neurons x 3 bins, +6 spikes/bin); query carries the answer block'))
