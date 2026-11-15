"""Export natural_movie_three frames for the mouse demo items (k4-000..009, k8-000..009)."""
import json, os, base64, io, sys
import numpy as np, h5py
from PIL import Image

HOME = os.path.expanduser('~')
MOUSE = f'{HOME}/work/vals-mouse'
OUT = f'{HOME}/work/vals-hyp/demo-frames'
MOVIE = f'{OUT}/_dl/natural_movie_3.h5'  # actually .npy (3600,304,608) uint8
LET = 'ABCDEFGH'
FPS = 30; SEG = 30  # 1 s clips

mov = np.load(MOVIE, mmap_mode='r')
assert mov.shape == (3600, 304, 608), mov.shape

items = [json.loads(l) for l in open(f'{MOUSE}/tasks.jsonl')]
k4 = [x for x in items if x['k'] == 4][:10]; k8 = [x for x in items if x['k'] == 8][:10]
sel = k4 + k8
assert [x['id'] for x in k4] == [f'k4-{i:03d}' for i in range(10)]
assert [x['id'] for x in k8] == [f'k8-{i:03d}' for i in range(10)]

# model preds
preds = {}
for l in open(f'{HOME}/work/vals-hyp/pilot/results.jsonl'):
    d = json.loads(l)
    if d['hyp'] in ('mouse', 'mouse8'):
        preds.setdefault(d['id'], {})[d['model']] = d['pred']

# ---- independent check: recompute spike counts from NWB and compare to prompt matrices ----
f = h5py.File(f'{MOUSE}/data/session_732592105.nwb', 'r')
ids = f['units/id'][:]; st = f['units/spike_times'][:]; sti = f['units/spike_times_index'][:]
starts = np.concatenate([[0], sti[:-1]])
spk = {int(i): st[s:e] for i, s, e in zip(ids, starts, sti)}
g = f['intervals/natural_movie_three_presentations']
t0 = g['start_time'][:]; fr = g['frame'][:].astype(int)
rep_starts = np.where(fr == 0)[0]
onsets = []
for i0 in rep_starts:
    blk = fr[i0:i0+3600]
    if len(blk) == 3600 and np.all(blk == np.arange(3600)):
        onsets.append(t0[i0:i0+3600][np.arange(120)*SEG])
onsets = np.array(onsets); assert onsets.shape == (10, 120)
edges = np.arange(11) * 0.1

def counts(unit_ids, rep, seg):
    e = onsets[rep, seg] + edges
    return np.array([np.diff(np.searchsorted(spk[u], e)) for u in unit_ids])

def parse_prompt(p, K):
    lines = p.split('\n'); mats = {}; q = None; i = 0
    while i < len(lines):
        L = lines[i]
        if L.startswith('Clip ') and ', example ' in L:
            lab = L[5]; m = np.array([[int(v) for v in lines[i+1+r].split()] for r in range(40)])
            mats.setdefault(lab, []).append(m); i += 41; continue
        if L.startswith('Query trial'):
            q = np.array([[int(v) for v in lines[i+1+r].split()] for r in range(40)]); i += 41; continue
        i += 1
    assert sorted(mats) == list(LET[:K]) and all(len(v) == 3 for v in mats.values()) and q is not None
    return mats, q

def corr_d(x, y):
    x = x - x.mean(); y = y - y.mean(); d = np.linalg.norm(x)*np.linalg.norm(y)
    return 1 - (x@y)/d if d > 0 else 1.0

def jpg(arr, w, q=70):
    im = Image.fromarray(np.asarray(arr)); h = round(im.height * w / im.width)
    im = im.resize((w, h), Image.LANCZOS); b = io.BytesIO(); im.save(b, 'JPEG', quality=q, optimize=True)
    return b.getvalue()

os.makedirs(f'{OUT}/frames', exist_ok=True); os.makedirs(f'{OUT}/strips', exist_ok=True)
out = []; checks = 0; dec_ok = 0
for it in sel:
    K = it['k']; segs = it['segment_ids']; U = it['unit_ids']
    mats, q = parse_prompt(it['prompt'], K)
    # mapping check: letter j -> segs[j]; prompt examples == NWB counts at (repeat, segs[j])
    for j in range(K):
        for m, r in zip(mats[LET[j]], it['example_repeats'][LET[j]]):
            assert np.array_equal(m, counts(U, r, segs[j])), (it['id'], LET[j]); checks += 1
    a = LET.index(it['answer'])
    assert np.array_equal(q, counts(U, it['query_repeat'], segs[a])), it['id']; checks += 1
    # decoder: nearest centroid (corr distance) on in-prompt examples
    cents = [np.mean([m.ravel().astype(float) for m in mats[LET[j]]], 0) for j in range(K)]
    dists = [corr_d(q.ravel().astype(float), c) for c in cents]
    dec = LET[int(np.argmin(dists))]; dec_ok += dec == it['answer']
    opts = []
    for j in range(K):
        s = segs[j]; f0 = s*SEG; fm = f0 + SEG//2; f1 = f0 + SEG - 1
        fn = f'frames/{it["id"]}_{LET[j]}_clip{s:03d}.jpg'
        open(f'{OUT}/{fn}', 'wb').write(jpg(mov[fm], 240))
        strip = np.concatenate([mov[f0], np.full((304, 8), 255, np.uint8), mov[fm], np.full((304, 8), 255, np.uint8), mov[f1]], 1)
        sn = f'strips/{it["id"]}_{LET[j]}_clip{s:03d}.jpg'
        open(f'{OUT}/{sn}', 'wb').write(jpg(strip, 480))
        opts.append(dict(letter=LET[j], clip_index=int(s), start_s=float(s), frames=[f0, fm, f1],
                         frame_file=fn, strip_file=sn))
    out.append(dict(id=it['id'], k=K, answer=it['answer'], query_repeat=it['query_repeat'], options=opts,
                    decoder=dict(method='nearest_centroid_corr', pick=dec, dist={LET[j]: round(float(d), 4) for j, d in enumerate(dists)}),
                    preds=preds.get(it['id'], {})))

meta = dict(session_id=732592105, stimulus='natural_movie_three', movie_source='s3://allen-brain-observatory/visual-coding-neuropixels/ecephys-cache/natural_movie_templates/natural_movie_3.h5 (npy, 3600x304x608 uint8)',
            clip_def='clip s = frames 30*s..30*s+29 (1 s at 30 Hz); frame_file = middle frame 30*s+15, width 240, JPEG q70; strip = start/middle/end',
            mapping_check=f'{checks} prompt matrices matched NWB spike counts at (repeat, segment_ids[letter])')
json.dump(dict(meta=meta, items=out), open(f'{OUT}/items.json', 'w'), indent=1)
for o in out:
    for op in o['options']:
        op['frame_data'] = 'data:image/jpeg;base64,' + base64.b64encode(open(f'{OUT}/{op["frame_file"]}', 'rb').read()).decode()
json.dump(dict(meta=meta, items=out), open(f'{OUT}/items_inline.json', 'w'))
print('mapping checks passed:', checks, '| decoder correct', dec_ok, '/', len(out))
for o in out[:3] + out[10:12]:
    print(o['id'], 'ans', o['answer'], 'dec', o['decoder']['pick'], o['preds'], [ (p['letter'], p['clip_index']) for p in o['options']])
print('inline bytes', os.path.getsize(f'{OUT}/items_inline.json'))
