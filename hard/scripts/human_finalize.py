"""Finalize human-hard from build_pseudo candidates (K=1 shot/class, M=40). Recomputes decoders from the prompt rows only.
python3 -I human_finalize.py <hard_dir> <picked_seed> <seed>..."""
import sys, json, re, os, numpy as np
H = sys.argv[1]; pick = sys.argv[2]; seeds = sys.argv[3:]
def cd(x, y):
    x = x - x.mean(); y = y - y.mean(); d = np.linalg.norm(x)*np.linalg.norm(y); return 1 - x@y/d if d > 0 else 2
def rows(p):
    ex = [list(map(float, m.split())) for m in re.findall(r'^  ex\d+:\s+(.*)$', p, re.M)]
    q = list(map(float, re.search(r'^  query:\s+(.*)$', p, re.M).group(1).split())); return np.array(ex), np.array(q)
def score(items):
    acc = dict(nc_corr_sqrt=0, nc_corr=0, nc_euclid_sqrt=0, nc_euclid=0)
    for it in items:
        ex, q = rows(it['prompt']); a = 'ABCD'.index(it['answer']); kk = len(ex)//4
        for s, tf in (('_sqrt', np.sqrt), ('', lambda x: x)):
            E = tf(ex).reshape(4, kk, -1).mean(1); qq = tf(q)
            acc['nc_corr'+s] += int(np.argmin([cd(qq, e) for e in E])) == a; acc['nc_euclid'+s] += int(np.argmin([np.linalg.norm(qq-e) for e in E])) == a
    return {k: round(v/len(items), 3) for k, v in acc.items()}
cands = {s: [json.loads(l) for l in open(f'{H}/human-hard/cand_seed{s}.jsonl')] for s in seeds}
cs = {s: score(v) for s, v in cands.items()}
out = []
for it in cands[pick]:
    it['id'] = it['id'].replace('human-mtl-category', 'human-hard'); it['k'] = 4; it['meta']['tier'] = 'hard'; out.append(it)
with open(f'{H}/human-hard/tasks.jsonl', 'w') as f:
    for it in out: f.write(json.dumps(it)+'\n')
sweep = json.load(open(f'{H}/human_sweep.json'))
from collections import Counter
rep = dict(set='human-hard', config=dict(classes=4, shots_per_class=1, neurons=40, window_ms=[200, 1200], group='cars/food/people/animals', build_seed=int(pick)),
    items=len(out), chance=0.25, decoders_on_set=cs[pick], candidate_seeds_on_30_items=cs, picked_seed_reason='set NC (corr, sqrt) closest to the 400-item estimate',
    decoders_400_item_estimates=sweep, letters=dict(sorted(Counter(i['answer'] for i in out).items())),
    query_categories=dict(Counter(i['meta']['query_category'] for i in out)), prompt_chars_max=max(len(i['prompt']) for i in out),
    reference_existing='human K=4 3-shot 40 neurons: NC 0.65 (corr sqrt); Opus 0.68')
json.dump(rep, open(f'{H}/human-hard/report.json', 'w'), indent=1); print(json.dumps({k: rep[k] for k in ('decoders_on_set', 'candidate_seeds_on_30_items', 'letters', 'query_categories')}))
