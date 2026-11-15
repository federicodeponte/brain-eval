import sys, os, json, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import ROOT, write_jsonl, balanced_positions
from sklearn.ensemble import RandomForestRegressor
d = pd.read_csv(ROOT + '/_raw/bh/data_table.csv')
d = d[d.additive.notna() & d.aryl_halide.notna() & d['yield'].notna()].reset_index(drop=True)
print('reactions', len(d), 'halides', d.aryl_halide.nunique(), 'ligands', d.ligand.nunique(), 'bases', d.base.nunique(), 'additives', d.additive.nunique())
F = ['aryl_halide', 'ligand', 'base', 'additive']
rng = np.random.default_rng(11); L = 'ABCD'; pos = balanced_positions(40, 4, seed=2)
halides = sorted(d.aryl_halide.unique()); used = set(); rows = []; groups = []
k = 0
while len(rows) < 40:
    h = halides[k % len(halides)]; sub = d[(d.aryl_halide == h) & ~d.index.isin(used)]
    for _ in range(2000):
        s = sub.loc[rng.choice(sub.index, 4, replace=False)]
        y = np.sort(s['yield'].values)[::-1]
        if y[0] >= 15 and y[0] - y[1] >= 10 and s.ligand.nunique() >= 2 and s.base.nunique() >= 2: break
    else:
        k += 1; continue
    best = s['yield'].idxmax(); others = [i for i in s.index if i != best]; rng.shuffle(others)
    order = others[:]; order.insert(pos[len(rows)], best)
    used.update(order)
    choices = {L[i]: f"ligand {d.ligand[j]}, base {d.base[j]}, additive {d.additive[j]}" for i, j in enumerate(order)}
    prompt = ("Buchwald-Hartwig amination run in a 96-well high-throughput screen: aryl halide + p-toluidine, "
              "Pd precatalyst (Buchwald G3 palladacycle of the listed ligand), DMSO, 60 C, 16 h. "
              "One equivalent of an isoxazole-type additive is present in each well.\n\n"
              f"Aryl halide: {h} (SMILES {d.aryl_halide_smiles[order[0]]})\n\n"
              "Which condition set gave the HIGHEST yield?\n" + '\n'.join(f"{l}) {c}" for l, c in choices.items()) +
              "\n\nAnswer with a single letter.")
    rows.append({'id': f'yield-{len(rows):03d}', 'hypothesis': 'yield', 'prompt': prompt, 'answer': L[pos[len(rows)]],
                 'choices': choices, 'meta': {'yields': {L[i]: round(float(d['yield'][j]), 1) for i, j in enumerate(order)},
                 'row_idx': [int(j) for j in order], 'source': 'Ahneman et al. Science 2018 via doylelab/rxnpredict (MIT)'}})
    groups.append((h, order)); k += 1
write_jsonl(ROOT + '/yield/tasks.jsonl', rows)
for i in (0, 1): open(ROOT + f'/yield/examples/example_{i+1}.txt', 'w').write(rows[i]['prompt'] + f"\n\n[answer: {rows[i]['answer']}  yields: {rows[i]['meta']['yields']}]\n")
oh = pd.get_dummies(d[F].astype(str)).astype(np.uint8)
def ev(mask_fn, tag):
    c_rf = c_lig = 0
    cache = {}
    for r, (h, order) in zip(rows, groups):
        tr = mask_fn(h); key = tr.tobytes()
        if key not in cache:
            cache[key] = RandomForestRegressor(300, min_samples_leaf=2, n_jobs=8, random_state=0).fit(oh[tr], d['yield'][tr])
        p = cache[key].predict(oh.loc[order]); c_rf += L[int(np.argmax(p))] == r['answer']
        lm = d[tr].groupby('ligand')['yield'].mean(); bm = d[tr].groupby('base')['yield'].mean(); am = d[tr].groupby('additive')['yield'].mean()
        sc = [lm[d.ligand[j]] + bm[d.base[j]] + am[d.additive[j]] for j in order]; c_lig += L[int(np.argmax(sc))] == r['answer']
    print(f'{tag}: RF={c_rf/40:.3f} additive-main-effects heuristic={c_lig/40:.3f}')
notitem = ~d.index.isin(used)
ev(lambda h: notitem, 'random-split (train = all non-item reactions)')
ev(lambda h: notitem & (d.aryl_halide != h).values, 'strict leave-aryl-halide-out')
lig_only = sum(r['answer'] == max(r['choices'], key=lambda l: {'AdBrettPhos':3,'t-BuBrettPhos':2,'t-BuXPhos':1,'XPhos':0}.get(r['choices'][l].split(',')[0].replace('ligand ',''),0)) for r in rows)
print('textbook ligand-rank rule (AdBrettPhos>tBuBrettPhos>tBuXPhos>XPhos, first wins ties):', lig_only/40)
print('ligands', d.ligand.unique().tolist())
print('letters', pd.Series([r['answer'] for r in rows]).value_counts().to_dict(), 'halides used', len(set(g[0] for g in groups)))
