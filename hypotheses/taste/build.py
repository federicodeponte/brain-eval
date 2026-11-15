import sys, os, json, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
from sklearn.linear_model import LogisticRegression
t = pd.read_excel(ROOT + '/_raw/taste/ChemTastesDB_database.xlsx', dtype=str)
cls = {'Sweetness': 'sweet', 'Bitterness': 'bitter', 'Umaminess': 'umami'}
t = t[t['Class taste'].isin(cls)].copy(); t['y'] = t['Class taste'].map(cls)
t['can'] = t['canonical SMILES'].map(lambda x: canon(x) if isinstance(x, str) else None)
t = t[t.can.notna()]
amb = t.groupby('can').y.nunique(); t = t[t.can.map(amb) == 1].drop_duplicates('can').reset_index(drop=True)
t['fp'] = [fp(x) for x in t.can]
print('dataset', t.y.value_counts().to_dict())
t['cidn'] = pd.to_numeric(t['PubChem CID'], errors='coerce')
cand = t[t.cidn.notna() & ~t.Name.str.lower().isin(FAMOUS)].sample(frac=1, random_state=3)
picked = []
for y, n in (('sweet', 14), ('bitter', 13), ('umami', 13)):
    c = cand[cand.y == y].head(150)
    syn = synonym_counts([int(x) for x in c.cidn], ROOT + '/taste/pubchem_syn_cache.json')
    c = c.assign(nsyn=[syn[str(int(x))] for x in c.cidn]); c = c[(c.nsyn >= 0) & (c.nsyn <= 15)]
    print(y, 'obscure', len(c)); picked.append(c.head(n))
items = pd.concat(picked).sample(frac=1, random_state=5)
LET = {'sweet': 'A', 'bitter': 'B', 'umami': 'C'}
choices = {'A': 'sweet', 'B': 'bitter', 'C': 'umami (savory)'}
rows = []
for k, (_, r) in enumerate(items.iterrows()):
    prompt = ("What is the dominant taste of the molecule below, as reported by human taste panels?\n\n"
              f"SMILES: {r.can}\n\nA) sweet\nB) bitter\nC) umami (savory)\n\nAnswer with a single letter.")
    rows.append({'id': f'taste-{k:03d}', 'hypothesis': 'taste', 'prompt': prompt, 'answer': LET[r.y], 'choices': choices,
                 'meta': {'name': r.Name, 'cid': int(r.cidn), 'pubchem_synonyms': int(r.nsyn), 'chemtastesdb_id': r.ID,
                          'taste_raw': r.Taste, 'source': 'ChemTastesDB v2.1 (Zenodo 15051366, CC-BY-4.0)'}})
write_jsonl(ROOT + '/taste/tasks.jsonl', rows)
for i in (0, 1): open(ROOT + f'/taste/examples/example_{i+1}.txt', 'w').write(rows[i]['prompt'] + f"\n\n[answer: {rows[i]['answer']}]\n")
ids = set(items.index); ifps = list(items.fp)
X = lambda d: np.array([list(f) for f in d.fp], dtype=np.uint8)
def ev(train, tag):
    yt = np.array([LET[v] for v in train.y]); yi = np.array([r['answer'] for r in rows])
    lr = LogisticRegression(C=0.1, max_iter=3000, class_weight='balanced').fit(X(train), yt)
    a_lr = (lr.predict(X(items)) == yi).mean()
    pk = []
    for f in items.fp:
        s = np.array(DataStructs.BulkTanimotoSimilarity(f, list(train.fp))); top = np.argsort(-s)[:5]
        v = pd.Series(s[top], index=yt[top]).groupby(level=0).sum(); pk.append(v.idxmax())
    a_knn = (np.array(pk) == yi).mean()
    print(f'{tag}: n_train={len(train)} LR={a_lr:.3f} kNN5={a_knn:.3f}')
train = t[~t.index.isin(ids)]
ev(train, 'baseline (train = all non-item tastants)')
far = [max(DataStructs.BulkTanimotoSimilarity(f, ifps)) < 0.6 for f in train.fp]
ev(train[far], 'strict (train Tc>=0.6 to any item removed)')
print('letters', pd.Series([r['answer'] for r in rows]).value_counts().to_dict())
print('heavy-atom heuristic check: umami items with peptide bond', sum('N' in r['prompt'] and 'C(=O)N' in r['prompt'] for r in rows if r['answer']=='C'))
