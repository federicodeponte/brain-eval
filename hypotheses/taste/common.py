import json, time, os, requests, numpy as np
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem
RDLogger.DisableLog('rdApp.*')
ROOT = os.path.expanduser('~/work/vals-hyp')
FAMOUS = set(x.lower() for x in """vanillin limonene menthol linalool geraniol citral eugenol coumarin camphor
benzaldehyde cinnamaldehyde ethyl vanillin carvone anethole thymol nerol citronellal citronellol
ionone damascone muscone civetone indole skatole furaneol maltol ethyl maltol sotolon
sucrose glucose fructose saccharin aspartame sucralose stevioside rebaudioside caffeine quinine
theobromine nicotine strychnine naringin denatonium monosodium glutamate glutamic acid
cyclamate acesulfame neotame alitame thaumatin glycyrrhizin""".replace('\n',' ').split('  ') ) | set(
 ['vanillin','limonene','menthol','linalool','geraniol','citral','eugenol','coumarin','camphor','benzaldehyde',
  'cinnamaldehyde','carvone','anethole','thymol','indole','skatole','maltol','sucrose','glucose','fructose',
  'saccharin','aspartame','sucralose','caffeine','quinine','nicotine','strychnine','naringin','cyclamate',
  'acesulfame','neotame','glycyrrhizin','stevioside','muscone','furaneol','sotolon','nootkatone','ambroxide'])

def fp(smi, n=2048):
    m = Chem.MolFromSmiles(smi)
    if m is None: return None
    return AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=n)

def canon(smi):
    m = Chem.MolFromSmiles(smi)
    return Chem.MolToSmiles(m) if m else None

def tanimoto_matrix(fa, fb):
    return np.array([DataStructs.BulkTanimotoSimilarity(a, fb) for a in fa])

def synonym_counts(cids, cache_path):
    cache = json.load(open(cache_path)) if os.path.exists(cache_path) else {}
    todo = [str(c) for c in cids if str(c) not in cache and int(c) > 0]
    for i in range(0, len(todo), 100):
        chunk = todo[i:i+100]
        try:
            r = requests.get('https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/%s/synonyms/JSON' % ','.join(chunk), timeout=60)
            for info in r.json().get('InformationList', {}).get('Information', []):
                cache[str(info['CID'])] = len(info.get('Synonym', []))
        except Exception as e:
            print('pubchem err', e)
        for c in chunk: cache.setdefault(c, -1)
        time.sleep(0.3)
    json.dump(cache, open(cache_path, 'w'))
    return {str(c): (0 if int(c) <= 0 else cache.get(str(c), -1)) for c in cids}

def balanced_positions(n, k=4, seed=0):
    pos = [i % k for i in range(n)]
    rng = np.random.default_rng(seed); rng.shuffle(pos)
    return pos

def write_jsonl(path, rows):
    with open(path, 'w') as f:
        for r in rows: f.write(json.dumps(r) + '\n')
