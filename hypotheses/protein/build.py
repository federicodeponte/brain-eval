import json, glob, os, numpy as np, pandas as pd
from Bio.Align import substitution_matrices
B = substitution_matrices.load("BLOSUM62")
ref = pd.read_csv("raw/DMS_substitutions.csv").set_index("DMS_id")
ASSAYS = ["MLAC_ECOLI_MacRae_2023", "RNC_ECOLI_Weeks_2023", "ESTA_BACSU_Nutschel_2020", "A0A247D711_LISMN_Stadelmann_2021"]
L = "ABCD"; rng = np.random.default_rng(11)
items, bl, pos_hit, hyd_hit = [], 0, 0, 0
KD = dict(A=1.8,R=-4.5,N=-3.5,D=-3.5,C=2.5,Q=-3.5,E=-3.5,G=-0.4,H=-3.2,I=4.5,L=3.8,K=-3.9,M=1.9,F=2.8,P=-1.6,S=-0.8,T=-0.7,W=-0.9,Y=-1.3,V=4.2)
k = 0
for a in ASSAYS:
    d = pd.read_csv(f"raw/assays/{a}.csv"); d = d[~d.mutant.str.contains(":")].copy()
    wt = ref.loc[a, "target_seq"]; name = ref.loc[a, "molecule_name"]; org = ref.loc[a, "source_organism"]; assay = ref.loc[a, "selection_assay"]
    d["wt"] = d.mutant.str[0]; d["mt"] = d.mutant.str[-1]; d["pos"] = d.mutant.str[1:-1].astype(int)
    assert all(wt[p - 1] == w for p, w in zip(d.pos, d.wt)), a
    lo, hi = d.DMS_score.quantile(0.10), d.DMS_score.quantile(0.50)
    bad = d[d.DMS_score <= lo]; good = d[d.DMS_score >= hi]
    for j in range(10):
        while True:
            b = bad.sample(1, random_state=int(rng.integers(1e9))).iloc[0]
            g = good[good.pos != b.pos].sample(3, random_state=int(rng.integers(1e9)))
            if g.pos.nunique() == 3: break
        ans = L[k % 4]; k += 1
        others = list(g.itertuples()); rng.shuffle(others)
        opts = []; oi = 0
        for li in L:
            if li == ans: opts.append(b)
            else: opts.append(others[oi]); oi += 1
        muts = [o.mutant for o in opts]; scores = [float(o.DMS_score) for o in opts]
        prompt = (f"Protein sequence ({len(wt)} residues, 1-indexed):\n{wt}\n\n"
                  "Below are four single amino-acid substitutions (format: wild-type residue, position, new residue). "
                  "A deep mutational scan measured how each one affects the protein's function. Exactly one of them is strongly damaging; the other three are tolerated.\n"
                  + "\n".join(f"{l}. {m}" for l, m in zip(L, muts)) +
                  "\n\nWhich substitution damages function the most? Answer with a single letter.")
        # baselines (no query labels)
        bs = [B[m[0]][m[-1]] for m in muts]; bl += L[int(np.argmin(bs))] == ans
        q = set(muts); rest = d[~d.mutant.isin(q)]
        pm = [rest[rest.pos == int(m[1:-1])].DMS_score.mean() for m in muts]; pos_hit += L[int(np.nanargmin(pm))] == ans
        hd = [abs(KD[m[0]] - KD[m[-1]]) + (3 if m[-1] == "P" else 0) for m in muts]; hyd_hit += L[int(np.argmax(hd))] == ans
        items.append({"id": f"protein-{len(items):03d}", "hypothesis": "protein-mutations", "prompt": prompt, "answer": ans,
                      "choices": dict(zip(L, muts)),
                      "meta": {"assay": a, "uniprot": ref.loc[a, "UniProt_ID"], "molecule": name, "organism": org, "selection_assay": assay,
                               "dms_scores": dict(zip(L, scores)), "q10": float(lo), "q50": float(hi), "blosum": dict(zip(L, map(int, bs)))}})
with open("tasks.jsonl", "w") as f:
    for o in items: f.write(json.dumps(o) + "\n")
for o in (items[0], items[25]): open(f"examples/{o['id']}.txt", "w").write(o["prompt"] + f"\n\n[answer: {o['answer']}]\n")
n = len(items)
print("items", n, {l: sum(o["answer"] == l for o in items) for l in L}, "max chars", max(len(o["prompt"]) for o in items))
print("chance 0.25 | blosum62_min", round(bl / n, 3), "| hydropathy+proline", round(hyd_hit / n, 3), "| position_mean_other_variants", round(pos_hit / n, 3))
for a in ASSAYS:
    sub = [o for o in items if o["meta"]["assay"] == a]
    print(a, "blosum", sum(min(o["meta"]["blosum"], key=o["meta"]["blosum"].get) == o["answer"] for o in sub), "/", len(sub))
