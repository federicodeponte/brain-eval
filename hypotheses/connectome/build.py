import json, sys, numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
VARIANT = sys.argv[1]  # "mb" or "super"
if VARIANT == "mb":
    COL = "cell_class"; CLS = ["Kenyon_Cell", "MBON", "DAN", "ALPN"]
    NAMES = {"Kenyon_Cell": "Kenyon cell (mushroom body intrinsic neuron)", "MBON": "mushroom body output neuron (MBON)",
             "DAN": "dopaminergic neuron (DAN)", "ALPN": "antennal lobe projection neuron (ALPN)"}
else:
    COL = "super_class"; CLS = ["central", "descending", "ascending", "visual_projection"]
    NAMES = {"central": "central brain intrinsic neuron", "descending": "descending neuron (brain to ventral nerve cord)",
             "ascending": "ascending neuron (ventral nerve cord to brain)", "visual_projection": "visual projection neuron (optic lobe to central brain)"}
L = "ABCD"
a = pd.read_csv("raw/ann.tsv", sep="\t", low_memory=False, usecols=["root_id", COL, "cell_type", "side"])
a = a[a[COL].isin(CLS)]
rng = np.random.default_rng(5)
sel = pd.concat([g.sample(min(len(g), 400), random_state=1) for _, g in a.groupby(COL)])
ids = set(sel.root_id)
pre = pd.read_feather("raw/per_neuron_neuropil_count_pre_783.feather"); pre = pre[pre.pre_pt_root_id.isin(ids)]
post = pd.read_feather("raw/per_neuron_neuropil_count_post_783.feather"); post = post[post.post_pt_root_id.isin(ids)]
out_m = pre.pivot_table(index="pre_pt_root_id", columns="neuropil", values="count", aggfunc="sum", fill_value=0)
in_m = post.pivot_table(index="post_pt_root_id", columns="neuropil", values="count", aggfunc="sum", fill_value=0)
sel = sel[sel.root_id.isin(out_m.index) & sel.root_id.isin(in_m.index)].copy()
regs = sorted(set(out_m.columns) | set(in_m.columns))
out_m = out_m.reindex(index=sel.root_id, columns=regs, fill_value=0); in_m = in_m.reindex(index=sel.root_id, columns=regs, fill_value=0)
sel = sel[(out_m.values.sum(1) >= 20) & (in_m.values.sum(1) >= 20)]
def summ(rid, k=12):
    def top(m):
        r = m.loc[rid]; r = r[r > 0].sort_values(ascending=False); tot = int(r.sum())
        s = ", ".join(f"{n} {int(v)}" for n, v in r.head(k).items())
        return tot, s + (f", other regions {int(r.iloc[k:].sum())}" if len(r) > k else "")
    ti, si = top(in_m); to, so = top(out_m)
    return f"  inputs (synapses received, {ti} total): {si}\n  outputs (synapses made, {to} total): {so}"
shots, tests, pool = [], [], []
for c in CLS:
    s = sel[sel[COL] == c].sample(frac=1, random_state=2)
    shots.append(s.iloc[:2]); tests.append(s.iloc[2:12]); pool.append(s.iloc[12:])
shots = pd.concat(shots).sample(frac=1, random_state=3); tests = pd.concat(tests).sample(frac=1, random_state=4); pool = pd.concat(pool)
legend = "\n".join(f"{L[i]} = {NAMES[c]}" for i, c in enumerate(CLS))
head = ("Below is a summary of one neuron from the adult fruit fly (Drosophila) whole-brain connectome (FlyWire). For each brain region "
        "(standard FlyWire neuropil abbreviations; _L/_R = left/right hemisphere) you get the number of synapses where the neuron receives input "
        "and where it makes output. Regions are sorted by count.\n\nClasses:\n" + legend + "\n\nLabeled examples:\n" +
        "\n\n".join(f"Example (class {L[CLS.index(r[COL])]}):\n" + summ(r.root_id) for _, r in shots.iterrows()))
items = []
for i, (_, r) in enumerate(tests.iterrows()):
    items.append({"id": f"connectome-{VARIANT}-{i:03d}", "hypothesis": f"fly-connectome-{VARIANT}",
                  "prompt": head + "\n\nQuery neuron:\n" + summ(r.root_id) + "\n\nWhich class is the query neuron? Answer with a single letter.",
                  "answer": L[CLS.index(r[COL])], "choices": {L[j]: NAMES[c] for j, c in enumerate(CLS)},
                  "meta": {"root_id": str(r.root_id), "class": r[COL], "cell_type": None if pd.isna(r.cell_type) else r.cell_type, "side": r.side}})
fn = "tasks.jsonl" if VARIANT == "super" else f"tasks_{VARIANT}.jsonl"
with open(fn, "w") as f:
    for o in items: f.write(json.dumps(o) + "\n")
for o in items[:2]: open(f"examples/{o['id']}.txt", "w").write(o["prompt"] + f"\n\n[answer: {o['answer']}]\n")
def X(df):
    i = in_m.loc[df.root_id].values; o = out_m.loc[df.root_id].values
    return np.hstack([i / i.sum(1, keepdims=True), o / o.sum(1, keepdims=True)])
y = lambda df: np.array(df[COL].tolist()); yt = y(tests)
print(VARIANT, "items", len(items), {l: sum(o["answer"] == l for o in items) for l in L}, "max chars", max(len(o["prompt"]) for o in items), "pool", len(pool))
print(" knn1_on_8_shots", round(float((KNeighborsClassifier(1).fit(X(shots), y(shots)).predict(X(tests)) == yt).mean()), 3),
      "| logreg_pool", round(float((LogisticRegression(max_iter=3000, C=10).fit(X(pool), y(pool)).predict(X(tests)) == yt).mean()), 3),
      "| knn5_pool", round(float((KNeighborsClassifier(5).fit(X(pool), y(pool)).predict(X(tests)) == yt).mean()), 3))
