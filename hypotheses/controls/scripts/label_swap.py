"""label-swap control: permute category names consistently (example headers + options) per item.
Correct answer = example-consistent letter (unchanged). prior_answer = letter whose shown name is the true query category."""
import json, re, sys, random
from collections import Counter
src, out = sys.argv[1], sys.argv[2]
rng = random.Random(7)
def derange(xs):
    while True:
        p = xs[:]; rng.shuffle(p)
        if all(a != b for a, b in zip(xs, p)): return p
items = []
for line in open(src):
    t = json.loads(line); L = "ABCD"
    true = [c.split(") ", 1)[1] for c in t["choices"]]          # true category per letter
    shown = derange(true)                                        # displayed name per letter
    m = dict(zip(true, shown))
    p = t["prompt"]
    for li, c in enumerate(true):
        old_h = f"Class {L[li]} = photo of {c}:"; assert p.count(old_h) == 1
        p = p.replace(old_h, f"Class {L[li]} = photo of @@{li}@@:")
        old_o = f"{L[li]}) {c}"; assert p.count(old_o) == 1, (old_o, t["id"])
        p = p.replace(old_o, f"{L[li]}) ##{li}##")
    for li in range(4):
        p = p.replace(f"@@{li}@@", shown[li]).replace(f"##{li}##", shown[li])
    ans = t["answer"]; qcat = t["meta"]["query_category"]; assert true[L.index(ans)] == qcat
    prior = L[shown.index(qcat)]; assert prior != ans
    meta = dict(t["meta"]); meta.update(answer_example_consistent=ans, answer_prior_consistent=prior, query_true_category=qcat, query_shown_category=shown[L.index(ans)], true_labels=dict(zip(L, true)), shown_labels=dict(zip(L, shown)), source_item=t["id"])
    items.append(dict(id=t["id"].replace("human-mtl-category", "label-swap"), k=4, prompt=p, answer=ans,
        choices=[f"{L[i]}) {shown[i]}" for i in range(4)], meta=meta))
with open(out, "w") as f:
    for it in items: f.write(json.dumps(it) + "\n")
nc = sum(it["meta"]["nc_correlation_pred"] == it["answer"] for it in items) / len(items)
ncp = sum(it["meta"]["nc_correlation_pred"] == it["meta"]["answer_prior_consistent"] for it in items) / len(items)
print(json.dumps(dict(items=len(items), chance=0.25, nc_example_consistent=nc, nc_prior_consistent=ncp,
    letters=dict(sorted(Counter(i["answer"] for i in items).items())),
    prior_letters=dict(sorted(Counter(i["meta"]["answer_prior_consistent"] for i in items).items())))))
