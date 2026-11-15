"""Summarize gen.py --validate jsonl files: python summarize.py val_*.jsonl"""
import json, sys, collections
rows = [json.loads(l) for f in sys.argv[1:] for l in open(f) if l.strip()]
g = collections.defaultdict(list)
for r in rows:
    g["+".join(r["faults"]) or "clean"].append(r)
print("| fault | n | ref test | naive val | naive CV | naive test | CV-test gap | min gap | aware test | fix applies |")
print("|---|---|---|---|---|---|---|---|---|---|")
m = lambda v: sum(v) / len(v)
for k in sorted(g, key=lambda k: (k.count("+"), k != "clean", k)):
    v = g[k]
    gaps = [r["naive_cv"] - r["naive_test"] for r in v]
    print(f"| {k} | {len(v)} | {m([r['ref_test'] for r in v]):.2f} | {m([r['naive_val'] for r in v]):.2f} | {m([r['naive_cv'] for r in v]):.2f} | "
          f"{m([r['naive_test'] for r in v]):.2f} | {m(gaps):+.2f} | {min(gaps):+.2f} | {m([r['aware_test'] for r in v]):.2f} | {sum(r['fix_applies'] for r in v)}/{len(v)} |")
