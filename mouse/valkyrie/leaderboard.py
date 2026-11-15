#!/usr/bin/env python3
"""Leaderboard from `valkyrie run results` files. Usage: python3 leaderboard.py results-*.json [--md]
Accuracy = exact-match letter; also reports unparsed answers, errored tasks, and accuracy by k."""
import json, re, sys
from collections import defaultdict
rows = []
files = [f for f in sys.argv[1:] if not f.startswith("--")]
for f in files:
    d = json.load(open(f))
    model = d["benchmark_arguments"]["contract"].get("model", "?")
    ev = d["evaluation_results"]
    n = len(ev); correct = unparsed = errors = 0
    byk = defaultdict(lambda: [0, 0])
    for tid, r in ev.items():
        if not r or "rewards" not in r:
            errors += 1; continue
        ok = r["rewards"].get("reward", 0) >= 1
        correct += ok
        out = (r.get("metadata") or {}).get("verifier_output", "")
        if "predicted=None" in out: unparsed += 1
    acc = correct / n * 100 if n else 0
    rows.append((acc, model, n, correct, unparsed, errors, d["final_evaluation"]["final_score"], f))
rows.sort(reverse=True)
md = "--md" in sys.argv
hdr = ["model", "accuracy %", "correct/n", "unparsed", "errored", "valkyrie score", "file"]
if md:
    print("| " + " | ".join(hdr) + " |"); print("|" + "---|" * len(hdr))
for acc, model, n, c, u, e, vs, f in rows:
    cells = [model, f"{acc:.1f}", f"{c}/{n}", str(u), str(e), f"{vs:.1f}", f]
    print(("| " + " | ".join(cells) + " |") if md else "  ".join(cells))
