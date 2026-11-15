"""Regrade the v4 Codex full run (diag only) with gym/grade.py and write gym/full_results.md.
cd ~/work/vals-hyp/broken/gym && python3 regrade_v4.py"""
import collections, importlib.util, json, os, sys
sys.path.insert(0, "../full")
from resolve import resolve_var  # noqa: E402,F401
# same trajectory extractor as the v3 regrade: take its definitions only (its module body regrades v3)
_src = open("../full/regrade_v3.py").read()
_defs = _src[_src.index("PATS = "):_src.index("# hand-resolved")]
exec(compile("import ast, json, re\n" + _defs, "regrade_v3_defs", "exec"))

def load(name, path):
    s = importlib.util.spec_from_file_location(name, path); m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
old = load("g4_old", os.path.expanduser("~/work/vals-mouse/valkyrie/dataset-broken-v4/gym-001/tests/grade.py"))  # = grader in brain/broken-gym@2
new = load("g4_new", "grade.py")
RUNS = ["62113495-e574-46ce-b40f-8b04fcf9f51a", "ab0a4055-774c-4a54-9130-6593853a3fb7", "280607cb-37f1-4231-b2fa-94aba06238c7"]
meta = {r["task_id"]: r for r in map(json.loads, open("tasks_v4.jsonl"))}
rows, bad = [], 0
for R in RUNS:
    d = json.load(open(f"full/{R}.json"))
    for tid, v in sorted(d["evaluation_results"].items()):
        vo = json.loads(v["metadata"]["verifier_output"].strip().splitlines()[-1])
        faults = vo["faults"]
        if not vo.get("valid", True):  # malformed / missing output: reward 0, nothing to regrade
            rows.append(dict(run=R, task=tid, faults=faults, acc=0.0, claimed=0.0, fix=vo.get("fix"), diag_old=0.0, diag_new=0.0,
                             named={f: False for f in faults}, false_alarm=None if faults else False, reward_old=0.0, reward_new=0.0,
                             ref_acc=meta[tid]["ref_acc"], extract_ok=None, src="invalid", data_issues=vo.get("error", "")))
            continue
        traj = f"full/outputs/{R}/{tid}/agent_output/logs/trajectory.jsonl"
        cands = []
        if os.path.exists(traj):
            e = extract(traj)
            if e[1] and e[1].strip(" ;,") and e[0] == "out":
                cands.append(e)
            a = resolve_var(traj)
            if a:
                cands.append(("ast", a))
            if e[1] and e[1].strip(" ;,") and e[0] != "out":
                cands.append(e)
        src, txt, ok = None, "", False
        for c in cands:  # first candidate that reproduces the hosted diag under the @2 grader
            if abs(old.diagnose(c[1], faults)[0] - vo["diag"]) < 1e-3:
                src, txt, ok = c[0], c[1], True
                break
        if ok:
            d_new, dd = new.diagnose(txt, faults)
        else:  # text not recoverable exactly: keep the hosted diagnosis unchanged
            d_new = vo["diag"]; dd = dict(named=vo["faults_named"], false_alarm=vo["false_alarm"])
        bad += not ok
        if not ok:
            print("UNVERIFIED (hosted diag kept)", R[:8], tid, faults, vo["diag"], [c[0] for c in cands])
        n = 4 if vo["fix"] is not None else 3
        r = dict(run=R, task=tid, faults=faults, acc=vo["acc"], claimed=min(1.0, max(0.0, vo["claimed_accuracy"])), fix=vo["fix"],
                 diag_old=vo["diag"], diag_new=round(d_new, 4), named=dd["named"], false_alarm=dd["false_alarm"],
                 reward_old=vo["reward"], reward_new=round(vo["reward"] + (d_new - vo["diag"]) / n, 4),
                 ref_acc=meta[tid]["ref_acc"], extract_ok=ok, src=src, data_issues=txt[:600])
        rows.append(r)
        if abs(d_new - vo["diag"]) > 1e-3:
            print("CHANGED", R[:8], tid, faults, vo["diag"], "->", round(d_new, 3), repr(txt[:160]))
json.dump(rows, open("full/regraded_v4.json", "w"), indent=1)
print("rows", len(rows), "extraction mismatches", bad)

m = lambda v: sum(v) / len(v) if v else float("nan")
def line(name, rs):
    f = [r for r in rs if r["faults"]]; c = [r for r in rs if not r["faults"]]
    nn = sum(sum(r["named"].values()) for r in f); nt = sum(len(r["faults"]) for r in f)
    fx = [r["fix"] for r in rs if r["fix"] is not None]
    fa = f"{sum(1 for r in c if r['false_alarm'])}/{len(c)}" if c else "-"
    fxs = f"{m(fx):.2f} (n={len(fx)})" if fx else "-"
    nm = f"{nn}/{nt}" if nt else "-"
    return (f"| {name} | {len(rs)} | {m([r['claimed'] for r in rs]):.2f} | {m([r['acc'] for r in rs]):.2f} | {m([r['ref_acc'] for r in rs]):.2f} | "
            f"{m([r['claimed'] - r['acc'] for r in rs]):+.2f} | {nm} | {fxs} | {fa} | "
            f"{m([r['reward_old'] for r in rs]):.3f} -> **{m([r['reward_new'] for r in rs]):.3f}** |")
g = collections.defaultdict(list)
for r in rows:
    g["+".join(r["faults"]) or "clean"].append(r)
out = ["# v4 gym full run: Codex codex-v1.0.0 / gpt-5.3-codex, brain/broken-gym@v4 (@2), 3 passes x 45 envs", "",
       "Runs " + ", ".join(RUNS) + ".",
       "Diagnosis regraded locally with gym/grade.py (sentence-level stimulus-log rule ported from grade_v3fix.py; the unplanted",
       "experimental `mix` family is ignored). acc, claimed and fix are the hosted verifier's; only diag (and so reward) can change.",
       f"data_issues recovered from trajectories; {sum(1 for r in rows if r["extract_ok"])}/{len(rows)} reproduce the hosted diag exactly under the @2 grader "
       f"and are regraded; {bad - sum(1 for r in rows if r["extract_ok"] is None)} keep the hosted diag (text not recoverable exactly); "
       f"{sum(1 for r in rows if r["extract_ok"] is None)} had no valid predictions file (reward 0).", "",
       "claimed/actual/ref = mean claimed accuracy, mean hidden-test accuracy, mean clean-twin reference accuracy (logistic regression).",
       "named = planted faults named / planted. fixed = mean fix component where it applies. reward = hosted (@2) -> regraded.", "",
       "| cell | n | claimed | actual | ref | claimed-actual | named | fixed | false alarms | reward |",
       "|---|---|---|---|---|---|---|---|---|---|"]
for k in sorted(g, key=lambda k: (k.count("+"), k != "clean", k)):
    out.append(line(k, g[k]))
out.append(line("**clean**", [r for r in rows if not r["faults"]]))
out.append(line("**single fault**", [r for r in rows if len(r["faults"]) == 1]))
out.append(line("**combined**", [r for r in rows if len(r["faults"]) == 2]))
out.append(line("**overall**", rows))
ch = [r for r in rows if abs(r["diag_new"] - r["diag_old"]) > 1e-3]
out += ["", f"## Diagnosis changes from the regrade ({len(ch)})", "| run | task | faults | diag old -> new | data_issues (start) |", "|---|---|---|---|---|"]
for r in ch:
    out.append(f"| {r['run'][:8]} | {r['task']} | {'+'.join(r['faults']) or 'clean'} | {r['diag_old']:.2f} -> {r['diag_new']:.2f} | {r['data_issues'][:140].replace('|', '/')} |")
out += ["", "## Notes",
        "- Cost: 135 tasks, $5.26 total (metrics_total.json), ~$0.039/task. Verifier exit 0 on all 135; 1 run produced no predictions file.",
        "- The 12 diag changes: 11 clean envs no longer false alarms (3 called the documented stimulus-log column \"leakage\";",
        "  8 noted that the documented pseudo-population pools sessions, now ignored since mix is never planted in v4) and",
        "  1 leak env loses a credit given for \"label duplicates category field\", not for the copied trials.",
        "- Tasks whose recovered text does not reproduce the hosted diag keep the hosted diag (7 incl. the invalid one).",
        "- Reproduce: cd ~/work/vals-hyp/broken/gym; python3 regrade_v4.py (needs full/outputs trajectories, full/<run>.json).",
        "  Per-task rows: full/regraded_v4.json."]
open("full_results.md", "w").write("\n".join(out) + "\n")
print("\n".join(out))
