"""Regrade the 72 Codex v3 results with grade_v3fix.py (diag only changes; acc/claimed come from the hosted verifier).
python3 regrade_v3.py  (run in ~/work/vals-hyp/broken/full) -> regraded.json + stdout checks"""
import ast, json, os, re, sys
sys.path.insert(0, "..")
import importlib.util
from resolve import resolve_var
def load(name, path):
    s = importlib.util.spec_from_file_location(name, path); m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
old = load("g_old", "../grade.py"); new = load("g_new", "../grade_v3fix.py")
RUNS = ["e02d6e7c-b985-4fe4-8edd-49b6d1396d20", "96d54ab0-51ec-4d9c-8fa2-9efd2196b356", "9c79db2c-ecfb-4aef-8cac-ac7d76a2081f"]
PATS = [re.compile(r'"data_issues"\s*:\s*"((?:[^"\\]|\\.)*)"'),
        re.compile(r"'data_issues'\s*:\s*'((?:[^'\\]|\\.)*)'"),
        re.compile(r"'data_issues'\s*:\s*\"((?:[^\"\\]|\\.)*)\"")]
def unq(g, i):
    try:
        return json.loads('"' + g + '"') if i == 0 else ast.literal_eval(("'" if i == 1 else '"') + g + ("'" if i == 1 else '"'))
    except Exception:
        return g
def extract(traj):
    cands = []
    for l in open(traj):
        e = json.loads(l); it = e.get("item", {})
        if e.get("type") != "item.completed" or it.get("type") != "command_execution":
            continue
        for src in (it.get("aggregated_output") or "",):  # printed file / dict wins: it is what was written
            for i, p in enumerate(PATS):
                for m in p.finditer(src):
                    cands.append(("out", unq(m.group(1), i)))
        cmd = (it.get("command") or "").replace('\\"', '"')
        for i, p in enumerate(PATS):
            for m in p.finditer(cmd):
                v = unq(m.group(1), i)
                if v and "join(" not in v:
                    cands.append(("cmd", v))
    outs = [c for c in cands if c[0] == "out"]
    if outs or cands:
        return (outs or cands)[-1]
    v = resolve_var(traj)
    return ("ast", v) if v else (None, None)
# hand-resolved from the trajectory source (bash quote-splicing defeats ast): f-string placeholders shown as {}
MANUAL = {("9c79db2c-ecfb-4aef-8cac-ac7d76a2081f", "decode-02a"):
          "Label/log mismatch: y does not match movie_time_s mapping (expected y = movie_time_s - 10). Mismatches: train {}/{}, val {}/{}."}
rows, bad = [], 0
for R in RUNS:
    d = json.load(open(R + ".json"))
    for tid, v in sorted(d["evaluation_results"].items()):
        vo = json.loads(v["metadata"]["verifier_output"])
        traj = f"outputs/{R}/{tid}/agent_output/logs/trajectory.jsonl"
        src, txt = ("manual", MANUAL[(R, tid)]) if (R, tid) in MANUAL else (extract(traj) if os.path.exists(traj) else (None, None))
        txt = txt or ""
        fam_old = sorted(old.flagged(txt)); fam_new = sorted(new.flagged(txt))
        match = fam_old == sorted(vo["families_flagged"])
        bad += not match
        defect = vo["twin"] == "defect"
        diag_old = (1.0 if vo["fault"] in vo["families_flagged"] else 0.0) if defect else (0.0 if vo["families_flagged"] else 1.0)
        diag_new = (1.0 if vo["fault"] in fam_new else 0.0) if defect else (0.0 if fam_new else 1.0)
        if not match:  # recovered text does not reproduce the hosted verdict: keep the hosted diag
            diag_new = diag_old; fam_new = sorted(vo["families_flagged"])
        acc = vo["acc"]; K = {"dead": None}
        rows.append(dict(run=R, task=tid, base=vo["task_base"], twin=vo["twin"], fault=vo["fault"], acc=acc, claimed=vo["claimed_accuracy"],
                         gap=vo["gap"], reward_old=vo["reward"], reward_new=round(vo["reward"] + (diag_new - diag_old) / 3, 4),
                         diag_old=diag_old, diag_new=diag_new, fam_hosted=vo["families_flagged"], fam_new=fam_new, src=src,
                         extract_ok=match, data_issues=txt[:600]))
        if not match:
            print("UNVERIFIED (hosted diag kept)", R[:8], tid, vo["families_flagged"], fam_old, src, repr(txt[:150]))
        if diag_new != diag_old:
            print("CHANGED", R[:8], tid, vo["twin"], vo["fault"], diag_old, "->", diag_new, repr(txt[:200]))
json.dump(rows, open("regraded.json", "w"), indent=1)
print("rows", len(rows), "extraction mismatches", bad)
