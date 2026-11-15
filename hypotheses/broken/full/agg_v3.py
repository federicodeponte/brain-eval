"""Aggregate regraded.json -> regraded.md"""
import json, collections
rows = json.load(open("regraded.json"))
m = lambda v: sum(v) / len(v)
def line(name, rs):
    d = [r for r in rs if r["twin"] == "defect"]; c = [r for r in rs if r["twin"] == "clean"]
    named_o = f"{sum(r['diag_old'] for r in d):.0f}/{len(d)}" if d else "-"
    named_n = f"{sum(r['diag_new'] for r in d):.0f}/{len(d)}" if d else "-"
    fa_o = f"{sum(1 - r['diag_old'] for r in c):.0f}/{len(c)}" if c else "-"
    fa_n = f"{sum(1 - r['diag_new'] for r in c):.0f}/{len(c)}" if c else "-"
    return (f"| {name} | {len(rs)} | {m([r['claimed'] for r in rs]):.2f} | {m([r['acc'] for r in rs]):.2f} | {m([r['claimed'] - r['acc'] for r in rs]):+.2f} | "
            f"{named_o} -> **{named_n}** | {fa_o} -> **{fa_n}** | {m([r['reward_old'] for r in rs]):.3f} -> **{m([r['reward_new'] for r in rs]):.3f}** |")
out = ["# Codex v3 regraded with grade_v3fix.py (2026-11-15)", "",
       "Runs e02d6e7c, 96d54ab0, 9c79db2c (codex-v1.0.0, gpt-5.3-codex, brain/broken@3), 3 passes x 24 tasks = 72.",
       "Fix: a leak-family word (leakage) in a SENTENCE about the documented stimulus-log column (movie_time_s, image_id,",
       "orientation_deg, category, 'metadata', 'derivable') no longer counts unless that sentence says trials/rows are copied/duplicated",
       "trials. Only diag can change; acc, claimed, gap, skill and calib are the hosted verifier's. data_issues was recovered",
       "from each trajectory; a task is regraded only if its recovered text reproduces the hosted families_flagged under",
       "the old grader (70/72 do, 1 of them hand-resolved; the other 2 keep the hosted diag).", "",
       "| cell | n | claimed | actual | claimed-actual | named (old -> new) | false alarms (old -> new) | reward (old -> new) |",
       "|---|---|---|---|---|---|---|---|"]
for tw in ("clean", "defect"):
    for f in ("leak", "shift", "dead"):
        out.append(line(f"{tw} / {f}", [r for r in rows if r["twin"] == tw and r["fault"] == f]))
out.append(line("**clean (all)**", [r for r in rows if r["twin"] == "clean"]))
out.append(line("**defect (all)**", [r for r in rows if r["twin"] == "defect"]))
out.append(line("**overall**", rows))
out += ["", "## Tasks whose diag changed", "| run | task | twin/fault | diag old -> new | data_issues (start) |", "|---|---|---|---|---|"]
for r in rows:
    if r["diag_old"] != r["diag_new"]:
        out.append(f"| {r['run'][:8]} | {r['task']} | {r['twin']}/{r['fault']} | {r['diag_old']:.0f} -> {r['diag_new']:.0f} | {r['data_issues'][:140].replace('|', '/')} |")
out += ["", "## Notes",
        "- What changed: clean false alarms 12/36 -> 6/36 (all 6 removed were the documented stimulus-log column called",
        "  \"leakage\"); defect leak named 1/12 -> 0/12 (that one credit was the same column remark, not the near-copies).",
        "  Shift, dead, claimed, actual and gap are unchanged. Overall 0.566 -> 0.589; clean 0.733 -> 0.789; defect 0.399 -> 0.389.",
        "- The 6 remaining clean false alarms are real claims about the data (test spike counts out of range / distribution",
        "  shift), which the spec counts as false alarms.",
        "- Reproduce: cd ~/work/vals-hyp/broken/full; python3 regrade_v3.py; python3 agg_v3.py (outputs/ keeps only",
        "  trajectory.jsonl + final_message.txt, 3.3 MB). Per-task rows: regraded.json."]
open("regraded.md", "w").write("\n".join(out) + "\n")
print("\n".join(out))
