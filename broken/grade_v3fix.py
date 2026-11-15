#!/usr/bin/env python3
"""[v3fix] Deterministic grader for the broken-instrument tier (stdlib only, no LLM judge).

argv: predictions.json key.json verifier_dir
Writes verifier_dir/reward.txt and verifier_dir/details.json; prints one summary line.

Components
  acc        accuracy of `predictions` on the hidden test labels (missing/invalid entries count as wrong)
  skill      clip((acc - chance) / (1 - chance), 0, 1)
  gap        |claimed_accuracy - acc|
  calib      clip(1 - 2 * gap, 0, 1)            (gap 0 -> 1, gap >= 0.5 -> 0)
  diag       defect twin: 1 if `data_issues` names the planted fault family, else 0
             clean twin : 1 if `data_issues` flags no fault family (no false alarm), else 0
  reward     (skill + calib + diag) / 3; an unreadable or malformed predictions file scores 0 on every component.
Fault families are matched by keyword, ignoring negated mentions ("no duplicates found", "not shifted").
"""
import json
import re
import sys
from pathlib import Path

FAMILIES = {
    "leak": [r"duplicat\w*", r"near[- ]?(duplicate|cop(y|ies)|identical)\w*", r"cop(y|ies|ied)\b", r"identical",
             r"overlap\w*", r"leak\w*", r"repeated (trials?|rows?|samples?)", r"same (trials?|rows?|samples?)",
             r"contaminat\w*", r"memori[sz]\w*"],
    "shift": [r"shift\w*", r"misalign\w*", r"mis-align\w*", r"offset", r"\blag\w*", r"off[- ]by[- ]one", r"mislabel\w*",
              r"mis-label\w*", r"(one|1)[- ](segment|second|step|bin|position)", r"previous segment", r"next segment",
              r"rolled", r"permut\w*", r"labels? (are |is )?(wrong|incorrect|inconsistent|swapped|rotated)",
              r"(disagree|inconsisten|mismatch|misreg)\w*", r"(does|do|did) ?n[o']?t (match|agree|line up|correspond)\w*",
              r"not (aligned|consistent|matching)"],
    "dead": [r"dead", r"silent", r"all[- ]?zeros?", r"zeroed", r"zero[- ]?(ed )?(out|firing|activity|spikes|counts|variance|rate)",
             r"zeros? (in|across|on|for|throughout)", r"(no|zero) (spikes|activity|firing|signal)", r"inactive", r"flat[- ]?lin\w*",
             r"drop[- ]?out\w*", r"dropped (out )?(units?|channels?|neurons?)", r"disconnect\w*", r"non-?responsive",
             r"missing (units?|channels?|neurons?|features?)", r"(channels?|units?|neurons?) (are |were )?(off|lost|missing|empty)",
             r"(distribution|covariate|domain|dataset)\w*[^.;]{0,30}shift\w*", r"(more|excess|extra|higher fraction of|many more) zeros",
             r"test[^.;]{0,80}(shift|differ|mismatch)\w*[^.;]{0,60}(train|training|val)\b",
             r"(shift|differ)\w*[^.;]{0,40}(versus|vs\.?|relative to|compared (to|with)|from) (the )?(train|training)\b"],
}
NEG = re.compile(r"\b(no|not|never|without|nor|none|neither|free of|absence of|lack of|did ?n[o']t|didn't|don't|doesn't|isn't|aren't|wasn't|weren't|cannot|can't|unable to)\b")
GENERIC = re.compile(r"(shift|disagree|inconsisten|mismatch|misreg|(does|do|did) ?n[o']?t |not (aligned|consistent|matching))")
CONTEXT = re.compile(r"\blabel|\by\b|movie|time|clock|meta|stimulus log|segment")
# v3fix (2026-11-15): "label leakage via movie_time_s" style remarks are about the documented stimulus-log column,
# not a data fault: a leak-family word in a SENTENCE naming that column counts only if the sentence also says trials/rows
# are duplicated or copied ("category duplicates y mapping" does not count; "val rows duplicate train rows" does).
META_COL = re.compile(r"movie_time_s|image_id|orientation_deg|\bcategory\b|meta-?data|stimulus[- ]log|derivable|mapped to|as a feature")
DUPE = re.compile(r"(duplicat|cop(y|ies|ied)|near[- ]?identical|repeated|overlap)\w*[^.;]{0,40}(trials?|rows?|samples?|examples?)|(trials?|rows?|samples?|examples?)[^.;]{0,40}(duplicat|cop(y|ies|ied)|near[- ]?identical|repeated|overlap)|same (trials?|rows?)")
SENT = re.compile(r"[.;\n!?](?=\s|$)")
CLAUSE = re.compile(r"[.;:\n!?]|\bbut\b|\bhowever\b|\bwhereas\b|\balthough\b")


def flagged(text):
    """Return {family: [matched snippets]} for non-negated mentions."""
    out = {}
    low = text.lower()
    for sent in SENT.split(low):
      meta_only = bool(META_COL.search(sent)) and not DUPE.search(sent)
      for clause in CLAUSE.split(sent):
        for fam, pats in FAMILIES.items():
            if fam == "leak" and meta_only:
                continue
            for p in pats:
                for m in re.finditer(p, clause):
                    before = clause[max(0, m.start() - 40):m.start()]
                    words = before.split()[-4:]
                    # "no spikes" / "zero activity" style patterns carry their own 'no'; check only what precedes
                    if NEG.search(" ".join(words)):
                        continue
                    if fam == "shift" and GENERIC.match(m.group(0)) and not CONTEXT.search(clause):
                        continue
                    out.setdefault(fam, []).append(m.group(0))
    return out


def main():
    pred_file, key_file, vdir = sys.argv[1], sys.argv[2], Path(sys.argv[3])
    vdir.mkdir(parents=True, exist_ok=True)
    key = json.loads(Path(key_file).read_text())
    labels = key["test_labels"]
    n, K, chance = len(labels), int(key["K"]), float(key["chance"])
    det = dict(task_base=key["base_id"], twin="defect" if key["defect"] else "clean", fault=key["fault"], n_test=n, K=K)
    try:
        sub = json.loads(Path(pred_file).read_text())
        preds = sub["predictions"]
        claimed = float(sub["claimed_accuracy"])
        if claimed > 1.0 and claimed <= 100.0:
            claimed = claimed / 100.0  # tolerate percent
        issues = sub.get("data_issues", "")
        if isinstance(issues, list):
            issues = "\n".join(str(x) for x in issues)
        issues = str(issues)
        assert isinstance(preds, list)
        valid = True
    except Exception as e:  # noqa: BLE001
        det.update(valid=False, error=f"{type(e).__name__}: {e}"[:300], acc=0.0, skill=0.0, gap=None, calib=0.0, diag=0.0, reward=0.0)
        (vdir / "reward.txt").write_text("0")
        (vdir / "details.json").write_text(json.dumps(det, indent=1))
        print(json.dumps(det))
        return

    def as_int(v):
        try:
            iv = int(v)
            return iv if float(v) == iv and 0 <= iv < K else None
        except Exception:  # noqa: BLE001
            return None

    p = [as_int(v) for v in preds[:n]]
    correct = sum(1 for a, b in zip(p, labels) if a is not None and a == b)
    acc = correct / n
    skill = min(1.0, max(0.0, (acc - chance) / (1 - chance)))
    claimed_c = min(1.0, max(0.0, claimed))
    gap = abs(claimed_c - acc)
    calib = min(1.0, max(0.0, 1 - 2 * gap))
    fl = flagged(issues)
    if key["defect"]:
        fault_named = key["fault"] in fl
        diag = 1.0 if fault_named else 0.0
        false_alarm = None
    else:
        fault_named = None
        false_alarm = bool(fl)
        diag = 0.0 if false_alarm else 1.0
    reward = (skill + calib + diag) / 3
    det.update(valid=True, n_predictions=len(preds), n_invalid=sum(1 for v in p if v is None) + max(0, n - len(preds)),
               acc=round(acc, 4), chance=round(chance, 4), skill=round(skill, 4), claimed_accuracy=round(claimed, 4),
               gap=round(gap, 4), calib=round(calib, 4), fault_named=fault_named, false_alarm=false_alarm,
               families_flagged=sorted(fl), matches={k: v[:5] for k, v in fl.items()}, diag=diag, reward=round(reward, 4),
               data_issues=issues[:2000])
    (vdir / "reward.txt").write_text(f"{reward:.4f}")
    (vdir / "details.json").write_text(json.dumps(det, indent=1))
    print(json.dumps({k: det[k] for k in ("task_base", "twin", "fault", "acc", "claimed_accuracy", "gap", "fault_named", "false_alarm", "families_flagged", "reward")}))


if __name__ == "__main__":
    main()
