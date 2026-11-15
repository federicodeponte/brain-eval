#!/usr/bin/env python3
"""Deterministic grader for the broken-instrument gym (v4; stdlib only, no LLM judge).

argv: predictions.json key.json verifier_dir  -> verifier_dir/reward.txt, verifier_dir/details.json, one summary line.

Components
  acc     accuracy of `predictions` on the hidden test labels (missing/invalid entries count as wrong)
  skill   clip((acc - chance) / (1 - chance), 0, 1)
  calib   clip(1 - 2 * |claimed_accuracy - acc|, 0, 1)
  diag    clean env : 1 if `data_issues` flags no fault family, else 0 (false alarm)
          faulted   : fraction of planted faults named, minus 0.25 per flagged fault group that matches no planted fault
  fix     only when key.fix_applies (the fault hurts a naive decoder and can be undone from the visible files):
          1 if acc >= fix_target - fix_tol, falling linearly to 0 at 0.2 below that. fix_target = the clean twin's
          reference accuracy (with dead units: the best accuracy reachable after dropping them).
  reward  mean of the applicable components: (skill + calib + diag [+ fix]) / 3 or 4. Malformed output = 0.
Fault families are matched by keyword with negation handling ("no duplicates found" does not count). Families
belong to groups; a generic phrase names every family of its group:
  label group  = shift, noise      (generic: labels disagree with the stimulus log)
  input group  = dead, perm, mix   (generic: test/train input distribution shift)
  leak group   = leak
"""
import json
import re
import sys
from pathlib import Path

LAB = r"(labels?|\by\b|targets?|class(es)?|movie|stimulus|log|meta|image_id|orientation|category|segment)"
FAMILIES = {
    "leak": [r"duplicat\w*", r"near[- ]?(duplicate|cop(y|ies)|identical)\w*", r"cop(y|ies|ied)\b", r"identical (trials?|rows?|samples?)",
             r"overlap\w*", r"leak\w*", r"repeated (trials?|rows?|samples?)", r"same (trials?|rows?|samples?)",
             r"contaminat\w*", r"memori[sz]\w*"],
    "shift": [r"(?<!distribution )(?<!covariate )(?<!domain )(?<!dataset )(?<!input )\bshifted? (by|of) (one|1|two|2|-?\d) ",
              r"labels?[^.;]{0,40}\bshift\w*", r"shift\w*[^.;]{0,40}labels?", r"misalign\w*", r"mis-align\w*", r"\boffset\b",
              r"\blag\w*", r"off[- ]by[- ](one|two|1|2)", r"(one|1|two|2)[- ](segment|second|step|bin|position|class)",
              r"previous segment", r"next segment", r"rolled", r"rotated", r"cyclic\w*", r"(y|labels?) ?= ?[^.;]{0,30}[+-] ?[12]\b"],
    "noise": [r"label[- ]noise", r"noisy labels?", r"labels? (are |is |were )?(noisy|corrupt\w*|flipped|scrambled|shuffled|permuted|randomi[sz]ed|random)",
              r"(random|corrupt\w*|flipped|scrambled|shuffled|wrong|incorrect)[^.;]{0,20}labels?", r"mislabel\w*", r"mis-label\w*",
              r"(subset|fraction|portion|some|part|\d+(\.\d+)? ?%|\d+ ?(of|/) ?\d+)[^.;]{0,40}(labels?|trials?)[^.;]{0,40}(wrong|incorrect|mismatch|disagree|inconsisten|do ?n[o']?t match|corrupt|random)\w*",
              r"(labels?|trials?)[^.;]{0,30}(mismatch|disagree|inconsisten|do ?n[o']?t match)\w*[^.;]{0,50}(\d+(\.\d+)? ?%|\d+ ?(of|/) ?\d+|subset|fraction|portion|some)"],
    "dead": [r"\bdead\b", r"silent", r"all[- ]?zeros?", r"zeroed", r"zero[- ]?(ed )?(out|firing|activity|spikes|counts|variance|rate)",
             r"zeros? (in|across|on|for|throughout)", r"(no|zero) (spikes|activity|firing|signal)", r"inactive", r"flat[- ]?lin\w*",
             r"drop[- ]?out\w*", r"dropped (out )?(units?|channels?|neurons?)", r"disconnect\w*", r"non-?responsive",
             r"missing (units?|channels?|neurons?|features?)", r"(channels?|units?|neurons?) (are |were )?(off|lost|missing|empty)"],
    "perm": [r"permut\w*[^.;]{0,30}(units?|columns?|channels?|neurons?|features?|order)", r"(units?|columns?|channels?|neurons?|features?)[^.;]{0,30}(permut|shuffl|reorder|scrambl|swapp|misorder)\w*",
             r"(shuffl|reorder|scrambl|swapp|misorder)\w*[^.;]{0,20}(units?|columns?|channels?|neurons?|features?)", r"(unit|column|channel|neuron|feature) (order|ordering|mapping|map|index)\w*",
             r"channel[- ]?map\w*", r"(order|mapping) of (the )?(units?|columns?|channels?|neurons?)", r"re-?indexed"],
    "mix": [r"(different|another|other|second|separate|foreign|wrong) (recording|session|animal|subject|mouse|patient|probe|population|set of (units|neurons)|units|neurons|electrode|dataset)\w*",
            r"(mix|mixed|mixture|merg|concatenat|combin|pool)\w*[^.;]{0,40}(sessions?|recordings?|populations?|sources?|animals?|subjects?|datasets?|probes?)",
            r"(two|2|multiple|distinct) (clusters?|populations?|groups|subpopulations?|modes|regimes|sources|batches)", r"bimodal\w*",
            r"batch[- ]effect\w*", r"heterogene\w*", r"(rows?|trials?)[^.;]{0,40}(from|belong to|come from)[^.;]{0,30}(different|another|other) "],
}
GENERIC = {
    "label": [LAB + r"[^.;]{0,60}(disagree|inconsisten|mismatch|misreg|do ?n[o']?t (match|agree|line up|correspond)|not (aligned|consistent|matching)|wrong|incorrect|corrupt)\w*",
              r"(disagree|inconsisten|mismatch|misreg|do ?n[o']?t (match|agree|line up|correspond)|not (aligned|consistent|matching))\w*[^.;]{0,60}" + LAB],
    "input": [r"(distribution|covariate|domain|dataset|input)\w*[^.;]{0,30}shift\w*", r"(more|excess|extra|higher fraction of|many more) zeros",
              r"test[^.;]{0,80}(shift|differ|mismatch|inconsisten)\w*[^.;]{0,60}(train|training|val)\b",
              r"(shift|differ|mismatch)\w*[^.;]{0,40}(versus|vs\.?|relative to|compared (to|with)|from) (the )?(train|training)\b",
              r"(firing|rate|statistics|distribution)[^.;]{0,40}(differ|mismatch|inconsisten)\w*[^.;]{0,40}(test|train)"],
}
GROUP_OF = {"leak": "leak", "shift": "label", "noise": "label", "dead": "input", "perm": "input", "mix": "input"}
NEG = re.compile(r"\b(no|not|never|without|nor|none|neither|free of|absence of|lack of|did ?n[o']t|didn't|don't|doesn't|isn't|aren't|wasn't|weren't|cannot|can't|unable to|rule[sd]? out|ruled out)\b")
# "label leakage via orientation_deg" style remarks are about the documented stimulus-log column, not a data fault:
# a leak-family word in a sentence that names that column counts only if the sentence says trials/rows are copied.
META_COL = re.compile(r"movie_time_s|image_id|orientation_deg|\bcategory\b|meta-?data|stimulus[- ]log|derivable|mapped to|as a feature")
DUPE = re.compile(r"(duplicat|cop(y|ies|ied)|near[- ]?identical|repeated|overlap)\w*[^.;]{0,40}(trials?|rows?|samples?|examples?)|"
                  r"(trials?|rows?|samples?|examples?)[^.;]{0,40}(duplicat|cop(y|ies|ied)|near[- ]?identical|repeated|overlap)|same (trials?|rows?)")
SENT = re.compile(r"[.;\n!?](?=\s|$)")  # the stimulus-log rule works per sentence, so "Label leakage: image_id ..." is covered
CLAUSE = re.compile(r"[.;:\n!?]|\bbut\b|\bhowever\b|\bwhereas\b|\balthough\b")


def flagged(text):
    """Return ({family: snippets}, {group: snippets}) for non-negated mentions."""
    fam, gen = {}, {}
    for sent in SENT.split(text.lower()):
      meta_only = bool(META_COL.search(sent)) and not DUPE.search(sent)
      for clause in CLAUSE.split(sent):
        for table, out in ((FAMILIES, fam), (GENERIC, gen)):
            for name, pats in table.items():
                if table is FAMILIES and name == "leak" and meta_only:
                    continue
                for p in pats:
                    for m in re.finditer(p, clause):
                        words = clause[max(0, m.start() - 40):m.start()].split()[-4:]
                        if NEG.search(" ".join(words)):
                            continue
                        out.setdefault(name, []).append(m.group(0)[:80])
    return fam, gen


def diagnose(text, faults):
    fam, gen = flagged(text)
    if "mix" not in faults:
        # mix is experimental (not in the v4 dataset) and human envs document that neurons are pooled across sessions,
        # so "combined across sessions" remarks are neither credit nor false alarm unless mix is actually planted.
        fam.pop("mix", None)
    groups_flagged = {GROUP_OF[f] for f in fam} | set(gen)
    if not faults:
        return (0.0 if groups_flagged else 1.0), dict(false_alarm=bool(groups_flagged), named=None, extra=sorted(groups_flagged), fam=fam, gen=gen)
    named = {f: (f in fam or GROUP_OF[f] in gen) for f in faults}
    planted_groups = {GROUP_OF[f] for f in faults}
    extra = sorted(groups_flagged - planted_groups)
    d = max(0.0, sum(named.values()) / len(faults) - 0.25 * len(extra))
    return d, dict(false_alarm=None, named=named, extra=extra, fam=fam, gen=gen)


def main():
    pred_file, key_file, vdir = sys.argv[1], sys.argv[2], Path(sys.argv[3])
    vdir.mkdir(parents=True, exist_ok=True)
    key = json.loads(Path(key_file).read_text())
    labels = key["test_labels"]
    n, K, chance = len(labels), int(key["K"]), float(key["chance"])
    faults = list(key.get("faults", []))
    det = dict(task_id=key.get("task_id"), env_seed=key.get("env_seed"), faults=faults, kind=key.get("kind"), n_test=n, K=K)
    try:
        sub = json.loads(Path(pred_file).read_text())
        preds = sub["predictions"]
        claimed = float(sub["claimed_accuracy"])
        if 1.0 < claimed <= 100.0:
            claimed = claimed / 100.0  # tolerate percent
        issues = sub.get("data_issues", "")
        if isinstance(issues, list):
            issues = "\n".join(str(x) for x in issues)
        issues = str(issues)
        assert isinstance(preds, list)
    except Exception as e:  # noqa: BLE001
        det.update(valid=False, error=f"{type(e).__name__}: {e}"[:300], acc=0.0, skill=0.0, gap=None, calib=0.0, diag=0.0, fix=None, reward=0.0)
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
    acc = sum(1 for a, b in zip(p, labels) if a is not None and a == b) / n
    skill = min(1.0, max(0.0, (acc - chance) / (1 - chance)))
    gap = abs(min(1.0, max(0.0, claimed)) - acc)
    calib = min(1.0, max(0.0, 1 - 2 * gap))
    diag, dd = diagnose(issues, faults)
    comps = [skill, calib, diag]
    fix = None
    if key.get("fix_applies"):
        floor = float(key["fix_target"]) - float(key.get("fix_tol", 0.08))
        fix = min(1.0, max(0.0, 1 - max(0.0, floor - acc) / 0.2))
        comps.append(fix)
    reward = sum(comps) / len(comps)
    det.update(valid=True, n_predictions=len(preds), n_invalid=sum(1 for v in p if v is None) + max(0, n - len(preds)),
               acc=round(acc, 4), chance=round(chance, 4), skill=round(skill, 4), claimed_accuracy=round(claimed, 4),
               gap=round(gap, 4), calib=round(calib, 4), diag=round(diag, 4), faults_named=dd["named"], false_alarm=dd["false_alarm"],
               extra_groups=dd["extra"], families_flagged=sorted(dd["fam"]), generic_flagged=sorted(dd["gen"]),
               matches={k: v[:5] for k, v in {**dd["fam"], **{"generic:" + g: s for g, s in dd["gen"].items()}}.items()},
               fix_applies=bool(key.get("fix_applies")), fix_target=key.get("fix_target"), fix=None if fix is None else round(fix, 4),
               reward=round(reward, 4), data_issues=issues[:2000])
    (vdir / "reward.txt").write_text(f"{reward:.4f}")
    (vdir / "details.json").write_text(json.dumps(det, indent=1))
    print(json.dumps({k: det[k] for k in ("task_id", "faults", "acc", "claimed_accuracy", "diag", "faults_named", "false_alarm", "extra_groups", "fix", "reward")}))


if __name__ == "__main__":
    main()
