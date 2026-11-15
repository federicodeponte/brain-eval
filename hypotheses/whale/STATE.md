# speak-whale: sperm whale coda type from inter-click intervals

Status: GO (data + validity done, no model calls yet)

## Source + license
- DominicaCodas.csv (8,719 codas, Dominica Sperm Whale Project 2005-2010), from github.com/pratyushasharma/sw-combinatoriality/data
  (Sharma et al. 2024, Nature Communications). Same coda set as Gero, Whitehead & Rendell 2016 (R. Soc. Open Sci.), whose data is on Dryad (CC0).
  The GitHub repo has no license file; cite both papers. Local copy: raw/DominicaCodas.csv.

## Construction (build.py, fixed seeds)
- 4-way: 5-click coda types 1+1+3, 5R1, 5R2, 5R3 (same click count, so counting clicks gives nothing; rhythm and tempo decide).
- Prompt: 24 labeled few-shot codas (6 per type, ICIs in ms) + 1 query coda. 40 test items, 10 per class, letters A-D 10 each.
- Shots and tests are disjoint codas; the remaining codas form a pool used only for pool baselines. Prompts ~1.25k chars.
- Rebuild: `../_tools/.v4/bin/python -I build.py raw/DominicaCodas.csv`

## Validity
| method | acc |
|---|---|
| chance | 0.25 |
| 1-NN on the 24 in-prompt shots only | 1.00 |
| logistic regression on in-prompt shots | 1.00 |
| kNN-5 / random forest on held-out pool | 1.00 / 1.00 |

## Caveats
- Baseline at ceiling: the information is fully in the prompt and a 1-NN solves it. This is a clean "can the model do nearest-neighbour
  reasoning over 4 numbers" test. A frontier model failing is a strong demo; passing is plausible.
- Type names (1+1+3, 5R3) describe rhythm, so a model can partly use prior knowledge. 5R3 is the EC2 clan identity coda; 1+1+3 the EC1 one.
- Memorization: low risk for per-coda labels.
- DROPPED harder variant (build_hard.py, dropped_tasks_hard_unitid.jsonl): which family unit (A/F/J/U) made a 5R1 coda. Random-split CV
  looked fine (0.55), but with recording dates disjoint between shots and tests the baselines fall to 0.25-0.35, so unit identity is
  mostly recording-session confounded. Not valid as a 4-way eval.
