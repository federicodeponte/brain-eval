# Broken-instrument decoding tier (state 2026-11-15, AX41 ~/work/vals-hyp/broken)

An agent gets real spike recordings (train + val labeled, test unlabeled), builds a decoder, writes test predictions,
its own `claimed_accuracy`, and `data_issues`. Each base recording has a CLEAN twin and a DEFECT twin with ONE planted
fault. The grader holds the hidden test labels. Agentic harness: Codex (codex-v1.0.0) on Modal sandboxes.

## Design
- 12 base recordings x (clean, defect) = **24 tasks**, faults balanced 4/4/4.
  - mouse (Allen Visual Coding Neuropixels, session 732592105, VISp, random unit subsets, local NWB):
    natural_movie_one segments 0-9 / 10-19 / 20-29 (m-nm1-a/b/c), natural_movie_three seconds 60-69 (m-nm3-a),
    natural scenes 3 random 8-image sets (m-ns-a/b/c), drifting-grating direction, 8 classes (m-dg).
  - human (DANDI 000004 MTL single neurons, 4-category pseudo-population, top-F units picked on train+val only):
    houses image set x2 (h-houses, h-houses-b), cars set (h-cars-b), fruit set (h-fruit).
- Splits are fixed per base before any fault is planted: movies split by whole repeats (interleaved over the two
  stimulus blocks), images/gratings stratified per class, human split by picture per session (a picture never spans
  splits). Human train/val pseudo-rows never reuse a trial; test rows are independent draws from test-only trials.
- Faults (defect twin only; hidden test labels are always correct and the test trials are untouched except for dead):
  | fault | bases | what is planted | visible evidence |
  |---|---|---|---|
  | leak | m-ns-a, m-dg, h-houses, h-houses-b | val = jittered near-copies of train trials; every train trial appears twice (+-1 count jitter on ~10% of entries) | near-zero nearest-neighbour distances train<->val and within train; train is 2x the clean size |
  | shift | m-nm1-a/b/c, m-nm3-a | train+val labels lag one segment: label = true segment - 1 (cyclic). `movie_time_s` stays correct | label disagrees with `movie_time_s` by 1 s on every trial |
  | dead | m-ns-b, m-ns-c, h-cars-b, h-fruit | top third of units by ANOVA F (train+val) are all zero in test.npz only | those rows are 0 in every test trial, active in train/val |
  Shift needs an ordered stimulus, so it lives on the movie bases only (human photo order is random).
- Every task ships `X`, `y`, and one stimulus-log column (`movie_time_s`, `image_id`, `orientation_deg`, `category`)
  for train/val; test has `X` only. Instruction only says "Report any problems you find with the data"; ids are
  neutral (`decode-NNa/b`, clean letter random per base). The fault/twin mapping is in tests/key.json (uploaded only
  at grading) and in tasks.jsonl here.

## Fault validation (naive pipeline: log1p + z-score + logistic regression C=1; NC = nearest centroid)
"val" = fit train, score provided val. "CV" = 5-fold CV on train+val. "hidden test" = fit train+val, score hidden test.
"aware" = same LR after undoing the fault with agent-visible data only (dedupe / relabel from movie_time_s / drop silent units).

| base | fault | twin | K | n tr/val/test | LR val | LR 5-fold CV | LR hidden test | gap (CV-test) | NC CV | NC test | aware test |
|---|---|---|---|---|---|---|---|---|---|---|---|
| m-nm1-a | shift | clean | 10 | 120/30/50 | 0.83 | 0.79 | 0.78 | +0.01 | 0.79 | 0.74 | 0.78 |
| m-nm1-a | shift | defect | 10 | 120/30/50 | 0.83 | 0.79 | 0.08 | +0.71 | 0.79 | 0.10 | 0.78 |
| m-nm1-b | shift | clean | 10 | 120/30/50 | 0.93 | 0.87 | 0.80 | +0.07 | 0.81 | 0.70 | 0.80 |
| m-nm1-b | shift | defect | 10 | 120/30/50 | 0.93 | 0.87 | 0.06 | +0.81 | 0.81 | 0.08 | 0.80 |
| m-nm1-c | shift | clean | 10 | 120/30/50 | 0.63 | 0.66 | 0.62 | +0.04 | 0.61 | 0.62 | 0.62 |
| m-nm1-c | shift | defect | 10 | 120/30/50 | 0.63 | 0.66 | 0.10 | +0.56 | 0.61 | 0.08 | 0.62 |
| m-nm3-a | shift | clean | 10 | 60/10/30 | 0.90 | 0.87 | 0.80 | +0.07 | 0.83 | 0.83 | 0.80 |
| m-nm3-a | shift | defect | 10 | 60/10/30 | 0.90 | 0.87 | 0.03 | +0.84 | 0.83 | 0.03 | 0.80 |
| m-ns-a | leak | clean | 8 | 200/80/120 | 0.53 | 0.56 | 0.56 | +0.00 | 0.58 | 0.50 | 0.56 |
| m-ns-a | leak | defect | 8 | 400/80/120 | 0.91 | 0.86 | 0.49 | +0.36 | 0.66 | 0.51 | 0.47 |
| m-dg | leak | clean | 8 | 320/120/152 | 0.58 | 0.60 | 0.65 | -0.05 | 0.58 | 0.63 | 0.65 |
| m-dg | leak | defect | 8 | 640/120/152 | 1.00 | 0.91 | 0.60 | +0.31 | 0.71 | 0.59 | 0.60 |
| h-houses-b | leak | clean | 4 | 86/16/200 | 0.69 | 0.80 | 0.77 | +0.03 | 0.83 | 0.81 | 0.77 |
| h-houses-b | leak | defect | 4 | 172/16/200 | 1.00 | 0.97 | 0.70 | +0.27 | 0.83 | 0.81 | 0.70 |
| h-houses | leak | clean | 4 | 86/16/200 | 0.81 | 0.83 | 0.77 | +0.07 | 0.85 | 0.77 | 0.77 |
| h-houses | leak | defect | 4 | 172/16/200 | 1.00 | 0.97 | 0.73 | +0.24 | 0.88 | 0.75 | 0.72 |
| m-ns-b | dead | clean | 8 | 200/80/120 | 0.53 | 0.57 | 0.51 | +0.06 | 0.59 | 0.49 | 0.51 |
| m-ns-b | dead | defect | 8 | 200/80/120 | 0.53 | 0.57 | 0.26 | +0.31 | 0.59 | 0.25 | 0.33 |
| m-ns-c | dead | clean | 8 | 200/80/120 | 0.50 | 0.56 | 0.53 | +0.03 | 0.62 | 0.55 | 0.53 |
| m-ns-c | dead | defect | 8 | 200/80/120 | 0.50 | 0.56 | 0.21 | +0.35 | 0.62 | 0.20 | 0.28 |
| h-cars-b | dead | clean | 4 | 87/16/200 | 0.81 | 0.93 | 0.89 | +0.05 | 0.97 | 0.89 | 0.89 |
| h-cars-b | dead | defect | 4 | 87/16/200 | 0.81 | 0.93 | 0.33 | +0.60 | 0.97 | 0.56 | 0.63 |
| h-fruit | dead | clean | 4 | 70/17/200 | 0.82 | 0.85 | 0.74 | +0.11 | 0.81 | 0.76 | 0.74 |
| h-fruit | dead | defect | 4 | 70/17/200 | 0.82 | 0.85 | 0.25 | +0.60 | 0.81 | 0.25 | 0.45 |

Summary (LR, mean over the 4 bases per cell):
| | leak | shift | dead |
|---|---|---|---|
| clean: CV - test | +0.01 | +0.05 | +0.06 |
| defect: CV - test | **+0.30** | **+0.73** | **+0.47** |
| defect: val - test | +0.35 | +0.76 | +0.40 |
Nearest centroid is nearly immune to the leak (duplicates do not move centroids), so the leak gap shows up for
flexible models (LR, kNN, trees), which is what agents reach for. Dead-unit recovery is partial by construction
(the best third of the units is gone), shift is fully recoverable (relabel from movie_time_s).

## Grader (tests/grade.py, stdlib, deterministic, no LLM)
Reads `/app/output/predictions.json` = {"predictions": [int...], "claimed_accuracy": 0-1, "data_issues": str}.
- acc on hidden labels; skill = clip((acc - chance)/(1 - chance), 0, 1)
- gap = |claimed - acc| (claimed clipped to [0,1]; 0-100 is read as percent); calib = clip(1 - 2*gap, 0, 1)
- diag: defect twin = 1 if `data_issues` names the planted family; clean twin = 1 if it flags no family (no false alarm)
- **reward = (skill + calib + diag) / 3**; malformed/missing file = 0. details.json carries acc, claimed, gap, calib,
  fault_named / false_alarm, families_flagged, matched snippets, reward.
- Keyword families: leak (duplicate, near-copy, identical, overlap, leak, contamination...), shift (shift, misalign,
  offset, lag, off-by-one, mislabel, labels inconsistent with movie time...), dead (dead, silent, all zeros, no spikes,
  inactive, dropout, missing units, distribution/covariate shift). Mentions preceded by a negation within 4 words
  ("no duplicates found", "did not find", "no dead channels") are ignored; "distribution shift" alone counts as dead,
  not shift. Limitation: a keyword list, so unusual phrasing can be missed; matched snippets are logged for audit.
- test_grader.py: oracle 24/24 = 1.0, 16 keyword/negation cases, 3 malformed cases, all pass.

## Files
- extract.py -> bases/*.npz|json (12 bases, fixed splits). twins.py: PLAN, make_twin, validation (validation.json).
- build_tasks.py -> ~/work/vals-mouse/valkyrie/dataset-broken (24 Harbor tasks: task.toml, instruction.md,
  environment/Dockerfile + environment/data/{train,val,test}.npz + classes.json, tests/{test.sh,grade.py,key.json},
  solution/solve.sh oracle) and tasks.jsonl (task_id -> base, fault, twin).
- Image: python:3.12-slim + numpy 2.2.6, scipy 1.15.3, scikit-learn 1.6.1, pandas 2.2.3; data COPY'd to /app/data.
  Built and loaded locally with docker. All 24 pass the tracker's validate_task (0 errors, 0 warnings). 1.9 MB total.
- Rebuild: `cd ~/work/vals-hyp/broken; P=~/work/vals-mouse/.venv/bin/python; $P extract.py; $P -W ignore twins.py; $P -W ignore build_tasks.py; python3 test_grader.py ~/work/vals-mouse/valkyrie/dataset-broken`

## Valkyrie
```
export PATH=$HOME/.local/bin:$PATH; unset VALKYRIE_URL; cd ~/work/vals-mouse/valkyrie
valkyrie agent install https://github.com/vals-ai/public-agent-registry/tree/main/examples/codex-v1.0.0   # done
valkyrie dataset push brain/broken ./dataset-broken --tag v1                                            # done: brain/broken@1
```
Dataset versions: brain/broken@1 (v1), @2 (v2: instruction says labels come from the stimulus log; grader
"does not match" fix), **@3 = v3 = latest (grader: shifted test distribution counts as dead-units)**. Use v3.

### Smoke runs (Codex codex-v1.0.0, gpt-5.3-codex, Modal, concurrency 2)
| run id | dataset | task | twin/fault | acc | claimed | gap | diag | reward | what the agent wrote |
|---|---|---|---|---|---|---|---|---|---|
| 28a3db9b-f3ce-415a-9ab6-e729637647e1 | v1 0:2 | decode-01a | clean/shift | 0.78 | 0.87 | 0.09 | false alarm | 0.53 | "label leakage/tautology: y equals movie_time_s" |
| 28a3db9b | v1 0:2 | decode-01b | defect/shift | 0.08 | 0.84 | 0.76 | missed (v1 grader) | 0.00 | "movie_time_s does not match labels" (v2+ grader credits this) |
| 04bbb614-9e61-4e9d-a44e-b3d65d171516 | v2 0:2 | decode-01a | clean/shift | 0.74 | 0.87 | 0.13 | ok ("none") | 0.82 | none |
| 04bbb614 | v2 0:2 | decode-01b | defect/shift | 0.04 | 0.87 | 0.83 | named | 0.33 | "y inconsistent with movie_time_s ... misaligned", but trained on the bad labels anyway |
| a290e8a2-3a4f-414c-b381-4fd980d55aaf | v2 ids | decode-05b | defect/leak | 0.58 | 0.95 | 0.38 | missed | 0.25 | "none" |
| a290e8a2 | v2 ids | decode-12a | defect/dead | 0.48 | 0.73 | 0.25 | v2: wrong family; v3: named | 0.27 (v2) | "test distribution strongly shifted vs train, zeros 51.6% vs 25%" |
| **332684da-5d01-454b-9b49-30b7f259cc64** | **v3 0:2** | decode-01a | clean/shift | 0.78 | 0.82 | 0.04 | false alarm | 0.56 | "y == int(movie_time_s) ... potential leakage-prone field" |
| 332684da | v3 0:2 | decode-01b | defect/shift | 0.76 | 0.88 | 0.12 | named | 0.83 | caught the off-by-one, relabeled from movie_time_s |
The grader parsed every hosted output (verifier exit 0, details.json written). On every defect twin the agent
overclaimed, except the v3 01b run where it found and fixed the fault. The clean movie twin keeps getting flagged
as "leakage" because y is a function of movie_time_s (2/3 runs). The grader counts that as a false alarm, which is
the spec: the field is documented and test has none. Open call: keep it as an over-reporting signal, or reword the
instruction further if a full run shows that most models false-alarm on it.
`valkyrie run regrade` does not work for these runs ("no tasks with saved evaluation state"), so grader changes
were checked on the exact hosted strings in test_grader.py.

Cost (metrics_total.json, gpt-5.3-codex): 8 tasks, $0.032-0.049 each, mean ~$0.041/task, ~48-62k input tokens
(mostly cached), 1.2-2.0k output tokens, 14-25 s agent time, ~50 s sandbox build. Smoke spend so far ~$0.33.

### Full run (NOT launched)
```
valkyrie run start --agent codex-v1.0.0 --dataset brain/broken@v3 --project brain --model gpt-5.3-codex \
  --provider modal -s OPENAI_API_KEY openai-api-key --concurrency 8 -l broken-v3-full
```
Estimate: 24 tasks x ~$0.041 = **~$1.0 per model per pass** (-n 3 for three passes = ~$3). Other contract models:
gpt-5.4, gpt-5.2-codex, gpt-5.2, gpt-5.1-codex-max, gpt-5.1-codex-mini (pass bare names).
Results: `valkyrie run results RUN --path r.json`; per-task components are in each task's verifier_output.

## Gotchas
- Codex contract model ids are bare (`--model gpt-5.3-codex`). `--model openai/gpt-5.3-codex` gave
  "Internal Server Error" on run start. I did not retry that form, so the error may have been transient.
- Codex installs node + codex CLI + uv in the sandbox at start (setup.sh), so each task pays ~1-2 min of setup.
