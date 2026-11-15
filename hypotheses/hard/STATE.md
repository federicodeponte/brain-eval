# HARD tier: brain-reading eval (2026-11-14)

Status: built and validated offline. No model calls made. No new downloads. Mouse sets reuse the cached NWB; the human set reuses the cached data004 counts.
Goal: a simple in-prompt decoder still clearly beats chance, but the task gets much harder for an LLM (Opus tied nearest centroid on the existing sets: human K=4 3-shot 68% vs 65%, mouse K=4 87% vs 87%, mouse K=8 67% vs 67%).

## Picked sets (30 items each, same prompt format, final line "Answer with a single letter.")
| set | config | chance | NC corr (set) | NC euclid (set) | NC corr, 600/400-item estimate | other |
|---|---|---|---|---|---|---|
| mouse-hard | K=8 clips, 1 example/clip, 40 units x 10 bins | 0.125 | 0.500 | 0.567 | 0.558 (euclid 0.538) | NC corr sqrt 0.667, LR 0.733 on the set |
| human-hard | K=4 classes, 1 example/class, 40 neurons | 0.25 | 0.500 raw / 0.533 sqrt (existing gate metric) | 0.567 raw / 0.633 sqrt | 0.482 raw / 0.517 sqrt (euclid 0.512 / 0.542) | held-out LR (240 pseudo-trials) 0.967 |
Letters: mouse A-F 4 each, G/H 3 each; human A/B 8, C/D 7; human query categories 8/7/7/8. Prompt sizes: mouse 8.1k chars, human 2.1k chars.

## Candidate sweep (nearest centroid on in-prompt examples only)
Mouse, 600 items each (mouse_sweep.json):
| K | shots | units | chance | NC corr | NC euclid | NC corr sqrt | LR |
|---|---|---|---|---|---|---|---|
| 8 | 1 | 40 | 0.125 | **0.558** | 0.538 | 0.693 | 0.698 |
| 8 | 1 | 20 | 0.125 | 0.480 | 0.478 | 0.573 | 0.665 |
| 8 | 2 | 40 | 0.125 | 0.677 | 0.608 | 0.788 | 0.808 |
| 10 | 1 | 40 | 0.100 | 0.510 | 0.463 | 0.653 | 0.697 |
| 12 | 1 | 40 | 0.083 | 0.483 | 0.418 | 0.573 | 0.652 |
| 12 | 1 | 20 | 0.083 | 0.390 | 0.408 | 0.528 | 0.635 |
| 8 | 1 | 10 | 0.125 | 0.392 | 0.390 | 0.505 | 0.590 |
Human, 400 items each (human_sweep.json):
| neurons | shots | NC corr sqrt | NC corr | NC euclid sqrt | NC euclid |
|---|---|---|---|---|---|
| 40 | 1 | **0.517** | 0.482 | 0.542 | 0.512 |
| 40 | 2 | 0.595 | 0.598 | 0.637 | 0.600 |
| 20 | 1 | 0.495 | 0.487 | 0.535 | 0.480 |
| 20 | 2 | 0.608 | 0.593 | 0.657 | 0.580 |
| 20 | 3 | 0.685 | 0.680 | 0.713 | 0.662 |
| 10 | 3 | 0.655 | 0.620 | 0.680 | 0.585 |
Why these picks: K=8/1-shot/40 units is in the 45-60% band, stays within letters A-H (no grader/parser change), and changes only the shot count from the existing K=8 set. Human 40 neurons/1 shot changes only the shot count from the main human set. Human build seeds 21/22/23 gave NC corr sqrt 0.567/0.533/0.533 on 30 items; seed 22 was picked as the one closest to the 400-item estimate (all three in report.json).

## Construction
- mouse-hard: same pipeline as tasks_v2 (session 732592105, natural_movie_three, 1 s clips, 10 x 100 ms bins). Per item, the top 40 VISp units by split-half PSTH reliability on the 9 non-query repeats; the single example per clip comes from those repeats; the query repeat is never seen by selection or examples (asserted). Seed 7.
- human-hard: scripts/build_pseudo.py from human-neurons with SEED=22 K=1 M=40 N=30 (selection half picks neurons, item half supplies every row; query trial/picture never in the examples, asserted). Then scripts/human_finalize.py renames ids to human-hard-NNN, adds k=4, and recomputes all decoders from the prompt rows.

## Files
- mouse-hard/, human-hard/: tasks.jsonl + report.json (meta holds decoder predictions; no answer-revealing fields go into the prompt)
- mouse_sweep.json, human_sweep.json; scripts/mouse_hard.py (sweep | build K NEX U, run from ~/work/vals-mouse with .venv), scripts/human_sweep.py, scripts/human_finalize.py
- Rebuild human: cd ../human-neurons; . ./env.sh; SEED=22 K=1 M=40 N=30 .venv/bin/python -I scripts/build_pseudo.py $PWD/scripts ../hard/human-hard/cand_seed22.jsonl; cd ../hard; python3 -I scripts/human_finalize.py $PWD 22 22

## Runners
- Valkyrie: ~/work/vals-mouse/valkyrie/dataset-mouse-hard, dataset-human-hard (build_dataset.py). Grader checked: all 60 "Answer: **X**" oracle replies score 1; letters are A-H (mouse) and A-D (human); "I" on a k=8 task is unparsed, as designed. Push: valkyrie dataset push hard/<name> ./dataset-<name> --tag v1
- Pilot: pilot.py now picks up hard/*/tasks.jsonl as --hyp mouse-hard / human-hard (backup pilot.py.bak-hard). Its parse() covers A-H, so both sets parse.

## Caveats
- n=30: one item is 3.3 points; a 95% CI near 0.5 is about +/-18 points. Compare models to the decoder on the same items (meta.nc_corr_pred / nc_correlation_pred).
- With 1 shot, NC = 1-NN. sqrt-count correlation and LR are clearly stronger (mouse 0.67-0.73, human held-out LR 0.97), so the signal is there for a smarter in-context reader.
- K=10/12 were not picked; they would need letters beyond H in pilot.py parse() (grade.py already handles any k).
