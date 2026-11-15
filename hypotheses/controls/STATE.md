# Control sets for "Can a model read a brain?" (2026-11-14)

Status: built and validated offline. No model calls made. No new downloads (reused data/session_732592105.nwb; natural_movie_one is in the same NWB).

## Sets
| set | dir | items | k | chance | decoder baseline (in-prompt examples only) | result that shows learning |
|---|---|---|---|---|---|---|
| main v2 K=4 | ~/work/vals-mouse/tasks_v2.jsonl | 60 | 4 | 0.25 | NC 0.817, LR 0.967 (v1: 0.867 / 0.933) | well above 0.25 |
| main v2 K=8 | same file | 30 | 8 | 0.125 | NC 0.833, LR 0.867 (v1: 0.667 / 0.767) | well above 0.125 |
| swap-out | swap-out/ | 40 (20 normal, 10 swap NM1, 10 swap NM3) | 5 | 0.20 uniform; always-E 0.50; never-E ceiling 0.50 | NC + distance threshold: 0.675 (in-prompt tau), 0.70 (sqrt counts, tau tuned on 400 separate items) | > 0.50, with E chosen on swap items AND the right clip on normal items |
| label-swap | label-swap/ | 40 | 4 | 0.25 | NC: 0.65 example-consistent, 0.125 prior-consistent | example-consistent rate >> prior-consistent rate |
| shot-curve | shot-curve/ | 160 (40 per level) | 4 | 0.25 | NC 0.25 / 0.65 / 0.925 / 0.925 at 0/1/3/5 shots (LR 0.95 at 3 and 5) | ~0.25 at 0 shots, rising with shots |
| null | null/ | 40 (20 shuffled-label real trials, 20 Poisson) | 4 | 0.25 | NC 0.25 | ~0.25 (above chance = harness leak or letter bias) |
| planted | planted/ | 40 | 4 | 0.25 | NC 1.00 | ~1.00 (harness and parsing work) |

## Construction
- Units (all mouse sets): per item, top 40 VISp units by split-half PSTH reliability on the 9 non-query repeats of natural_movie_three. The query repeat is never seen by selection or examples. `unit_ids` per item.
- swap-out: K=4 options + "E) none of these" (k=5). Swap items replace the true clip's 3 examples with either a natural_movie_one 1 s clip (3 of its 20 repeats) or a different natural_movie_three clip not in the item. Prompt says the query may match none. Normal letters A-D 5 each, E 20. Threshold rule: E if correlation distance to the nearest centroid > tau.
- label-swap: the human tasks (DANDI 000004) with a per-item derangement of the category names applied consistently to example headers and options. Spikes and letters unchanged. Answer = example-consistent. meta.answer_prior_consistent = letter showing the true query category (never equal to the answer).
- shot-curve: same 40 queries at 0/1/3/5 shots, nested examples (1 in 3 in 5). 0-shot prompt lists only clip letters. Field `shots` and `base_item`.
- null: shuffled-label = every example and the query are real trials from 13 distinct random clips. Poisson = draws at real unit x bin mean rates. planted: Poisson background + each clip has its own block (4 neurons x 3 bins, +6 spikes/bin); the query carries the answer block.
- Answer-revealing fields (swap_type, prior answer, nc_pred, planted blocks) live in the nested `meta` dict, so they stay out of task.toml.

## Valkyrie
- ~/work/vals-mouse/valkyrie/dataset-{swap-out,label-swap,shot-curve,null,planted,v2}/ built with build_dataset.py.
- grade.py takes E when k=5 (letters = first k). Checked: oracle 410/410, "E" and "Answer: **E**" score 1 on a swap task, "Answer: F" is unparsed.
- Push: `valkyrie dataset push controls/<name> ./dataset-<name> --tag v1`
- Scoring the controls: per-task predicted letters are in the verifier details.json (valkyrie run results). Join on id with tasks.jsonl to get the label-swap prior rate, shot-curve by `shots`, and swap-out by meta.swap_type.

## Files
- scripts/mouse_controls.py (run from ~/work/vals-mouse: `.venv/bin/python ~/work/vals-hyp/controls/scripts/mouse_controls.py`). It writes all mouse sets plus tasks_v2.jsonl and build_report_v2.json.
- scripts/label_swap.py (`python3 -I scripts/label_swap.py ../human-neurons/tasks.jsonl label-swap/tasks.jsonl`)
- <set>/tasks.jsonl, <set>/report.json

## Caveats
- swap-out NM1 examples come from a different stimulus block. Rate or adaptation differences could let a model flag the "odd" option. The NM3 swaps (the same movie, a clip outside the item) guard against that, so compare the E rate by swap type.
- n=40 per set: one item is 2.5 points, and a 95% CI is about ±15 points near 0.5.
- The v2 baselines differ from v1 partly by sampling (new items), not only by the selection change.

## HARD tier
- ~/work/vals-hyp/hard/ (mouse-hard, human-hard, 1-shot). See its STATE.md.
