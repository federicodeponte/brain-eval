# Can a model read a brain?

Vals AI Hackathon (Nov 14-15 2026), Build Your Own Eval track, built on [Valkyrie](https://github.com/vals-ai/Valkyrie).

Frontier models get raw spike counts from real neurons plus a few labeled examples in the prompt, and must decode what the animal or person was seeing. A 10-line decoder (nearest centroid on the same in-prompt examples) is the baseline.

## Main tasks
| Task | Data | Chance | Decoder | Haiku | Opus (pilot, n=10) |
|---|---|---|---|---|---|
| Human neurons: car / food / person / animal photo | DANDI 000004, 40 amygdala/hippocampus neurons, epilepsy patients | 25% | 65-78% | 5/10 | 5/10 |
| Mouse: which of 4 movie clips | Allen Neuropixels 732592105, 40 VISp units x 10 bins | 25% | 82-97% | 6/10 | 8/10 |
| Mouse: which of 8 movie clips | same | 12.5% | 83-87% | - | 7/10 |

## Controls (does the model learn from the examples?)
In `hypotheses/controls/` and `mouse/valkyrie/dataset-*`:
- **swap-out**: true clip examples replaced by a clip from another stimulus; correct = E) none of these
- **label-swap**: human category names permuted; does the model follow the examples or its priors?
- **shot-curve**: 0/1/3/5 examples per option; learning curve
- **null**: shuffled / Poisson data, expect chance
- **planted**: obvious planted pattern, expect ~100% (harness check)

## Layout
- `mouse/` build scripts, tasks (`tasks_v2.jsonl` = units chosen from training repeats only), `valkyrie/` datasets, one-shot MCQ agent, grader, leaderboard
- `hypotheses/human-neurons/` human set; `controls/` learning controls; `pilot/` claude -p pilot runner + results
- `hypotheses/*` other explored hypotheses (taste, yield, protein, connectome, room, whale, heartbeat, sleep, pulse) with STATE.md each

## Run on Valkyrie
See `mouse/valkyrie/STATE.md`: `valkyrie dataset push ...`, `valkyrie agent push mouse/valkyrie/agents/oneshot-mcq`, `valkyrie run start ...`, `python3 leaderboard.py results-*.json --md`.

Raw recordings are not included (download scripts in each folder). Data licenses: Allen Institute terms, DANDI CC-BY-4.0, PhysioNet ODC-By.
